"""
finetune_qwenvl.py — Qwen2.5-VL LoRA Fine-tuning for Fruit-Clearing VLM Tasks

This script fine-tunes Qwen2.5-VL-7B-Instruct using LoRA (via `peft`) for the
Multi-Fruit Table Clearing task. It supports two training stages:

  Stage 1: Train only on subtask completion verification data (Requirement 3).
  Stage 2: Jointly train on verification + scanning + termination judgment data
           (Requirements 3 + 4).

Key components:
  - QwenVLFinetuneConfig  : dataclass holding all hyperparameters and paths.
  - QwenVLConversationDataset : PyTorch Dataset that loads conversations from
    JSONL, applies the Qwen2.5-VL chat template, processes images, and builds
    labels with prompt positions masked to -100.
  - finetune_qwenvl()     : main training loop (LoRA + AdamW + W&B logging).

Usage:
    python vla-scripts/finetune_qwenvl.py \\
        --pretrained_checkpoint Qwen/Qwen2.5-VL-7B-Instruct \\
        --stage 1 \\
        --train_data_path data/vlm/stage1_train.jsonl \\
        --val_data_path   data/vlm/stage1_val.jsonl

Requirements: 6.1, 6.2, 6.3
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Optional heavy dependencies — handled gracefully so the module can be
# imported even in environments where the GPU stack is not installed.
# ---------------------------------------------------------------------------

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Provide a stub so type annotations don't break at import time.
    class Dataset:  # type: ignore[no-redef]
        pass

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

try:
    from qwen_vl_utils import process_vision_info
    _QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    _QWEN_VL_UTILS_AVAILABLE = False
    process_vision_info = None  # type: ignore[assignment]

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Suppress tokenizer parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class QwenVLFinetuneConfig:
    """
    Configuration for Qwen2.5-VL LoRA fine-tuning.

    All fields have sensible defaults that match the design document
    (Requirement 6.1).
    """

    # ── Model ────────────────────────────────────────────────────────────
    pretrained_checkpoint: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # ── Data ─────────────────────────────────────────────────────────────
    stage: int = 1                  # 1 = completion only; 2 = all tasks
    train_data_path: str = ""       # Path to train conversations JSONL
    val_data_path: str = ""         # Path to val conversations JSONL

    # ── LoRA hyperparameters (Requirement 6.1) ───────────────────────────
    lora_rank: int = 16
    learning_rate: float = 1e-4
    batch_size: int = 4
    max_steps: int = 5000
    save_steps: int = 500
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # ── Output ───────────────────────────────────────────────────────────
    run_root_dir: str = "runs/qwen_vl"
    wandb_project: str = "fruit-clearing-vlm"
    run_id_note: Optional[str] = None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QwenVLConversationDataset(Dataset):
    """
    Dataset for Qwen2.5-VL fine-tuning.

    Loads conversations from a list of dicts (each with a ``"messages"`` key
    in Qwen2.5-VL conversation format), applies the chat template via
    ``processor.apply_chat_template``, processes images, and builds labels
    with prompt token positions masked to ``-100`` so that the loss is
    computed only on the assistant response tokens.

    Each conversation dict is expected to follow the format produced by
    ``vqa_sample_to_conversation()`` in ``data/build_vlm_dataset.py``
    (Requirement 6.2):

    .. code-block:: json

        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "<path_or_url>"},
                        {"type": "image", "image": "<path_or_url>"},
                        {"type": "text",  "text": "<question>"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "<answer>"}]
                }
            ]
        }

    Parameters
    ----------
    conversations : List[dict]
        List of conversation dicts loaded from a JSONL file.
    processor : AutoProcessor or None
        Qwen2.5-VL processor.  When ``None`` (e.g. in unit tests without GPU
        dependencies), ``__getitem__`` returns a placeholder dict.
    """

    def __init__(self, conversations: List[dict], processor) -> None:
        self.conversations = conversations
        self.processor = processor

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.conversations)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with:
            input_ids      : torch.Tensor  — token IDs for the full conversation
            attention_mask : torch.Tensor  — 1 for real tokens, 0 for padding
            pixel_values   : torch.Tensor  — processed image tensor(s)
            labels         : torch.Tensor  — same as input_ids but with prompt
                             positions (user turn + system tokens) masked to -100
        """
        conversation = self.conversations[idx]
        messages: List[dict] = conversation.get("messages", [])

        # ── Fallback: processor not available ─────────────────────────
        if self.processor is None or not _TORCH_AVAILABLE:
            return self._placeholder_item()

        try:
            return self._process_item(messages)
        except Exception as exc:  # noqa: BLE001
            # Gracefully degrade to a placeholder so DataLoader doesn't crash
            # on a single bad sample.
            print(f"[QwenVLConversationDataset] Warning: failed to process "
                  f"item {idx}: {exc}")
            return self._placeholder_item()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_images_from_messages(self, messages: List[dict]) -> List:
        """
        Extract image paths from the messages content and load them as PIL
        Images.  Returns a list of PIL Image objects (one per image block).
        """
        if not _PIL_AVAILABLE:
            return []

        images = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                # Plain-text content — no images
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "image":
                    img_ref = block.get("image", "")
                    if not img_ref:
                        continue
                    try:
                        if isinstance(img_ref, str) and os.path.isfile(img_ref):
                            images.append(_PILImage.open(img_ref).convert("RGB"))
                        elif hasattr(img_ref, "convert"):
                            # Already a PIL Image
                            images.append(img_ref.convert("RGB"))
                        # else: URL or base64 — skip for now (handled by
                        # process_vision_info when available)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[QwenVLConversationDataset] Could not load "
                              f"image '{img_ref}': {exc}")
        return images

    def _build_labels(
        self,
        input_ids: "torch.Tensor",
        prompt_length: int,
    ) -> "torch.Tensor":
        """
        Build labels tensor: copy input_ids, then mask the first
        ``prompt_length`` positions (the user/system prompt) to -100 so the
        loss is only computed on the assistant response.
        """
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        return labels

    def _process_item(self, messages: List[dict]) -> dict:
        """
        Full processing pipeline for a single conversation.

        Steps
        -----
        1. Apply chat template to get the full text (with generation prompt).
        2. Apply chat template *without* generation prompt to measure the
           prompt length (for label masking).
        3. Load images as PIL Images.
        4. Use ``process_vision_info`` (if available) or pass PIL images
           directly to the processor.
        5. Tokenize via the processor.
        6. Build labels with prompt positions masked to -100.
        """
        processor = self.processor

        # ── Step 1 & 2: apply chat template ───────────────────────────
        # Full text (prompt + generation prompt marker)
        full_text: str = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Prompt-only text (no generation prompt) — used to measure length
        prompt_text: str = processor.apply_chat_template(
            messages[:-1],          # exclude the assistant turn
            tokenize=False,
            add_generation_prompt=True,
        )

        # ── Step 3: load images ────────────────────────────────────────
        if _QWEN_VL_UTILS_AVAILABLE and process_vision_info is not None:
            # process_vision_info handles paths, URLs, and PIL images
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            # Fallback: load PIL images manually
            image_inputs = self._load_images_from_messages(messages) or None
            video_inputs = None

        # ── Step 4 & 5: tokenize ──────────────────────────────────────
        processor_kwargs: dict = dict(
            text=[full_text],
            padding=True,
            return_tensors="pt",
        )
        if image_inputs is not None:
            processor_kwargs["images"] = image_inputs
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs

        encoding = processor(**processor_kwargs)

        input_ids: "torch.Tensor" = encoding["input_ids"][0]          # (seq_len,)
        attention_mask: "torch.Tensor" = encoding["attention_mask"][0] # (seq_len,)

        # pixel_values may be a Tensor or a list of Tensors (multi-image)
        pixel_values = encoding.get("pixel_values")
        if pixel_values is not None and hasattr(pixel_values, "__getitem__"):
            # Some processor versions return a list; keep as-is for collation
            if not isinstance(pixel_values, list):
                pixel_values = pixel_values[0]  # (num_patches, C, H, W) or similar

        # ── Step 6: build labels ───────────────────────────────────────
        # Tokenize the prompt-only text to find the boundary
        prompt_encoding = processor(
            text=[prompt_text],
            padding=False,
            return_tensors="pt",
        )
        prompt_length: int = prompt_encoding["input_ids"].shape[1]

        labels = self._build_labels(input_ids, prompt_length)

        result: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values

        return result

    def _placeholder_item(self) -> dict:
        """
        Return a minimal placeholder dict when the processor is unavailable
        or an error occurs.  Allows the DataLoader to continue without
        crashing.
        """
        if _TORCH_AVAILABLE:
            return {
                "input_ids": torch.zeros(1, dtype=torch.long),
                "attention_mask": torch.zeros(1, dtype=torch.long),
                "pixel_values": torch.zeros(1, 3, 224, 224, dtype=torch.float32),
                "labels": torch.full((1,), -100, dtype=torch.long),
            }
        # Absolute fallback (no torch)
        return {
            "input_ids": [0],
            "attention_mask": [0],
            "pixel_values": None,
            "labels": [-100],
        }


# ---------------------------------------------------------------------------
# Utility: load conversations from JSONL
# ---------------------------------------------------------------------------

def load_conversations(jsonl_path: str) -> List[dict]:
    """
    Load a list of conversation dicts from a JSON Lines file.

    Each line must be a valid JSON object with a ``"messages"`` key.

    Parameters
    ----------
    jsonl_path : str
        Path to the ``.jsonl`` file.

    Returns
    -------
    List[dict]
        Parsed conversation dicts.
    """
    conversations: List[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                conversations.append(obj)
            except json.JSONDecodeError as exc:
                print(f"[load_conversations] Skipping malformed line {lineno} "
                      f"in '{jsonl_path}': {exc}")
    return conversations


# ---------------------------------------------------------------------------
# Experiment ID helper
# ---------------------------------------------------------------------------

def build_exp_id(cfg: QwenVLFinetuneConfig) -> str:
    """Construct a unique experiment identifier string from config."""
    model_name = Path(cfg.pretrained_checkpoint).name or \
        cfg.pretrained_checkpoint.split("/")[-1]
    exp_id = (
        f"{model_name}"
        f"+stage{cfg.stage}"
        f"+b{cfg.batch_size}"
        f"+lr-{cfg.learning_rate}"
        f"+lora-r{cfg.lora_rank}"
    )
    if cfg.run_id_note:
        exp_id += f"--{cfg.run_id_note}"
    return exp_id


# ---------------------------------------------------------------------------
# Main fine-tuning loop
# ---------------------------------------------------------------------------

def finetune_qwenvl(cfg: QwenVLFinetuneConfig) -> None:
    """
    Main fine-tuning loop for Qwen2.5-VL.

    Uses ``transformers`` + ``peft`` (LoRA).  Supports two-stage training
    (Requirement 6.3):

    * Stage 1 — trains only on subtask completion verification data.
    * Stage 2 — jointly trains on verification + scanning + termination data.

    Parameters
    ----------
    cfg : QwenVLFinetuneConfig
        All hyperparameters and paths.
    """
    assert _TORCH_AVAILABLE, "PyTorch is required for fine-tuning."
    assert _TRANSFORMERS_AVAILABLE, "transformers is required for fine-tuning."
    assert _PEFT_AVAILABLE, "peft is required for LoRA fine-tuning."
    assert torch.cuda.is_available(), "Fine-tuning requires at least one GPU."

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    exp_id = build_exp_id(cfg)
    run_dir = Path(cfg.run_root_dir) / exp_id
    adapter_dir = run_dir / "adapter"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    print(f"[finetune_qwenvl] Stage {cfg.stage} fine-tuning")
    print(f"  Model      : {cfg.pretrained_checkpoint}")
    print(f"  Train data : {cfg.train_data_path}")
    print(f"  Val data   : {cfg.val_data_path}")
    print(f"  Run dir    : {run_dir}")

    # ------------------------------------------------------------------
    # Load processor and model
    # ------------------------------------------------------------------
    print("[*] Loading processor ...")
    processor = AutoProcessor.from_pretrained(
        cfg.pretrained_checkpoint,
        trust_remote_code=True,
    )

    print("[*] Loading model ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ------------------------------------------------------------------
    # LoRA wrapping (Requirement 6.1)
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        init_lora_weights="gaussian",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    assert cfg.train_data_path, "train_data_path must be set."
    train_conversations = load_conversations(cfg.train_data_path)
    train_dataset = QwenVLConversationDataset(train_conversations, processor)

    val_dataset = None
    if cfg.val_data_path:
        val_conversations = load_conversations(cfg.val_data_path)
        val_dataset = QwenVLConversationDataset(val_conversations, processor)

    def _collate_fn(batch: List[dict]) -> dict:
        """Simple collate: stack tensors, keep pixel_values as list."""
        collated: dict = {}
        for key in ("input_ids", "attention_mask", "labels"):
            tensors = [item[key] for item in batch if key in item]
            if tensors:
                # Pad to the same length
                max_len = max(t.shape[0] for t in tensors)
                padded = []
                for t in tensors:
                    pad_len = max_len - t.shape[0]
                    if pad_len > 0:
                        pad_val = -100 if key == "labels" else 0
                        t = torch.cat([t, torch.full((pad_len,), pad_val,
                                                     dtype=t.dtype)])
                    padded.append(t)
                collated[key] = torch.stack(padded)
        # pixel_values: keep as list (variable number of patches per image)
        pv_list = [item["pixel_values"] for item in batch
                   if "pixel_values" in item and item["pixel_values"] is not None]
        if pv_list:
            collated["pixel_values"] = pv_list
        return collated

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    if _WANDB_AVAILABLE:
        wandb.init(
            project=cfg.wandb_project,
            name=f"qwenvl-ft+{exp_id}",
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    global_step = 0
    data_iter = iter(train_loader)

    import tqdm as _tqdm
    with _tqdm.tqdm(total=cfg.max_steps, desc=f"Stage {cfg.stage}") as pbar:
        while global_step < cfg.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Move tensors to device
            device = next(model.parameters()).device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            forward_kwargs: dict = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if "pixel_values" in batch:
                # pixel_values may be a list of tensors (multi-image batches)
                pv = batch["pixel_values"]
                if isinstance(pv, list):
                    forward_kwargs["pixel_values"] = [
                        p.to(torch.bfloat16).to(device) for p in pv
                    ]
                else:
                    forward_kwargs["pixel_values"] = pv.to(torch.bfloat16).to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**forward_kwargs)
                loss = output.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # ── Logging ───────────────────────────────────────────────
            if global_step % 10 == 0 and _WANDB_AVAILABLE:
                wandb.log({"train/loss": loss.item()}, step=global_step)

            # ── Checkpoint ────────────────────────────────────────────
            if global_step % cfg.save_steps == 0:
                ckpt_path = run_dir / f"checkpoint-{global_step}"
                os.makedirs(ckpt_path, exist_ok=True)
                model.save_pretrained(str(ckpt_path))
                processor.save_pretrained(str(ckpt_path))
                print(f"  Saved checkpoint → {ckpt_path}")

    # ------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------
    model.save_pretrained(str(run_dir / "final"))
    processor.save_pretrained(str(run_dir / "final"))
    print(f"Training complete. Final checkpoint → {run_dir / 'final'}")

    if _WANDB_AVAILABLE:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL with LoRA for fruit-clearing VLM tasks."
    )
    parser.add_argument("--pretrained_checkpoint", type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--stage", type=int, default=1,
                        help="Training stage: 1=completion only, 2=all tasks")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, default="")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="all-linear")
    parser.add_argument("--run_root_dir", type=str, default="runs/qwen_vl")
    parser.add_argument("--wandb_project", type=str, default="fruit-clearing-vlm")
    parser.add_argument("--run_id_note", type=str, default=None)

    args = parser.parse_args()
    cfg = QwenVLFinetuneConfig(**vars(args))
    finetune_qwenvl(cfg)
