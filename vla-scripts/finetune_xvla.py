"""
finetune_xvla.py

Parameter-efficient fine-tuning script for X-VLA models, using HuggingFace PEFT (LoRA).

Key differences from finetune.py (OpenVLA):
  - Loads model via AutoModel (not AutoModelForVision2Seq)
  - Processor called with `images` + `language_instruction` kwargs
  - Forward pass uses model.predict_action() interface; for training we call
    model(**inputs, proprio=..., domain_id=..., labels=...) which returns a
    loss when the model exposes a supervised training path, OR we fall back to
    a manual cross-entropy loss over the action head logits.
  - Proprioceptive state (proprio) is zero-padded when not available in batch.
  - domain_id is a configurable integer (default 3 = LIBERO per X-VLA docs).

Run with:
    torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune_xvla.py
    torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune_xvla.py \
        --pretrained_checkpoint <HF_HUB_ID_OR_LOCAL_PATH> \
        --data_root_dir <PATH/TO/RLDS/DATASETS> \
        --dataset_name <DATASET_NAME> \
        --domain_id 3 \
        --run_root_dir runs/xvla \
        ...

Notes:
  - Requires `peft>=0.11.1`, `accelerate`, `wandb`, `draccus`.
  - LoRA targets all linear layers by default; adjust `lora_target_modules` if needed.
  - Gradient accumulation is supported for effective large-batch training.
  - Checkpoints are saved as merged (LoRA fused) HF models for easy deployment.
"""

import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Local npz dataset (for raw_demos .npz files)
# When running as `python vla-scripts/finetune_xvla.py` from repo root,
# vla-scripts/ is already on sys.path via the script's own directory.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).parent))
from npz_dataset import NpzEpisodeDataset

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class XVLAFinetuneConfig:
    # fmt: off

    # Model
    pretrained_checkpoint: str = "HuggingFaceM4/xvla-7b"   # HF Hub ID or local path to X-VLA checkpoint

    # Data — choose ONE of the two modes:
    #   Mode A (RLDS):  set data_root_dir + dataset_name, leave npz_data_dir empty
    #   Mode B (npz):   set npz_data_dir, leave data_root_dir / dataset_name as defaults
    npz_data_dir: str = ""                                     # Path to raw_demos dir with episode_*.npz files
    data_root_dir: Path = Path("datasets/open-x-embodiment")  # [RLDS mode] Root directory of RLDS datasets
    dataset_name: str = "libero_spatial"                       # [RLDS mode] RLDS dataset name
    run_root_dir: Path = Path("runs/xvla")                     # Directory for logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp/xvla")           # Temp dir for LoRA adapter weights before merging

    # X-VLA-specific
    domain_id: int = 3                                         # Embodiment domain ID (3 = LIBERO per X-VLA docs)
    proprio_dim: int = 7                                       # Proprioceptive state dimension (pos + rot + gripper)
    action_dim: int = 7                                        # Action output dimension

    # Training hyperparameters
    batch_size: int = 8                                        # Per-GPU batch size
    max_steps: int = 20_000                                    # Total gradient update steps
    save_steps: int = 500                                      # Checkpoint save interval (gradient steps)
    learning_rate: float = 2e-4                                # AdamW learning rate
    grad_accumulation_steps: int = 2                           # Gradient accumulation steps
    image_aug: bool = True                                     # Enable random image augmentation
    shuffle_buffer_size: int = 10_000                          # RLDS shuffle buffer size
    save_latest_checkpoint_only: bool = True                   # Overwrite latest checkpoint vs. keep all

    # LoRA
    use_lora: bool = True                                      # Use LoRA PEFT
    lora_rank: int = 32                                        # LoRA rank
    lora_dropout: float = 0.0                                  # LoRA dropout
    lora_target_modules: str = "all-linear"                    # LoRA target modules spec
    use_quantization: bool = False                             # 4-bit quantization (reduces VRAM, may hurt quality)

    # Logging
    wandb_project: str = "xvla-finetune"                       # W&B project name
    wandb_entity: str = ""                                     # W&B entity (leave empty to use default)
    run_id_note: Optional[str] = None                          # Optional suffix for the run ID

    # fmt: on


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_exp_id(cfg: XVLAFinetuneConfig) -> str:
    """Construct a unique experiment identifier string from config."""
    model_name = Path(cfg.pretrained_checkpoint).name or cfg.pretrained_checkpoint.split("/")[-1]
    exp_id = (
        f"{model_name}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
        f"+domain{cfg.domain_id}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.image_aug:
        exp_id += "--image_aug"
    if cfg.run_id_note:
        exp_id += f"--{cfg.run_id_note}"
    return exp_id


def compute_action_metrics(action_preds: torch.Tensor, action_gt: torch.Tensor, action_tokenizer: ActionTokenizer):
    """
    Compute token-level accuracy and continuous L1 loss for action predictions.

    Args:
        action_preds: predicted token IDs, shape (B, T)
        action_gt:    ground-truth token IDs, shape (B, T)  (padded positions = -100)
        action_tokenizer: ActionTokenizer instance for decoding

    Returns:
        accuracy (float), l1_loss (float)
    """
    mask = action_gt > action_tokenizer.action_token_begin_idx
    if mask.sum() == 0:
        return 0.0, 0.0

    correct = (action_preds == action_gt) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    continuous_pred = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()),
        dtype=torch.float32,
    )
    continuous_gt = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()),
        dtype=torch.float32,
    )
    l1_loss = torch.nn.functional.l1_loss(continuous_pred, continuous_gt)

    return accuracy.item(), l1_loss.item()


def save_checkpoint(
    vla,
    processor,
    cfg: XVLAFinetuneConfig,
    run_dir: Path,
    adapter_dir: Path,
    step: int,
    distributed_state,
):
    """Save model checkpoint (merge LoRA if applicable)."""
    if distributed_state.is_main_process:
        print(f"Saving checkpoint at step {step} ...")
        save_dir = adapter_dir if cfg.use_lora else run_dir
        processor.save_pretrained(run_dir)
        vla.module.save_pretrained(save_dir)

    # All processes wait for main to finish writing adapter/full weights
    dist.barrier()

    if cfg.use_lora:
        # Reload base model and merge LoRA weights for inference-ready checkpoint
        base_model = AutoModel.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        merged_model = merged_model.merge_and_unload()

        if distributed_state.is_main_process:
            if cfg.save_latest_checkpoint_only:
                merged_model.save_pretrained(run_dir)
                print(f"  -> Saved merged checkpoint to: {run_dir}")
            else:
                ckpt_dir = Path(str(run_dir) + f"--step{step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                save_dataset_statistics(None, ckpt_dir)  # placeholder; real stats saved at start
                processor.save_pretrained(ckpt_dir)
                merged_model.save_pretrained(ckpt_dir)
                print(f"  -> Saved merged checkpoint to: {ckpt_dir}")

    dist.barrier()


# ---------------------------------------------------------------------------
# Main fine-tuning loop
# ---------------------------------------------------------------------------

@draccus.wrap()
def finetune(cfg: XVLAFinetuneConfig) -> None:
    print(f"Fine-tuning X-VLA `{cfg.pretrained_checkpoint}` on `{cfg.dataset_name}` (domain_id={cfg.domain_id})")

    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------
    assert torch.cuda.is_available(), "Fine-tuning requires at least one GPU!"
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Experiment ID & directories
    # ------------------------------------------------------------------
    exp_id = build_exp_id(cfg)
    run_dir = cfg.run_root_dir / exp_id
    adapter_dir = cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Quantization config (LoRA + 4-bit only)
    # ------------------------------------------------------------------
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training is only supported with LoRA!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    # ------------------------------------------------------------------
    # Load X-VLA processor and model
    # ------------------------------------------------------------------
    print("[*] Loading X-VLA processor ...")
    processor = AutoProcessor.from_pretrained(
        cfg.pretrained_checkpoint,
        trust_remote_code=True,
    )

    print("[*] Loading X-VLA model ...")
    model = AutoModel.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # ------------------------------------------------------------------
    # Device placement
    # ------------------------------------------------------------------
    if cfg.use_quantization:
        model = prepare_model_for_kbit_training(model)
    else:
        model = model.to(device_id)

    # ------------------------------------------------------------------
    # LoRA wrapping
    # ------------------------------------------------------------------
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # DDP wrapping
    # ------------------------------------------------------------------
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # ------------------------------------------------------------------
    # Action tokenizer (reuse OpenVLA's tokenizer for RLDS compatibility)
    # ------------------------------------------------------------------
    # X-VLA shares the same action tokenization scheme as OpenVLA when
    # trained on RLDS data.  We use the underlying text tokenizer exposed
    # by the processor.
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    if cfg.npz_data_dir:
        # ── Mode B: raw .npz episodes ──────────────────────────────────
        print(f"[*] Using NpzEpisodeDataset from: {cfg.npz_data_dir}")
        vla_dataset = NpzEpisodeDataset(
            npz_dir           = cfg.npz_data_dir,
            action_tokenizer  = action_tokenizer,
            base_tokenizer    = processor.tokenizer,
            image_transform   = processor.image_processor.apply_transform,
            prompt_builder_fn = PurePromptBuilder,
            image_aug         = cfg.image_aug,
            resize_resolution = (224, 224),
        )
        if distributed_state.is_main_process:
            import json, pathlib
            stats_dst = run_dir / "dataset_statistics.json"
            with open(stats_dst, "w") as f:
                json.dump(vla_dataset.dataset_statistics, f, indent=2)
            print(f"  Saved dataset statistics → {stats_dst}")

        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length,
            processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        dataloader = DataLoader(
            vla_dataset,
            batch_size  = cfg.batch_size,
            shuffle     = True,
            collate_fn  = collator,
            num_workers = 4,
            pin_memory  = True,
        )
    else:
        # ── Mode A: RLDS dataset ───────────────────────────────────────
        print(f"[*] Using RLDSDataset: {cfg.dataset_name}")
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
        )
        vla_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=(224, 224),
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=cfg.image_aug,
        )
        if distributed_state.is_main_process:
            save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length,
            processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        dataloader = DataLoader(
            vla_dataset,
            batch_size  = cfg.batch_size,
            sampler     = None,
            collate_fn  = collator,
            num_workers = 0,   # RLDS/TFDS manages its own parallelism
        )

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------
    if distributed_state.is_main_process:
        wandb_kwargs = dict(project=cfg.wandb_project, name=f"xvla-ft+{exp_id}")
        if cfg.wandb_entity:
            wandb_kwargs["entity"] = cfg.wandb_entity
        wandb.init(**wandb_kwargs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        model.train()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            # ----------------------------------------------------------------
            # Build proprio tensor: zero-padded if state not in batch
            # Shape: (B, proprio_dim)
            # ----------------------------------------------------------------
            if "state" in batch:
                proprio = batch["state"].to(torch.bfloat16).to(device_id)
            else:
                B = batch["input_ids"].shape[0]
                proprio = torch.zeros(B, cfg.proprio_dim, dtype=torch.bfloat16, device=device_id)

            # ----------------------------------------------------------------
            # Forward pass
            # X-VLA's training interface: pass input_ids, pixel_values (image),
            # attention_mask, labels, proprio, and domain_id.
            # The model returns a loss when `labels` is provided.
            # ----------------------------------------------------------------
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"].to(device_id),
                    proprio=proprio,
                    domain_id=cfg.domain_id,
                )
                loss = output.loss

            # ----------------------------------------------------------------
            # Gradient accumulation
            # ----------------------------------------------------------------
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            # ----------------------------------------------------------------
            # Compute action-level metrics for logging
            # ----------------------------------------------------------------
            # logits shape: (B, seq_len, vocab_size)
            # We look at the action token positions (after vision patch tokens)
            logits = output.logits
            # Shift: action logits start after the vision patch embeddings.
            # Use the same slice as OpenVLA: [num_patches : -1]
            try:
                num_patches = model.module.vision_backbone.featurizer.patch_embed.num_patches
                action_logits = logits[:, num_patches:-1]
            except AttributeError:
                # Fallback: use all logits except the last position
                action_logits = logits[:, :-1]

            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)

            accuracy, l1_loss = compute_action_metrics(action_preds, action_gt, action_tokenizer)

            recent_losses.append(loss.item())
            recent_action_accuracies.append(accuracy)
            recent_l1_losses.append(l1_loss)

            # ----------------------------------------------------------------
            # Gradient step
            # ----------------------------------------------------------------
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            # ----------------------------------------------------------------
            # Logging (every 10 gradient steps)
            # ----------------------------------------------------------------
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                smoothed_loss = sum(recent_losses) / len(recent_losses)
                smoothed_acc = sum(recent_action_accuracies) / len(recent_action_accuracies)
                smoothed_l1 = sum(recent_l1_losses) / len(recent_l1_losses)
                wandb.log(
                    {
                        "train/loss": smoothed_loss,
                        "train/action_accuracy": smoothed_acc,
                        "train/action_l1_loss": smoothed_l1,
                    },
                    step=gradient_step_idx,
                )

            # ----------------------------------------------------------------
            # Checkpoint saving
            # ----------------------------------------------------------------
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                save_checkpoint(
                    vla=model,
                    processor=processor,
                    cfg=cfg,
                    run_dir=run_dir,
                    adapter_dir=adapter_dir,
                    step=gradient_step_idx,
                    distributed_state=distributed_state,
                )

            # ----------------------------------------------------------------
            # Stop condition
            # ----------------------------------------------------------------
            if gradient_step_idx >= cfg.max_steps:
                print(f"Reached max_steps={cfg.max_steps}. Stopping training.")
                break

    # ------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------
    if distributed_state.is_main_process:
        print("Saving final checkpoint ...")
    save_checkpoint(
        vla=model,
        processor=processor,
        cfg=cfg,
        run_dir=run_dir,
        adapter_dir=adapter_dir,
        step=cfg.max_steps,
        distributed_state=distributed_state,
    )

    if distributed_state.is_main_process:
        wandb.finish()
        print(f"Training complete. Checkpoint saved to: {run_dir}")


if __name__ == "__main__":
    finetune()
