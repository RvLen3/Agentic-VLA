"""
finetune_xvla.py

Fine-tuning script for X-VLA models with THREE selectable strategies
(set via --finetune_mode):

  1. "soft_prompt" (default) — X-VLA's signature adaptation method.
       Freeze the ENTIRE backbone (VLM + policy transformer + action head) and
       train ONLY the embodiment-specific *soft-prompt* embeddings (the learnable
       per-domain tokens indexed by `domain_id`). This is the official Phase-II
       recipe: adapt to a new robot (your UR7E) by learning a fresh prompt while
       the pretrained backbone stays frozen. ~1% of params, best for few demos.

  2. "lora" — Official X-VLA fine-tuning recipe (recommended for a new robot).
       Injects LoRA adapters into all linear layers AND fully trains the
       "Action Expert" (transformer.action_encoder + transformer.action_decoder)
       together with the soft prompt hub, via PEFT `modules_to_save`. Optionally
       warms up the action expert first (freeze_steps) before ramping the LoRA
       adapters. This mirrors official_ft_xvla.py from the X-VLA repo and gives
       the model enough capacity to remap vision→action for a new embodiment.

  3. "full" — Full fine-tuning of every parameter. Following the official recipe,
       VLM (vision-language) layers use 1/10 of the base LR (--vlm_lr_scale) for
       stable optimization while all other components use the full LR.

X-VLA background (arXiv:2510.10274):
  - Flow-matching VLA built on soft-prompted Transformer encoders.
  - `domain_id` selects the soft-prompt set for an embodiment (LIBERO = 3).
  - Inference uses model.generate_actions(...); training uses model.forward(...).

Training forward pass (aligned to the real `modeling_xvla.py`):
  The model's forward signature is
      forward(input_ids, image_input, image_mask, domain_id, proprio, action)
  and it returns a DICT of flow-matching losses, e.g.
      {"position_loss", "rotate6D_loss", "gripper_loss"}.
  Internally it builds the diffusion-style noisy action mixture
  x_t = t*noise + (1-t)*action, predicts the velocity/target, and computes the
  per-component loss via the registered action space (EE6D by default, dim=20).
  We sum the component losses into one scalar for backprop (per-component
  weighting is already applied inside the model). There is NO action tokenizer,
  NO logits, and NO min/max action normalization — actions are continuous EE6D.

Data:
  `xvla_npz_dataset.XVLANpzDataset` converts raw_demos .npz episodes into the
  EE6D continuous format (xyz Δ + 6D relative rotation + {0,1} gripper) as an
  action chunk of length `num_actions`, with the current pose as proprio.

Validation & checkpoints (small-data friendly):
  - `--val_ratio` holds out a fraction of EPISODES (not frames) for validation.
    Validation runs every `--val_every_steps`; with `--early_stop_patience > 0`
    training stops when val loss stops improving. This is the reliable signal on
    small datasets where train loss alone overfits.
  - Lightweight `.pth` checkpoints (trainable params + metadata) are written to
    `<run_dir>/pth/XVLA_<mode>_step<step>_loss<loss>.pth`, plus `_best` (lowest
    val loss) and `_final`. For deployment, a full HF model dir is exported at
    the end (load with AutoModel.from_pretrained), see `--export_hf_on_finish`.

Run with:
    torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune_xvla.py \
        --finetune_mode soft_prompt \
        --pretrained_checkpoint <HF_HUB_ID_OR_LOCAL_PATH> \
        --npz_data_dir raw_demos_left_third \
        --domain_id 3 \
        --run_root_dir runs/xvla

Notes:
  - Requires `peft>=0.11.1`, `accelerate`, `wandb`, `draccus`.
  - Gradient accumulation is supported for effective large-batch training.
  - soft_prompt/full save full HF checkpoints; lora saves merged (fused) weights.
"""

import os
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# HuggingFace cache → store downloads under <repo_root>/.cache
# (i.e. ../.cache relative to this file in vla-scripts/).
# MUST be set BEFORE importing transformers / huggingface_hub so the env vars
# take effect. Uses setdefault so an externally-provided HF_HOME still wins.
# ---------------------------------------------------------------------------
_CACHE_DIR = (Path(__file__).parent / ".." / ".cache").resolve()
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_CACHE_DIR / "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_DIR / "datasets"))

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

# X-VLA-native npz dataset (continuous EE6D flow-matching inputs).
# When running as `python vla-scripts/finetune_xvla.py` from repo root,
# vla-scripts/ is already on sys.path via the script's own directory.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).parent))
from xvla_npz_dataset import (
    XVLANpzDataset,
    XVLACollator,
    EpisodeBatchSampler,
    make_train_val_split,
)

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class XVLAFinetuneConfig:
    # fmt: off

    # Model
    pretrained_checkpoint: str = "2toINF/X-VLA-Pt"             # HF Hub ID or local path to X-VLA checkpoint

    # Data (raw_demos .npz mode)
    npz_data_dir: str = "raw_demos_left_third"                 # Path to dir with episode_*.npz files
    dataset_name: str = "ur_fruit"                             # Short tag used only in the experiment id
    run_root_dir: Path = Path("runs/xvla")                     # Directory for logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp/xvla")           # Temp dir for LoRA adapter weights before merging

    # X-VLA-specific (must match the loaded checkpoint's action space / config)
    domain_id: int = 3                                         # Soft-prompt / embodiment index (3 = LIBERO/Franka EE6D)
    num_actions: int = 30                                      # Action chunk length (X-VLA config: num_actions)
    use_wrist_image: bool = True                               # Feed wrist camera as a 2nd view (images_wrist)
    frame_stride: int = 1                                      # Sub-sample future steps in each action chunk (1 = every frame)
    min_chunk_motion: float = 0.0                              # Skip chunk-starts whose total path < this (m); 0 = keep all

    # Training hyperparameters
    batch_size: int = 8                                        # Per-GPU batch size
    max_steps: int = 20_000                                    # Total gradient update steps
    save_steps: int = 500                                      # Checkpoint save interval (gradient steps)
    learning_rate: float = 1e-4                                # AdamW learning rate (X-VLA recipe base LR)
    grad_accumulation_steps: int = 2                           # Gradient accumulation steps
    num_workers: int = 4                                       # DataLoader worker processes
    cache_episodes: int = 2                                    # Per-worker LRU cache size (decoded episodes)
    group_by_episode: bool = True                              # Batch within one episode (maximizes cache hits → big speedup)
    save_latest_checkpoint_only: bool = True                   # Overwrite latest checkpoint vs. keep all

    # ── LR schedule (official X-VLA: linear warmup → cosine decay) ────────
    # Stabilises training when fully training the action expert; without it the
    # large-capacity groups can oscillate (position_loss spiking).
    use_lr_schedule: bool = True                               # Enable warmup + (optional) cosine decay
    warmup_steps: int = 1000                                   # Linear warmup length (gradient steps)
    use_cosine_decay: bool = True                              # Cosine-decay after warmup (else hold flat)
    min_lr_ratio: float = 0.1                                  # Cosine floor as a fraction of base LR

    # ── Validation & early stopping (recommended for small datasets) ──────
    val_ratio: float = 0.15                                    # Fraction of EPISODES held out for validation (0 = disable)
    val_every_steps: int = 200                                 # Run validation every N gradient steps
    val_max_batches: int = 50                                  # Cap val batches per eval (speed)
    early_stop_patience: int = 0                               # Stop if val loss doesn't improve for N evals (0 = disable)
    early_stop_min_delta: float = 1e-4                         # Min val-loss improvement to count as "better"
    split_seed: int = 0                                        # RNG seed for the train/val episode split

    # ── Checkpoint export ────────────────────────────────────────────────
    save_pth: bool = True                                      # Save lightweight .pth (trainable params + metadata)
    export_hf_on_finish: bool = True                           # Also export a full HF model dir at the end (for deployment)
    export_best_hf: bool = True                                # Snapshot the best-val model and export it as a merged HF dir (best_hf/)

    # ── Fine-tuning strategy ─────────────────────────────────────────────
    # Choose ONE of three modes:
    #   "soft_prompt" : freeze the entire backbone; train ONLY the embodiment
    #                   soft-prompt embeddings (X-VLA Phase-II adaptation, the
    #                   model's core innovation). Smallest footprint (~1% of
    #                   params); best fit for adapting to ONE new robot (UR7E)
    #                   with limited demos. Only the soft-prompt row indexed by
    #                   `domain_id` actually receives gradients.
    #   "lora"        : official X-VLA recipe — inject LoRA adapters into all
    #                   linear layers AND fully train the Action Expert
    #                   (action_encoder + action_decoder) + soft prompts. An
    #                   optional freeze phase warms up the action expert first.
    #   "full"        : train every parameter. VLM (vision-language) layers use
    #                   a reduced LR (vlm_lr_scale x lr) per the official recipe.
    finetune_mode: str = "soft_prompt"                         # {"soft_prompt", "lora", "full"}

    # Substrings (comma-separated, case-insensitive) used to locate the
    # soft-prompt parameters by name. For the official 2toINF/X-VLA checkpoint
    # the soft prompts live in `transformer.soft_prompt_hub`. Inspect your
    # checkpoint's `model.named_parameters()` and adjust if no params match.
    soft_prompt_name_patterns: str = "soft_prompt_hub,soft_prompt,soft_prompts,prompt_emb,domain_emb,embodiment"
    # If True, also train the domain-conditioned projection / action en-decoder
    # layers (DomainAwareLinear: action_encoder/action_decoder/vlm_proj). These
    # carry per-embodiment parameters too, so training them alongside the soft
    # prompts usually helps adaptation to a genuinely new robot. Soft-prompt mode only.
    soft_prompt_include_domain_layers: bool = True
    domain_layer_name_patterns: str = "action_encoder,action_decoder,vlm_proj,aux_visual_proj"
    # Substrings used to locate VLM/backbone params (reduced-LR group, full mode).
    # The official backbone is Florence2 (vision + language encoder).
    vlm_name_patterns: str = "vlm,florence,vision,visual,image_encoder,language_model,text,encoder.embed,siglip,dino"
    vlm_lr_scale: float = 0.1                                  # [full mode] VLM LR multiplier (official recipe = 1/10)

    # LoRA (used only when finetune_mode == "lora")
    # Official X-VLA recipe: LoRA on all-linear + FULLY train the action expert
    # (action_encoder + action_decoder) and the soft prompts via `modules_to_save`.
    lora_rank: int = 8                                         # LoRA rank (official default = 8)
    lora_alpha: int = 16                                       # LoRA alpha (official default = 16)
    lora_dropout: float = 0.0                                  # LoRA dropout
    lora_target_modules: str = "all-linear"                    # LoRA target modules spec
    use_quantization: bool = False                             # 4-bit quantization (reduces VRAM, may hurt quality)
    # Modules fully trained alongside LoRA (PEFT modules_to_save). The action
    # expert = action_encoder + action_decoder; soft_prompt_hub = embodiment prompt.
    train_action_expert: bool = True                           # [lora] also fully-train the action expert + soft prompts
    action_expert_modules: str = "transformer.soft_prompt_hub,transformer.action_encoder,transformer.action_decoder"
    # Freeze schedule (official): for the first `freeze_steps` gradient steps the
    # LoRA adapters are held at lr=0 so the action expert + soft prompt warm up
    # first; afterwards the adapters train at full LR. 0 = no freeze phase.
    freeze_steps: int = 0
    max_grad_norm: float = 1.0                                 # Gradient clipping (0 = disabled)

    # Derived from finetune_mode — do NOT set manually (overwritten in __post_init__).
    use_lora: bool = True

    # Logging
    wandb_project: str = "xvla-finetune"                       # W&B project name
    wandb_entity: str = ""                                     # W&B entity (leave empty to use default)
    run_id_note: Optional[str] = None                          # Optional suffix for the run ID

    # fmt: on

    def __post_init__(self):
        valid_modes = {"soft_prompt", "lora", "full"}
        if self.finetune_mode not in valid_modes:
            raise ValueError(
                f"finetune_mode must be one of {valid_modes}, got '{self.finetune_mode}'"
            )
        # `use_lora` is kept for backward-compat with the checkpoint-saving path;
        # it is fully derived from finetune_mode.
        self.use_lora = self.finetune_mode == "lora"
        if self.use_quantization and self.finetune_mode != "lora":
            raise ValueError("use_quantization=True is only supported with finetune_mode='lora'.")


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
        f"+{cfg.finetune_mode}"
    )
    if cfg.finetune_mode == "lora":
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note:
        exp_id += f"--{cfg.run_id_note}"
    return exp_id


def _match_any(name: str, patterns: List[str]) -> bool:
    """Return True if `name` contains any of the (lower-cased) substrings."""
    low = name.lower()
    return any(p in low for p in patterns)


def configure_trainable_parameters(model, cfg: "XVLAFinetuneConfig", is_main: bool = True):
    """
    Set `requires_grad` on the model's parameters according to `cfg.finetune_mode`.

    Modes
    -----
    soft_prompt : Only the embodiment soft-prompt embeddings are trainable; the
                  entire backbone (VLM + policy transformer + action head) is frozen.
                  This mirrors X-VLA's official Phase-II domain-adaptation recipe.
    full        : Every parameter is trainable. The returned optimizer param-groups
                  put VLM/backbone params on a reduced LR (cfg.vlm_lr_scale * lr).
    lora        : Handled separately via PEFT (this function is not used).

    Returns
    -------
    param_groups : list[dict]  ready to hand to torch.optim.AdamW
    """
    sp_patterns  = [p.strip().lower() for p in cfg.soft_prompt_name_patterns.split(",") if p.strip()]
    vlm_patterns = [p.strip().lower() for p in cfg.vlm_name_patterns.split(",") if p.strip()]
    domain_patterns = [p.strip().lower() for p in cfg.domain_layer_name_patterns.split(",") if p.strip()]

    if cfg.finetune_mode == "soft_prompt":
        # Freeze everything, then re-enable only the soft-prompt (and optionally
        # the per-domain projection / action en-decoder) parameters.
        train_patterns = list(sp_patterns)
        if cfg.soft_prompt_include_domain_layers:
            train_patterns += domain_patterns

        soft_prompt_params, matched_names = [], []
        for name, p in model.named_parameters():
            if _match_any(name, train_patterns):
                p.requires_grad = True
                soft_prompt_params.append(p)
                matched_names.append(name)
            else:
                p.requires_grad = False

        if len(soft_prompt_params) == 0:
            raise RuntimeError(
                "finetune_mode='soft_prompt' but NO parameters matched "
                f"patterns={train_patterns}.\n"
                "Inspect your checkpoint with `for n,_ in model.named_parameters(): print(n)` "
                "and pass the correct substrings via --soft_prompt_name_patterns "
                "(and/or --domain_layer_name_patterns)."
            )
        if is_main:
            n_trainable = sum(p.numel() for p in soft_prompt_params)
            print(f"[soft_prompt] {len(soft_prompt_params)} tensors trainable "
                  f"({n_trainable/1e6:.3f}M params"
                  f"{', incl. domain-aware layers' if cfg.soft_prompt_include_domain_layers else ''}). "
                  f"Matched names (first 8):")
            for nm in matched_names[:8]:
                print(f"    - {nm}")
        return [{"params": soft_prompt_params, "lr": cfg.learning_rate}]

    if cfg.finetune_mode == "full":
        # Everything trainable; split into reduced-LR VLM group and full-LR rest.
        for p in model.parameters():
            p.requires_grad = True
        vlm_params, other_params = [], []
        for name, p in model.named_parameters():
            (vlm_params if _match_any(name, vlm_patterns) else other_params).append(p)

        if is_main:
            n_vlm   = sum(p.numel() for p in vlm_params)
            n_other = sum(p.numel() for p in other_params)
            print(f"[full] VLM group: {n_vlm/1e6:.1f}M params @ lr*{cfg.vlm_lr_scale} | "
                  f"rest: {n_other/1e6:.1f}M params @ lr")
        groups = [{"params": other_params, "lr": cfg.learning_rate}]
        if vlm_params:
            groups.append({"params": vlm_params, "lr": cfg.learning_rate * cfg.vlm_lr_scale})
        return groups

    raise RuntimeError(f"configure_trainable_parameters() should not be called for mode '{cfg.finetune_mode}'")


def reduce_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Sum the component losses returned by X-VLA's action space
    (e.g. {"position_loss", "rotate6D_loss", "gripper_loss"}) into a single
    scalar for backprop. The per-component weighting (XYZ_SCALE, ROT_SCALE,
    GRIPPER_SCALE) is already applied inside the model's compute_loss.
    """
    total = None
    for v in loss_dict.values():
        total = v if total is None else total + v
    if total is None:
        raise RuntimeError("X-VLA forward returned an empty loss dict.")
    return total


def lr_at_step(step: int, base_lr: float, cfg: "XVLAFinetuneConfig", freeze_until: int = 0) -> float:
    """
    Per-step learning rate following the official X-VLA schedule:
      - while step < freeze_until            → 0 (group is frozen)
      - linear warmup over `warmup_steps`    → ramp 0 → base_lr
      - then cosine decay to base_lr*min_lr_ratio (if use_cosine_decay) else hold

    The warmup/decay clock starts at `freeze_until` so a group that unfreezes
    later still gets its own warmup ramp.
    """
    if not cfg.use_lr_schedule:
        return 0.0 if step < freeze_until else base_lr
    if step < freeze_until:
        return 0.0
    progress = step - freeze_until
    if progress < cfg.warmup_steps:
        return base_lr * (progress / max(1, cfg.warmup_steps))
    if not cfg.use_cosine_decay:
        return base_lr
    remain = max(1, cfg.max_steps - (freeze_until + cfg.warmup_steps))
    frac = min(1.0, (progress - cfg.warmup_steps) / remain)
    cos = 0.5 * (1.0 + math.cos(math.pi * frac))
    return base_lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cos)


@torch.no_grad()
def evaluate(model, val_loader, device_id, cfg) -> Dict[str, float]:
    """
    Run validation over (a capped number of) batches and return mean total loss
    plus mean per-component losses. Used for early stopping / best-checkpoint
    selection on small datasets where train loss alone is misleading.
    """
    if val_loader is None:
        return {}
    was_training = model.training
    model.eval()

    totals: Dict[str, float] = {}
    n = 0
    for i, batch in enumerate(val_loader):
        if cfg.val_max_batches and i >= cfg.val_max_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss_dict = model(
                input_ids=batch["input_ids"].to(device_id),
                image_input=batch["image_input"].to(torch.bfloat16).to(device_id),
                image_mask=batch["image_mask"].to(device_id),
                domain_id=batch["domain_id"].to(device_id),
                proprio=batch["proprio"].to(torch.bfloat16).to(device_id),
                action=batch["action"].to(torch.bfloat16).to(device_id),
            )
            total = reduce_loss_dict(loss_dict)
        totals["loss"] = totals.get("loss", 0.0) + total.item()
        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1

    if was_training:
        model.train()
    if n == 0:
        return {}
    return {k: v / n for k, v in totals.items()}


def save_pth_checkpoint(model, cfg, run_dir: Path, step: int, loss: float,
                        kind: str = "step", val_metrics: Optional[Dict] = None) -> Path:
    """
    Save a lightweight .pth containing ONLY the trainable parameters plus
    metadata. Filename: XVLA_{mode}_step{step}_loss{loss}.pth (or _best/_final).

    This is for archiving/inspection and resuming; for deployment use the full
    HF model dir exported by export_hf_on_finish.
    """
    core = model.module if hasattr(model, "module") else model
    trainable_state = {
        name: p.detach().cpu()
        for name, p in core.named_parameters()
        if p.requires_grad
    }

    ckpt_dir = run_dir / "pth"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loss_str = f"{loss:.4f}".replace(".", "p")  # avoid extra dots in filename
    if kind == "step":
        fname = f"XVLA_{cfg.finetune_mode}_step{step}_loss{loss_str}.pth"
    else:
        fname = f"XVLA_{cfg.finetune_mode}_{kind}.pth"   # e.g. _best, _final
    fpath = ckpt_dir / fname

    torch.save(
        {
            "trainable_state_dict": trainable_state,
            "step": step,
            "loss": loss,
            "val_metrics": val_metrics or {},
            "finetune_mode": cfg.finetune_mode,
            "domain_id": cfg.domain_id,
            "num_actions": cfg.num_actions,
            "pretrained_checkpoint": cfg.pretrained_checkpoint,
        },
        fpath,
    )
    print(f"  -> Saved .pth checkpoint: {fpath}")
    return fpath


def save_loss_history(history: Dict, run_dir: Path) -> None:
    """
    Dump the recorded loss history to JSON and render a loss-curve PNG.

    Produces:
      <run_dir>/loss_history.json
      <run_dir>/loss_curve.png   (train vs val total loss + per-component panels)
    """
    import json
    # Raw data always saved (so curves can be re-plotted later without re-training).
    with open(run_dir / "loss_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  -> Saved loss history → {run_dir / 'loss_history.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed → skipping loss_curve.png; JSON still saved)")
        return

    comp_names = sorted(set(history["train_components"]) | set(history["val_components"]))
    n_panels = 1 + len(comp_names)
    ncols = 2
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    # Panel 0: total loss
    ax = axes[0]
    if history["train_step"]:
        ax.plot(history["train_step"], history["train_loss"], label="train", color="tab:blue")
    if history["val_step"]:
        ax.plot(history["val_step"], history["val_loss"], label="val", color="tab:orange", marker="o", ms=3)
    ax.set_title("total loss"); ax.set_xlabel("gradient step"); ax.set_ylabel("loss")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Per-component panels
    for i, name in enumerate(comp_names):
        ax = axes[i + 1]
        tr = history["train_components"].get(name, [])
        va = history["val_components"].get(name, [])
        if tr:
            ax.plot([s for s, _ in tr], [v for _, v in tr], label="train", color="tab:blue")
        if va:
            ax.plot([s for s, _ in va], [v for _, v in va], label="val", color="tab:orange", marker="o", ms=3)
        ax.set_title(name); ax.set_xlabel("gradient step"); ax.set_ylabel("loss")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Hide any unused panels
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle("X-VLA fine-tuning loss curves")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=120)
    plt.close(fig)
    print(f"  -> Saved loss curve → {run_dir / 'loss_curve.png'}")


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
                processor.save_pretrained(ckpt_dir)
                merged_model.save_pretrained(ckpt_dir)
                print(f"  -> Saved merged checkpoint to: {ckpt_dir}")

    dist.barrier()


def snapshot_best_adapter(vla, cfg: XVLAFinetuneConfig, best_dir: Path, distributed_state):
    """
    Save the CURRENT model state as the 'best' snapshot (main process only).
      - lora mode : save the PEFT adapter (small) → merged later into best_hf/.
      - else      : save the full HF model directly into best_dir.
    Called whenever validation improves, so best_dir always holds the best model.
    """
    if not distributed_state.is_main_process:
        return
    best_dir = Path(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    vla.module.save_pretrained(str(best_dir))


def export_best_hf(processor, cfg: XVLAFinetuneConfig, best_adapter_dir: Path,
                   best_hf_dir: Path, distributed_state):
    """
    Produce a deployment-ready HF model dir for the best snapshot.
      - lora mode : reload base + merge the best adapter → save merged model.
      - else      : the best snapshot IS already a full HF model → copy/save.
    After this, evaluate with `--model_path <best_hf_dir>` and NO `--pth`.
    """
    if not distributed_state.is_main_process:
        return
    best_adapter_dir = Path(best_adapter_dir)
    best_hf_dir = Path(best_hf_dir)
    if not best_adapter_dir.exists():
        print(f"[best] no best snapshot at {best_adapter_dir}; skipping best_hf export.")
        return
    best_hf_dir.mkdir(parents=True, exist_ok=True)

    if cfg.use_lora:
        base_model = AutoModel.from_pretrained(
            cfg.pretrained_checkpoint, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
        merged = PeftModel.from_pretrained(base_model, str(best_adapter_dir)).merge_and_unload()
        merged.save_pretrained(str(best_hf_dir))
    else:
        # best snapshot is already a full HF model; just (re)save under best_hf.
        m = AutoModel.from_pretrained(
            str(best_adapter_dir), torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
        m.save_pretrained(str(best_hf_dir))
    processor.save_pretrained(str(best_hf_dir))
    print(f"  -> Exported BEST merged HF model → {best_hf_dir}  (eval with --model_path this, no --pth)")


# ---------------------------------------------------------------------------
# Main fine-tuning loop
# ---------------------------------------------------------------------------

@draccus.wrap()
def finetune(cfg: XVLAFinetuneConfig) -> None:
    print(f"Fine-tuning X-VLA `{cfg.pretrained_checkpoint}` on `{cfg.dataset_name}` "
          f"(mode={cfg.finetune_mode}, domain_id={cfg.domain_id})")

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
    best_adapter_dir = cfg.adapter_tmp_dir / (exp_id + "--best")   # best snapshot (adapter or full)
    best_hf_dir = run_dir / "best_hf"                              # deployable merged best model
    os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Quantization config (LoRA + 4-bit only)
    # ------------------------------------------------------------------
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training is only supported with finetune_mode='lora'!"
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
    # Fine-tuning strategy: configure trainable params / optimizer groups
    #   - "lora"        : inject PEFT LoRA adapters (backbone frozen) + train soft prompts
    #   - "soft_prompt" : freeze backbone, train ONLY soft-prompt embeddings
    #   - "full"        : train everything (VLM at reduced LR)
    # ------------------------------------------------------------------
    param_groups = None
    lora_param_group_names = None   # set in lora mode → used for the freeze schedule
    if cfg.finetune_mode == "lora":
        # ── Official X-VLA recipe ─────────────────────────────────────
        # LoRA on all-linear layers + FULLY train the action expert
        # (action_encoder + action_decoder) and the soft prompt hub via
        # PEFT `modules_to_save`. These three modules are the parts that carry
        # embodiment-specific behaviour, so full-training them (not just LoRA)
        # is what lets X-VLA actually adapt its action output to a new robot.
        modules_to_save = [m.strip() for m in cfg.action_expert_modules.split(",") if m.strip()] \
            if cfg.train_action_expert else None
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=cfg.lora_target_modules,
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_config)
        if distributed_state.is_main_process:
            model.print_trainable_parameters()

        # Build 2 param groups so we can (a) give soft-prompt/action-expert a
        # possibly different LR and (b) freeze the LoRA adapters during warmup.
        # PEFT names: LoRA tensors contain ".lora_"; the fully-trained
        # modules_to_save tensors contain ".modules_to_save".
        lora_params, save_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if ".lora_" in name:
                lora_params.append(p)
            else:
                # modules_to_save copies (action expert + soft prompts)
                save_params.append(p)
        param_groups = [
            {"name": "lora_adapters", "params": lora_params, "lr": cfg.learning_rate},
            {"name": "action_expert", "params": save_params, "lr": cfg.learning_rate},
        ]
        lora_param_group_names = ["lora_adapters"]
        if distributed_state.is_main_process:
            n_lora = sum(p.numel() for p in lora_params)
            n_save = sum(p.numel() for p in save_params)
            print(f"[lora] LoRA adapters: {n_lora/1e6:.3f}M | action-expert+soft-prompt "
                  f"(full): {n_save/1e6:.3f}M")
            if cfg.freeze_steps > 0:
                print(f"[lora] freeze schedule: LoRA held at lr=0 for first {cfg.freeze_steps} steps")
    else:
        # soft_prompt / full: select trainable params by name and build LR groups
        param_groups = configure_trainable_parameters(
            model, cfg, is_main=distributed_state.is_main_process
        )

    # ------------------------------------------------------------------
    # DDP wrapping
    # ------------------------------------------------------------------
    model = DDP(model, device_ids=[device_id], find_unused_parameters=False, gradient_as_bucket_view=True)

    # ------------------------------------------------------------------
    # Optimizer (all modes now use explicit param groups)
    # ------------------------------------------------------------------
    optimizer = AdamW(param_groups, lr=cfg.learning_rate)
    # Per-group schedule metadata: base (target) LR + the step until which the
    # group is frozen. Only the LoRA adapter group has a freeze phase.
    lora_names = set(lora_param_group_names or [])
    for g in optimizer.param_groups:
        g["base_lr"] = g["lr"]
        g["freeze_until"] = cfg.freeze_steps if g["name"] in lora_names else 0
    # Initialise LR at step 0 according to the schedule (warmup starts at 0).
    for g in optimizer.param_groups:
        g["lr"] = lr_at_step(0, g["base_lr"], cfg, g["freeze_until"])

    # ------------------------------------------------------------------
    # Dataset & DataLoader (X-VLA-native EE6D flow-matching inputs)
    # ------------------------------------------------------------------
    if not cfg.npz_data_dir:
        raise ValueError("npz_data_dir must be set (path to a dir of episode_*.npz files).")

    print(f"[*] Using XVLANpzDataset from: {cfg.npz_data_dir}")
    print(f"[*] Building datasets from: {cfg.npz_data_dir}")
    train_eps, val_eps = make_train_val_split(cfg.npz_data_dir, cfg.val_ratio, seed=cfg.split_seed)
    if distributed_state.is_main_process:
        print(f"  Train episodes: {len(train_eps)} | Val episodes: {len(val_eps)} "
              f"(val_ratio={cfg.val_ratio})")

    vla_dataset = XVLANpzDataset(
        npz_dir           = cfg.npz_data_dir,
        num_actions       = cfg.num_actions,
        domain_id         = cfg.domain_id,
        use_wrist_image   = cfg.use_wrist_image,
        resize_resolution = (224, 224),
        cache_episodes    = cfg.cache_episodes,
        episode_indices   = train_eps,
        split_tag         = "train",
        frame_stride      = cfg.frame_stride,
        min_chunk_motion  = cfg.min_chunk_motion,
    )
    val_dataset = None
    if val_eps:
        val_dataset = XVLANpzDataset(
            npz_dir           = cfg.npz_data_dir,
            num_actions       = cfg.num_actions,
            domain_id         = cfg.domain_id,
            use_wrist_image   = cfg.use_wrist_image,
            resize_resolution = (224, 224),
            cache_episodes    = cfg.cache_episodes,
            episode_indices   = val_eps,
            split_tag         = "val",
            frame_stride      = cfg.frame_stride,
            min_chunk_motion  = cfg.min_chunk_motion,
        )

    if distributed_state.is_main_process:
        import json
        stats_dst = run_dir / "xvla_dataset_statistics.json"
        with open(stats_dst, "w") as f:
            json.dump(vla_dataset.dataset_statistics, f, indent=2)
        # Record the exact split for reproducibility.
        with open(run_dir / "train_val_split.json", "w") as f:
            json.dump({"train_episodes": train_eps, "val_episodes": val_eps}, f, indent=2)
        print(f"  Saved dataset statistics → {stats_dst}")

    collator = XVLACollator(processor)
    if cfg.group_by_episode:
        # Episode-grouped batches → each episode decompressed ~once per batch,
        # reused via the per-worker LRU cache. This is the key data-loading speedup.
        batch_sampler = EpisodeBatchSampler(
            episode_spans = vla_dataset._episode_spans,
            batch_size    = cfg.batch_size,
            shuffle       = True,
            drop_last     = True,
            seed          = 0,
        )
        dataloader = DataLoader(
            vla_dataset,
            batch_sampler = batch_sampler,
            collate_fn    = collator,
            num_workers   = cfg.num_workers,
            pin_memory    = True,
            persistent_workers = cfg.num_workers > 0,
        )
    else:
        batch_sampler = None
        dataloader = DataLoader(
            vla_dataset,
            batch_size  = cfg.batch_size,
            shuffle     = True,
            collate_fn  = collator,
            num_workers = cfg.num_workers,
            pin_memory  = True,
            drop_last   = True,
            persistent_workers = cfg.num_workers > 0,
        )

    # Validation loader (no shuffle needed; smaller worker count is fine).
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size  = cfg.batch_size,
            shuffle     = False,
            collate_fn  = collator,
            num_workers = max(1, cfg.num_workers // 2),
            pin_memory  = True,
            drop_last   = False,
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
    # Training loop (flow-matching: model.forward returns a dict of losses)
    # ------------------------------------------------------------------
    recent_losses = deque(maxlen=10 * cfg.grad_accumulation_steps)
    recent_components: Dict[str, deque] = {}

    # Loss history for end-of-training curve plotting (main process only).
    # Each entry: (gradient_step, value). Components keep their own series.
    history = {
        "train_step": [], "train_loss": [],
        "train_components": {},               # name -> list[(step, val)]
        "val_step": [], "val_loss": [],
        "val_components": {},                 # name -> list[(step, val)]
    }

    # Early-stopping / best-checkpoint state (driven by validation loss).
    best_val_loss = float("inf")
    best_step = -1
    evals_since_improve = 0
    stop_early = False

    def run_validation_and_maybe_save(gstep: int, train_loss: float):
        """Validate, log, save best .pth, and update early-stop state."""
        nonlocal best_val_loss, best_step, evals_since_improve, stop_early
        val_metrics = evaluate(model, val_loader, device_id, cfg)
        if not val_metrics:
            return
        if distributed_state.is_main_process:
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=gstep)
            print(f"[val] step {gstep}: " + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
            # Record val history for the final loss curve.
            history["val_step"].append(gstep)
            history["val_loss"].append(val_metrics["loss"])
            for k, v in val_metrics.items():
                if k == "loss":
                    continue
                history["val_components"].setdefault(k, []).append((gstep, v))

        val_loss = val_metrics["loss"]
        improved = val_loss < (best_val_loss - cfg.early_stop_min_delta)
        if improved:
            best_val_loss = val_loss
            best_step = gstep
            evals_since_improve = 0
            if distributed_state.is_main_process and cfg.save_pth:
                save_pth_checkpoint(model, cfg, run_dir, gstep, val_loss,
                                    kind="best", val_metrics=val_metrics)
            # Snapshot the deployable best model (adapter for lora, full otherwise).
            if cfg.export_best_hf:
                snapshot_best_adapter(model, cfg, best_adapter_dir, distributed_state)
            dist.barrier()
        else:
            evals_since_improve += 1
            if cfg.early_stop_patience > 0 and evals_since_improve >= cfg.early_stop_patience:
                if distributed_state.is_main_process:
                    print(f"[early-stop] no val improvement for {evals_since_improve} evals "
                          f"(best={best_val_loss:.4f} @ step {best_step}). Stopping.")
                stop_early = True

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        model.train()
        optimizer.zero_grad()

        batch_idx = 0
        done = False
        epoch = 0
        last_train_loss = float("nan")
        while not done:
            if batch_sampler is not None:
                batch_sampler.set_epoch(epoch)
            for batch in dataloader:
                # ----------------------------------------------------------------
                # Move batch to device. Float tensors → bfloat16 to match the model;
                # integer tensors (input_ids, domain_id) and bool (image_mask) stay.
                # ----------------------------------------------------------------
                input_ids   = batch["input_ids"].to(device_id)
                image_input = batch["image_input"].to(torch.bfloat16).to(device_id)
                image_mask  = batch["image_mask"].to(device_id)
                domain_id   = batch["domain_id"].to(device_id)
                proprio     = batch["proprio"].to(torch.bfloat16).to(device_id)
                action      = batch["action"].to(torch.bfloat16).to(device_id)

                # ----------------------------------------------------------------
                # Forward pass — real X-VLA flow-matching interface.
                # Returns a dict like {"position_loss", "rotate6D_loss", "gripper_loss"}.
                # The diffusion-style noisy action mixture + per-component weighting
                # are handled INSIDE the model (see modeling_xvla.py / action_hub.py).
                # ----------------------------------------------------------------
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss_dict = model(
                        input_ids=input_ids,
                        image_input=image_input,
                        image_mask=image_mask,
                        domain_id=domain_id,
                        proprio=proprio,
                        action=action,
                    )
                    loss = reduce_loss_dict(loss_dict)

                # ----------------------------------------------------------------
                # Gradient accumulation
                # ----------------------------------------------------------------
                normalized_loss = loss / cfg.grad_accumulation_steps
                normalized_loss.backward()

                recent_losses.append(loss.item())
                last_train_loss = loss.item()
                for k, v in loss_dict.items():
                    recent_components.setdefault(k, deque(maxlen=10 * cfg.grad_accumulation_steps)).append(v.item())

                # ----------------------------------------------------------------
                # Gradient step
                # ----------------------------------------------------------------
                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
                is_step_boundary = (batch_idx + 1) % cfg.grad_accumulation_steps == 0

                if is_step_boundary:
                    # Per-group LR schedule (official X-VLA: warmup → cosine),
                    # with the LoRA group additionally frozen for `freeze_steps`.
                    # Stabilises full action-expert training (prevents position
                    # loss from oscillating).
                    for g in optimizer.param_groups:
                        g["lr"] = lr_at_step(gradient_step_idx, g["base_lr"], cfg, g["freeze_until"])
                    # Gradient clipping (official default max_grad_norm=1.0).
                    if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            (p for p in model.parameters() if p.requires_grad),
                            cfg.max_grad_norm,
                        )
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update()

                # ----------------------------------------------------------------
                # Logging (every 10 gradient steps)
                # ----------------------------------------------------------------
                if distributed_state.is_main_process and gradient_step_idx % 10 == 0 and is_step_boundary:
                    log_data = {"train/loss": sum(recent_losses) / len(recent_losses)}
                    for k, dq in recent_components.items():
                        log_data[f"train/{k}"] = sum(dq) / len(dq)
                    # Log current per-group LRs so the warmup/cosine schedule is visible.
                    for g in optimizer.param_groups:
                        log_data[f"lr/{g['name']}"] = g["lr"]
                    wandb.log(log_data, step=gradient_step_idx)
                    progress.set_postfix(loss=log_data["train/loss"], epoch=epoch)
                    # Record train history for the final loss curve.
                    history["train_step"].append(gradient_step_idx)
                    history["train_loss"].append(log_data["train/loss"])
                    for k, dq in recent_components.items():
                        history["train_components"].setdefault(k, []).append(
                            (gradient_step_idx, sum(dq) / len(dq))
                        )

                # ----------------------------------------------------------------
                # Validation + best-checkpoint + early stop
                # ----------------------------------------------------------------
                if val_loader is not None and is_step_boundary and gradient_step_idx > 0 \
                        and gradient_step_idx % cfg.val_every_steps == 0:
                    run_validation_and_maybe_save(gradient_step_idx, last_train_loss)

                # ----------------------------------------------------------------
                # Periodic .pth checkpoint (archival)
                # ----------------------------------------------------------------
                if distributed_state.is_main_process and cfg.save_pth and is_step_boundary \
                        and gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                    save_pth_checkpoint(model, cfg, run_dir, gradient_step_idx, last_train_loss, kind="step")

                batch_idx += 1

                # ----------------------------------------------------------------
                # Stop conditions
                # ----------------------------------------------------------------
                if stop_early:
                    done = True
                    break
                if gradient_step_idx >= cfg.max_steps:
                    print(f"Reached max_steps={cfg.max_steps}. Stopping training.")
                    done = True
                    break

            epoch += 1

    # ------------------------------------------------------------------
    # Final validation + final .pth
    # ------------------------------------------------------------------
    final_step = gradient_step_idx
    if val_loader is not None:
        final_val = evaluate(model, val_loader, device_id, cfg)
        if final_val and distributed_state.is_main_process:
            print(f"[val] final: " + ", ".join(f"{k}={v:.4f}" for k, v in final_val.items()))
            history["val_step"].append(final_step)
            history["val_loss"].append(final_val["loss"])
            for k, v in final_val.items():
                if k != "loss":
                    history["val_components"].setdefault(k, []).append((final_step, v))
    if distributed_state.is_main_process and cfg.save_pth:
        save_pth_checkpoint(model, cfg, run_dir, final_step, last_train_loss, kind="final")
        if best_step >= 0:
            print(f"Best val loss {best_val_loss:.4f} at step {best_step} "
                  f"(see pth/XVLA_{cfg.finetune_mode}_best.pth)")

    # ------------------------------------------------------------------
    # Save loss history + render loss curve
    # ------------------------------------------------------------------
    if distributed_state.is_main_process:
        save_loss_history(history, run_dir)

    # ------------------------------------------------------------------
    # Final HF model export (deployment-ready: AutoModel.from_pretrained)
    # ------------------------------------------------------------------
    if cfg.export_hf_on_finish:
        if distributed_state.is_main_process:
            print("Exporting final HF model directory ...")
        save_checkpoint(
            vla=model,
            processor=processor,
            cfg=cfg,
            run_dir=run_dir,
            adapter_dir=adapter_dir,
            step=final_step,
            distributed_state=distributed_state,
        )

    # ------------------------------------------------------------------
    # Best-model export → run_dir/best_hf (merged, deployment-ready).
    # Evaluate the BEST checkpoint with: --model_path <run_dir>/best_hf  (no --pth)
    # ------------------------------------------------------------------
    if cfg.export_best_hf and best_step >= 0:
        if distributed_state.is_main_process:
            print(f"Exporting BEST HF model (val loss {best_val_loss:.4f} @ step {best_step}) ...")
        export_best_hf(processor, cfg, best_adapter_dir, best_hf_dir, distributed_state)
    dist.barrier()

    if distributed_state.is_main_process:
        wandb.finish()
        print(f"Training complete. Artifacts in: {run_dir}")
        if cfg.export_best_hf and best_step >= 0:
            print(f"  Best model (deployable): {best_hf_dir}")


if __name__ == "__main__":
    finetune()
