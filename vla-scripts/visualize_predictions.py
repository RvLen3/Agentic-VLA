"""
visualize_predictions.py

Sanity-check a fine-tuned X-VLA checkpoint by overlaying its PREDICTED action
chunk against the GROUND-TRUTH actions from a recorded npz episode.

For small-data fine-tuning, this is far more informative than the raw loss
number: you can directly see whether the predicted end-effector trajectory and
gripper open/close timing track the demonstration.

What it does
------------
1. Loads the model + processor (either the exported HF dir from training, or the
   base checkpoint with a `--pth` adapter of trainable params applied on top).
2. Picks an episode and a start timestep, builds the same EE6D inputs the
   training dataset uses (via XVLANpzDataset), and calls model.generate_actions.
3. Decodes both predicted and ground-truth EE6D chunks to:
      - xyz position deltas (metres)
      - gripper open/close
   and plots them per-dimension to PNG.

Usage
-----
  # Using the HF model dir produced at end of training:
  python vla-scripts/visualize_predictions.py \
      --model_path runs/xvla/<exp_id> \
      --npz_dir raw_demos_left_third \
      --episode 0 --start_t 100 \
      --out pred_vs_gt.png

  # Using base checkpoint + a trainable-params .pth:
  python vla-scripts/visualize_predictions.py \
      --model_path 2toINF/X-VLA-Pt \
      --pth runs/xvla/<exp_id>/pth/XVLA_soft_prompt_best.pth \
      --npz_dir raw_demos_left_third --episode 0 --start_t 100

Notes
-----
- Requires matplotlib. Runs on GPU if available, else CPU (slow but fine for a
  one-off check).
- The model outputs 20-D EE6D actions; we only plot arm-1 (xyz + gripper) since
  the UR is single-arm and arm-2 channels are zero-padded.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Cache to repo-root .cache (consistent with finetune_xvla.py), set before HF import.
_CACHE_DIR = (Path(__file__).parent / ".." / ".cache").resolve()
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_CACHE_DIR / "hub"))

from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from xvla_npz_dataset import XVLANpzDataset, XVLACollator, ARM1_POS, ARM1_GRIPPER


def load_model(model_path: str, pth: str = None, device="cuda"):
    print(f"[*] Loading model from {model_path}")
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    )
    if pth:
        print(f"[*] Applying trainable params from {pth}")
        ckpt = torch.load(pth, map_location="cpu")
        state = ckpt.get("trainable_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"    loaded {len(state)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
        if ckpt.get("step") is not None:
            print(f"    checkpoint step={ckpt.get('step')} loss={ckpt.get('loss')}")
    model = model.to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="HF model dir or base checkpoint ID")
    ap.add_argument("--pth", default=None, help="Optional trainable-params .pth to apply on top")
    ap.add_argument("--npz_dir", default="raw_demos_left_third")
    ap.add_argument("--episode", type=int, default=0, help="Episode index within npz_dir")
    ap.add_argument("--start_t", type=int, default=0, help="Start timestep for the action chunk")
    ap.add_argument("--num_actions", type=int, default=30)
    ap.add_argument("--domain_id", type=int, default=3)
    ap.add_argument("--use_wrist_image", action="store_true", default=True)
    ap.add_argument("--steps", type=int, default=10, help="Flow-matching denoising steps")
    ap.add_argument("--out", default="pred_vs_gt.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_model(args.model_path, args.pth, device)

    # Build the same EE6D inputs the trainer uses, for one (episode, start_t).
    ds = XVLANpzDataset(
        npz_dir=args.npz_dir,
        num_actions=args.num_actions,
        domain_id=args.domain_id,
        use_wrist_image=args.use_wrist_image,
        episode_indices=[args.episode],
        split_tag="viz",
    )
    # Find the flat index for (episode, start_t).
    target = None
    for flat_i, (ep, t) in enumerate(ds._index):
        if t == args.start_t:
            target = flat_i
            break
    if target is None:
        raise ValueError(f"start_t={args.start_t} not found (episode length {ds._ep_lengths.get(args.episode)})")

    item = ds[target]
    collate = XVLACollator(processor)
    batch = collate([item])

    # Predict
    with torch.no_grad():
        proc_inputs = {
            "input_ids": batch["input_ids"].to(device),
            "image_input": batch["image_input"].to(torch.bfloat16).to(device),
            "image_mask": batch["image_mask"].to(device),
            "proprio": batch["proprio"].to(torch.bfloat16).to(device),
            "domain_id": batch["domain_id"].to(device),
        }
        pred = model.generate_actions(**proc_inputs, steps=args.steps)
        pred = pred.squeeze(0).float().cpu().numpy()   # (num_actions, 20)

    gt = item["action"]                                # (num_actions, 20)

    # Extract arm-1 xyz + gripper
    pos_pred, pos_gt = pred[:, ARM1_POS], gt[:, ARM1_POS]
    grip_pred, grip_gt = pred[:, ARM1_GRIPPER], gt[:, ARM1_GRIPPER]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: pip install matplotlib")
        # Fall back to a text summary
        mae = np.abs(pos_pred - pos_gt).mean()
        print(f"position MAE: {mae:.5f} m | gripper match: "
              f"{(np.round(grip_pred) == np.round(grip_gt)).mean():.2f}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = ["x", "y", "z"]
    # Actions are ABSOLUTE EEF positions → plot them directly (no cumsum).
    for i, lbl in enumerate(labels):
        axes[0, 0].plot(pos_gt[:, i], label=f"gt {lbl}", linestyle="-")
        axes[0, 0].plot(pos_pred[:, i], label=f"pred {lbl}", linestyle="--")
    axes[0, 0].set_title("Absolute position (base frame)")
    axes[0, 0].set_xlabel("chunk step"); axes[0, 0].set_ylabel("metres"); axes[0, 0].legend(fontsize=8)

    # Per-step movement (diff of absolute) — shows velocity profile.
    dgt = np.diff(pos_gt, axis=0, prepend=pos_gt[:1])
    dpred = np.diff(pos_pred, axis=0, prepend=pos_pred[:1])
    for i, lbl in enumerate(labels):
        axes[0, 1].plot(dgt[:, i], label=f"gt Δ{lbl}", linestyle="-")
        axes[0, 1].plot(dpred[:, i], label=f"pred Δ{lbl}", linestyle="--")
    axes[0, 1].set_title("Per-step movement (diff of absolute)")
    axes[0, 1].set_xlabel("chunk step"); axes[0, 1].set_ylabel("Δ metres"); axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(grip_gt, label="gt gripper", marker="o")
    axes[1, 0].plot(grip_pred, label="pred gripper", marker="x")
    axes[1, 0].set_title("Gripper (0=open, 1=closed)")
    axes[1, 0].set_xlabel("chunk step"); axes[1, 0].set_ylim(-0.1, 1.1); axes[1, 0].legend(fontsize=8)

    pos_mae = np.abs(pos_pred - pos_gt).mean()
    grip_acc = (np.round(np.clip(grip_pred, 0, 1)) == np.round(grip_gt)).mean()
    axes[1, 1].axis("off")
    axes[1, 1].text(0.05, 0.5,
                    f"episode: {args.episode}\nstart_t: {args.start_t}\n"
                    f"num_actions: {args.num_actions}\n\n"
                    f"position MAE: {pos_mae:.5f} m  (absolute)\n"
                    f"gripper match: {grip_acc:.2f}\n"
                    f"instruction:\n{item['instruction']}",
                    fontsize=11, va="center")

    fig.suptitle(f"X-VLA predicted vs ground-truth action chunk (absolute EEF)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"[*] Saved plot → {args.out}")
    print(f"    position MAE: {pos_mae:.5f} m | gripper match: {grip_acc:.2f}")


if __name__ == "__main__":
    main()
