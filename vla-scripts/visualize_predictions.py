"""
visualize_predictions.py

Diagnose a fine-tuned X-VLA checkpoint by comparing its PREDICTED action chunk
against the GROUND-TRUTH actions AND against a "predict-constant" baseline.

Why the baseline matters
-------------------------
On hand-collected data with many idle/near-static frames, imitation policies
often COLLAPSE: they learn to output "stay where you are" because that minimizes
loss on the (dominant) static frames. Such a model looks fine on a static window
but never actually moves.

To detect this we compare two MAEs on DYNAMIC windows (where the gt actually
moves a lot):
    model MAE       — error of the model's predicted chunk vs gt
    constant MAE    — error of "repeat the current pose for the whole chunk" vs gt
If model MAE ≈ constant MAE → the model has collapsed (learned nothing but "hold").
If model MAE << constant MAE → the model genuinely tracks motion.

What it does
------------
1. Loads model + processor (HF dir from training, or base ckpt + --pth adapter).
2. Selects evaluation windows:
     --auto_dynamic (default): picks the N windows with the LARGEST gt motion in
                               the episode (most informative; avoids idle frames).
     --start_t T            : use a single fixed window starting at T.
3. For each window: predicts, computes model MAE & constant-baseline MAE, and
   (for the first/most-dynamic window) plots pred vs gt vs baseline.
4. Prints an aggregate verdict over all windows.

Usage
-----
  python vla-scripts/visualize_predictions.py \
      --model_path runs/fruit_vla/<exp_id> \
      --npz_dir raw_demos_left_third --episode 0 \
      --auto_dynamic --num_windows 5 --out pred_vs_gt.png

  # base checkpoint + trainable-params .pth:
  python vla-scripts/visualize_predictions.py \
      --model_path 2toINF/X-VLA-Pt \
      --pth runs/fruit_vla/<exp_id>/pth/XVLA_soft_prompt_best.pth \
      --npz_dir raw_demos_left_third --episode 0 --auto_dynamic
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
        ckpt = torch.load(pth, map_location="cpu", weights_only=False)
        state = ckpt.get("trainable_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        n_loaded = len(state) - len(unexpected)
        print(f"    {len(state)} tensors in .pth | applied={n_loaded} "
              f"unexpected(unused)={len(unexpected)} missing(in-model)={len(missing)}")
        if n_loaded == 0:
            raise RuntimeError(
                "NONE of the .pth tensors matched the model parameter names — the "
                ".pth was NOT applied.\n"
                "This happens when --model_path points to a MERGED/exported HF dir "
                "(plain names like 'transformer....') while the .pth holds PEFT-style "
                "names ('base_model.model....lora_A...'). A LoRA .pth cannot be layered "
                "onto a merged model anyway.\n"
                "FIX: evaluate the merged BEST model directly with\n"
                "     --model_path <run_dir>/best_hf   (and DROP --pth)."
            )
        if n_loaded < len(state):
            print(f"    [warn] {len(unexpected)} .pth tensors were ignored (name mismatch).")
        if ckpt.get("step") is not None:
            print(f"    checkpoint step={ckpt.get('step')} loss={ckpt.get('loss')}")
    model = model.to(device).eval()
    return model


def gt_motion(item) -> float:
    """Total gt end-effector travel within the chunk (metres). Higher = more dynamic."""
    pos = item["action"][:, ARM1_POS]
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())


def predict_chunk(model, processor, item, device, steps):
    collate = XVLACollator(processor)
    batch = collate([item])
    with torch.no_grad():
        proc_inputs = {
            "input_ids": batch["input_ids"].to(device),
            "image_input": batch["image_input"].to(torch.bfloat16).to(device),
            "image_mask": batch["image_mask"].to(device),
            "proprio": batch["proprio"].to(torch.bfloat16).to(device),
            "domain_id": batch["domain_id"].to(device),
        }
        pred = model.generate_actions(**proc_inputs, steps=steps)
    return pred.squeeze(0).float().cpu().numpy()   # (num_actions, 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="HF model dir or base checkpoint ID")
    ap.add_argument("--pth", default=None, help="Optional trainable-params .pth to apply on top")
    ap.add_argument("--npz_dir", default="raw_demos_left_third")
    ap.add_argument("--episode", type=int, default=0, help="Episode index within npz_dir")
    ap.add_argument("--start_t", type=int, default=None, help="Fixed start timestep (disables auto_dynamic)")
    ap.add_argument("--auto_dynamic", action="store_true", default=True,
                    help="Auto-pick the most dynamic windows (default)")
    ap.add_argument("--num_windows", type=int, default=5, help="How many dynamic windows to evaluate")
    ap.add_argument("--num_actions", type=int, default=30)
    ap.add_argument("--frame_stride", type=int, default=1, help="Must match training frame_stride")
    ap.add_argument("--domain_id", type=int, default=3)
    ap.add_argument("--use_wrist_image", action="store_true", default=True)
    ap.add_argument("--steps", type=int, default=10, help="Flow-matching denoising steps")
    ap.add_argument("--out", default="pred_vs_gt.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_model(args.model_path, args.pth, device)

    ds = XVLANpzDataset(
        npz_dir=args.npz_dir,
        num_actions=args.num_actions,
        domain_id=args.domain_id,
        use_wrist_image=args.use_wrist_image,
        episode_indices=[args.episode],
        split_tag="viz",
        frame_stride=args.frame_stride,
    )

    # ---- choose windows (flat indices into ds) -----------------------------
    if args.start_t is not None:
        chosen = [next(i for i, (ep, t) in enumerate(ds._index) if t == args.start_t)]
        print(f"[*] Using fixed window start_t={args.start_t}")
    else:
        # Rank all windows by gt motion, take the top-N most dynamic.
        motions = [(i, gt_motion(ds[i])) for i in range(len(ds._index))]
        motions.sort(key=lambda x: x[1], reverse=True)
        chosen = [i for i, _ in motions[:args.num_windows]]
        print(f"[*] Auto-selected {len(chosen)} most-dynamic windows "
              f"(gt travel {motions[0][1]:.3f} … {motions[len(chosen)-1][1]:.3f} m)")

    # ---- evaluate each window: model MAE vs constant-baseline MAE -----------
    model_maes, base_maes, grip_accs = [], [], []
    first_window_data = None
    for rank, flat in enumerate(chosen):
        item = ds[flat]
        gt = item["action"]
        pos_gt = gt[:, ARM1_POS]
        grip_gt = gt[:, ARM1_GRIPPER]

        # Constant baseline: repeat the CURRENT pose (proprio) for the whole chunk.
        proprio = item["proprio"]
        pos_const = np.tile(proprio[ARM1_POS], (args.num_actions, 1))

        pred = predict_chunk(model, processor, item, device, args.steps)
        pos_pred = pred[:, ARM1_POS]
        grip_pred = pred[:, ARM1_GRIPPER]

        model_mae = np.abs(pos_pred - pos_gt).mean()
        base_mae = np.abs(pos_const - pos_gt).mean()
        grip_acc = (np.round(np.clip(grip_pred, 0, 1)) == np.round(grip_gt)).mean()
        model_maes.append(model_mae); base_maes.append(base_mae); grip_accs.append(grip_acc)

        _, t = ds._index[flat]
        print(f"  window#{rank} start_t={t:4d} | model MAE={model_mae:.4f}  "
              f"const MAE={base_mae:.4f}  ratio={model_mae/max(base_mae,1e-9):.2f}  "
              f"grip_acc={grip_acc:.2f}")

        if rank == 0:
            first_window_data = dict(
                t=t, pos_gt=pos_gt, pos_pred=pos_pred, pos_const=pos_const,
                grip_gt=grip_gt, grip_pred=grip_pred,
                instruction=item["instruction"],
            )

    # ---- aggregate verdict --------------------------------------------------
    m = float(np.mean(model_maes)); b = float(np.mean(base_maes))
    ratio = m / max(b, 1e-9)
    print("\n" + "=" * 60)
    print(f"AGGREGATE over {len(chosen)} dynamic windows:")
    print(f"  model position MAE : {m:.4f} m")
    print(f"  constant baseline  : {b:.4f} m")
    print(f"  ratio (model/base) : {ratio:.2f}")
    print(f"  gripper match      : {float(np.mean(grip_accs)):.2f}")
    if ratio > 0.9:
        print("  VERDICT: ⚠️  model ≈ constant baseline → likely COLLAPSED "
              "(not tracking motion). Suspect idle-frame dominance / underfitting.")
    elif ratio > 0.6:
        print("  VERDICT: 🟡 model only mildly beats 'hold still' → weak motion learning.")
    else:
        print("  VERDICT: ✅ model clearly beats baseline → it IS tracking motion.")
    print("=" * 60)

    # ---- plot the most dynamic window --------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: pip install matplotlib (skipping plot)")
        return

    d = first_window_data
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = ["x", "y", "z"]
    colors = ["tab:blue", "tab:green", "tab:purple"]
    for i, lbl in enumerate(labels):
        axes[0, 0].plot(d["pos_gt"][:, i], color=colors[i], linestyle="-", label=f"gt {lbl}")
        axes[0, 0].plot(d["pos_pred"][:, i], color=colors[i], linestyle="--", label=f"pred {lbl}")
        axes[0, 0].plot(d["pos_const"][:, i], color=colors[i], linestyle=":", alpha=0.5)
    axes[0, 0].set_title("Absolute position: gt(—) pred(- -) const(··)")
    axes[0, 0].set_xlabel("chunk step"); axes[0, 0].set_ylabel("metres"); axes[0, 0].legend(fontsize=7, ncol=3)

    dgt = np.diff(d["pos_gt"], axis=0, prepend=d["pos_gt"][:1])
    dpred = np.diff(d["pos_pred"], axis=0, prepend=d["pos_pred"][:1])
    for i, lbl in enumerate(labels):
        axes[0, 1].plot(dgt[:, i], color=colors[i], linestyle="-", label=f"gt Δ{lbl}")
        axes[0, 1].plot(dpred[:, i], color=colors[i], linestyle="--", label=f"pred Δ{lbl}")
    axes[0, 1].set_title("Per-step movement (diff of absolute)")
    axes[0, 1].set_xlabel("chunk step"); axes[0, 1].set_ylabel("Δ metres"); axes[0, 1].legend(fontsize=7, ncol=3)

    axes[1, 0].plot(d["grip_gt"], label="gt gripper", marker="o")
    axes[1, 0].plot(d["grip_pred"], label="pred gripper", marker="x")
    axes[1, 0].set_title("Gripper (0=open, 1=closed)")
    axes[1, 0].set_xlabel("chunk step"); axes[1, 0].set_ylim(-0.1, 1.1); axes[1, 0].legend(fontsize=8)

    axes[1, 1].axis("off")
    verdict = ("COLLAPSED" if ratio > 0.9 else "weak" if ratio > 0.6 else "tracking motion")
    axes[1, 1].text(0.05, 0.5,
                    f"episode: {args.episode}  (most dynamic window: start_t={d['t']})\n"
                    f"num_actions: {args.num_actions}\n\n"
                    f"model MAE   : {m:.4f} m\n"
                    f"const MAE   : {b:.4f} m\n"
                    f"ratio       : {ratio:.2f}  → {verdict}\n"
                    f"gripper match: {float(np.mean(grip_accs)):.2f}\n\n"
                    f"instruction:\n{d['instruction']}",
                    fontsize=11, va="center", family="monospace")

    fig.suptitle("X-VLA pred vs ground-truth vs constant-baseline (dynamic window)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"[*] Saved plot → {args.out}")


if __name__ == "__main__":
    main()
