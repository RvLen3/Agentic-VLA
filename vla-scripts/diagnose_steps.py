"""
diagnose_steps.py

Answer one question definitively: is X-VLA's denoising actually iterating, or is
`steps` effectively a no-op for this (fine-tuned) checkpoint?

It runs the SAME observation through generate_actions with different `steps`
values AND with controlled random seeds, then reports element-wise differences.

Interpretation
--------------
A) Different `steps`, SAME seed → if outputs are bit-identical for steps>=2, the
   model is insensitive to the denoising trajectory (predict-x0 collapsed to a
   one-shot regressor). Expected to differ at least slightly if iteration matters.
B) Same `steps`, DIFFERENT seed (different initial noise x1) → if outputs barely
   change, the model ignores the noise input entirely (flow-matching degenerated
   into deterministic regression conditioned only on vision+proprio).
C) We also re-implement the denoising loop locally and print the per-iteration
   change ‖action_k - action_{k-1}‖ so you can SEE whether later iterations move.

Usage
-----
  python vla-scripts/diagnose_steps.py \
      --model_path runs/fruit_vla/<EXP> \
      --pth runs/fruit_vla/<EXP>/pth/XVLA_lora_best.pth \
      --npz_dir raw_demos_left_third --episode 66 --start_t 215
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_CACHE_DIR = (Path(__file__).parent / ".." / ".cache").resolve()
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_CACHE_DIR / "hub"))

from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from xvla_npz_dataset import XVLANpzDataset, XVLACollator, ARM1_POS

# Reuse the (now fixed) loader so the diagnostic uses exactly the eval weights.
from visualize_predictions import load_model


def build_inputs(model_path, npz_dir, episode, start_t, num_actions, domain_id,
                 use_wrist_image, frame_stride, device):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    ds = XVLANpzDataset(
        npz_dir=npz_dir, num_actions=num_actions, domain_id=domain_id,
        use_wrist_image=use_wrist_image, episode_indices=[episode],
        split_tag="diag", frame_stride=frame_stride,
    )
    flat = next((i for i, (ep, t) in enumerate(ds._index) if t == start_t), None)
    if flat is None:
        flat = 0
        print(f"[!] start_t={start_t} not found; using first window (t={ds._index[0][1]})")
    item = ds[flat]
    batch = XVLACollator(processor)([item])
    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "image_input": batch["image_input"].to(torch.bfloat16).to(device),
        "image_mask": batch["image_mask"].to(device),
        "proprio": batch["proprio"].to(torch.bfloat16).to(device),
        "domain_id": batch["domain_id"].to(device),
    }
    return inputs, item


def gen_with_seed(model, inputs, steps, seed, device):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    out = model.generate_actions(**inputs, steps=steps)
    return out.squeeze(0).float().cpu().numpy()


def manual_denoise_trace(model, inputs, steps, seed, device):
    """
    Re-implement generate_actions' loop locally to print per-iteration movement.
    Mirrors modeling_xvla.generate_actions exactly.
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    core = model.module if hasattr(model, "module") else model
    enc = core.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
    B = inputs["input_ids"].shape[0]
    D = core.action_space.dim_action
    proprio = inputs["proprio"]
    x1 = torch.randn(B, core.num_actions, D, device=proprio.device, dtype=proprio.dtype)
    action = torch.zeros_like(x1)
    prev = None
    moves = []
    for i in range(steps, 0, -1):
        t = torch.full((B,), i / steps, device=proprio.device, dtype=proprio.dtype)
        x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, x_t_m = core.action_space.preprocess(proprio, x_t)
        action = core.transformer(domain_id=inputs["domain_id"], action_with_noise=x_t_m,
                                  proprio=proprio_m, t=t, **enc)
        cur = action.float()
        if prev is not None:
            moves.append(float((cur - prev).abs().max().item()))
        prev = cur.clone()
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--pth", default=None)
    ap.add_argument("--npz_dir", default="raw_demos_left_third")
    ap.add_argument("--episode", type=int, default=66)
    ap.add_argument("--start_t", type=int, default=215)
    ap.add_argument("--num_actions", type=int, default=30)
    ap.add_argument("--frame_stride", type=int, default=3)
    ap.add_argument("--domain_id", type=int, default=3)
    ap.add_argument("--use_wrist_image", action="store_true", default=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_path, args.pth, device)
    inputs, item = build_inputs(args.model_path, args.npz_dir, args.episode, args.start_t,
                                args.num_actions, args.domain_id, args.use_wrist_image,
                                args.frame_stride, device)

    print("\n" + "=" * 64)
    print("TEST A — different steps, SAME seed (=0). Compare to steps=1 output.")
    print("=" * 64)
    ref = gen_with_seed(model, inputs, steps=1, seed=0, device=device)
    for s in [2, 5, 10, 50, 200]:
        out = gen_with_seed(model, inputs, steps=s, seed=0, device=device)
        max_abs = np.abs(out - ref).max()
        mean_abs = np.abs(out - ref).mean()
        print(f"  steps={s:4d} vs steps=1 : max|Δ|={max_abs:.3e}  mean|Δ|={mean_abs:.3e}")

    print("\n" + "=" * 64)
    print("TEST B — SAME steps (=10), DIFFERENT seed (initial noise x1).")
    print("If outputs barely change → model IGNORES the noise (deterministic).")
    print("=" * 64)
    base = gen_with_seed(model, inputs, steps=10, seed=0, device=device)
    for sd in [1, 2, 3]:
        out = gen_with_seed(model, inputs, steps=10, seed=sd, device=device)
        max_abs = np.abs(out - base).max()
        print(f"  seed={sd} vs seed=0 : max|Δ|={max_abs:.3e}")

    print("\n" + "=" * 64)
    print("TEST C — per-iteration movement within one generate (steps=50, seed=0).")
    print("Each value = max|action_k - action_{k-1}|. If they drop to ~0 fast,")
    print("the model converges in 1-2 steps (extra steps are wasted).")
    print("=" * 64)
    moves = manual_denoise_trace(model, inputs, steps=50, seed=0, device=device)
    shown = moves[:8] + (["..."] if len(moves) > 12 else []) + moves[-4:] if len(moves) > 12 else moves
    print("  per-step movement:", [f"{m:.3e}" if isinstance(m, float) else m for m in shown])
    if moves:
        print(f"  first step move={moves[0]:.3e} | last step move={moves[-1]:.3e}")

    print("\nVERDICT GUIDE:")
    print("  TEST A all ~0  + TEST C drops to ~0 after step 1  → steps genuinely")
    print("        don't matter for THIS model (predict-x0 one-shot). Not a bug.")
    print("  TEST A all EXACTLY 0.000e+00 → suspicious; steps may not reach compute.")
    print("  TEST B ~0 → model ignores initial noise (flow-matching degenerated to")
    print("        deterministic regression conditioned on vision+proprio).")


if __name__ == "__main__":
    main()
