"""
test_npz_pipeline.py

Dry-run test for the NpzEpisodeDataset → DataLoader pipeline.

Does NOT require a real X-VLA model or GPU.  Uses a tiny mock tokenizer /
image transform so the full data path can be validated quickly.

Run:
    python vla-scripts/test_npz_pipeline.py \
        --npz_dir  e:/ACoT-VLA-Test/raw_demos \
        --batch_size 4 \
        --num_batches 3
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Make sure the repo root is on sys.path ─────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "vla-scripts"))
from npz_dataset import (          # noqa: E402
    NpzEpisodeDataset,
    compute_dataset_statistics,
    ActionNormalizer,
    IGNORE_INDEX,
    ACTION_DIM,
    PROPRIO_DIM,
)

# ── Mock objects (no real model needed) ───────────────────────────────────

class MockTokenizer:
    """Minimal tokenizer stub that maps strings to fixed-length token IDs."""
    model_max_length = 512
    pad_token_id     = 0
    eos_token_id     = 2

    # Simulate a vocab where the last 256 IDs are action tokens
    VOCAB_SIZE = 32_000

    def __call__(self, text: str, add_special_tokens: bool = True):
        # Deterministic fake encoding: hash each char to a token ID
        ids = [hash(c) % (self.VOCAB_SIZE - 256) + 1 for c in text[:60]]
        if add_special_tokens:
            ids = [1] + ids + [self.eos_token_id]
        return type("Enc", (), {"input_ids": ids})()

    def convert_ids_to_tokens(self, ids):
        return [f"<tok_{i}>" for i in ids]

    def get_vocab(self):
        return {f"<tok_{i}>": i for i in range(self.VOCAB_SIZE)}


class MockActionTokenizer:
    """Stub that mimics ActionTokenizer interface."""
    action_token_begin_idx = MockTokenizer.VOCAB_SIZE - 256

    def __call__(self, action: np.ndarray) -> str:
        # Return a fixed-length string of fake action tokens
        return " ".join([f"<tok_{self.action_token_begin_idx + i}>" for i in range(len(action))])

    def decode_token_ids_to_actions(self, token_ids: np.ndarray) -> np.ndarray:
        bin_indices = token_ids - self.action_token_begin_idx
        bin_indices = np.clip(bin_indices, 0, 255)
        return (bin_indices / 255.0 * 2.0 - 1.0).astype(np.float32)


class MockPromptBuilder:
    """Stub PromptBuilder."""
    def __init__(self, model_family: str):
        self._turns = []

    def add_turn(self, role: str, content: str):
        self._turns.append(f"{role}: {content}")

    def get_prompt(self) -> str:
        return " | ".join(self._turns)


def mock_image_transform(img):
    """Convert PIL image to a (3, 224, 224) float tensor."""
    import torchvision.transforms.functional as TF
    return TF.to_tensor(img)


# ── Collator (mirrors PaddedCollatorForActionPrediction) ──────────────────

def simple_collate(instances):
    """Pad input_ids / labels, stack pixel_values and state."""
    from torch.nn.utils.rnn import pad_sequence

    input_ids    = pad_sequence([x["input_ids"]    for x in instances], batch_first=True, padding_value=0)
    labels       = pad_sequence([x["labels"]       for x in instances], batch_first=True, padding_value=IGNORE_INDEX)
    pixel_values = torch.stack([x["pixel_values"]  for x in instances])
    state        = torch.stack([x["state"]         for x in instances])
    attention_mask = input_ids.ne(0)

    return dict(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        state=state,
    )


# ── Main test ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir",    default="e:/ACoT-VLA-Test/raw_demos")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches",type=int, default=3)
    parser.add_argument("--image_aug",  action="store_true", default=False)
    args = parser.parse_args()

    print("=" * 60)
    print("NpzEpisodeDataset pipeline dry-run")
    print(f"  npz_dir    : {args.npz_dir}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  image_aug  : {args.image_aug}")
    print("=" * 60)

    # ── 1. Statistics ──────────────────────────────────────────────────
    print("\n[1] Dataset statistics")
    stats_path = os.path.join(args.npz_dir, "dataset_statistics.json")
    if not os.path.exists(stats_path):
        print("  Computing statistics ...")
        stats = compute_dataset_statistics(args.npz_dir)
        import json
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved → {stats_path}")
    else:
        import json
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"  Loaded from {stats_path}")

    print(f"  episodes    : {stats['num_episodes']}")
    print(f"  transitions : {stats['num_transitions']}")
    action_labels = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]
    print(f"\n  {'dim':<10} {'q01':>10} {'q99':>10}  (normalization range)")
    for i, lbl in enumerate(action_labels):
        print(f"  {lbl:<10} {stats['action']['q01'][i]:>10.5f} {stats['action']['q99'][i]:>10.5f}")

    # ── 2. Normalizer sanity check ─────────────────────────────────────
    print("\n[2] Normalizer round-trip check")
    normalizer = ActionNormalizer(stats)
    rng = np.random.default_rng(42)
    raw_action = np.array(stats["action"]["mean"], dtype=np.float32)
    normed     = normalizer.normalize(raw_action)
    recovered  = normalizer.denormalize(normed)
    print(f"  raw    : {raw_action}")
    print(f"  normed : {normed}")
    print(f"  recovered : {recovered}")
    max_err = np.abs(raw_action - recovered).max()
    assert max_err < 1e-4, f"Round-trip error too large: {max_err}"
    print(f"  max round-trip error : {max_err:.2e}  ✓")

    # ── 3. Build dataset ───────────────────────────────────────────────
    print("\n[3] Building NpzEpisodeDataset ...")
    t0 = time.perf_counter()
    dataset = NpzEpisodeDataset(
        npz_dir          = args.npz_dir,
        action_tokenizer = MockActionTokenizer(),
        base_tokenizer   = MockTokenizer(),
        image_transform  = mock_image_transform,
        prompt_builder_fn= MockPromptBuilder,
        image_aug        = args.image_aug,
        resize_resolution= (224, 224),
    )
    print(f"  Dataset built in {time.perf_counter() - t0:.2f}s")
    print(f"  len(dataset) = {len(dataset)}")

    # ── 4. Single item check ───────────────────────────────────────────
    print("\n[4] Single item check (idx=0)")
    item = dataset[0]
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:<15} shape={tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:<15} {v}")

    # Verify labels masking
    n_valid = (item["labels"] != IGNORE_INDEX).sum().item()
    n_total = len(item["labels"])
    print(f"\n  labels: {n_valid}/{n_total} positions are action tokens (rest = IGNORE_INDEX)")
    assert n_valid > 0, "No action tokens found in labels!"
    assert n_valid < n_total, "Prompt masking not applied!"
    print("  Label masking ✓")

    # Verify state dimension
    assert item["state"].shape == (PROPRIO_DIM,), f"Expected state shape ({PROPRIO_DIM},), got {item['state'].shape}"
    print(f"  state shape ✓  values: {item['state'].numpy()}")

    # ── 5. DataLoader batch check ──────────────────────────────────────
    print(f"\n[5] DataLoader — {args.num_batches} batches of size {args.batch_size}")
    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        collate_fn  = simple_collate,
        num_workers = 0,
    )

    t0 = time.perf_counter()
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.num_batches:
            break

        B = batch["input_ids"].shape[0]
        L = batch["input_ids"].shape[1]
        print(f"\n  Batch {batch_idx}:")
        for k, v in batch.items():
            print(f"    {k:<16} shape={tuple(v.shape)}  dtype={v.dtype}")

        # Sanity checks
        assert batch["pixel_values"].shape == (B, 3, 224, 224), \
            f"Unexpected pixel_values shape: {batch['pixel_values'].shape}"
        assert batch["attention_mask"].shape == (B, L)
        assert batch["state"].shape == (B, PROPRIO_DIM)
        assert batch["labels"].shape == (B, L)

        # Check that attention_mask is 1 for non-pad tokens
        pad_mask = batch["input_ids"] == 0
        assert (batch["attention_mask"][pad_mask] == 0).all(), "Padding tokens should have attention_mask=0"

        print(f"    attention_mask non-zero: {batch['attention_mask'].sum().item()} / {B * L}")
        print(f"    labels non-ignore: {(batch['labels'] != IGNORE_INDEX).sum().item()} / {B * L}")

    elapsed = time.perf_counter() - t0
    print(f"\n  {args.num_batches} batches loaded in {elapsed:.2f}s  "
          f"({elapsed / args.num_batches * 1000:.0f} ms/batch)")

    # ── 6. Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("All checks passed ✓")
    print("\nDataset is ready for finetune_xvla.py.")
    print("Key dimensions for finetune_xvla.py config:")
    print(f"  proprio_dim = {PROPRIO_DIM}")
    print(f"  action_dim  = {ACTION_DIM}")
    print(f"  image size  = 224×224 (resize from 256×256)")
    print(f"  domain_id   = 3  (LIBERO / custom; adjust as needed)")
    print("=" * 60)


if __name__ == "__main__":
    main()
