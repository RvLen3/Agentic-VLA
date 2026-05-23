"""
npz_dataset.py

PyTorch Dataset for raw_demos .npz files.

Each .npz episode contains:
    images          uint8  (T, 256, 256, 3)   — main camera RGB
    images_wrist    uint8  (T, 256, 256, 3)   — wrist camera RGB
    tcp_poses       float64 (T, 6)            — [x, y, z, roll, pitch, yaw]
    joint_positions float64 (T, 6)            — joint angles
    gripper         float64 (T,)              — 0=open, 1=closed
    instruction     str    scalar             — language task description
    fps             int64  scalar             — recording frame rate

Each sample returned by __getitem__ is a single (obs, action) transition:
    pixel_values    torch.Tensor  — image processed by image_transform
    input_ids       torch.Tensor  — tokenized prompt + action tokens (1D, long)
    labels          torch.Tensor  — same as input_ids, prompt part masked to -100
    state           torch.Tensor  — proprio state [tcp_pose(6) + gripper(1)] = 7-dim

Action construction:
    action[t] = [Δtcp_pose(6), gripper_cmd(1)]
    where Δtcp_pose = tcp_poses[t+1] - tcp_poses[t]
    and   gripper_cmd = gripper[t+1]   (next-step gripper state as command)

Action normalization:
    Each action dimension is normalized to [-1, 1] using per-dimension
    statistics (mean + std, or min/max) computed over the full dataset.
    Statistics are saved to / loaded from a JSON file for reproducibility.
"""

import glob
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# IGNORE_INDEX matches HuggingFace / LLaMA convention
IGNORE_INDEX = -100

# Action dimension layout
ACTION_DIM = 7          # [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
PROPRIO_DIM = 7         # [x, y, z, roll, pitch, yaw, gripper]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_dataset_statistics(npz_dir: str) -> Dict:
    """
    Compute per-dimension mean, std, min, max over all transitions in the dataset.
    Returns a dict that can be serialised to JSON.
    """
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "episode_*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No episode_*.npz files found in {npz_dir}")

    all_actions: List[np.ndarray] = []
    all_states: List[np.ndarray] = []

    for fpath in npz_files:
        d = np.load(fpath, allow_pickle=True)
        tcp   = d["tcp_poses"].astype(np.float32)    # (T, 6)
        grip  = d["gripper"].astype(np.float32)      # (T,)
        d.close()

        T = tcp.shape[0]
        # Transitions: t = 0 … T-2
        delta_tcp = tcp[1:] - tcp[:-1]               # (T-1, 6)
        grip_cmd  = grip[1:].reshape(-1, 1)           # (T-1, 1)
        actions   = np.concatenate([delta_tcp, grip_cmd], axis=1)  # (T-1, 7)

        state = np.concatenate([tcp[:-1], grip[:-1].reshape(-1, 1)], axis=1)  # (T-1, 7)

        all_actions.append(actions)
        all_states.append(state)

    actions_np = np.concatenate(all_actions, axis=0)  # (N, 7)
    states_np  = np.concatenate(all_states,  axis=0)  # (N, 7)

    stats = {
        "action": {
            "mean":  actions_np.mean(axis=0).tolist(),
            "std":   actions_np.std(axis=0).tolist(),
            "min":   actions_np.min(axis=0).tolist(),
            "max":   actions_np.max(axis=0).tolist(),
            "q01":   np.percentile(actions_np, 1, axis=0).tolist(),
            "q99":   np.percentile(actions_np, 99, axis=0).tolist(),
        },
        "state": {
            "mean":  states_np.mean(axis=0).tolist(),
            "std":   states_np.std(axis=0).tolist(),
            "min":   states_np.min(axis=0).tolist(),
            "max":   states_np.max(axis=0).tolist(),
        },
        "num_transitions": int(actions_np.shape[0]),
        "num_episodes":    len(npz_files),
    }
    return stats


def load_or_compute_statistics(npz_dir: str, stats_path: Optional[str] = None) -> Dict:
    """Load statistics from JSON if it exists, otherwise compute and save."""
    if stats_path is None:
        stats_path = os.path.join(npz_dir, "dataset_statistics.json")

    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        print(f"[NpzDataset] Loaded dataset statistics from {stats_path}")
    else:
        print(f"[NpzDataset] Computing dataset statistics (this may take a moment) ...")
        stats = compute_dataset_statistics(npz_dir)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[NpzDataset] Saved dataset statistics to {stats_path}")

    return stats


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class ActionNormalizer:
    """
    Normalizes continuous actions to [-1, 1] using per-dimension min/max
    clipped at the 1st / 99th percentile to be robust to outliers.

    Gripper dimension (index 6) is already in {0, 1} — we map it to {-1, +1}
    via a simple linear transform instead of percentile clipping.
    """

    def __init__(self, stats: Dict):
        q01 = np.array(stats["action"]["q01"], dtype=np.float32)
        q99 = np.array(stats["action"]["q99"], dtype=np.float32)

        # Gripper: override with exact [0, 1] range
        q01[-1] = 0.0
        q99[-1] = 1.0

        # Avoid division by zero for near-constant dimensions
        rng = q99 - q01
        rng = np.where(rng < 1e-8, 1.0, rng)

        self.low  = q01
        self.high = q99
        self.rng  = rng

    def normalize(self, action: np.ndarray) -> np.ndarray:
        """Map action from original range to [-1, 1]."""
        normed = 2.0 * (action - self.low) / self.rng - 1.0
        return np.clip(normed, -1.0, 1.0).astype(np.float32)

    def denormalize(self, normed: np.ndarray) -> np.ndarray:
        """Map action from [-1, 1] back to original range."""
        return ((normed + 1.0) / 2.0 * self.rng + self.low).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NpzEpisodeDataset(Dataset):
    """
    Flat dataset over all (obs, action) transitions from a directory of .npz episodes.

    Each item is a dict with keys:
        pixel_values  torch.Tensor  — processed image (C, H, W) or whatever image_transform returns
        input_ids     torch.Tensor  — 1D long tensor, tokenized [prompt + action tokens]
        labels        torch.Tensor  — 1D long tensor, prompt positions = IGNORE_INDEX(-100)
        state         torch.Tensor  — float32 (PROPRIO_DIM,) proprio state at time t

    Parameters
    ----------
    npz_dir : str | Path
        Directory containing episode_*.npz files.
    action_tokenizer : ActionTokenizer
        Converts normalized continuous actions → token string.
    base_tokenizer : PreTrainedTokenizerBase
        LLM tokenizer for encoding the full prompt.
    image_transform : Callable
        Converts PIL.Image → torch.Tensor (e.g. processor.image_processor.apply_transform).
    prompt_builder_fn : Type
        PromptBuilder class (e.g. PurePromptBuilder).
    stats_path : str, optional
        Path to dataset_statistics.json. Auto-computed if missing.
    image_aug : bool
        If True, apply random horizontal flip + colour jitter to images.
    resize_resolution : Tuple[int, int]
        (H, W) to resize images before passing to image_transform.
    use_wrist_image : bool
        If True, use wrist camera instead of main camera.
    """

    def __init__(
        self,
        npz_dir: str,
        action_tokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: Callable,
        prompt_builder_fn: Type,
        stats_path: Optional[str] = None,
        image_aug: bool = False,
        resize_resolution: Tuple[int, int] = (224, 224),
        use_wrist_image: bool = False,
    ):
        self.npz_dir          = str(npz_dir)
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer   = base_tokenizer
        self.image_transform  = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.image_aug        = image_aug
        self.resize_resolution = resize_resolution
        self.use_wrist_image  = use_wrist_image

        # Load / compute statistics and build normalizer
        self.stats      = load_or_compute_statistics(self.npz_dir, stats_path)
        self.normalizer = ActionNormalizer(self.stats)

        # Build flat index: list of (episode_path, timestep_t)
        # Also cache episode lengths to avoid re-opening files later.
        self._index: List[Tuple[str, int]] = []
        self._ep_lengths: Dict[str, int] = {}
        npz_files = sorted(glob.glob(os.path.join(self.npz_dir, "episode_*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"No episode_*.npz files found in {self.npz_dir}")

        # Cache episode lengths in a sidecar JSON to avoid re-scanning on every run
        index_cache_path = os.path.join(self.npz_dir, "_episode_lengths.json")
        if os.path.exists(index_cache_path):
            with open(index_cache_path) as f:
                cached = json.load(f)
        else:
            cached = {}

        needs_save = False
        for fpath in npz_files:
            key = os.path.basename(fpath)
            if key in cached:
                T = cached[key]
            else:
                # Only open the file if not cached — npz is zip-compressed so
                # mmap_mode is ignored; we must fully decompress to get shape.
                d = np.load(fpath, allow_pickle=True)
                T = int(d["images"].shape[0])
                d.close()
                cached[key] = T
                needs_save = True

            self._ep_lengths[fpath] = T
            for t in range(T - 1):
                self._index.append((fpath, t))

        if needs_save:
            with open(index_cache_path, "w") as f:
                json.dump(cached, f)
            print(f"[NpzEpisodeDataset] Cached episode lengths → {index_cache_path}")

        print(
            f"[NpzEpisodeDataset] {len(npz_files)} episodes, "
            f"{len(self._index)} transitions, "
            f"image_aug={image_aug}"
        )

        # Optional image augmentation transforms
        if image_aug:
            try:
                import torchvision.transforms as T_vis
                self._aug = T_vis.Compose([
                    T_vis.RandomHorizontalFlip(p=0.5),
                    T_vis.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                ])
            except ImportError:
                print("[NpzEpisodeDataset] torchvision not available; image_aug disabled.")
                self._aug = None
                self.image_aug = False
        else:
            self._aug = None

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fpath, t = self._index[idx]

        # Load only the two timesteps we need.
        # Note: .npz files are zip-compressed; mmap_mode is not supported.
        # We load the full file but only index the rows we need.
        d = np.load(fpath, allow_pickle=True)

        # ── Image ──────────────────────────────────────────────────────
        img_key = "images_wrist" if self.use_wrist_image else "images"
        img_np  = np.array(d[img_key][t])                # force read: (256, 256, 3) uint8
        img_pil = Image.fromarray(img_np).convert("RGB")
        img_pil = img_pil.resize(
            (self.resize_resolution[1], self.resize_resolution[0]),
            Image.BILINEAR,
        )
        if self.image_aug and self._aug is not None:
            # Apply augmentation before image_transform
            import torchvision.transforms.functional as TF
            img_tensor_uint8 = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)  # (3, H, W)
            img_tensor_uint8 = self._aug(img_tensor_uint8)
            img_pil = Image.fromarray(img_tensor_uint8.permute(1, 2, 0).numpy())

        pixel_values = self.image_transform(img_pil)

        # ── Proprio state at time t ────────────────────────────────────
        tcp_t   = np.array(d["tcp_poses"][t],   dtype=np.float32)   # (6,)
        tcp_t1  = np.array(d["tcp_poses"][t+1], dtype=np.float32)   # (6,)
        grip_t  = float(d["gripper"][t])
        grip_t1 = float(d["gripper"][t+1])

        # ── Language instruction ───────────────────────────────────────
        instruction = str(d["instruction"])

        d.close()

        state   = np.concatenate([tcp_t, [grip_t]], axis=0).astype(np.float32)  # (7,)

        # ── Action: delta_tcp + next gripper ──────────────────────────
        delta_tcp  = tcp_t1 - tcp_t                                  # (6,)
        action_raw = np.concatenate([delta_tcp, [grip_t1]], axis=0)  # (7,)

        # ── Normalize action to [-1, 1] ────────────────────────────────
        action_norm = self.normalizer.normalize(action_raw)  # (7,) float32

        # ── Tokenize: build prompt + action token string ───────────────
        action_str = self.action_tokenizer(action_norm)

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {
                "from": "human",
                "value": f"What action should the robot take to {instruction}?",
            },
            {
                "from": "gpt",
                "value": action_str,
            },
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        prompt_text = prompt_builder.get_prompt()

        # Full sequence token IDs
        input_ids = torch.tensor(
            self.base_tokenizer(prompt_text, add_special_tokens=True).input_ids,
            dtype=torch.long,
        )

        # Labels: mask out everything except the action tokens
        # Strategy: tokenize the prompt-only part (without action), measure its length,
        # then set labels[:prompt_len] = IGNORE_INDEX
        action_token_ids = self.base_tokenizer(
            action_str, add_special_tokens=False
        ).input_ids
        n_action_tokens = len(action_token_ids) + 1  # +1 for EOS

        labels = input_ids.clone()
        prompt_len = len(input_ids) - n_action_tokens
        if prompt_len > 0:
            labels[:prompt_len] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            state=torch.tensor(state, dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # Expose statistics for saving (compatible with finetune_xvla.py)
    # ------------------------------------------------------------------

    @property
    def dataset_statistics(self) -> Dict:
        return self.stats
