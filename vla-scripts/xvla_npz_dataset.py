"""
xvla_npz_dataset.py

X-VLA-native PyTorch Dataset for raw_demos .npz files.

Unlike `npz_dataset.py` (which tokenises actions into text for an OpenVLA-style
autoregressive model), this dataset produces the *continuous* inputs that the
real X-VLA flow-matching model consumes in its `forward()`:

    model.forward(
        input_ids,      # [B, L]                 (from XVLAProcessor / BartTokenizer)
        image_input,    # [B, num_views, C, H, W] (from XVLAProcessor image processor)
        image_mask,     # [B, num_views]          (bool: which views are valid)
        domain_id,      # [B]                     (embodiment / soft-prompt index)
        proprio,        # [B, dim_action]         (EE6D current state, gripper-zeroed internally)
        action,         # [B, num_actions, dim_action]  (EE6D action chunk target)
    ) -> {"position_loss", "rotate6D_loss", "gripper_loss"}

Reference: modeling_xvla.py / action_hub.py (EE6DActionSpace) from the
`2toINF/X-VLA-Pt` checkpoint.

EE6D action layout (dim_action = 20):
    arm-1 :  [ 0: 3] xyz position (ABSOLUTE, in robot base frame, metres)
             [ 3: 9] 6D rotation (ABSOLUTE: first two columns of the rotation matrix)
             [ 9   ] gripper (0 = open, 1 = closed)  ── trained with BCE
    arm-2 :  [10:13] xyz position
             [13:19] 6D rotation
             [19   ] gripper
    For a single-arm UR robot, arm-2 channels [10:20] are zero-padded.

Action chunking (ABSOLUTE targets — matches official X-VLA convention):
    For each start timestep t the target is a chunk of `num_actions` future
    steps (default 30). Each step is the ABSOLUTE target end-effector pose:
        position   = p[t+k]                              (absolute xyz, metres)
        rotation6D = R6D( R[t+k] )                        (absolute rotation as 6D)
        gripper    = gripper[t+k]                         (absolute next state)
    If fewer than `num_actions` steps remain, the last valid action is repeated
    (standard action-chunk padding).

    IMPORTANT: X-VLA is trained on ABSOLUTE EEF actions, NOT deltas. The official
    LIBERO preprocessing converts relative dataset actions into absolute EEF
    poses before EE6D encoding, and the deployment controller runs with
    `use_delta=False` (the predicted pose IS the controller goal). See:
      https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/preprocess.md
      https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/libero_client.py
    Our raw_demos tcp_poses are already absolute (UR getActualTCPPose), so we use
    them directly — no delta computation.

NOTE on normalization:
    X-VLA does NOT use min/max action normalization. The EE6D action space
    applies fixed loss-weight *scales* (XYZ_SCALE=500, ROT_SCALE=10) internally,
    and the gripper is supervised with BCE on {0,1} targets. We therefore feed
    RAW absolute metric poses + 6D rotations + {0,1} gripper. At deployment the
    predicted EE6D pose is converted back (rotate6d → axis-angle) and sent to the
    controller directly as the absolute goal — there is no de-normalization step.
"""

import glob
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

try:
    from scipy.spatial.transform import Rotation
except Exception as _e:  # pragma: no cover - scipy is a hard dependency here
    Rotation = None


# EE6D layout constants (mirror EE6DActionSpace in action_hub.py)
DIM_ACTION = 20
ARM1_POS = slice(0, 3)
ARM1_ROT6D = slice(3, 9)
ARM1_GRIPPER = 9


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to the 6D representation of Zhou et al. (2019):
    the first two COLUMNS of R, concatenated → 6 values [r11,r21,r31, r12,r22,r32].

    This matches the official X-VLA `Mat_to_Rotate6D` (libero_client.py):
        concat([R[:3, 0], R[:3, 1]])
    so checkpoints decode our actions with the same convention.
    """
    return np.concatenate([R[:3, 0], R[:3, 1]]).astype(np.float32)


def _axisangle_to_rotmat(rvec: np.ndarray) -> np.ndarray:
    """UR axis-angle (rotation vector) → 3x3 rotation matrix."""
    if Rotation is None:
        raise ImportError("scipy is required for rotation conversion (pip install scipy).")
    return Rotation.from_rotvec(np.asarray(rvec, dtype=np.float64)).as_matrix()


def _chunk_motion(pos: np.ndarray, t: int, num_actions: int, stride: int) -> float:
    """
    Total path length (metres) of the action chunk starting at t with the given
    stride, measured on absolute xyz positions. Used to filter near-static chunks.
        steps used = t+stride, t+2*stride, ... (num_actions of them, clipped to end)
    """
    T = pos.shape[0]
    prev = pos[t]
    total = 0.0
    for k in range(num_actions):
        step = t + (k + 1) * stride
        if step >= T:
            break
        total += float(np.linalg.norm(pos[step] - prev))
        prev = pos[step]
    return total


# ---------------------------------------------------------------------------
# Statistics (lightweight: position-delta ranges only, for inspection/logging)
# ---------------------------------------------------------------------------

def compute_dataset_statistics(npz_dir: str, num_actions: int = 30) -> Dict:
    """
    Compute simple statistics over ABSOLUTE end-effector positions and gripper
    usage. These are informational only (X-VLA does not normalise with them) but
    are saved alongside the checkpoint for reproducibility and sanity checking.
    """
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "episode_*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No episode_*.npz files found in {npz_dir}")

    all_pos: List[np.ndarray] = []
    grip_vals: List[np.ndarray] = []
    total_transitions = 0

    for fpath in npz_files:
        d = np.load(fpath, allow_pickle=True)
        tcp = d["tcp_poses"].astype(np.float32)   # (T, 6) absolute [x,y,z,rx,ry,rz]
        grip = d["gripper"].astype(np.float32)    # (T,)
        d.close()

        if tcp.shape[0] < 2:
            continue
        all_pos.append(tcp[:, :3])                # absolute positions
        grip_vals.append(grip)
        total_transitions += tcp.shape[0] - 1

    pos_np = np.concatenate(all_pos, axis=0) if all_pos else np.zeros((1, 3), np.float32)
    grip_np = np.concatenate(grip_vals, axis=0) if grip_vals else np.zeros((1,), np.float32)

    return {
        "format": "xvla_ee6d_absolute",
        "dim_action": DIM_ACTION,
        "num_actions": num_actions,
        "position_abs": {
            "mean": pos_np.mean(axis=0).tolist(),
            "std": pos_np.std(axis=0).tolist(),
            "min": pos_np.min(axis=0).tolist(),
            "max": pos_np.max(axis=0).tolist(),
            "q01": np.percentile(pos_np, 1, axis=0).tolist(),
            "q99": np.percentile(pos_np, 99, axis=0).tolist(),
        },
        "gripper": {
            "mean": float(grip_np.mean()),
            "min": float(grip_np.min()),
            "max": float(grip_np.max()),
        },
        "num_transitions": int(total_transitions),
        "num_episodes": len(npz_files),
    }


def make_train_val_split(npz_dir: str, val_ratio: float, seed: int = 0):
    """
    Split episodes (NOT frames) into train/val index lists.

    Splitting by episode is mandatory: consecutive frames within an episode are
    highly correlated, so a frame-level split would leak near-duplicate states
    into validation and give a misleadingly low val loss.

    Returns (train_indices, val_indices) — lists of positions into the sorted
    episode_*.npz file list.
    """
    npz_files = sorted(glob.glob(os.path.join(str(npz_dir), "episode_*.npz")))
    n = len(npz_files)
    if n == 0:
        raise FileNotFoundError(f"No episode_*.npz files found in {npz_dir}")

    idx = list(range(n))
    if val_ratio <= 0.0 or n < 2:
        return idx, []

    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1)   # always keep at least 1 episode for training
    val_indices = sorted(idx[:n_val])
    train_indices = sorted(idx[n_val:])
    return train_indices, val_indices


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class XVLANpzDataset(Dataset):
    """
    Flat dataset over all start-timesteps of every .npz episode, emitting the
    continuous inputs required by the X-VLA flow-matching model.

    __getitem__ returns a dict of RAW (un-processed) fields:
        images       List[PIL.Image]   (main view first, then wrist if available)
        instruction  str
        proprio      np.ndarray (DIM_ACTION,)               float32
        action       np.ndarray (num_actions, DIM_ACTION)   float32
        domain_id    int

    Image tokenisation/normalisation is deferred to the collator, which calls
    the real `XVLAProcessor` once per batch.

    Parameters
    ----------
    npz_dir : str
    num_actions : int
        Length of the predicted action chunk (X-VLA config: num_actions, default 30).
    domain_id : int
        Embodiment / soft-prompt index used for this dataset.
    use_wrist_image : bool
        If True, append the wrist camera as a second view.
    resize_resolution : (H, W)
        Target size to which images are resized before the processor runs.
    """

    def __init__(
        self,
        npz_dir: str,
        num_actions: int = 30,
        domain_id: int = 3,
        use_wrist_image: bool = True,
        resize_resolution: Tuple[int, int] = (224, 224),
        stats_path: Optional[str] = None,
        cache_episodes: int = 2,
        episode_indices: Optional[List[int]] = None,
        split_tag: str = "all",
        frame_stride: int = 1,
        min_chunk_motion: float = 0.0,
    ):
        self.npz_dir = str(npz_dir)
        self.num_actions = int(num_actions)
        self.domain_id = int(domain_id)
        self.use_wrist_image = use_wrist_image
        self.resize_resolution = resize_resolution
        self.cache_episodes = int(cache_episodes)
        self.split_tag = split_tag
        self.frame_stride = max(1, int(frame_stride))
        self.min_chunk_motion = float(min_chunk_motion)

        npz_files = sorted(glob.glob(os.path.join(self.npz_dir, "episode_*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"No episode_*.npz files found in {self.npz_dir}")
        self._npz_files = npz_files

        # Which episodes belong to THIS split (None = all). Indices refer to
        # positions in `_npz_files`, so `_load_episode(ep_idx)` keeps working.
        if episode_indices is None:
            episode_indices = list(range(len(npz_files)))
        self._episode_indices = list(episode_indices)

        # Cache episode lengths in a sidecar JSON to avoid re-decompressing on every run.
        index_cache_path = os.path.join(self.npz_dir, "_episode_lengths.json")
        cached = {}
        if os.path.exists(index_cache_path):
            with open(index_cache_path) as f:
                cached = json.load(f)

        needs_save = False
        # Flat index of (episode_idx, t) and per-episode contiguous spans of the
        # flat index (used by EpisodeBatchSampler to keep a batch within one file).
        self._index: List[Tuple[int, int]] = []
        self._ep_lengths: Dict[int, int] = {}
        self._episode_spans: List[Tuple[int, int]] = []   # (start, end) into self._index
        n_total_starts = 0
        n_kept = 0
        for ep_idx in self._episode_indices:
            fpath = npz_files[ep_idx]
            key = os.path.basename(fpath)
            if key in cached:
                T = cached[key]
            else:
                d = np.load(fpath, allow_pickle=True)
                T = int(d["images"].shape[0])
                d.close()
                cached[key] = T
                needs_save = True
            self._ep_lengths[ep_idx] = T

            # If motion filtering is on, we need this episode's positions (cheap:
            # tcp_poses is T x 6, ~KB — we do NOT decode the images here).
            pos = None
            if self.min_chunk_motion > 0.0:
                d = np.load(fpath, allow_pickle=True)
                pos = d["tcp_poses"][:, :3].astype(np.float32)
                d.close()

            span_start = len(self._index)
            # A start t is valid if at least the FIRST strided action step exists
            # (t + frame_stride < T); remaining steps are pad-repeated if needed.
            for t in range(T - 1):
                n_total_starts += 1
                if pos is not None:
                    if _chunk_motion(pos, t, self.num_actions, self.frame_stride) < self.min_chunk_motion:
                        continue  # skip near-static chunk
                self._index.append((ep_idx, t))
                n_kept += 1
            self._episode_spans.append((span_start, len(self._index)))

        if needs_save:
            with open(index_cache_path, "w") as f:
                json.dump(cached, f)

        # Per-worker LRU cache of decoded episode arrays. Populated lazily in
        # __getitem__ so each worker process keeps its own copy (npz is zip-
        # compressed and cannot be memory-mapped, so we decode once per episode
        # instead of once per sample — the key fix for slow data loading).
        self._ep_cache: "OrderedDict[int, Dict]" = OrderedDict()

        # Statistics (informational) — compute/load once over the FULL dataset
        # (not just this split) so train/val share the same reference numbers.
        if stats_path is None:
            stats_path = os.path.join(self.npz_dir, "xvla_dataset_statistics.json")
        self.stats = None
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                loaded = json.load(f)
            # Recompute if the cached stats came from an older (delta) format.
            if loaded.get("format") == "xvla_ee6d_absolute":
                self.stats = loaded
            else:
                print(f"[XVLANpzDataset] stale stats format in {stats_path}; recomputing.")
        if self.stats is None:
            self.stats = compute_dataset_statistics(self.npz_dir, self.num_actions)
            with open(stats_path, "w") as f:
                json.dump(self.stats, f, indent=2)

        print(
            f"[XVLANpzDataset:{self.split_tag}] {len(self._episode_indices)} episodes, "
            f"{len(self._index)} chunk-starts, num_actions={self.num_actions}, "
            f"stride={self.frame_stride}, min_chunk_motion={self.min_chunk_motion}, "
            f"domain_id={self.domain_id}, wrist={self.use_wrist_image}, "
            f"cache_episodes={self.cache_episodes}"
        )
        if self.min_chunk_motion > 0.0:
            kept_pct = 100.0 * n_kept / max(1, n_total_starts)
            print(f"    motion filter kept {n_kept}/{n_total_starts} starts ({kept_pct:.1f}%)")

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    def _load_episode(self, ep_idx: int) -> Dict:
        """
        Decode one episode ONCE and cache the precomputed per-frame EE6D poses
        plus the raw image arrays. Subsequent samples from the same episode are
        served from this cache (no re-decompression).
        """
        cache = self._ep_cache
        if ep_idx in cache:
            cache.move_to_end(ep_idx)
            return cache[ep_idx]

        fpath = self._npz_files[ep_idx]
        d = np.load(fpath, allow_pickle=True)
        images = np.asarray(d["images"])                       # (T, H, W, 3) uint8
        images_wrist = (
            np.asarray(d["images_wrist"])
            if (self.use_wrist_image and "images_wrist" in d.files)
            else None
        )
        tcp = np.asarray(d["tcp_poses"], dtype=np.float64)     # (T, 6)
        grip = np.asarray(d["gripper"], dtype=np.float64)      # (T,)
        instruction = str(d["instruction"])
        d.close()

        T = tcp.shape[0]
        # Precompute absolute rotation matrices once for the whole episode.
        rotmats = np.stack([_axisangle_to_rotmat(tcp[i, 3:6]) for i in range(T)], axis=0)  # (T,3,3)

        entry = dict(
            images=images,
            images_wrist=images_wrist,
            pos=tcp[:, :3],          # (T,3)
            rotmats=rotmats,         # (T,3,3)
            grip=grip,               # (T,)
            instruction=instruction,
            T=T,
        )
        cache[ep_idx] = entry
        cache.move_to_end(ep_idx)
        while len(cache) > max(1, self.cache_episodes):
            cache.popitem(last=False)
        return entry

    # ------------------------------------------------------------------
    def _ee6d_from_pose(self, xyz: np.ndarray, rotmat: np.ndarray, gripper: float) -> np.ndarray:
        """Pack a single pose into the 20-dim EE6D vector (arm-2 zero-padded)."""
        vec = np.zeros(DIM_ACTION, dtype=np.float32)
        vec[ARM1_POS] = xyz.astype(np.float32)
        vec[ARM1_ROT6D] = _rotmat_to_6d(rotmat)
        vec[ARM1_GRIPPER] = float(gripper)
        return vec

    def __getitem__(self, idx: int) -> Dict:
        ep_idx, t = self._index[idx]
        ep = self._load_episode(ep_idx)   # decoded ONCE, then cached per worker

        T = ep["T"]
        pos = ep["pos"]
        rotmats = ep["rotmats"]
        grip = ep["grip"]

        # ── Images (main view, optional wrist view) ───────────────────
        images = [self._to_pil(ep["images"][t])]
        if ep["images_wrist"] is not None:
            images.append(self._to_pil(ep["images_wrist"][t]))

        # ── Current pose (proprio) in EE6D — ABSOLUTE (base frame) ─────
        pos_t = pos[t]
        R_t = rotmats[t]
        proprio = self._ee6d_from_pose(pos_t, R_t, float(grip[t]))

        # ── Action chunk: ABSOLUTE target poses (NOT deltas) ───────────
        # X-VLA is trained on absolute EEF actions; the predicted pose is the
        # controller goal directly (use_delta=False). See module docstring.
        # frame_stride sub-samples future steps so a chunk covers more real
        # motion (helps when data is recorded at high fps with little movement).
        action = np.zeros((self.num_actions, DIM_ACTION), dtype=np.float32)
        last_valid = None
        for k in range(self.num_actions):
            step = t + (k + 1) * self.frame_stride
            if step < T:
                vec = self._ee6d_from_pose(pos[step], rotmats[step], float(grip[step]))
                action[k] = vec
                last_valid = vec
            else:
                # Pad by repeating the last valid action (chunk padding).
                action[k] = last_valid if last_valid is not None else action[k]

        return dict(
            images=images,
            instruction=ep["instruction"],
            proprio=proprio,
            action=action,
            domain_id=self.domain_id,
        )

    # ------------------------------------------------------------------
    def _to_pil(self, img_np: np.ndarray) -> Image.Image:
        img = Image.fromarray(img_np).convert("RGB")
        if self.resize_resolution is not None:
            img = img.resize(
                (self.resize_resolution[1], self.resize_resolution[0]),
                Image.BILINEAR,
            )
        return img

    @property
    def dataset_statistics(self) -> Dict:
        return self.stats


# ---------------------------------------------------------------------------
# Episode-grouped batch sampler
# ---------------------------------------------------------------------------

class EpisodeBatchSampler(Sampler):
    """
    Yields batches whose samples all come from the SAME episode.

    This is the companion to the per-worker episode cache: because every sample
    in a batch belongs to one episode, that episode is decompressed at most once
    per batch (and reused across batches via the LRU cache) instead of being
    re-decompressed for every single sample. Combined, they turn the data path
    from O(num_samples) decompressions per epoch into ~O(num_batches).

    Order is still well shuffled for training:
      - episode visitation order is reshuffled every epoch
      - chunk-start order within each episode is reshuffled every epoch

    Parameters
    ----------
    episode_spans : list[(start, end)]
        Half-open [start, end) ranges into the dataset's flat index, one per episode.
    batch_size : int
    shuffle : bool
    drop_last : bool
        If True, drop the trailing partial batch of each episode.
    seed : int
        Base RNG seed; combined with `set_epoch` for per-epoch reshuffling.
    """

    def __init__(self, episode_spans, batch_size, shuffle=True, drop_last=True, seed=0):
        self.episode_spans = list(episode_spans)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self._epoch = 0

        n_batches = 0
        for start, end in self.episode_spans:
            n = end - start
            if n <= 0:
                continue
            n_batches += n // self.batch_size if self.drop_last else -(-n // self.batch_size)
        self._num_batches = n_batches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        g = np.random.default_rng(self.seed + self._epoch) if self.shuffle else None

        ep_order = list(range(len(self.episode_spans)))
        if self.shuffle:
            g.shuffle(ep_order)

        for ep in ep_order:
            start, end = self.episode_spans[ep]
            indices = list(range(start, end))
            if not indices:
                continue
            if self.shuffle:
                g.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class XVLACollator:
    """
    Collate raw dataset items into a model-ready batch by invoking the real
    XVLAProcessor on the batched images + instructions.

    Returns a dict with:
        input_ids   [B, L]
        image_input [B, num_views, C, H, W]
        image_mask  [B, num_views]
        proprio     [B, DIM_ACTION]               float32
        action      [B, num_actions, DIM_ACTION]  float32
        domain_id   [B]                            long
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_images = [inst["images"] for inst in instances]            # List[List[PIL]]
        instructions = [inst["instruction"] for inst in instances]       # List[str]

        proc = self.processor(images=batch_images, language_instruction=instructions)

        proprio = torch.from_numpy(np.stack([inst["proprio"] for inst in instances])).float()
        action = torch.from_numpy(np.stack([inst["action"] for inst in instances])).float()
        domain_id = torch.tensor([inst["domain_id"] for inst in instances], dtype=torch.long)

        out = {
            "input_ids": proc["input_ids"],
            "image_input": proc["image_input"],
            "image_mask": proc["image_mask"],
            "proprio": proprio,
            "action": action,
            "domain_id": domain_id,
        }
        return out
