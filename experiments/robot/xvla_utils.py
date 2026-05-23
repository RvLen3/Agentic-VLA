"""Utils for evaluating the X-VLA policy."""

import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# X-VLA domain ID for LIBERO (domain_id=3 per official docs).
# Adjust if using a different embodiment/dataset domain.
XVLA_DOMAIN_ID = 3


def get_vla(cfg):
    """Loads and returns an X-VLA model from checkpoint."""
    print("[*] Instantiating Pretrained X-VLA model")
    print("[*] Loading with trust_remote_code=True")

    model = AutoModel.from_pretrained(
        cfg.pretrained_checkpoint,
        trust_remote_code=True,
    )

    # Move model to device.
    model = model.to(DEVICE)
    model.eval()

    return model


def get_processor(cfg):
    """Get X-VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(
        cfg.pretrained_checkpoint,
        trust_remote_code=True,
    )
    return processor


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key=None, center_crop=False):
    """
    Generates an action with the X-VLA policy.

    X-VLA processor expects:
        images: List[PIL.Image]  (single-view or multi-view)
        language_instruction: str

    X-VLA model.predict_action expects:
        **inputs  (output of processor: input_ids, image_input, image_mask)
        proprio: torch.Tensor of shape [1, state_dim]
        domain_id: int

    Returns:
        action: np.ndarray of shape (ACTION_DIM,)
    """
    # Prepare image
    image = Image.fromarray(obs["full_image"]).convert("RGB")

    # Prepare proprioceptive state: shape [1, state_dim]
    state = obs.get("state", None)
    if state is not None:
        proprio = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    else:
        # Fallback: zero proprio if state is not provided
        proprio = torch.zeros(1, 7, dtype=torch.float32).to(DEVICE)

    # Process inputs via X-VLA processor
    inputs = processor(
        images=[image],
        language_instruction=task_label,
    )

    # Move tensors to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        # predict_action returns action chunk of shape [chunk_size, action_dim]
        # or directly [action_dim]; we take the first step.
        action_output = vla.predict_action(
            **inputs,
            proprio=proprio,
            domain_id=XVLA_DOMAIN_ID,
        )

    # Handle both chunk output [T, D] and single-step output [D]
    if isinstance(action_output, torch.Tensor):
        action_np = action_output.detach().cpu().float().numpy()
        if action_np.ndim == 2:
            action_np = action_np[0]  # take first step of the chunk
    else:
        action_np = np.array(action_output, dtype=np.float32)
        if action_np.ndim == 2:
            action_np = action_np[0]

    return action_np
