"""
qwenvl.py  —  VLM utilities (Qwen2.5-VL)

Functions
---------
load_qwen_vl_model          — load model + processor
check_completion_with_qwen_vl  — judge whether a linear subtask is done
scan_targets_with_qwen_vl   — (NEW) scan an area and return a list of visible targets
check_termination_with_qwen_vl — (NEW) judge whether a REPEAT…UNTIL condition is met
"""

import collections
import traceback
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

print("Successfully imported qwen_vl_utils.process_vision_info.")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_qwen_vl_model(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device: str = "auto",
):
    """Load Qwen2.5-VL model and processor."""
    print(f"Loading VLM: {model_id}  device={device}  (flash_attention_2)")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("VLM loaded successfully.")
        return model, processor
    except Exception as e:
        print(f"Failed to load VLM: {e}")
        traceback.print_exc()
        return None, None


# ---------------------------------------------------------------------------
# Internal helper: run a single VLM query
# ---------------------------------------------------------------------------

def _run_vlm_query(
    vlm_model,
    vlm_processor,
    images: List[Image.Image],
    image_labels: List[str],
    question: str,
    max_new_tokens: int = 64,
) -> str:
    """
    Build a multi-image message, run inference, return the decoded response string.

    images       : list of PIL images to include
    image_labels : one label string per image (shown before each image in the prompt)
    question     : the question appended after all images
    """
    content_list = []
    for label, img in zip(image_labels, images):
        content_list.append({"type": "text",  "text": label})
        content_list.append({"type": "image", "image": img})
    content_list.append({"type": "text", "text": question})

    messages = [{"role": "user", "content": content_list}]

    text_template = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    if image_inputs is None:
        return ""

    inputs = vlm_processor(
        text=[text_template],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(vlm_model.device)

    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=vlm_processor.tokenizer.eos_token_id,
        )

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    responses = vlm_processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return responses[0].strip() if responses else ""


def _collect_images_from_queue(
    image_pair_queue: collections.deque,
) -> Tuple[List[Image.Image], List[str]]:
    """Flatten an image-pair queue into (images, labels) lists."""
    images, labels = [], []
    for i, pair in enumerate(image_pair_queue):
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            main_img, eih_img = pair
            images.append(main_img)
            labels.append(f"Image Pair {i+1} — Main View:")
            images.append(eih_img)
            labels.append(f"Image Pair {i+1} — Hand View:")
        else:
            # Single image fallback
            images.append(pair)
            labels.append(f"Image {i+1}:")
    return images, labels


# ---------------------------------------------------------------------------
# 1. check_completion_with_qwen_vl  (original, extended)
# ---------------------------------------------------------------------------

def check_completion_with_qwen_vl(
    vlm_model,
    vlm_processor,
    image_pair_queue: collections.deque,
    current_subtask_instruction: str,
) -> bool:
    """
    Judge whether a linear subtask has been completed.
    Returns True if the VLM answers "Yes", False otherwise.

    Supports all atomic operations:
        pick up / place / open / close / turn on / turn off / scan
    """
    if vlm_model is None or vlm_processor is None:
        print("[VLM] Error: model not loaded.")
        return False
    if not image_pair_queue:
        print("[VLM] Warning: image queue empty.")
        return False

    try:
        images, labels = _collect_images_from_queue(image_pair_queue)
        instr = current_subtask_instruction
        instr_lower = instr.lower()

        # ── Build a task-specific question ────────────────────────────
        prefix = (
            f"Observe the following {len(images)} image(s). "
            f"The robot instruction is: '{instr}'. "
        )

        if instr_lower.startswith("pick up"):
            obj = instr.split("pick up", 1)[-1].strip().rstrip(".")
            question = (
                f"{prefix}Has '{obj}' been securely grasped and clearly lifted "
                f"off the table surface by the end of the sequence? "
                f"Answer strictly 'Yes' or 'No'."
            )
        elif instr_lower.startswith("place"):
            # place [object] in/on [location]
            for sep in [" into ", " in ", " on ", " onto "]:
                if sep in instr_lower:
                    obj_part, loc_part = instr.split(sep, 1)
                    obj = re.sub(r"^place\s+", "", obj_part, flags=re.IGNORECASE).strip()
                    loc = loc_part.strip().rstrip(".")
                    question = (
                        f"{prefix}Has '{obj}' been successfully placed inside/onto "
                        f"'{loc}', and does the gripper appear open and moving away "
                        f"from the object? "
                        f"Answer strictly 'Yes' or 'No'."
                    )
                    break
            else:
                question = (
                    f"{prefix}Has the placement action been completed successfully "
                    f"(object released at the target location)? "
                    f"Answer strictly 'Yes' or 'No'."
                )
        elif instr_lower.startswith("open"):
            obj = instr.split("open", 1)[-1].strip().rstrip(".")
            question = (
                f"{prefix}Is '{obj}' now visibly open (lid/door/drawer moved)? "
                f"Answer strictly 'Yes' or 'No'."
            )
        elif instr_lower.startswith("close"):
            obj = instr.split("close", 1)[-1].strip().rstrip(".")
            question = (
                f"{prefix}Is '{obj}' now visibly closed? "
                f"Answer strictly 'Yes' or 'No'."
            )
        elif instr_lower.startswith("turn on"):
            dev = instr.split("turn on", 1)[-1].strip().rstrip(".")
            question = (
                f"{prefix}Is '{dev}' now turned on (indicator light, visible change)? "
                f"Answer strictly 'Yes' or 'No'."
            )
        elif instr_lower.startswith("turn off"):
            dev = instr.split("turn off", 1)[-1].strip().rstrip(".")
            question = (
                f"{prefix}Is '{dev}' now turned off? "
                f"Answer strictly 'Yes' or 'No'."
            )
        elif instr_lower.startswith("scan"):
            # scan is a VLM perception step, not a robot action.
            # Always considered "done" immediately; real output comes from
            # scan_targets_with_qwen_vl().
            return True
        else:
            question = (
                f"{prefix}Has this action been successfully completed by the end "
                f"of the sequence? Answer strictly 'Yes' or 'No'."
            )

        response = _run_vlm_query(
            vlm_model, vlm_processor, images, labels, question, max_new_tokens=10
        )
        print(f"[VLM check_completion] '{instr}' → '{response}'")
        return response.lower().startswith("yes")

    except Exception as e:
        print(f"[VLM] Exception in check_completion: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# 2. scan_targets_with_qwen_vl  (NEW)
# ---------------------------------------------------------------------------

def scan_targets_with_qwen_vl(
    vlm_model,
    vlm_processor,
    image_pair_queue: collections.deque,
    target_description: str,
    max_targets: int = 20,
) -> List[str]:
    """
    Scan the current view and return an ordered list of visible target descriptions.

    Called when the executor encounters a `scan [area]` step in a REPEAT block.
    The returned list is used to populate the subsequent `pick up` instruction
    with a concrete target (e.g. "the apple on the left side of the table").

    Parameters
    ----------
    target_description : str
        What to look for, e.g. "fruit", "object on the table", "apple".
        For the table-clearing task, pass "fruit" or "object".
    max_targets : int
        Upper bound on how many targets to report.

    Returns
    -------
    List[str]
        Ordered list of target descriptions, nearest/easiest first, e.g.:
        ["the apple on the left side of the table",
         "the orange near the center of the table",
         "the banana on the right side of the table"]
        Returns an empty list if none are found or on error.
    """
    if vlm_model is None or vlm_processor is None:
        print("[VLM scan] Error: model not loaded.")
        return []
    if not image_pair_queue:
        print("[VLM scan] Warning: image queue empty.")
        return []

    try:
        images, labels = _collect_images_from_queue(image_pair_queue)

        question = (
            f"Look at the table in the image(s) carefully. "
            f"List ALL visible '{target_description}' objects currently on the table "
            f"that the robot arm can reach, ordered from nearest/easiest to grasp first. "
            f"For each object, include its type and position "
            f"(e.g. 'the apple on the left side of the table', "
            f"'the orange near the center', 'the banana on the right'). "
            f"Output ONLY a numbered list, one item per line. "
            f"If the table is clear (no {target_description} remaining), "
            f"output exactly: NONE"
        )

        response = _run_vlm_query(
            vlm_model, vlm_processor, images, labels, question, max_new_tokens=256
        )
        print(f"[VLM scan] raw response:\n{response}")

        if not response or response.strip().upper() == "NONE":
            return []

        # Parse numbered list
        targets = []
        for line in response.splitlines():
            line = line.strip()
            m = re.match(r"^\d+[\.\)]\s*(.*)", line)
            if m:
                item = m.group(1).strip()
                if item:
                    targets.append(item)
            elif line and not line.upper().startswith("NONE"):
                targets.append(line)
            if len(targets) >= max_targets:
                break

        print(f"[VLM scan] found {len(targets)} target(s): {targets}")
        return targets

    except Exception as e:
        print(f"[VLM] Exception in scan_targets: {e}")
        traceback.print_exc()
        return []


# ---------------------------------------------------------------------------
# 3. check_termination_with_qwen_vl  (NEW)
# ---------------------------------------------------------------------------

def check_termination_with_qwen_vl(
    vlm_model,
    vlm_processor,
    image_pair_queue: collections.deque,
    until_condition: str,
) -> bool:
    """
    Judge whether the UNTIL termination condition of a REPEAT…UNTIL loop is met.

    For the table-clearing task the typical condition is:
        "no fruits remain on the table"

    Parameters
    ----------
    until_condition : str
        Natural-language condition from the plan, e.g.
        "no fruits remain on the table".

    Returns
    -------
    bool
        True  → condition is met, exit the loop (table is clear).
        False → condition not yet met, continue looping.
    """
    if vlm_model is None or vlm_processor is None:
        print("[VLM termination] Error: model not loaded.")
        return False
    if not image_pair_queue:
        print("[VLM termination] Warning: image queue empty.")
        return False

    try:
        images, labels = _collect_images_from_queue(image_pair_queue)

        question = (
            f"Observe the image(s) carefully. "
            f"Termination condition: '{until_condition}'. "
            f"Based ONLY on what is visible in the image(s), "
            f"is this termination condition currently TRUE? "
            f"Answer strictly 'Yes' or 'No'."
        )

        response = _run_vlm_query(
            vlm_model, vlm_processor, images, labels, question, max_new_tokens=10
        )
        print(f"[VLM termination] '{until_condition}' → '{response}'")
        return response.lower().startswith("yes")

    except Exception as e:
        print(f"[VLM] Exception in check_termination: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Keep re import at module level (used in check_completion)
# ---------------------------------------------------------------------------
import re
