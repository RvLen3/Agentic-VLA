"""
main.py  —  PEV (Plan → Execute → Verify) evaluation loop

Supports two plan structures produced by the LRM (ds.py):

  Linear plan:
      Steps are executed in order; VLM verifies each step before advancing.

  REPEAT…UNTIL plan (for harvesting / long-horizon collection tasks):
      A REPEAT block is executed repeatedly; after each full pass through the
      loop body the VLM checks the UNTIL termination condition.
      Inside the loop, `scan [area]` steps call scan_targets_with_qwen_vl()
      to discover visible targets and dynamically rewrite the subsequent
      `pick [object] from [location]` step with the nearest found target.

Architecture
------------
  LRM  (ds.py / DeepSeek)   →  structured Plan
  VLA  (X-VLA / OpenVLA)    →  low-level actions
  VLM  (Qwen2.5-VL)         →  subtask completion + loop termination + scan

Usage
-----
  python main.py \
      --model_family xvla \
      --pretrained_checkpoint <path> \
      --task_suite_name libero_spatial \
      --use_vlm True \
      --use_lrm True \
      --task_description "pick all the blueberries from the bush and put them in the basket"
"""

import collections
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import draccus
import numpy as np
import torch
import tqdm
from PIL import Image

import wandb

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
    resize_image,
)
from experiments.robot.openvla_utils import get_processor as get_openvla_processor
from experiments.robot.xvla_utils import get_processor as get_xvla_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.libero.ds import (
    Plan,
    RepeatBlock,
    decompose_task_with_llm,
)
from experiments.robot.libero.qwenvl import (
    load_qwen_vl_model,
    check_completion_with_qwen_vl,
    scan_targets_with_qwen_vl,
    check_termination_with_qwen_vl,
)

try:
    from libero.libero import benchmark
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GenerateConfig:
    # fmt: off

    # ── Model ──────────────────────────────────────────────────────────────
    model_family: str = "xvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    # ── LIBERO environment ─────────────────────────────────────────────────
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 5

    # ── LRM (task decomposition) ───────────────────────────────────────────
    use_lrm: bool = True                    # Use LRM to decompose task dynamically
    task_description: str = ""              # Override task description for LRM
                                            # (if empty, uses env task description)
    lrm_model_name: str = "deepseek-chat"
    lrm_base_url: str = "https://api.deepseek.com"
    force_harvest_mode: bool = False        # Force REPEAT…UNTIL plan structure

    # ── VLM (verification) ────────────────────────────────────────────────
    use_vlm: bool = True                    # Enable VLM-based subtask verification
    vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    verify_frequency: int = 5              # Steps between VLM checks
    # Max steps to spend on a single subtask before forcing advance
    subtask_timeout_steps: int = 150
    # Max iterations of a REPEAT loop (safety cap)
    max_repeat_iterations: int = 50

    # ── Fallback hardcoded plan (used when use_lrm=False) ─────────────────
    hardcoded_plan: List[str] = field(default_factory=lambda: [
        "put both the alphabet soup and the tomato sauce in the basket",
    ])

    # ── Logging ───────────────────────────────────────────────────────────
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "YOUR_WANDB_PROJECT"
    wandb_entity: str = "YOUR_WANDB_ENTITY"
    seed: int = 42
    save_frames: bool = True
    frames_save_root_dir: str = "./experiments/saved_frames"

    # fmt: on


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------

def build_plan(cfg: GenerateConfig, task_description: str) -> Plan:
    """
    Build a Plan either via LRM or from the hardcoded list.
    Always returns a valid Plan object.
    """
    if cfg.use_lrm:
        desc = cfg.task_description if cfg.task_description else task_description
        print(f"[LRM] Decomposing: '{desc}'")
        plan = decompose_task_with_llm(
            desc,
            model_name=cfg.lrm_model_name,
            base_url=cfg.lrm_base_url,
            force_harvest_mode=cfg.force_harvest_mode,
        )
        if plan.steps:
            print(f"[LRM] Plan ({len(plan.steps)} top-level steps):")
            for s in plan.steps:
                if isinstance(s, str):
                    print(f"  [linear] {s}")
                elif isinstance(s, RepeatBlock):
                    print(f"  [REPEAT] body={s.body}  UNTIL: {s.until_condition}")
            return plan

        print("[LRM] Empty plan returned; falling back to hardcoded plan.")

    # Fallback: wrap hardcoded list in a linear Plan
    return Plan(steps=list(cfg.hardcoded_plan))


# ---------------------------------------------------------------------------
# Image queue helper
# ---------------------------------------------------------------------------

def update_image_queue(
    queue: collections.deque,
    obs: dict,
    resize_size: int,
):
    """Append the latest (main, wrist) image pair to the queue."""
    img_main = get_libero_image(obs, resize_size)
    img_eih  = obs["robot0_eye_in_hand_image"]
    queue.append((Image.fromarray(img_main), Image.fromarray(img_eih)))
    return img_main  # return numpy array for replay video


# ---------------------------------------------------------------------------
# Subtask executor
# ---------------------------------------------------------------------------

class SubtaskExecutor:
    """
    Executes a single subtask instruction until the VLM confirms completion
    or the step budget is exhausted.

    Returns (obs, done, timed_out, steps_used).
    """

    def __init__(self, cfg: GenerateConfig, vla_model, vla_processor,
                 vlm_model, vlm_processor, resize_size: int,
                 log_file, replay_images: list,
                 episode_frame_save_dir: Optional[Path]):
        self.cfg = cfg
        self.vla_model = vla_model
        self.vla_processor = vla_processor
        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        self.resize_size = resize_size
        self.log_file = log_file
        self.replay_images = replay_images
        self.frame_dir = episode_frame_save_dir

    def run(
        self,
        obs: dict,
        instruction: str,
        env,
        global_t: int,
        max_steps: int,
    ):
        """
        Run the VLA on `instruction` until VLM says done or budget exhausted.

        Returns
        -------
        obs         : latest observation dict
        env_done    : bool, True if environment signalled success
        timed_out   : bool, True if step budget exhausted without VLM confirmation
        global_t    : updated global timestep counter
        """
        image_queue = collections.deque(maxlen=2)
        steps_this_subtask = 0
        env_done = False

        self._log(f"  >> Subtask: '{instruction}'  (budget={self.cfg.subtask_timeout_steps})")

        while steps_this_subtask < self.cfg.subtask_timeout_steps:
            if global_t >= max_steps:
                break

            # ── Collect images ────────────────────────────────────────
            img_main_np = update_image_queue(image_queue, obs, self.resize_size)
            self.replay_images.append(img_main_np)

            # ── Save frames ───────────────────────────────────────────
            if self.frame_dir and (global_t % self.cfg.verify_frequency == 0):
                self._save_frame(obs, global_t, img_main_np)

            # ── VLM completion check ──────────────────────────────────
            if (
                self.cfg.use_vlm
                and self.vlm_model is not None
                and steps_this_subtask > 0
                and steps_this_subtask % self.cfg.verify_frequency == 0
            ):
                completed = check_completion_with_qwen_vl(
                    self.vlm_model, self.vlm_processor,
                    image_queue, instruction,
                )
                if completed:
                    self._log(f"  [VLM] '{instruction}' COMPLETE at t={global_t}")
                    return obs, False, False, global_t

            # ── VLA action ────────────────────────────────────────────
            observation = {
                "full_image": img_main_np,
                "state": np.concatenate((
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )),
            }
            action = get_action(
                self.cfg, self.vla_model, observation,
                instruction, self.vla_processor,
            )
            action = normalize_gripper_action(action, binarize=True)
            if self.cfg.model_family == "openvla":
                action = invert_gripper_action(action)

            obs, reward, env_done, info = env.step(action.tolist())
            global_t += 1
            steps_this_subtask += 1

            if env_done:
                self._log(f"  [ENV] Done signal at t={global_t}")
                return obs, True, False, global_t

        self._log(f"  [TIMEOUT] '{instruction}' timed out after {steps_this_subtask} steps")
        return obs, env_done, True, global_t

    def _log(self, msg: str):
        print(msg)
        self.log_file.write(msg + "\n")

    def _save_frame(self, obs: dict, t: int, img_main_np: np.ndarray):
        try:
            Image.fromarray(img_main_np).save(
                self.frame_dir / f"frame_{t:04d}.png"
            )
            Image.fromarray(obs["robot0_eye_in_hand_image"]).save(
                self.frame_dir / f"eyeinhand_{t:04d}.png"
            )
        except Exception as e:
            print(f"[warn] frame save failed at t={t}: {e}")


# ---------------------------------------------------------------------------
# Plan executor
# ---------------------------------------------------------------------------

def execute_plan(
    plan: Plan,
    obs: dict,
    env,
    executor: SubtaskExecutor,
    cfg: GenerateConfig,
    vlm_model,
    vlm_processor,
    resize_size: int,
    max_steps: int,
    log_file,
) -> tuple:
    """
    Execute a full Plan.

    Returns (obs, env_done, global_t).
    """
    global_t = cfg.num_steps_wait  # already consumed by wait loop
    env_done = False
    image_queue = collections.deque(maxlen=2)  # shared queue for termination checks

    def refresh_image_queue():
        img = get_libero_image(obs, resize_size)
        image_queue.append((
            Image.fromarray(img),
            Image.fromarray(obs["robot0_eye_in_hand_image"]),
        ))

    for step_idx, step in enumerate(plan.steps):

        # ── Linear subtask ────────────────────────────────────────────
        if isinstance(step, str):
            obs, env_done, _, global_t = executor.run(
                obs, step, env, global_t, max_steps
            )
            if env_done or global_t >= max_steps:
                return obs, env_done, global_t

        # ── REPEAT…UNTIL block ────────────────────────────────────────
        elif isinstance(step, RepeatBlock):
            log_file.write(
                f"[REPEAT] body={step.body}  UNTIL: {step.until_condition}\n"
            )
            print(f"[REPEAT] Starting loop. UNTIL: '{step.until_condition}'")

            for iteration in range(cfg.max_repeat_iterations):
                print(f"[REPEAT] Iteration {iteration + 1}")
                log_file.write(f"[REPEAT] Iteration {iteration + 1}\n")

                # ── Resolve dynamic targets from scan steps ────────────
                # Build a mutable copy of the loop body for this iteration,
                # substituting concrete target names discovered by scan.
                resolved_body = list(step.body)
                last_scan_targets: List[str] = []

                for body_idx, subtask in enumerate(resolved_body):
                    subtask_lower = subtask.lower()

                    # ── scan step: call VLM, get target list ──────────
                    if subtask_lower.startswith("scan"):
                        # Extract what to scan for from the instruction
                        # e.g. "scan the blueberry tree" → target = "blueberry"
                        area = subtask.split("scan", 1)[-1].strip().rstrip(".")
                        # Infer target object from subsequent pick step
                        target_hint = _infer_target_from_body(resolved_body, body_idx)

                        refresh_image_queue()
                        last_scan_targets = scan_targets_with_qwen_vl(
                            vlm_model, vlm_processor,
                            image_queue,
                            target_description=target_hint or area,
                        )
                        log_file.write(
                            f"  [scan] found {len(last_scan_targets)} target(s): "
                            f"{last_scan_targets}\n"
                        )

                        # If no targets found, check termination immediately
                        if not last_scan_targets:
                            print("[REPEAT] scan found 0 targets → checking termination")
                            refresh_image_queue()
                            if check_termination_with_qwen_vl(
                                vlm_model, vlm_processor,
                                image_queue, step.until_condition,
                            ):
                                print("[REPEAT] Termination confirmed. Exiting loop.")
                                log_file.write("[REPEAT] Terminated (no targets + VLM confirmed).\n")
                                goto_next_step = True
                                break
                        # Mark scan as done (no VLA action needed)
                        continue

                    # ── pick step: substitute nearest target ──────────
                    if (subtask_lower.startswith("pick") and last_scan_targets):
                        # Replace generic object reference with the first
                        # (nearest/easiest) target from the scan result
                        nearest = last_scan_targets[0]
                        subtask = _substitute_target(subtask, nearest)
                        resolved_body[body_idx] = subtask
                        log_file.write(f"  [pick] resolved to: '{subtask}'\n")

                    # ── Execute this body step ────────────────────────
                    obs, env_done, timed_out, global_t = executor.run(
                        obs, subtask, env, global_t, max_steps
                    )
                    if env_done or global_t >= max_steps:
                        return obs, env_done, global_t

                else:
                    # Normal end of loop body (no early break)
                    goto_next_step = False

                if goto_next_step:
                    break

                # ── Check UNTIL termination condition ─────────────────
                refresh_image_queue()
                terminated = check_termination_with_qwen_vl(
                    vlm_model, vlm_processor,
                    image_queue, step.until_condition,
                )
                if terminated:
                    print(f"[REPEAT] UNTIL condition met after iteration {iteration + 1}.")
                    log_file.write(
                        f"[REPEAT] Terminated after {iteration + 1} iteration(s).\n"
                    )
                    break
            else:
                print(f"[REPEAT] Reached max iterations ({cfg.max_repeat_iterations}).")
                log_file.write(
                    f"[REPEAT] Max iterations ({cfg.max_repeat_iterations}) reached.\n"
                )

    return obs, env_done, global_t


# ---------------------------------------------------------------------------
# Helpers for dynamic target substitution
# ---------------------------------------------------------------------------

def _infer_target_from_body(body: List[str], scan_idx: int) -> str:
    """
    Look ahead in the loop body after a scan step to find the object being picked.
    e.g. "pick the blueberry from the tree" → "blueberry"
    """
    for i in range(scan_idx + 1, len(body)):
        s = body[i].lower()
        if s.startswith("pick"):
            # Extract object between "pick" and "from"
            if " from " in s:
                obj_part = s.split("pick", 1)[1].split(" from ")[0].strip()
                # Remove articles
                for art in ["the ", "a ", "an "]:
                    obj_part = obj_part.replace(art, "")
                return obj_part.strip()
    return ""


def _substitute_target(pick_instruction: str, nearest_target: str) -> str:
    """
    Replace the generic object in a pick instruction with a specific target.

    e.g. ("pick the blueberry from the tree",
          "the leftmost blueberry on the upper branch")
      → "pick the leftmost blueberry on the upper branch from the tree"
    """
    instr_lower = pick_instruction.lower()
    if " from " in instr_lower:
        _, loc_part = pick_instruction.split(" from ", 1)
        return f"pick {nearest_target} from {loc_part}"
    return f"pick {nearest_target}"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, "cfg.pretrained_checkpoint must not be empty!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit)

    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name

    # ── Load VLA ──────────────────────────────────────────────────────────
    vla_model = get_model(cfg)
    if cfg.model_family == "openvla":
        if (cfg.unnorm_key not in vla_model.norm_stats
                and f"{cfg.unnorm_key}_no_noops" in vla_model.norm_stats):
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in vla_model.norm_stats

    vla_processor = (
        get_openvla_processor(cfg) if cfg.model_family == "openvla"
        else get_xvla_processor(cfg)
    )

    # ── Load VLM ──────────────────────────────────────────────────────────
    vlm_model, vlm_processor = None, None
    if cfg.use_vlm:
        vlm_model, vlm_processor = load_qwen_vl_model(cfg.vlm_model_id)
        if vlm_model is None:
            print("[warn] VLM failed to load; disabling VLM verification.")
            cfg.use_vlm = False

    # ── Logging setup ─────────────────────────────────────────────────────
    run_id = f"PEV-EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    log_path = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(log_path, "w")
    log_file.write(f"use_lrm={cfg.use_lrm}  use_vlm={cfg.use_vlm}\n")
    print(f"Logging → {log_path}")

    frames_run_dir = None
    if cfg.save_frames:
        frames_run_dir = Path(cfg.frames_save_root_dir) / run_id
        os.makedirs(frames_run_dir, exist_ok=True)

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
            config=draccus.encode(cfg),
        )

    # ── LIBERO task suite ─────────────────────────────────────────────────
    assert LIBERO_AVAILABLE, "libero package not installed."
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    resize_size = get_image_resize_size(cfg)

    max_steps_map = {
        "libero_spatial": 220,
        "libero_object":  280,
        "libero_goal":    300,
        "libero_10":      520,
        "libero_90":      400,
    }

    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, original_task_description = get_libero_env(
            task, cfg.model_family, resolution=256
        )
        max_steps = max_steps_map.get(cfg.task_suite_name, 400)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(
            range(cfg.num_trials_per_task),
            desc=f"Task {task_id}",
            leave=False,
        ):
            print(f"\n{'='*60}")
            print(f"Task: {original_task_description}")
            log_file.write(f"\nTask: {original_task_description}\n")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            # ── Build plan for this episode ────────────────────────────
            plan = build_plan(cfg, original_task_description)
            log_file.write(f"Plan: {plan.steps}\n")

            # ── Frame save directory ───────────────────────────────────
            episode_frame_dir = None
            if cfg.save_frames and frames_run_dir:
                episode_frame_dir = (
                    frames_run_dir / f"task_{task_id}" / f"episode_{episode_idx}"
                )
                os.makedirs(episode_frame_dir, exist_ok=True)

            replay_images: list = []

            # ── Wait for environment to stabilise ─────────────────────
            for _ in range(cfg.num_steps_wait):
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

            # ── Build executor ────────────────────────────────────────
            executor = SubtaskExecutor(
                cfg=cfg,
                vla_model=vla_model,
                vla_processor=vla_processor,
                vlm_model=vlm_model,
                vlm_processor=vlm_processor,
                resize_size=resize_size,
                log_file=log_file,
                replay_images=replay_images,
                episode_frame_save_dir=episode_frame_dir,
            )

            # ── Execute plan ──────────────────────────────────────────
            try:
                obs, env_done, _ = execute_plan(
                    plan=plan,
                    obs=obs,
                    env=env,
                    executor=executor,
                    cfg=cfg,
                    vlm_model=vlm_model,
                    vlm_processor=vlm_processor,
                    resize_size=resize_size,
                    max_steps=max_steps,
                    log_file=log_file,
                )
            except Exception as e:
                import traceback as tb
                print(f"[error] Episode exception: {e}")
                log_file.write(f"Exception: {e}\n")
                tb.print_exc(file=log_file)
                env_done = False

            # ── Episode bookkeeping ───────────────────────────────────
            task_episodes += 1
            total_episodes += 1
            if env_done:
                task_successes += 1
                total_successes += 1

            save_rollout_video(
                replay_images, total_episodes,
                success=env_done,
                task_description=original_task_description,
                log_file=log_file,
            )

            sr = total_successes / total_episodes * 100
            print(f"Episode done. success={env_done}  "
                  f"total={total_successes}/{total_episodes} ({sr:.1f}%)")
            log_file.write(
                f"success={env_done}  "
                f"total={total_successes}/{total_episodes} ({sr:.1f}%)\n"
            )
            log_file.flush()

            if cfg.use_wandb:
                wandb.log({
                    f"episode_success/{original_task_description}": int(env_done),
                    "step": total_episodes,
                })

        # ── Task summary ──────────────────────────────────────────────
        task_sr = task_successes / task_episodes if task_episodes else 0.0
        print(f"Task {task_id} done. SR={task_sr:.2f} ({task_successes}/{task_episodes})")
        log_file.write(f"Task {task_id} SR={task_sr:.2f}\n")
        if cfg.use_wandb:
            wandb.log({"task_summary/success_rate": task_sr}, step=task_id)

    # ── Final summary ─────────────────────────────────────────────────────
    total_sr = total_successes / total_episodes if total_episodes else 0.0
    print(f"\nEvaluation complete. Total SR={total_sr:.2f} "
          f"({total_successes}/{total_episodes})")
    log_file.write(f"\nTotal SR={total_sr:.2f} ({total_successes}/{total_episodes})\n")
    log_file.close()

    if cfg.use_wandb:
        wandb.log({
            "total_summary/success_rate": total_sr,
            "total_summary/num_successes": total_successes,
            "total_summary/num_episodes": total_episodes,
        })
        wandb.save(log_path)
        wandb.finish()


if __name__ == "__main__":
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t0 = time.perf_counter()
    eval_libero()
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"elapsed={time.perf_counter()-t0:.1f}s")
