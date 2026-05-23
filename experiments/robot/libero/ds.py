"""
ds.py  —  LRM (Language-based Robot Planner)

Decomposes a high-level task description into a structured plan of atomic
operations that a VLA executor can carry out step by step.

Supported plan structures
--------------------------
Linear plan  (original):
    1. pick up the red pepper
    2. place the red pepper in the left basket

REPEAT…UNTIL plan  (for tasks with unknown/variable target count):
    REPEAT:
      1. scan the table
      2. pick up the nearest visible vegetable
      3. place the vegetable in the left basket
    UNTIL: no vegetables remain on the table

Atomic operations
-----------------
Linear:
    pick up [object]
    place [object] in the left basket
    place [object] in the right basket
    open [object]
    close [object]
    turn on [device]
    turn off [device]

Collection / long-horizon additions:
    scan [area]          — VLM perceives area and returns visible target list
                           (no robot motion; handled by scan_targets_with_qwen_vl)

Target objects for this task
-----------------------------
Vegetables: red pepper, green pepper, yellow pepper, corn, purple sweet potato, pumpkin
Baskets: left basket, right basket (two baskets on the table, one on each side)
"""

import os
import re
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import openai


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RepeatBlock:
    """Represents a REPEAT…UNTIL loop in the plan."""
    body: List[str]          # ordered list of subtask strings inside the loop
    until_condition: str     # natural-language termination condition


@dataclass
class Plan:
    """
    A structured plan returned by the LRM.

    steps: list of either str (linear subtask) or RepeatBlock.
    The executor iterates through steps in order; RepeatBlock steps are
    executed repeatedly until the VLM confirms the until_condition is met.
    """
    steps: List  # List[str | RepeatBlock]

    def is_linear(self) -> bool:
        return all(isinstance(s, str) for s in self.steps)

    def flat_subtasks(self) -> List[str]:
        """Flatten to a simple list (for linear plans only)."""
        assert self.is_linear(), "Cannot flatten a plan that contains REPEAT blocks."
        return list(self.steps)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_NUMBERED_LINE = re.compile(r"^\s*\d+\.\s+(.*)")
_REPEAT_START  = re.compile(r"^\s*REPEAT\s*:\s*$", re.IGNORECASE)
_UNTIL_LINE    = re.compile(r"^\s*UNTIL\s*:\s*(.*)", re.IGNORECASE)


def parse_llm_plan(raw_plan_string: str) -> Plan:
    """
    Parse the raw LLM output into a Plan object.

    Handles both linear plans and REPEAT…UNTIL blocks.
    Gracefully falls back to a linear plan if the format is unexpected.
    """
    lines = raw_plan_string.splitlines()
    steps: List = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # ── REPEAT block ──────────────────────────────────────────────
        if _REPEAT_START.match(line):
            body: List[str] = []
            until_condition = ""
            i += 1
            while i < len(lines):
                inner = lines[i].strip()
                until_match = _UNTIL_LINE.match(inner)
                if until_match:
                    until_condition = until_match.group(1).strip()
                    i += 1
                    break
                numbered = _NUMBERED_LINE.match(inner)
                if numbered:
                    body.append(numbered.group(1).strip())
                i += 1
            if body:
                steps.append(RepeatBlock(body=body, until_condition=until_condition))
            continue

        # ── Numbered linear step ──────────────────────────────────────
        numbered = _NUMBERED_LINE.match(line)
        if numbered:
            instruction = numbered.group(1).strip()
            if instruction:
                steps.append(instruction)

        i += 1

    return Plan(steps=steps)


# ---------------------------------------------------------------------------
# LRM prompt templates
# ---------------------------------------------------------------------------

_LINEAR_PROMPT = """\
You are a planning assistant for a fixed robotic arm. Break down the high-level \
task into a minimal sequence of **essential high-level commands** that a capable \
Vision-Language-Action (VLA) model can execute directly.

Output Format:
Generate a numbered list. Each line = one atomic command.

Allowed atomic commands:
  pick up [object]
  place [object] in/on [location]
  open [object/container/drawer]
  close [object/container/drawer]
  turn on [device]
  turn off [device]

Rules:
- Do NOT include: locate, move to, lift, lower, grasp, release, push, pull, rotate, adjust.
- Assume the VLA handles all implicit sub-motions internally.
- Use descriptive names from the task description.
- Generate the minimal sequence needed.

Task: {task}
Output:
"""

_HARVEST_PROMPT = """\
You are a planning assistant for a robotic arm performing a **table-clearing or \
collection task** where the number of target objects is unknown and must be \
discovered visually at runtime.

The current task involves clearing vegetables from a table into two baskets:
  - Left basket  (on the left side of the table)
  - Right basket (on the right side of the table)
Target vegetables: red pepper, green pepper, yellow pepper, corn, purple sweet potato, pumpkin.

Output Format:
Use a REPEAT…UNTIL block for the repeated pick-and-place loop, plus optional
linear steps before/after it.

Allowed atomic commands:
  Linear steps (before/after the loop):
    pick up [object]
    place [object] in the left basket
    place [object] in the right basket
    open [object/container/drawer]
    close [object/container/drawer]
    turn on [device]
    turn off [device]

  Inside REPEAT block only:
    scan [area/surface]     — VLM perceives the area and returns a list of
                              visible objects; NO robot motion occurs
    pick up [object]        — grasp and lift the object from the surface
    place [object] in the left basket   — move object to the left basket and release
    place [object] in the right basket  — move object to the right basket and release

REPEAT…UNTIL format (use EXACTLY this):
REPEAT:
  1. scan [area]
  2. pick up the nearest visible [object type]
  3. place the [object type] in the left basket
UNTIL: [natural-language termination condition verifiable from a camera image]

Rules:
- Use REPEAT…UNTIL when the number of objects is unknown or variable.
- The UNTIL condition MUST be checkable from a camera image
  (e.g. "no vegetables remain on the table", "the table is clear").
- For the place step, choose "left basket" or "right basket" based on the
  object's position: objects on the left side of the table go to the left
  basket; objects on the right side go to the right basket. When position is
  unknown at plan time, default to "left basket".
- Do NOT enumerate individual objects — use the loop instead.
- Do NOT use `pick from`, `deposit into`, `move to`, or `place in the basket`
  (always specify left or right).
- The scan step discovers which specific object to pick next; the subsequent
  pick up step uses that information.

Example for "put all the vegetables on the table into the baskets":
REPEAT:
  1. scan the table
  2. pick up the nearest visible vegetable on the table
  3. place the vegetable in the left basket
UNTIL: no vegetables remain on the table

Example for "clear all objects from the table and put them in the baskets":
REPEAT:
  1. scan the table
  2. pick up the nearest visible object on the table
  3. place the object in the left basket
UNTIL: no objects remain on the table

Task: {task}
Output:
"""


def _is_harvest_task(task: str) -> bool:
    """
    Heuristic: decide whether to use the collection/table-clearing prompt.
    Triggers on keywords suggesting repeated pick-and-place of multiple unknown targets.
    """
    keywords = [
        # English — quantity / completeness
        "all", "every", "each", "all fruits", "all objects", "all items",
        "all vegetables", "all veggies",
        "clear the table", "clear all", "collect all", "gather all",
        # English — action verbs for collection
        "harvest", "gather", "collect",
        # English — vegetable-specific
        "peppers", "vegetables", "veggies",
        # Chinese — quantity / completeness
        "所有", "全部", "所有水果", "所有蔬菜", "清空", "收集所有", "摘", "采",
        # Chinese — vegetable-specific
        "辣椒", "蔬菜", "玉米", "紫薯", "南瓜",
    ]
    task_lower = task.lower()
    return any(kw in task_lower for kw in keywords)


# ---------------------------------------------------------------------------
# Main LRM function
# ---------------------------------------------------------------------------

def decompose_task_with_llm(
    task_description: str,
    model_name: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    force_harvest_mode: bool = False,
) -> Plan:
    """
    Call the LLM to decompose a task description into a structured Plan.

    Parameters
    ----------
    task_description : str
        High-level task in natural language.
    model_name : str
        LLM model identifier.
    base_url : str
        API base URL (DeepSeek by default).
    force_harvest_mode : bool
        If True, always use the harvest/collection prompt regardless of heuristic.

    Returns
    -------
    Plan
        Structured plan with linear steps and/or REPEAT…UNTIL blocks.
        Returns an empty Plan on failure.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        warnings.warn(
            "Environment variable 'DEEPSEEK_API_KEY' not set. Cannot call LLM.",
            RuntimeWarning,
        )
        return Plan(steps=[])

    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        warnings.warn(f"Failed to initialize OpenAI client: {e}", RuntimeWarning)
        return Plan(steps=[])

    use_harvest = force_harvest_mode or _is_harvest_task(task_description)
    template = _HARVEST_PROMPT if use_harvest else _LINEAR_PROMPT
    prompt = template.format(task=task_description)

    answer_content = ""
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content is not None:
                    answer_content += delta.content

    except openai.APIConnectionError as e:
        warnings.warn(f"API connection error: {e}", RuntimeWarning)
        return Plan(steps=[])
    except openai.RateLimitError as e:
        warnings.warn(f"API rate limit exceeded: {e}", RuntimeWarning)
        return Plan(steps=[])
    except openai.APIStatusError as e:
        warnings.warn(f"API status error {e.status_code}: {e.response}", RuntimeWarning)
        return Plan(steps=[])
    except Exception as e:
        warnings.warn(f"Unexpected error during LLM call: {e}", RuntimeWarning)
        return Plan(steps=[])

    if not answer_content:
        warnings.warn("LLM returned an empty response.", RuntimeWarning)
        return Plan(steps=[])

    plan = parse_llm_plan(answer_content)
    return plan


# ---------------------------------------------------------------------------
# Convenience: legacy flat-list interface (backward compat)
# ---------------------------------------------------------------------------

def decompose_task_with_llm_flat(
    task_description: str,
    model_name: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
) -> List[str]:
    """
    Legacy interface that returns a flat list of subtask strings.
    For linear plans only — raises ValueError if the plan contains REPEAT blocks.
    """
    plan = decompose_task_with_llm(task_description, model_name, base_url)
    if not plan.is_linear():
        raise ValueError(
            "Plan contains REPEAT…UNTIL blocks; use decompose_task_with_llm() instead."
        )
    return plan.flat_subtasks()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    tests = [
        # Linear task
        "pick up the red pepper and place it in the left basket",
        # Table-clearing task (English)
        "put all the vegetables on the table into the baskets",
        # Table-clearing task (Chinese)
        "将桌上所有蔬菜放入篮子",
        # Specific vegetable types
        "clear all peppers from the table and put them in the baskets",
        # Mixed types
        "clear all objects from the table and put them in the box",
    ]

    for task in tests:
        print("\n" + "=" * 60)
        print(f"Task: {task}")
        plan = decompose_task_with_llm(task)
        print(f"Steps ({len(plan.steps)}):")
        for step in plan.steps:
            if isinstance(step, str):
                print(f"  [linear] {step}")
            elif isinstance(step, RepeatBlock):
                print(f"  [REPEAT]")
                for s in step.body:
                    print(f"    {s}")
                print(f"  [UNTIL]  {step.until_condition}")
