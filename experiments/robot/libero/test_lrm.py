"""
test_lrm.py — Unit tests for LRM prompt engineering validation.

Covers:
  - Task 5.1: _is_harvest_task keyword routing tests
  - Task 5.3: parse_llm_plan REPEAT block parsing tests

Requirements: 7.1, 7.2, 7.4
"""

import pytest

from experiments.robot.libero.ds import (
    _is_harvest_task,
    parse_llm_plan,
    Plan,
    RepeatBlock,
)


# ---------------------------------------------------------------------------
# Sample plan strings
# ---------------------------------------------------------------------------

SAMPLE_HARVEST_PLAN = """\
REPEAT:
  1. scan the table
  2. pick up the nearest visible fruit on the table
  3. place the fruit in the basket
UNTIL: no fruits remain on the table
"""

SAMPLE_LINEAR_PLAN = """\
1. pick up the apple
2. place the apple in the basket
"""


# ---------------------------------------------------------------------------
# Task 5.1: _is_harvest_task keyword routing tests
# Requirements: 7.1
# ---------------------------------------------------------------------------

class TestIsHarvestTask:
    """Tests for _is_harvest_task heuristic keyword routing."""

    # --- English keywords that should return True ---

    def test_english_keyword_all(self):
        """Task containing 'all' should trigger harvest mode."""
        assert _is_harvest_task("put all the fruits on the table into the basket") is True

    def test_english_keyword_clear_the_table(self):
        """Task containing 'clear the table' should trigger harvest mode."""
        assert _is_harvest_task("clear the table of all objects") is True

    def test_english_keyword_harvest(self):
        """Task containing 'harvest' should trigger harvest mode."""
        assert _is_harvest_task("harvest all the blueberries") is True

    def test_english_keyword_every(self):
        """Task containing 'every' should trigger harvest mode."""
        assert _is_harvest_task("gather every fruit from the table") is True

    def test_english_keyword_collect(self):
        """Task containing 'collect' should trigger harvest mode."""
        assert _is_harvest_task("collect all items") is True

    # --- Chinese keywords that should return True ---

    def test_chinese_keyword_suoyou(self):
        """Task containing '所有' should trigger harvest mode."""
        assert _is_harvest_task("将桌上所有水果放入篮子") is True

    def test_chinese_keyword_qingkong(self):
        """Task containing '清空' should trigger harvest mode."""
        assert _is_harvest_task("清空桌面上的所有物品") is True

    def test_chinese_keyword_quanbu(self):
        """Task containing '全部' should trigger harvest mode."""
        assert _is_harvest_task("全部水果放入篮子") is True

    # --- Tasks without keywords that should return False ---

    def test_no_keyword_pick_up(self):
        """Simple pick-up task without harvest keywords should return False."""
        assert _is_harvest_task("pick up the apple") is False

    def test_no_keyword_place(self):
        """Simple place task without harvest keywords should return False."""
        assert _is_harvest_task("place the cup in the microwave") is False

    def test_no_keyword_open(self):
        """Simple open task without harvest keywords should return False."""
        assert _is_harvest_task("open the drawer") is False

    # --- Case-insensitivity check ---

    def test_case_insensitive_all_uppercase(self):
        """Keyword matching should be case-insensitive."""
        assert _is_harvest_task("Put ALL the fruits into the basket") is True

    def test_case_insensitive_harvest_mixed(self):
        """'Harvest' with mixed case should trigger harvest mode."""
        assert _is_harvest_task("Harvest the blueberries") is True


# ---------------------------------------------------------------------------
# Task 5.3: parse_llm_plan REPEAT block parsing tests
# Requirements: 7.2, 7.4
# ---------------------------------------------------------------------------

class TestParseLlmPlan:
    """Tests for parse_llm_plan REPEAT…UNTIL block parsing."""

    # --- SAMPLE_HARVEST_PLAN: single REPEAT block ---

    def test_harvest_plan_returns_plan_instance(self):
        """Parsing a harvest plan should return a Plan object."""
        result = parse_llm_plan(SAMPLE_HARVEST_PLAN)
        assert isinstance(result, Plan)

    def test_harvest_plan_has_one_step(self):
        """Harvest plan should produce exactly one step (the RepeatBlock)."""
        result = parse_llm_plan(SAMPLE_HARVEST_PLAN)
        assert len(result.steps) == 1

    def test_harvest_plan_step_is_repeat_block(self):
        """The single step in the harvest plan should be a RepeatBlock."""
        result = parse_llm_plan(SAMPLE_HARVEST_PLAN)
        assert isinstance(result.steps[0], RepeatBlock)

    def test_harvest_plan_body_has_three_subtasks(self):
        """RepeatBlock body should contain exactly 3 subtasks."""
        result = parse_llm_plan(SAMPLE_HARVEST_PLAN)
        block: RepeatBlock = result.steps[0]
        assert len(block.body) == 3

    def test_harvest_plan_body_first_step_is_scan(self):
        """First body step should be the scan instruction."""
        result = parse_llm_plan(SAMPLE_HARVEST_PLAN)
        block: RepeatBlock = result.steps[0]
        assert block.body[0] == "scan the table"

    def test_harvest_plan_body_second_step_is_pick_up(self):
        """Second body step should be the pick-up instruction."""
        result = parse_llm_plan(SAMPLE_HARVEST_PLAN)
        block: RepeatBlock = result.steps[0]
        assert block.body[1] == "pick up the nearest visible fruit on the table"

    def test_harvest_plan_body_third_step_is_place(self):
        """Third body step should be the place instruction."""
        result = parse_llm_plan(SAMPLE_HARVEST_PLAN)
        block: RepeatBlock = result.steps[0]
        assert block.body[2] == "place the fruit in the basket"

    def test_harvest_plan_until_condition(self):
        """UNTIL condition should match the UNTIL line content."""
        result = parse_llm_plan(SAMPLE_HARVEST_PLAN)
        block: RepeatBlock = result.steps[0]
        assert block.until_condition == "no fruits remain on the table"

    def test_harvest_plan_is_not_linear(self):
        """A plan with a RepeatBlock should not be considered linear."""
        result = parse_llm_plan(SAMPLE_HARVEST_PLAN)
        assert result.is_linear() is False

    # --- SAMPLE_LINEAR_PLAN: linear plan ---

    def test_linear_plan_returns_plan_instance(self):
        """Parsing a linear plan should return a Plan object."""
        result = parse_llm_plan(SAMPLE_LINEAR_PLAN)
        assert isinstance(result, Plan)

    def test_linear_plan_is_linear(self):
        """A plan with only string steps should be considered linear."""
        result = parse_llm_plan(SAMPLE_LINEAR_PLAN)
        assert result.is_linear() is True

    def test_linear_plan_has_two_steps(self):
        """Linear plan should produce exactly 2 steps."""
        result = parse_llm_plan(SAMPLE_LINEAR_PLAN)
        assert len(result.steps) == 2

    def test_linear_plan_steps_are_strings(self):
        """All steps in a linear plan should be strings."""
        result = parse_llm_plan(SAMPLE_LINEAR_PLAN)
        for step in result.steps:
            assert isinstance(step, str)

    def test_linear_plan_first_step(self):
        """First step of linear plan should be the pick-up instruction."""
        result = parse_llm_plan(SAMPLE_LINEAR_PLAN)
        assert result.steps[0] == "pick up the apple"

    def test_linear_plan_second_step(self):
        """Second step of linear plan should be the place instruction."""
        result = parse_llm_plan(SAMPLE_LINEAR_PLAN)
        assert result.steps[1] == "place the apple in the basket"

    # --- Empty string ---

    def test_empty_string_returns_empty_plan(self):
        """Parsing an empty string should return an empty Plan."""
        result = parse_llm_plan("")
        assert isinstance(result, Plan)
        assert len(result.steps) == 0

    def test_empty_plan_is_linear(self):
        """An empty plan (no steps) should be considered linear."""
        result = parse_llm_plan("")
        assert result.is_linear() is True

    # --- Edge cases ---

    def test_whitespace_only_string_returns_empty_plan(self):
        """Parsing a whitespace-only string should return an empty Plan."""
        result = parse_llm_plan("   \n\n  ")
        assert len(result.steps) == 0

    def test_repeat_block_case_insensitive(self):
        """REPEAT keyword matching should be case-insensitive."""
        plan_str = "repeat:\n  1. scan the table\nuntil: table is clear\n"
        result = parse_llm_plan(plan_str)
        assert len(result.steps) == 1
        assert isinstance(result.steps[0], RepeatBlock)
        assert result.steps[0].until_condition == "table is clear"
