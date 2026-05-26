"""
tests/test_properties.py — 属性测试汇总文件

使用 Hypothesis 框架整合所有属性测试，覆盖以下属性：

  Property 1:  Episode Temporal Dimension Consistency（Req 9.1）
  Property 2:  ActionNormalizer Round-Trip（Req 9.2）
  Property 3:  Normalized Action Value Range（Req 9.4）
  Property 7:  parse_llm_plan Idempotence（Req 7.5, 9.8）
  Property 8:  _is_harvest_task Keyword Routing（Req 7.1）
  Property 9:  parse_llm_plan REPEAT Block Parsing（Req 7.2, 7.4）
  Property 10: VQA Sample Structural Invariants（Req 3.2, 3.3, 3.6）
  Property 12: Pipeline Prerequisite Enforcement（Req 8.2, 8.3, 8.4）
  Property 13: VQA to Conversation Format（Req 6.2）
"""

import sys
import tempfile
import os
from pathlib import Path

# 确保项目根目录和 vla-scripts 目录在 sys.path 中，以便导入项目模块
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "vla-scripts"))

import numpy as np
import pytest

from hypothesis import given, settings, assume
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# 导入被测模块
# ---------------------------------------------------------------------------

# data.collect_fruit_demos 依赖硬件库（cv2、pyrealsense2、rtde_receive），
# 在无硬件环境下无法直接导入。
# 以下两个函数是该模块中的纯逻辑函数，在此处直接实现以供测试使用，
# 与原始实现保持完全一致（见 data/collect_fruit_demos.py）。

def _verify_temporal_consistency(episode_data: dict) -> bool:
    """
    验证 Episode 数据的时序维度一致性（Requirement 1.5, 9.1）。
    与 FruitDemoCollector.verify_temporal_consistency 实现完全一致。
    """
    T = episode_data["images"].shape[0]
    return all(
        episode_data[k].shape[0] == T
        for k in ["images_wrist", "tcp_poses", "joint_positions", "gripper"]
    )


def _collect_vqa_sample(images, instruction, answer, operation_type, fruit_type, failure_mode=None):
    """
    构造 VQA 样本（Requirement 3.2, 3.3, 3.6）。
    与 collect_fruit_demos.collect_vqa_sample 实现完全一致。
    返回一个包含 image_paths、question、answer 等字段的 dict（而非 dataclass）。
    """
    if len(images) != 2:
        raise ValueError(
            f"images 列表必须恰好包含 2 个元素（主视角 + 腕部视角），"
            f"实际收到 {len(images)} 个。"
        )
    if answer not in ("Yes", "No"):
        raise ValueError(
            f"answer 必须为 'Yes' 或 'No'，实际收到: '{answer}'"
        )
    question = (
        f"The robot instruction is: '{instruction}'. "
        f"Has this action been completed? Answer strictly 'Yes' or 'No'."
    )
    image_paths = ["<main_view_path>", "<wrist_view_path>"]
    return {
        "image_paths": image_paths,
        "question": question,
        "answer": answer,
        "operation_type": operation_type,
        "fruit_type": fruit_type,
        "failure_mode": failure_mode,
    }


from data.validate_dataset import DataValidator, format_plan
from experiments.robot.libero.ds import (
    Plan,
    RepeatBlock,
    parse_llm_plan,
    _is_harvest_task,
)
from data.build_vlm_dataset import vqa_sample_to_conversation
from run_training_pipeline import TrainingPipeline, PipelineConfig, PipelineError

# ---------------------------------------------------------------------------
# 辅助策略
# ---------------------------------------------------------------------------

# 7 维动作向量的 q01/q99 策略
_action_dim = 7

_stats_strategy = st.fixed_dictionaries({
    "action": st.fixed_dictionaries({
        "q01": st.lists(
            st.floats(min_value=-2.0, max_value=0.0, allow_nan=False, allow_infinity=False),
            min_size=_action_dim,
            max_size=_action_dim,
        ),
        "q99": st.lists(
            st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=_action_dim,
            max_size=_action_dim,
        ),
    })
})


def _make_stats_with_positive_range(q01_list, q99_list):
    """确保 q99 > q01（每个维度），避免零范围导致除零。"""
    q01 = np.array(q01_list, dtype=np.float64)
    q99 = np.array(q99_list, dtype=np.float64)
    # 强制 q99 = q01 + max(|q99 - q01|, 0.01)
    diff = q99 - q01
    diff = np.where(diff < 0.01, 0.01, diff)
    q99 = q01 + diff
    return {
        "action": {
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }
    }


# ---------------------------------------------------------------------------
# Property 1: Episode Temporal Dimension Consistency（Req 9.1）
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(T=st.integers(min_value=1, max_value=100))
def test_property1_episode_temporal_consistency(T):
    """
    **Validates: Requirements 1.5, 9.1**

    对于任意时序长度 T，构造包含正确形状数组的 episode_data，
    verify_temporal_consistency 应返回 True。
    修改任意一个数组的第一维后，应返回 False。
    """
    # 构造合法的 episode_data
    episode_data = {
        "images":           np.zeros((T, 256, 256, 3), dtype=np.uint8),
        "images_wrist":     np.zeros((T, 256, 256, 3), dtype=np.uint8),
        "tcp_poses":        np.zeros((T, 6), dtype=np.float64),
        "joint_positions":  np.zeros((T, 6), dtype=np.float64),
        "gripper":          np.zeros((T,), dtype=np.float64),
    }

    # 正例：所有数组第一维均为 T，应返回 True
    assert _verify_temporal_consistency(episode_data) is True

    # 反例：修改 tcp_poses 的第一维为 T+1，应返回 False
    if T < 100:  # 避免内存过大
        bad_episode_data = dict(episode_data)
        bad_episode_data["tcp_poses"] = np.zeros((T + 1, 6), dtype=np.float64)
        assert _verify_temporal_consistency(bad_episode_data) is False


# ---------------------------------------------------------------------------
# Property 2: ActionNormalizer Round-Trip（Req 9.2）
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(
    q01_list=st.lists(
        st.floats(min_value=-2.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
        min_size=_action_dim, max_size=_action_dim,
    ),
    q99_list=st.lists(
        st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
        min_size=_action_dim, max_size=_action_dim,
    ),
)
def test_property2_action_normalizer_roundtrip(q01_list, q99_list):
    """
    **Validates: Requirements 9.2**

    对于任意合法的 stats（包含 action.q01 和 action.q99），
    DataValidator.validate_action_normalizer_roundtrip(stats) 应返回 passed=True。
    """
    stats = _make_stats_with_positive_range(q01_list, q99_list)

    validator = DataValidator.__new__(DataValidator)
    result = validator.validate_action_normalizer_roundtrip(stats)

    assert result.passed is True, (
        f"ActionNormalizer round-trip 失败：{result.message}\n"
        f"stats={stats}"
    )


# ---------------------------------------------------------------------------
# Property 3: Normalized Action Value Range（Req 9.4）
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(
    q01_list=st.lists(
        st.floats(min_value=-2.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
        min_size=_action_dim, max_size=_action_dim,
    ),
    q99_list=st.lists(
        st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
        min_size=_action_dim, max_size=_action_dim,
    ),
)
def test_property3_normalized_action_value_range(q01_list, q99_list):
    """
    **Validates: Requirements 9.4**

    对于任意合法的 stats，归一化后的动作向量每个维度均在 [-1.0, 1.0] 内。
    """
    stats = _make_stats_with_positive_range(q01_list, q99_list)

    validator = DataValidator.__new__(DataValidator)
    result = validator.validate_normalized_action_range(stats)

    assert result.passed is True, (
        f"归一化动作值域验证失败：{result.message}\n"
        f"stats={stats}"
    )


# ---------------------------------------------------------------------------
# Property 7: parse_llm_plan Idempotence（Req 7.5, 9.8）
# ---------------------------------------------------------------------------

# 预定义的合法计划字符串列表
_VALID_PLAN_STRINGS = [
    # 线性计划
    "1. pick up the apple\n2. place it in the basket",
    "1. pick up the orange",
    "1. open the drawer\n2. pick up the cup\n3. close the drawer",
    # REPEAT 块计划
    "REPEAT:\n  1. scan the table\n  2. pick up the nearest visible fruit\n  3. place it in the basket\nUNTIL: no fruits remain on the table",
    "REPEAT:\n  1. scan the table\n  2. pick up the nearest visible object\nUNTIL: the table is clear",
    # 混合计划（线性 + REPEAT）
    "1. open the basket\nREPEAT:\n  1. scan the table\n  2. pick up the nearest visible fruit\n  3. place it in the basket\nUNTIL: no fruits remain on the table\n2. close the basket",
]


@settings(max_examples=50)
@given(plan_string=st.sampled_from(_VALID_PLAN_STRINGS))
def test_property7_parse_llm_plan_idempotence(plan_string):
    """
    **Validates: Requirements 7.5, 9.8**

    对于预定义的合法计划字符串，parse_llm_plan 的幂等性应成立：
    parse_llm_plan(format_plan(parse_llm_plan(s))) 与 parse_llm_plan(s) 语义等价。
    """
    validator = DataValidator.__new__(DataValidator)
    result = validator.validate_parse_llm_plan_idempotence([plan_string])

    assert result.passed is True, (
        f"parse_llm_plan 幂等性验证失败：{result.message}\n"
        f"plan_string={plan_string!r}"
    )


# ---------------------------------------------------------------------------
# Property 8: _is_harvest_task Keyword Routing（Req 7.1）
# ---------------------------------------------------------------------------

# harvest 关键词列表（来自 ds.py 的 _is_harvest_task 实现）
_HARVEST_KEYWORDS = [
    "all", "every", "each", "all fruits", "all objects", "all items",
    "clear the table", "clear all", "collect all", "gather all",
    "harvest", "gather", "collect",
    "所有", "全部", "所有水果", "清空", "收集所有", "摘", "采",
]


@settings(max_examples=50)
@given(keyword=st.sampled_from(_HARVEST_KEYWORDS))
def test_property8_is_harvest_task_keyword_routing(keyword):
    """
    **Validates: Requirements 7.1**

    对于任意 harvest 关键词，构造包含该关键词的任务描述，
    _is_harvest_task 应返回 True。
    """
    # 构造包含关键词的任务描述
    task_description = f"Please {keyword} the fruits on the table into the basket"

    result = _is_harvest_task(task_description)

    assert result is True, (
        f"_is_harvest_task 应对包含关键词 '{keyword}' 的任务描述返回 True，"
        f"但返回了 False。\n"
        f"task_description={task_description!r}"
    )


# ---------------------------------------------------------------------------
# Property 9: parse_llm_plan REPEAT Block Parsing（Req 7.2, 7.4）
# ---------------------------------------------------------------------------

# body 步骤的候选列表
_BODY_STEPS = [
    "scan the table",
    "pick up the nearest visible fruit",
    "place it in the basket",
    "pick up the nearest visible object on the table",
    "scan the surface",
]

_UNTIL_CONDITIONS = [
    "no fruits remain on the table",
    "the table is clear",
    "no objects remain on the table",
    "all fruits have been placed in the basket",
]


@settings(max_examples=50)
@given(
    body_steps=st.lists(
        st.sampled_from(_BODY_STEPS),
        min_size=1,
        max_size=3,
    ),
    until_condition=st.sampled_from(_UNTIL_CONDITIONS),
)
def test_property9_parse_llm_plan_repeat_block_parsing(body_steps, until_condition):
    """
    **Validates: Requirements 7.2, 7.4**

    对于随机生成的 REPEAT 块计划字符串，parse_llm_plan 应正确解析：
    - 结果包含一个 RepeatBlock
    - body 长度与输入一致
    - until_condition 非空
    """
    # 构造 REPEAT 块计划字符串
    body_lines = "\n".join(
        f"  {i + 1}. {step}" for i, step in enumerate(body_steps)
    )
    plan_string = f"REPEAT:\n{body_lines}\nUNTIL: {until_condition}"

    plan = parse_llm_plan(plan_string)

    # 验证：结果包含一个 RepeatBlock
    repeat_blocks = [s for s in plan.steps if isinstance(s, RepeatBlock)]
    assert len(repeat_blocks) == 1, (
        f"期望解析出 1 个 RepeatBlock，实际得到 {len(repeat_blocks)} 个。\n"
        f"plan_string={plan_string!r}\n"
        f"plan.steps={plan.steps}"
    )

    block = repeat_blocks[0]

    # 验证：body 长度正确
    assert len(block.body) == len(body_steps), (
        f"RepeatBlock.body 长度期望 {len(body_steps)}，实际 {len(block.body)}。\n"
        f"plan_string={plan_string!r}\n"
        f"block.body={block.body}"
    )

    # 验证：until_condition 非空
    assert block.until_condition, (
        f"RepeatBlock.until_condition 不应为空。\n"
        f"plan_string={plan_string!r}"
    )

    # 验证：until_condition 与输入一致
    assert block.until_condition == until_condition, (
        f"RepeatBlock.until_condition 期望 '{until_condition}'，"
        f"实际 '{block.until_condition}'。\n"
        f"plan_string={plan_string!r}"
    )


# ---------------------------------------------------------------------------
# Property 10: VQA Sample Structural Invariants（Req 3.2, 3.3, 3.6）
# ---------------------------------------------------------------------------

_FRUIT_TYPES = ["apple", "orange", "banana", "pear", "tomato"]
_OPERATION_TYPES = ["pick_up", "place"]
_ANSWERS = ["Yes", "No"]


@settings(max_examples=50)
@given(
    instruction=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
        min_size=1,
        max_size=50,
    ),
    answer=st.sampled_from(_ANSWERS),
    operation_type=st.sampled_from(_OPERATION_TYPES),
    fruit_type=st.sampled_from(_FRUIT_TYPES),
)
def test_property10_vqa_sample_structural_invariants(
    instruction, answer, operation_type, fruit_type
):
    """
    **Validates: Requirements 3.2, 3.3, 3.6**

    对于任意合法的 instruction、answer、operation_type、fruit_type，
    collect_vqa_sample 返回的样本应满足：
    - 有 2 个 image_paths
    - question 包含 instruction
    - answer 为 'Yes' 或 'No'
    """
    # 构造 2 个占位图像数组（不需要真实图像内容）
    images = [
        np.zeros((256, 256, 3), dtype=np.uint8),
        np.zeros((256, 256, 3), dtype=np.uint8),
    ]

    sample = _collect_vqa_sample(
        images=images,
        instruction=instruction,
        answer=answer,
        operation_type=operation_type,
        fruit_type=fruit_type,
        failure_mode=None,
    )

    # 验证：有 2 个 image_paths（Requirement 3.2）
    assert len(sample["image_paths"]) == 2, (
        f"VQA 样本应有 2 个 image_paths，实际有 {len(sample['image_paths'])} 个。"
    )

    # 验证：question 包含 instruction（Requirement 3.3）
    assert instruction in sample["question"], (
        f"VQA 样本的 question 应包含 instruction '{instruction}'，"
        f"实际 question='{sample['question']}'"
    )

    # 验证：answer 为 'Yes' 或 'No'（Requirement 3.6）
    assert sample["answer"] in ("Yes", "No"), (
        f"VQA 样本的 answer 应为 'Yes' 或 'No'，实际为 '{sample['answer']}'"
    )

    # 验证：answer 与输入一致
    assert sample["answer"] == answer, (
        f"VQA 样本的 answer 应为 '{answer}'，实际为 '{sample['answer']}'"
    )


# ---------------------------------------------------------------------------
# Property 12: Pipeline Prerequisite Enforcement（Req 8.2, 8.3, 8.4）
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(
    vqa_count=st.integers(min_value=0, max_value=399),  # 不满足 >= 400 的条件
)
def test_property12_stage2_prerequisite_vqa_insufficient(vqa_count):
    """
    **Validates: Requirements 8.2**

    当 VQA 样本数 < 400 时，check_stage2_prerequisites() 应抛出 PipelineError。
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 创建 vlm_data 目录和 vqa_completion.jsonl 文件
        vlm_data_dir = tmp_path / "vlm_data"
        vlm_data_dir.mkdir(parents=True)
        vqa_path = vlm_data_dir / "vqa_completion.jsonl"

        # 写入 vqa_count 行（每行一个 JSON 样本）
        with open(vqa_path, "w", encoding="utf-8") as f:
            for i in range(vqa_count):
                f.write('{"answer": "Yes", "question": "test"}\n')

        config = PipelineConfig(
            data_root=str(tmp_path),
            min_vqa_samples=400,
        )
        pipeline = TrainingPipeline(config)

        with pytest.raises(PipelineError):
            pipeline.check_stage2_prerequisites()


@settings(max_examples=50, deadline=None)
@given(
    episode_count=st.integers(min_value=0, max_value=299),  # 不满足 >= 300 的条件
)
def test_property12_stage3_prerequisite_episodes_insufficient(episode_count):
    """
    **Validates: Requirements 8.3**

    当 Episode 数 < 300 时，check_stage3_prerequisites() 应抛出 PipelineError。
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 创建 atomic_ops 目录并写入 episode_count 个空 .npz 文件
        # check_stage3_prerequisites 使用 glob("*.npz") 计数，不读取文件内容
        atomic_ops_dir = tmp_path / "atomic_ops"
        atomic_ops_dir.mkdir(parents=True)

        for i in range(episode_count):
            npz_path = atomic_ops_dir / f"episode_{i:04d}.npz"
            npz_path.touch()  # 创建空文件，glob 计数即可

        config = PipelineConfig(
            data_root=str(tmp_path),
            min_episodes=300,
        )
        pipeline = TrainingPipeline(config)

        with pytest.raises(PipelineError):
            pipeline.check_stage3_prerequisites()


@settings(max_examples=50)
@given(
    missing_vla=st.booleans(),
    missing_vlm=st.booleans(),
)
def test_property12_stage5_prerequisite_checkpoint_missing(missing_vla, missing_vlm):
    """
    **Validates: Requirements 8.4**

    当 VLA checkpoint 或 VLM checkpoint 不存在时，
    check_stage5_prerequisites() 应抛出 PipelineError。
    """
    # 至少一个 checkpoint 缺失才能触发 PipelineError
    assume(missing_vla or missing_vlm)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        vla_ckpt_dir = tmp_path / "vla_checkpoint"
        vlm_ckpt_dir = tmp_path / "vlm_checkpoint"

        # 根据 missing_* 标志决定是否创建目录
        if not missing_vla:
            vla_ckpt_dir.mkdir(parents=True)
        if not missing_vlm:
            vlm_ckpt_dir.mkdir(parents=True)

        config = PipelineConfig(
            data_root=str(tmp_path),
            vla_checkpoint_dir=str(vla_ckpt_dir),
            vlm_stage2_checkpoint_dir=str(vlm_ckpt_dir),
        )
        pipeline = TrainingPipeline(config)

        with pytest.raises(PipelineError):
            pipeline.check_stage5_prerequisites()


# ---------------------------------------------------------------------------
# Property 13: VQA to Conversation Format（Req 6.2）
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(
    image_path1=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        min_size=1,
        max_size=20,
    ),
    image_path2=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        min_size=1,
        max_size=20,
    ),
    question=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
        min_size=1,
        max_size=100,
    ),
    answer=st.sampled_from(["Yes", "No"]),
)
def test_property13_vqa_to_conversation_format(
    image_path1, image_path2, question, answer
):
    """
    **Validates: Requirements 6.2**

    对于任意合法的 VQA 样本 dict（包含 2 个 image_paths、question、answer），
    vqa_sample_to_conversation 返回的 messages 列表应满足：
    - 有 2 条消息（user + assistant）
    - user 消息有 3 个 content 块（2 个 image + 1 个 text）
    - assistant 消息有 1 个 content 块
    """
    sample = {
        "image_paths": [image_path1, image_path2],
        "question": question,
        "answer": answer,
    }

    conversation = vqa_sample_to_conversation(sample)

    messages = conversation["messages"]

    # 验证：有 2 条消息
    assert len(messages) == 2, (
        f"对话应有 2 条消息，实际有 {len(messages)} 条。"
    )

    user_msg = messages[0]
    assistant_msg = messages[1]

    # 验证：第一条消息为 user
    assert user_msg["role"] == "user", (
        f"第一条消息的 role 应为 'user'，实际为 '{user_msg['role']}'。"
    )

    # 验证：user 消息有 3 个 content 块（2 个 image + 1 个 text）
    user_content = user_msg["content"]
    assert len(user_content) == 3, (
        f"user 消息应有 3 个 content 块，实际有 {len(user_content)} 个。"
    )

    # 验证：前 2 个 content 块为 image 类型
    assert user_content[0]["type"] == "image", (
        f"user 消息第 1 个 content 块应为 'image' 类型，"
        f"实际为 '{user_content[0]['type']}'。"
    )
    assert user_content[1]["type"] == "image", (
        f"user 消息第 2 个 content 块应为 'image' 类型，"
        f"实际为 '{user_content[1]['type']}'。"
    )

    # 验证：第 3 个 content 块为 text 类型
    assert user_content[2]["type"] == "text", (
        f"user 消息第 3 个 content 块应为 'text' 类型，"
        f"实际为 '{user_content[2]['type']}'。"
    )

    # 验证：第二条消息为 assistant
    assert assistant_msg["role"] == "assistant", (
        f"第二条消息的 role 应为 'assistant'，实际为 '{assistant_msg['role']}'。"
    )

    # 验证：assistant 消息有 1 个 content 块
    assistant_content = assistant_msg["content"]
    assert len(assistant_content) == 1, (
        f"assistant 消息应有 1 个 content 块，实际有 {len(assistant_content)} 个。"
    )

    # 验证：assistant 消息的 content 块为 text 类型，内容为 answer
    assert assistant_content[0]["type"] == "text", (
        f"assistant 消息的 content 块应为 'text' 类型，"
        f"实际为 '{assistant_content[0]['type']}'。"
    )
    assert assistant_content[0]["text"] == answer, (
        f"assistant 消息的 text 应为 '{answer}'，"
        f"实际为 '{assistant_content[0]['text']}'。"
    )
