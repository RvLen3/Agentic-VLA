"""
VLM 数据集构建模块

本模块将 JSONL 格式的 VQA 样本和扫描样本转换为 Qwen2.5-VL 对话格式，
用于 VLM（Qwen2.5-VL）的 LoRA 微调训练。

主要功能：
  - load_vqa_samples()：从 JSONL 文件加载 VQA 样本，返回原始 dict 列表
  - vqa_sample_to_conversation()：将 VQASample（dict 格式）转换为 Qwen2.5-VL 对话格式
  - scan_sample_to_conversation()：将 ScanSample（dict 格式）转换为 Qwen2.5-VL 对话格式
  - build_stage1_dataset()：构建 Stage 1 训练集（仅子任务完成验证数据）
  - build_stage2_dataset()：构建 Stage 2 训练集（验证 + 扫描 + 终止判断数据）
  - compute_label_balance()：计算正负样本比例

对话格式遵循 Qwen2.5-VL 的 messages 列表规范（Requirement 6.2）：
  - user 消息：包含 image 块和 text 块
  - assistant 消息：包含 text 块（答案）

参考文档：
  - requirements.md Requirement 6.2
  - design.md Section 2: data/build_vlm_dataset.py
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

# 尝试从 data.collect_fruit_demos 导入 VQASample 和 ScanSample
# 若导入失败（例如缺少硬件依赖），则定义简单的替代类
try:
    from data.collect_fruit_demos import VQASample, ScanSample
except ImportError:
    try:
        from collect_fruit_demos import VQASample, ScanSample
    except ImportError:
        # 定义简单的替代类，保证模块在无硬件依赖环境下可用
        from dataclasses import dataclass, field

        @dataclass
        class VQASample:  # type: ignore[no-redef]
            """VQA 样本替代类（当 collect_fruit_demos 不可用时使用）"""
            image_paths: List[str]
            question: str
            answer: str
            operation_type: str
            fruit_type: str
            failure_mode: Optional[str] = None

        @dataclass
        class ScanSample:  # type: ignore[no-redef]
            """扫描样本替代类（当 collect_fruit_demos 不可用时使用）"""
            image_path: str
            visible_targets: List[str]
            target_count: int
            fruit_types_present: List[str]


logger = logging.getLogger(__name__)

# 扫描任务的标准化问题模板（Requirement 4.1, 4.3）
_SCAN_QUESTION = (
    "What fruits are visible on the table? "
    "List them in order from nearest to farthest, one per line. "
    "If the table is clear, respond with 'No fruits on the table'."
)

# 扫描任务空桌面的标准答案（Requirement 4.2）
_SCAN_EMPTY_ANSWER = "No fruits on the table"


# ============ 核心函数 ============

def load_vqa_samples(jsonl_path: str) -> List[dict]:
    """
    从 JSONL 文件加载 VQA 样本，返回原始 dict 列表。

    每行解析为一个 JSON 对象（dict），跳过空行和解析失败的行。

    Args:
        jsonl_path: JSONL 文件路径，每行为一个 JSON 对象。

    Returns:
        原始 dict 列表，每个 dict 对应 JSONL 文件中的一行。

    Raises:
        FileNotFoundError: 当文件不存在时抛出。

    Examples:
        >>> samples = load_vqa_samples("raw_demos/vlm_data/vqa_completion.jsonl")
        >>> len(samples)
        400
        >>> samples[0].keys()
        dict_keys(['image_paths', 'question', 'answer', 'operation_type', 'fruit_type', 'failure_mode'])
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL 文件不存在: {jsonl_path}")

    samples: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_num} 行 JSON 解析失败，已跳过: {e}")

    logger.info(f"从 {jsonl_path} 加载了 {len(samples)} 个样本")
    return samples


def vqa_sample_to_conversation(sample: dict) -> dict:
    """
    将 VQASample（dict 格式）转换为 Qwen2.5-VL 对话格式（Requirement 6.2）。

    输入 sample 包含字段：
      - image_paths (List[str])：图像路径列表，必须恰好包含 2 个路径
      - question (str)：问题字符串
      - answer (str)：答案字符串（"Yes" 或 "No"）

    返回格式：
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "<path1>"},
                    {"type": "image", "image": "<path2>"},
                    {"type": "text",  "text": "<question>"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<answer>"}]
            }
        ]
    }

    user 消息的 content 包含 2 个 image 块 + 1 个 text 块，共 3 个元素。
    assistant 消息的 content 包含 1 个 text 块。

    Args:
        sample: VQA 样本字典，包含 image_paths、question、answer 字段。

    Returns:
        Qwen2.5-VL 对话格式字典，包含 "messages" 键。

    Raises:
        KeyError: 当 sample 缺少必要字段时抛出。
        ValueError: 当 image_paths 长度不为 2 时抛出。

    Examples:
        >>> sample = {
        ...     "image_paths": ["main.png", "wrist.png"],
        ...     "question": "Has this action been completed? Answer strictly 'Yes' or 'No'.",
        ...     "answer": "Yes"
        ... }
        >>> conv = vqa_sample_to_conversation(sample)
        >>> len(conv["messages"])
        2
        >>> conv["messages"][0]["role"]
        'user'
        >>> len(conv["messages"][0]["content"])
        3
        >>> conv["messages"][1]["role"]
        'assistant'
    """
    image_paths: List[str] = sample["image_paths"]
    question: str = sample["question"]
    answer: str = sample["answer"]

    if len(image_paths) != 2:
        raise ValueError(
            f"vqa_sample_to_conversation 要求 image_paths 恰好包含 2 个路径，"
            f"实际收到 {len(image_paths)} 个。"
        )

    # 构造 user 消息的 content：2 个 image 块 + 1 个 text 块
    user_content = [
        {"type": "image", "image": image_paths[0]},
        {"type": "image", "image": image_paths[1]},
        {"type": "text",  "text": question},
    ]

    # 构造 assistant 消息的 content：1 个 text 块
    assistant_content = [
        {"type": "text", "text": answer},
    ]

    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def scan_sample_to_conversation(sample: dict) -> dict:
    """
    将 ScanSample（dict 格式）转换为 Qwen2.5-VL 对话格式（扫描任务）。

    输入 sample 包含字段：
      - image_path (str)：场景图像路径（主视角）
      - visible_targets (List[str])：可见水果目标的有序描述列表

    问题模板（固定）：
        "What fruits are visible on the table? List them in order from nearest
        to farthest, one per line. If the table is clear, respond with
        'No fruits on the table'."

    答案生成规则：
      - 若 visible_targets 为空列表：答案为 "No fruits on the table"
      - 否则：每行一个目标描述，用换行符连接
        例如："the apple on the left side of the table\\nthe orange near the center"

    返回格式与 vqa_sample_to_conversation 相同，但 user 消息的 content
    只包含 1 个 image 块 + 1 个 text 块（共 2 个元素）。

    Args:
        sample: 扫描样本字典，包含 image_path、visible_targets 字段。

    Returns:
        Qwen2.5-VL 对话格式字典，包含 "messages" 键。

    Raises:
        KeyError: 当 sample 缺少必要字段时抛出。

    Examples:
        >>> sample = {
        ...     "image_path": "frames/scene_001.png",
        ...     "visible_targets": [
        ...         "the apple on the left side of the table",
        ...         "the orange near the center"
        ...     ]
        ... }
        >>> conv = scan_sample_to_conversation(sample)
        >>> len(conv["messages"])
        2
        >>> conv["messages"][0]["role"]
        'user'
        >>> len(conv["messages"][0]["content"])
        2
        >>> conv["messages"][1]["content"][0]["text"]
        'the apple on the left side of the table\\nthe orange near the center'
    """
    image_path: str = sample["image_path"]
    visible_targets: List[str] = sample["visible_targets"]

    # 构造答案
    if not visible_targets:
        answer = _SCAN_EMPTY_ANSWER
    else:
        answer = "\n".join(visible_targets)

    # 构造 user 消息的 content：1 个 image 块 + 1 个 text 块
    user_content = [
        {"type": "image", "image": image_path},
        {"type": "text",  "text": _SCAN_QUESTION},
    ]

    # 构造 assistant 消息的 content：1 个 text 块
    assistant_content = [
        {"type": "text", "text": answer},
    ]

    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def build_stage1_dataset(
    vqa_completion_path: str,
    output_path: str,
    val_split: float = 0.1,
) -> Tuple[List[dict], List[dict]]:
    """
    构建 Stage 1 训练集（仅子任务完成验证数据）。

    从 vqa_completion.jsonl 加载样本，转换为对话格式，
    按 val_split 比例划分训练集和验证集，并保存到 output_path。

    Args:
        vqa_completion_path: VQA 完成验证数据的 JSONL 文件路径。
        output_path: 输出目录路径，将保存 train.jsonl 和 val.jsonl。
        val_split: 验证集比例，默认 0.1（10%）。

    Returns:
        (train_conversations, val_conversations) 元组，
        每个元素为对话格式 dict 列表。
    """
    samples = load_vqa_samples(vqa_completion_path)
    conversations = [vqa_sample_to_conversation(s) for s in samples]

    # 随机打乱后划分
    random.shuffle(conversations)
    split_idx = max(1, int(len(conversations) * (1 - val_split)))
    train_convs = conversations[:split_idx]
    val_convs = conversations[split_idx:]

    # 保存到输出目录
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_conversations(train_convs, output_dir / "train.jsonl")
    _save_conversations(val_convs, output_dir / "val.jsonl")

    logger.info(
        f"Stage 1 数据集构建完成：训练集 {len(train_convs)} 条，"
        f"验证集 {len(val_convs)} 条，保存至 {output_path}"
    )
    return train_convs, val_convs


def build_stage2_dataset(
    vqa_completion_path: str,
    scan_samples_path: str,
    vqa_termination_path: str,
    output_path: str,
    val_split: float = 0.1,
) -> Tuple[List[dict], List[dict]]:
    """
    构建 Stage 2 训练集（验证 + 扫描 + 终止判断数据）。

    合并三类数据：
      1. VQA 完成验证数据（vqa_completion.jsonl）
      2. 扫描样本数据（scan_samples.jsonl）
      3. VQA 终止判断数据（vqa_termination.jsonl）

    Args:
        vqa_completion_path: VQA 完成验证数据的 JSONL 文件路径。
        scan_samples_path: 扫描样本数据的 JSONL 文件路径。
        vqa_termination_path: VQA 终止判断数据的 JSONL 文件路径。
        output_path: 输出目录路径，将保存 train.jsonl 和 val.jsonl。
        val_split: 验证集比例，默认 0.1（10%）。

    Returns:
        (train_conversations, val_conversations) 元组，
        每个元素为对话格式 dict 列表。
    """
    # 加载并转换三类数据
    completion_samples = load_vqa_samples(vqa_completion_path)
    completion_convs = [vqa_sample_to_conversation(s) for s in completion_samples]

    scan_samples = load_vqa_samples(scan_samples_path)
    scan_convs = [scan_sample_to_conversation(s) for s in scan_samples]

    termination_samples = load_vqa_samples(vqa_termination_path)
    termination_convs = [vqa_sample_to_conversation(s) for s in termination_samples]

    # 合并所有对话
    all_convs = completion_convs + scan_convs + termination_convs
    random.shuffle(all_convs)

    # 划分训练集和验证集
    split_idx = max(1, int(len(all_convs) * (1 - val_split)))
    train_convs = all_convs[:split_idx]
    val_convs = all_convs[split_idx:]

    # 保存到输出目录
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_conversations(train_convs, output_dir / "train.jsonl")
    _save_conversations(val_convs, output_dir / "val.jsonl")

    logger.info(
        f"Stage 2 数据集构建完成：训练集 {len(train_convs)} 条，"
        f"验证集 {len(val_convs)} 条，保存至 {output_path}"
    )
    return train_convs, val_convs


def compute_label_balance(conversations: List[dict]) -> float:
    """
    计算正负样本比例（Requirement 3.7, 9.7）。

    遍历对话列表，统计 assistant 消息中答案为 "Yes" 的样本数量，
    计算正样本比例 = 正样本数 / 总样本数。

    仅统计 assistant 消息 content 中第一个 text 块的文本，
    忽略扫描任务等非 Yes/No 答案的对话。

    Args:
        conversations: 对话格式 dict 列表，每个 dict 包含 "messages" 键。

    Returns:
        正样本比例（float），范围 [0.0, 1.0]。
        若 conversations 为空，返回 0.0。

    Examples:
        >>> convs = [
        ...     {"messages": [{"role": "user", "content": [...]},
        ...                   {"role": "assistant", "content": [{"type": "text", "text": "Yes"}]}]},
        ...     {"messages": [{"role": "user", "content": [...]},
        ...                   {"role": "assistant", "content": [{"type": "text", "text": "No"}]}]},
        ... ]
        >>> compute_label_balance(convs)
        0.5
    """
    if not conversations:
        return 0.0

    positive_count = 0
    total_count = 0

    for conv in conversations:
        messages = conv.get("messages", [])
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if content and content[0].get("type") == "text":
                    answer = content[0].get("text", "").strip()
                    if answer in ("Yes", "No"):
                        total_count += 1
                        if answer == "Yes":
                            positive_count += 1

    if total_count == 0:
        return 0.0

    return positive_count / total_count


# ============ 内部辅助函数 ============

def _save_conversations(conversations: List[dict], output_path: Path) -> None:
    """
    将对话列表保存为 JSONL 文件。

    Args:
        conversations: 对话格式 dict 列表。
        output_path: 输出文件路径。
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    logger.debug(f"已保存 {len(conversations)} 条对话到 {output_path}")
