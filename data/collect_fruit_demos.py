"""
水果演示数据采集模块

本模块实现桌面水果收纳长时序任务（Multi-Fruit Table Clearing Task）的演示数据采集。
基于已有的 UR7e 采集脚本（data/dataset.py），扩展支持：
  - 多种水果类型（apple、orange、banana、pear、tomato）
  - 原子操作（pick_up / place）和完整 Episode 两种采集模式
  - Episode 元数据记录（EpisodeMetadata）
  - VQA 样本和扫描样本的构造与保存
  - 数据集统计量计算（dataset_statistics.json）

采集的数据用于后续 X-VLA LoRA 微调和 Qwen2.5-VL LoRA 微调。
"""

import os
import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs
from rtde_receive import RTDEReceiveInterface


# ============ 常量定义 ============

FRUIT_TYPES: List[str] = [
    "red pepper",
    "green pepper",
    "yellow pepper",
    "corn",
    "purple sweet potato",
    "pumpkin",
]
"""支持的蔬菜类型列表（红色辣椒、绿色辣椒、黄色辣椒、玉米、紫薯、南瓜）"""

BASKET_SIDES: List[str] = ["left", "right"]
"""篮子位置列表（左侧 / 右侧）"""

ATOMIC_TASK_TYPES: List[str] = ["pick_up", "place"]
"""原子操作类型列表"""

SPATIAL_POSITIONS: List[str] = ["left", "center", "right"]
"""桌面空间位置列表"""

SCENE_CONFIGS: List[str] = ["max_variety", "same_type", "spatial_layout"]
"""场景配置类型列表"""


# ============ 数据类定义 ============

@dataclass
class EpisodeMetadata:
    """
    完整 Episode 的元数据（Requirement 2.4）。

    记录每条演示 Episode 的任务类型、水果信息、采集统计等元数据，
    保存为 episode_NNNN_meta.json sidecar 文件。

    Fields:
        episode_id: Episode 的唯一编号。
        task_type: 任务类型，取值为 'atomic_ops' 或 'full_episodes'。
        fruit_type: 主要水果类型，如 'apple'、'orange' 等；完整 Episode 可为 'mixed'。
        total_fruit_count: Episode 开始时桌面上的水果总数。
        fruit_types_present: Episode 中出现的水果类型列表。
        successfully_placed_count: 成功放入篮子的水果数量。
        loop_iterations: REPEAT 循环的迭代次数（完整 Episode 专用，原子操作为 0）。
        total_timestep_count: Episode 的总时间步数（帧数）。
        spatial_position: 水果的空间位置，取值为 'left'、'center'、'right' 或 'mixed'。
        scene_config: 场景配置，取值为 'max_variety'、'same_type' 或 'spatial_layout'。
    """
    episode_id: int
    task_type: str          # "atomic_ops" | "full_episodes"
    fruit_type: str
    total_fruit_count: int
    fruit_types_present: List[str]
    successfully_placed_count: int
    loop_iterations: int
    total_timestep_count: int
    spatial_position: str   # "left" | "center" | "right" | "mixed"
    scene_config: str       # "max_variety" | "same_type" | "spatial_layout"


@dataclass
class VQASample:
    """
    视觉问答（VQA）样本（Requirement 3.6）。

    用于 VLM 子任务完成验证微调，每个样本包含双视角图像、
    标准化问题和 Yes/No 答案。

    Fields:
        image_paths: 图像路径列表，必须恰好包含 2 个路径（主视角 + 腕部视角）。
        question: 标准化问题模板字符串。
        answer: 答案，取值为 'Yes' 或 'No'。
        operation_type: 操作类型，取值为 'pick_up' 或 'place'。
        fruit_type: 水果类型。
        failure_mode: 失败模式描述（负样本专用），正样本为 None。
    """
    image_paths: List[str]
    question: str
    answer: str
    operation_type: str
    fruit_type: str
    failure_mode: Optional[str]


@dataclass
class ScanSample:
    """
    目标扫描标注样本（Requirement 4.6）。

    用于 VLM 目标扫描微调，记录场景图像和桌面可见水果的有序列表。

    Fields:
        image_path: 场景图像路径（主视角）。
        visible_targets: 可见水果目标的有序描述列表，按抓取优先级排序；空列表表示桌面已清空。
        target_count: 可见水果目标数量，等于 len(visible_targets)。
        fruit_types_present: 当前场景中出现的水果类型列表。
    """
    image_path: str
    visible_targets: List[str]
    target_count: int
    fruit_types_present: List[str]


# ============ 核心函数 ============

def build_instruction(task_type: str, fruit_type: str, basket_side: str = "left") -> str:
    """
    生成标准化指令字符串（Requirement 1.6）。

    根据原子操作类型、蔬菜类型和篮子位置，生成用于 VLA 训练的标准化语言指令。
    指令格式与 data/dataset.py 中的 TASK_LIST 保持一致。

    Args:
        task_type: 原子操作类型，支持 'pick_up' 和 'place'。
        fruit_type: 蔬菜类型，如 'red pepper'、'corn' 等。
        basket_side: 篮子位置，取值为 'left' 或 'right'，仅 place 操作使用。

    Returns:
        标准化指令字符串。
        - task_type='pick_up' -> 'pick up the {fruit_type}'
        - task_type='place'   -> 'place the {fruit_type} in the {basket_side} basket'

    Raises:
        ValueError: 当 task_type 不是 'pick_up' 或 'place' 时抛出。
        ValueError: 当 basket_side 不是 'left' 或 'right' 时抛出。

    Examples:
        >>> build_instruction('pick_up', 'red pepper')
        'pick up the red pepper'
        >>> build_instruction('place', 'corn', 'left')
        'place the corn in the left basket'
        >>> build_instruction('place', 'pumpkin', 'right')
        'place the pumpkin in the right basket'
    """
    if task_type == "pick_up":
        return f"pick up the {fruit_type}"
    elif task_type == "place":
        if basket_side not in BASKET_SIDES:
            raise ValueError(
                f"不支持的 basket_side: '{basket_side}'。"
                f"有效值为: {BASKET_SIDES}"
            )
        return f"place the {fruit_type} in the {basket_side} basket"
    else:
        raise ValueError(
            f"不支持的 task_type: '{task_type}'。"
            f"有效值为: {ATOMIC_TASK_TYPES}"
        )


def collect_vqa_sample(
    images: List[np.ndarray],
    instruction: str,
    answer: str,
    operation_type: str,
    fruit_type: str,
    failure_mode: Optional[str] = None,
) -> VQASample:
    """
    构造 VQA 样本（Requirement 3.2, 3.3, 3.6）。

    使用标准化问题模板构造视觉问答样本，确保 image_paths 恰好包含 2 个路径。
    注意：本函数接受图像数组列表，调用方需负责将图像保存到磁盘并提供路径。
    此函数主要用于验证样本结构的正确性。

    Args:
        images: 图像数组列表，必须恰好包含 2 个元素（主视角 + 腕部视角）。
        instruction: 机器人执行的指令字符串，用于构造问题模板。
        answer: VQA 答案，必须为 'Yes' 或 'No'。
        operation_type: 操作类型，取值为 'pick_up' 或 'place'。
        fruit_type: 水果类型。
        failure_mode: 失败模式描述（负样本专用），正样本传入 None。

    Returns:
        构造好的 VQASample 对象，image_paths 为占位符（需调用方替换为实际路径）。

    Raises:
        ValueError: 当 images 列表长度不为 2 时抛出。
        ValueError: 当 answer 不为 'Yes' 或 'No' 时抛出。
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

    # 标准化问题模板（Requirement 3.3）
    question = (
        f"The robot instruction is: '{instruction}'. "
        f"Has this action been completed? Answer strictly 'Yes' or 'No'."
    )

    # image_paths 使用占位符，调用方在保存图像后替换为实际路径
    image_paths = ["<main_view_path>", "<wrist_view_path>"]

    return VQASample(
        image_paths=image_paths,
        question=question,
        answer=answer,
        operation_type=operation_type,
        fruit_type=fruit_type,
        failure_mode=failure_mode,
    )


def compute_and_save_statistics(npz_dir: str) -> dict:
    """
    计算并保存数据集统计量（Requirement 1.7, 9.5）。

    调用 vla-scripts/npz_dataset.py 中的 compute_dataset_statistics()，
    将结果保存为 dataset_statistics.json。

    Args:
        npz_dir: 包含 episode_*.npz 文件的目录路径

    Returns:
        统计量字典，包含 action.mean/std/q01/q99、state.mean/std、
        num_transitions、num_episodes 字段
    """
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    vla_scripts_dir = repo_root / "vla-scripts"
    if str(vla_scripts_dir) not in sys.path:
        sys.path.insert(0, str(vla_scripts_dir))

    from npz_dataset import compute_dataset_statistics

    stats = compute_dataset_statistics(npz_dir)

    stats_path = Path(npz_dir) / "dataset_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


# ============ FruitDemoCollector 类 ============

class FruitDemoCollector:
    """
    遥操作演示数据采集器（Requirement 1.4, 1.5, 2.4）。

    复用 data/dataset.py 的 RealSense + RTDE 采集逻辑，
    增加水果任务专用的元数据记录和验证。

    Args:
        save_dir: 数据保存根目录。
        task_type: 任务类型，取值为 'atomic_ops' 或 'full_episodes'。
        fruit_type: 水果类型，取值为 FRUIT_TYPES 中的一个。

    Raises:
        ValueError: 当 task_type 或 fruit_type 不合法时抛出。
    """

    VALID_TASK_TYPES: List[str] = ["atomic_ops", "full_episodes"]

    def __init__(self, save_dir: str, task_type: str, fruit_type: str) -> None:
        # 验证 task_type
        if task_type not in self.VALID_TASK_TYPES:
            raise ValueError(
                f"不支持的 task_type: '{task_type}'。"
                f"有效值为: {self.VALID_TASK_TYPES}"
            )

        # 验证 fruit_type
        if fruit_type not in FRUIT_TYPES:
            raise ValueError(
                f"不支持的 fruit_type: '{fruit_type}'。"
                f"有效值为: {FRUIT_TYPES}"
            )

        self.task_type = task_type
        self.fruit_type = fruit_type

        # 保存目录：save_dir / task_type
        self.save_dir = Path(save_dir) / task_type
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 日志记录器
        self.logger = logging.getLogger(f"{__name__}.{task_type}.{fruit_type}")

        # Episode 计数器（从现有文件中推断起始值）
        self._episode_counter = self._get_next_episode_id()

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _get_next_episode_id(self) -> int:
        """从保存目录中已有的 npz 文件推断下一个 episode ID。"""
        existing = sorted(self.save_dir.glob("episode_*.npz"))
        if not existing:
            return 0
        max_id = -1
        for f in existing:
            try:
                idx = int(f.stem.split("_")[-1])
                max_id = max(max_id, idx)
            except ValueError:
                pass
        return max_id + 1

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def verify_temporal_consistency(self, episode_data: dict) -> bool:
        """
        验证 Episode 数据的时序维度一致性（Requirement 1.5, 9.1）。

        检查 images、images_wrist、tcp_poses、joint_positions、gripper
        的第一维（时间步数 T）是否完全相同。

        Args:
            episode_data: 包含各传感器数组的字典，键名与 npz 字段对应。

        Returns:
            True 表示所有时序数组的第一维相等；False 表示存在不一致。
        """
        T = episode_data["images"].shape[0]
        return all(
            episode_data[k].shape[0] == T
            for k in ["images_wrist", "tcp_poses", "joint_positions", "gripper"]
        )

    def save_episode(self, episode_data: dict, metadata: "EpisodeMetadata") -> Path:
        """
        保存 Episode 数据为 npz 文件，并写入元数据 JSON sidecar（Requirement 2.4）。

        先验证时序一致性，通过后保存：
          - episode_{episode_id:04d}.npz：压缩的 numpy 数组文件
          - episode_{episode_id:04d}_meta.json：元数据 sidecar 文件

        Args:
            episode_data: 包含 images、images_wrist、tcp_poses、
                          joint_positions、gripper 等数组的字典。
            metadata: EpisodeMetadata 实例，包含 episode_id 等元数据。

        Returns:
            保存的 npz 文件路径。

        Raises:
            ValueError: 当时序一致性验证失败时抛出。
        """
        import dataclasses

        # 1. 时序一致性验证（Requirement 1.5）
        if not self.verify_temporal_consistency(episode_data):
            msg = (
                f"Episode {metadata.episode_id} 时序维度不一致，已丢弃。"
                f" images.shape[0]={episode_data['images'].shape[0]}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        episode_id = metadata.episode_id

        # 2. 生成标准化指令字符串（Requirement 1.6）
        # 对于 atomic_ops，使用 build_instruction 生成指令；
        # 对于 full_episodes，使用通用指令
        try:
            if self.task_type in ATOMIC_TASK_TYPES:
                # place 操作随机选择左右篮子
                basket_side = random.choice(BASKET_SIDES)
                instruction = build_instruction(self.task_type, self.fruit_type, basket_side)
            else:
                instruction = build_instruction("pick_up", self.fruit_type)
        except ValueError:
            # full_episodes 模式下使用通用指令
            instruction = f"clear all {self.fruit_type} from the table"

        # 3. 保存 npz 文件（Requirement 1.4）
        npz_path = self.save_dir / f"episode_{episode_id:04d}.npz"
        np.savez_compressed(
            npz_path,
            images=episode_data["images"],
            images_wrist=episode_data["images_wrist"],
            tcp_poses=episode_data["tcp_poses"],
            joint_positions=episode_data["joint_positions"],
            gripper=episode_data["gripper"],
            instruction=instruction,
            fps=np.int64(30),
        )
        self.logger.info(f"已保存 npz: {npz_path}")

        # 4. 保存元数据 JSON sidecar（Requirement 2.4）
        meta_path = self.save_dir / f"episode_{episode_id:04d}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(metadata), f, indent=2, ensure_ascii=False)
        self.logger.info(f"已保存元数据: {meta_path}")

        return npz_path

    def collect_episode(self) -> Optional["EpisodeMetadata"]:
        """
        占位实现：实际采集需要硬件（RealSense + UR7e）。

        实际采集逻辑在 data/dataset.py 中实现。本方法仅记录日志并返回 None，
        供无硬件环境下的测试和流水线集成使用。

        Returns:
            None（占位实现，实际采集需要硬件）。
        """
        self.logger.info(
            "collect_episode() 为占位实现，实际采集需要 RealSense 摄像头和 UR7e 机器人。"
            " 实际采集逻辑请参考 data/dataset.py 中的 main() 函数。"
        )
        return None

    def compute_and_save_statistics(self) -> dict:
        """
        调用 compute_dataset_statistics()，将结果保存为 dataset_statistics.json。
        Requirement 1.7, 9.5

        确保输出包含 action.mean、action.std、action.q01、action.q99、
        state.mean、state.std、num_transitions、num_episodes 字段。

        Returns:
            统计量字典
        """
        import sys
        from pathlib import Path

        # 将 vla-scripts 目录加入 sys.path
        repo_root = Path(__file__).resolve().parent.parent
        vla_scripts_dir = repo_root / "vla-scripts"
        if str(vla_scripts_dir) not in sys.path:
            sys.path.insert(0, str(vla_scripts_dir))

        from npz_dataset import compute_dataset_statistics

        # 计算统计量
        stats = compute_dataset_statistics(str(self.save_dir))

        # 保存到 dataset_statistics.json
        stats_path = self.save_dir / "dataset_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"已保存数据集统计量: {stats_path}")
        return stats
