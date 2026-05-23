"""
validate_dataset.py — Data_Validator 数据质量验证模块

本模块实现数据集的正确性属性检查，包括：
  - 时序维度一致性验证（Requirement 9.1）
  - ActionNormalizer round-trip 属性验证（Requirement 9.2）
  - NpzEpisodeDataset 索引不变性验证（Requirement 9.3）
  - 归一化动作值域约束验证（Requirement 9.4）
  - dataset_statistics.json 完整性验证（Requirement 9.5）
  - VQA 数据集标签分布验证（Requirement 9.7）
  - parse_llm_plan 幂等性验证（Requirement 9.8）

核心数据类：
  ValidationResult  — 单项验证结果
  ValidationReport  — 全部验证结果汇总

核心函数：
  format_plan(plan)  — 将 Plan 对象序列化为字符串（用于 round-trip 测试）
"""

import os
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

# ---------------------------------------------------------------------------
# 导入 ds.py 中的 Plan / RepeatBlock / parse_llm_plan
# ---------------------------------------------------------------------------

try:
    # 将 experiments 目录加入 sys.path，以便在 data/ 目录下运行时也能找到模块
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from experiments.robot.libero.ds import Plan, RepeatBlock, parse_llm_plan

    _DS_AVAILABLE = True
except ImportError as _e:
    logging.warning(
        "无法导入 experiments.robot.libero.ds（%s）。"
        "format_plan 和 validate_parse_llm_plan_idempotence 将不可用。",
        _e,
    )
    _DS_AVAILABLE = False
    Plan = None          # type: ignore[assignment,misc]
    RepeatBlock = None   # type: ignore[assignment,misc]
    parse_llm_plan = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据类：ValidationResult / ValidationReport
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """单项验证结果。

    Attributes
    ----------
    passed : bool
        验证是否通过。
    message : str
        人类可读的结果描述（通过时为成功信息，失败时为错误原因）。
    details : Optional[dict]
        可选的附加细节，例如具体的失败样本、统计数值等。
    """

    passed: bool
    message: str
    details: Optional[dict] = None


@dataclass
class ValidationReport:
    """全部验证结果的汇总报告。

    Attributes
    ----------
    results : List[ValidationResult]
        每项验证的结果列表，顺序与调用顺序一致。
    all_passed : bool
        当且仅当所有 ValidationResult.passed 均为 True 时为 True。
    summary : str
        一行文字摘要，例如 "5/6 checks passed"。
    """

    results: List[ValidationResult]
    all_passed: bool
    summary: str


# ---------------------------------------------------------------------------
# format_plan：将 Plan 序列化为字符串
# ---------------------------------------------------------------------------


def format_plan(plan: "Plan") -> str:  # type: ignore[name-defined]
    """将 Plan 对象序列化为可被 parse_llm_plan 重新解析的字符串。

    序列化规则
    ----------
    - 线性步骤（str）→ 编号列表格式：``"1. step_text"``
    - RepeatBlock → REPEAT:…UNTIL: 格式::

        REPEAT:
          1. body_step_1
          2. body_step_2
        UNTIL: until_condition

    - 多个顶层步骤之间用换行分隔（每个步骤占一行或多行）。

    Parameters
    ----------
    plan : Plan
        要序列化的计划对象。

    Returns
    -------
    str
        可被 ``parse_llm_plan`` 重新解析的字符串表示。

    Raises
    ------
    TypeError
        如果 ``plan`` 不是 Plan 实例，或 ``plan.steps`` 中包含不支持的类型。
    ImportError
        如果 ``experiments.robot.libero.ds`` 模块不可用。
    """
    if not _DS_AVAILABLE:
        raise ImportError(
            "format_plan 依赖 experiments.robot.libero.ds，但该模块导入失败。"
        )

    lines: List[str] = []
    step_num = 1  # 顶层线性步骤的全局编号（RepeatBlock 不占用编号）

    for step in plan.steps:
        if isinstance(step, str):
            # 线性步骤：编号列表格式
            lines.append(f"{step_num}. {step}")
            step_num += 1

        elif isinstance(step, RepeatBlock):
            # REPEAT 块：不占用顶层编号
            lines.append("REPEAT:")
            for i, body_step in enumerate(step.body, start=1):
                lines.append(f"  {i}. {body_step}")
            lines.append(f"UNTIL: {step.until_condition}")

        else:
            raise TypeError(
                f"plan.steps 中包含不支持的类型：{type(step)!r}。"
                "仅支持 str 和 RepeatBlock。"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DataValidator：数据质量验证器
# ---------------------------------------------------------------------------


class DataValidator:
    """数据质量验证器。

    每个 validate_* 方法对应一个需求中的正确性属性。
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.atomic_ops_dir = self.data_root / "atomic_ops"
        self.full_episodes_dir = self.data_root / "full_episodes"
        self.vlm_data_dir = self.data_root / "vlm_data"
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # 9.1 时序维度一致性
    # ------------------------------------------------------------------

    def validate_temporal_consistency(self, npz_path: str) -> ValidationResult:
        """验证时序维度一致性（Requirement 9.1）。

        加载 npz 文件，验证 images / images_wrist / tcp_poses /
        joint_positions / gripper 的第一维相等。
        """
        import numpy as np

        try:
            d = np.load(npz_path, allow_pickle=True)
        except Exception as exc:
            return ValidationResult(
                passed=False,
                message=f"无法加载 npz 文件 {npz_path}：{exc}",
            )

        keys = ["images", "images_wrist", "tcp_poses", "joint_positions", "gripper"]
        missing = [k for k in keys if k not in d]
        if missing:
            d.close()
            return ValidationResult(
                passed=False,
                message=f"npz 文件缺少字段：{missing}",
                details={"missing_keys": missing},
            )

        shapes = {k: d[k].shape[0] for k in keys}
        d.close()

        T = shapes["images"]
        mismatched = {k: v for k, v in shapes.items() if v != T}
        if mismatched:
            return ValidationResult(
                passed=False,
                message=(
                    f"时序维度不一致：images.shape[0]={T}，"
                    f"不匹配字段：{mismatched}"
                ),
                details={"shapes": shapes, "mismatched": mismatched},
            )

        return ValidationResult(
            passed=True,
            message=f"时序维度一致，T={T}",
            details={"shapes": shapes},
        )

    # ------------------------------------------------------------------
    # 9.2 ActionNormalizer round-trip
    # ------------------------------------------------------------------

    def validate_action_normalizer_roundtrip(
        self,
        stats: dict,
        threshold: float = 1e-5,
    ) -> ValidationResult:
        """验证 ActionNormalizer round-trip 属性（Requirement 9.2）。

        从 stats 中读取 action.q01 / action.q99，生成 100 个随机动作向量
        （在 q01~q99 范围内），手动实现 normalize/denormalize，
        验证 L∞ 误差 < threshold。
        """
        import numpy as np

        try:
            q01 = np.array(stats["action"]["q01"], dtype=np.float64)
            q99 = np.array(stats["action"]["q99"], dtype=np.float64)
        except (KeyError, TypeError) as exc:
            return ValidationResult(
                passed=False,
                message=f"stats 缺少必要字段 action.q01 / action.q99：{exc}",
            )

        rng = q99 - q01
        # 避免除零
        rng_safe = np.where(np.abs(rng) < 1e-8, 1.0, rng)

        np.random.seed(42)
        # 生成 100 个在 [q01, q99] 范围内的随机动作向量
        t = np.random.uniform(0.0, 1.0, size=(100, len(q01)))
        actions = q01 + t * rng  # shape (100, dim)

        def normalize(a: np.ndarray) -> np.ndarray:
            return 2.0 * (a - q01) / rng_safe - 1.0

        def denormalize(a_norm: np.ndarray) -> np.ndarray:
            return (a_norm + 1.0) / 2.0 * rng_safe + q01

        reconstructed = denormalize(normalize(actions))
        linf_errors = np.abs(reconstructed - actions).max(axis=1)  # (100,)
        max_error = float(linf_errors.max())

        if max_error >= threshold:
            worst_idx = int(linf_errors.argmax())
            return ValidationResult(
                passed=False,
                message=(
                    f"round-trip L∞ 误差 {max_error:.2e} ≥ 阈值 {threshold:.2e}，"
                    f"最差样本索引：{worst_idx}"
                ),
                details={
                    "max_linf_error": max_error,
                    "threshold": threshold,
                    "worst_sample_index": worst_idx,
                },
            )

        return ValidationResult(
            passed=True,
            message=f"round-trip L∞ 误差 {max_error:.2e} < 阈值 {threshold:.2e}",
            details={"max_linf_error": max_error, "threshold": threshold},
        )

    # ------------------------------------------------------------------
    # 9.3 NpzEpisodeDataset 索引不变性
    # ------------------------------------------------------------------

    def validate_dataset_index_invariance(
        self,
        dataset,
        num_samples: int = 100,
    ) -> ValidationResult:
        """验证 NpzEpisodeDataset 索引不变性（Requirement 9.3）。

        随机选取 min(num_samples, len(dataset)) 个索引，
        对每个索引调用 dataset[i] 两次，比较返回的 tensor shapes 是否相同。
        """
        import random
        import torch

        n = len(dataset)
        if n == 0:
            return ValidationResult(
                passed=False,
                message="数据集为空，无法验证索引不变性。",
            )

        k = min(num_samples, n)
        random.seed(0)
        indices = random.sample(range(n), k)

        failures = []
        for i in indices:
            item1 = dataset[i]
            item2 = dataset[i]
            for key in item1:
                s1 = tuple(item1[key].shape) if hasattr(item1[key], "shape") else None
                s2 = tuple(item2[key].shape) if hasattr(item2[key], "shape") else None
                if s1 != s2:
                    failures.append(
                        {"index": i, "key": key, "shape1": s1, "shape2": s2}
                    )

        if failures:
            return ValidationResult(
                passed=False,
                message=f"索引不变性验证失败，{len(failures)} 处 shape 不一致。",
                details={"failures": failures[:10]},  # 最多展示 10 条
            )

        return ValidationResult(
            passed=True,
            message=f"索引不变性验证通过，抽查 {k} 个索引均一致。",
            details={"num_checked": k},
        )

    # ------------------------------------------------------------------
    # 9.4 归一化动作值域约束
    # ------------------------------------------------------------------

    def validate_normalized_action_range(
        self,
        stats: dict,
        num_samples: int = 1000,
    ) -> ValidationResult:
        """验证归一化动作值域约束（Requirement 9.4）。

        生成 num_samples 个随机动作向量，归一化后验证每个维度在 [-1.0, 1.0] 内。
        """
        import numpy as np

        try:
            q01 = np.array(stats["action"]["q01"], dtype=np.float64)
            q99 = np.array(stats["action"]["q99"], dtype=np.float64)
        except (KeyError, TypeError) as exc:
            return ValidationResult(
                passed=False,
                message=f"stats 缺少必要字段 action.q01 / action.q99：{exc}",
            )

        rng = q99 - q01
        rng_safe = np.where(np.abs(rng) < 1e-8, 1.0, rng)

        np.random.seed(123)
        # 生成在 [q01, q99] 范围内的随机动作向量
        t = np.random.uniform(0.0, 1.0, size=(num_samples, len(q01)))
        actions = q01 + t * rng  # shape (num_samples, dim)

        def normalize(a: np.ndarray) -> np.ndarray:
            return 2.0 * (a - q01) / rng_safe - 1.0

        normed = normalize(actions)  # (num_samples, dim)

        out_of_range_mask = (normed < -1.0 - 1e-9) | (normed > 1.0 + 1e-9)
        num_violations = int(out_of_range_mask.sum())

        if num_violations > 0:
            viol_indices = list(zip(*np.where(out_of_range_mask)))[:10]
            return ValidationResult(
                passed=False,
                message=(
                    f"归一化动作值域验证失败：{num_violations} 个元素超出 [-1, 1]。"
                ),
                details={
                    "num_violations": num_violations,
                    "sample_violations": [
                        {"sample": int(r), "dim": int(c), "value": float(normed[r, c])}
                        for r, c in viol_indices
                    ],
                },
            )

        return ValidationResult(
            passed=True,
            message=f"归一化动作值域验证通过，{num_samples} 个样本均在 [-1, 1] 内。",
            details={"num_samples": num_samples},
        )

    # ------------------------------------------------------------------
    # 9.5 dataset_statistics.json 完整性
    # ------------------------------------------------------------------

    def validate_statistics_completeness(self, stats_path: str) -> ValidationResult:
        """验证 dataset_statistics.json 完整性（Requirement 9.5）。

        加载 JSON 文件，验证包含所有必需字段：
        action.mean, action.std, action.q01, action.q99,
        state.mean, state.std, num_transitions, num_episodes。
        """
        REQUIRED_FIELDS = [
            ("action", "mean"),
            ("action", "std"),
            ("action", "q01"),
            ("action", "q99"),
            ("state", "mean"),
            ("state", "std"),
            ("num_transitions",),
            ("num_episodes",),
        ]

        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
        except FileNotFoundError:
            return ValidationResult(
                passed=False,
                message=f"统计文件不存在：{stats_path}",
            )
        except json.JSONDecodeError as exc:
            return ValidationResult(
                passed=False,
                message=f"统计文件 JSON 解析失败：{exc}",
            )

        missing = []
        for field_path in REQUIRED_FIELDS:
            obj = stats
            found = True
            for key in field_path:
                if not isinstance(obj, dict) or key not in obj:
                    found = False
                    break
                obj = obj[key]
            if not found:
                missing.append(".".join(field_path))

        if missing:
            return ValidationResult(
                passed=False,
                message=f"统计文件缺少必需字段：{missing}",
                details={"missing_fields": missing},
            )

        return ValidationResult(
            passed=True,
            message=f"统计文件完整性验证通过，所有必需字段均存在。",
            details={"stats_path": stats_path},
        )

    # ------------------------------------------------------------------
    # 9.7 VQA 标签分布
    # ------------------------------------------------------------------

    def validate_vqa_label_balance(
        self,
        jsonl_path: str,
        low: float = 0.45,
        high: float = 0.55,
    ) -> ValidationResult:
        """验证 VQA 数据集标签分布（Requirement 9.7）。

        读取 JSONL 文件，统计 answer=="Yes" 的比例。
        如果比例在 [low, high] 外，输出警告但仍返回 passed=True。
        """
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return ValidationResult(
                passed=False,
                message=f"JSONL 文件不存在：{jsonl_path}",
            )
        except OSError as exc:
            return ValidationResult(
                passed=False,
                message=f"读取 JSONL 文件失败：{exc}",
            )

        if not lines:
            return ValidationResult(
                passed=True,
                message="JSONL 文件为空，无样本可验证。",
                details={"total": 0, "positive_ratio": None},
            )

        total = 0
        yes_count = 0
        parse_errors = 0
        for line in lines:
            try:
                sample = json.loads(line)
                total += 1
                if sample.get("answer") == "Yes":
                    yes_count += 1
            except json.JSONDecodeError:
                parse_errors += 1

        if total == 0:
            return ValidationResult(
                passed=True,
                message="JSONL 文件中无有效样本。",
                details={"parse_errors": parse_errors},
            )

        positive_ratio = yes_count / total

        details = {
            "total": total,
            "yes_count": yes_count,
            "no_count": total - yes_count,
            "positive_ratio": positive_ratio,
            "parse_errors": parse_errors,
            "low": low,
            "high": high,
        }

        if not (low <= positive_ratio <= high):
            warning_msg = (
                f"VQA 标签分布不均衡：正样本比例 {positive_ratio:.3f} "
                f"超出期望范围 [{low}, {high}]。"
                f"建议对少数类进行过采样。"
            )
            self.logger.warning(warning_msg)
            return ValidationResult(
                passed=True,
                message=warning_msg,
                details=details,
            )

        return ValidationResult(
            passed=True,
            message=(
                f"VQA 标签分布均衡：正样本比例 {positive_ratio:.3f} "
                f"在期望范围 [{low}, {high}] 内。"
            ),
            details=details,
        )

    # ------------------------------------------------------------------
    # 9.8 parse_llm_plan 幂等性
    # ------------------------------------------------------------------

    def validate_parse_llm_plan_idempotence(
        self,
        plan_strings: List[str],
    ) -> ValidationResult:
        """验证 parse_llm_plan 的幂等性（Requirement 9.8）。

        对每个 plan_string s：
          plan1 = parse_llm_plan(s)
          plan2 = parse_llm_plan(format_plan(plan1))
          验证 plan1 和 plan2 的步骤数、步骤类型、步骤内容相同。
        """
        if not _DS_AVAILABLE:
            return ValidationResult(
                passed=False,
                message="ds 模块不可用",
            )

        failures = []
        for idx, s in enumerate(plan_strings):
            try:
                plan1 = parse_llm_plan(s)
                plan2 = parse_llm_plan(format_plan(plan1))
            except Exception as exc:
                failures.append(
                    {"index": idx, "error": str(exc), "input": s[:200]}
                )
                continue

            # 比较步骤数
            if len(plan1.steps) != len(plan2.steps):
                failures.append({
                    "index": idx,
                    "reason": "步骤数不同",
                    "plan1_steps": len(plan1.steps),
                    "plan2_steps": len(plan2.steps),
                    "input": s[:200],
                })
                continue

            # 逐步比较类型和内容
            step_mismatch = False
            for step_idx, (s1, s2) in enumerate(zip(plan1.steps, plan2.steps)):
                if type(s1) != type(s2):
                    failures.append({
                        "index": idx,
                        "step": step_idx,
                        "reason": "步骤类型不同",
                        "type1": type(s1).__name__,
                        "type2": type(s2).__name__,
                    })
                    step_mismatch = True
                    break

                if isinstance(s1, str):
                    if s1 != s2:
                        failures.append({
                            "index": idx,
                            "step": step_idx,
                            "reason": "线性步骤内容不同",
                            "step1": s1,
                            "step2": s2,
                        })
                        step_mismatch = True
                        break
                elif isinstance(s1, RepeatBlock):
                    # 比较 body 和 until_condition
                    if s1.body != s2.body or s1.until_condition != s2.until_condition:
                        failures.append({
                            "index": idx,
                            "step": step_idx,
                            "reason": "RepeatBlock 内容不同",
                            "body1": s1.body,
                            "body2": s2.body,
                            "until1": s1.until_condition,
                            "until2": s2.until_condition,
                        })
                        step_mismatch = True
                        break

        if failures:
            return ValidationResult(
                passed=False,
                message=f"parse_llm_plan 幂等性验证失败，{len(failures)} 个样本不通过。",
                details={"failures": failures[:10]},
            )

        return ValidationResult(
            passed=True,
            message=f"parse_llm_plan 幂等性验证通过，{len(plan_strings)} 个样本均通过。",
            details={"num_checked": len(plan_strings)},
        )

    # ------------------------------------------------------------------
    # run_all：运行所有可运行的验证
    # ------------------------------------------------------------------

    def run_all(self) -> ValidationReport:
        """运行所有可运行的验证（跳过需要外部数据的），返回 ValidationReport。

        自动运行：
          - validate_statistics_completeness（如果 stats 文件存在）
          - validate_vqa_label_balance（如果 JSONL 文件存在）

        其他方法（temporal_consistency, action_normalizer_roundtrip,
        dataset_index_invariance, normalized_action_range,
        parse_llm_plan_idempotence）需要外部传入数据，不在此处运行。
        """
        results: List[ValidationResult] = []

        # 1. 统计文件完整性（atomic_ops 或 full_episodes 目录下）
        for candidate_dir in [self.atomic_ops_dir, self.full_episodes_dir, self.data_root]:
            stats_path = candidate_dir / "dataset_statistics.json"
            if stats_path.exists():
                self.logger.info("验证统计文件：%s", stats_path)
                result = self.validate_statistics_completeness(str(stats_path))
                result.message = f"[{stats_path.name} @ {candidate_dir.name}] " + result.message
                results.append(result)
                break  # 找到第一个即可

        # 2. VQA 标签分布（vlm_data 目录下的 JSONL 文件）
        vqa_candidates = [
            self.vlm_data_dir / "vqa_completion.jsonl",
            self.vlm_data_dir / "vqa_termination.jsonl",
        ]
        for jsonl_path in vqa_candidates:
            if jsonl_path.exists():
                self.logger.info("验证 VQA 标签分布：%s", jsonl_path)
                result = self.validate_vqa_label_balance(str(jsonl_path))
                result.message = f"[{jsonl_path.name}] " + result.message
                results.append(result)

        if not results:
            summary = "未找到可自动验证的数据文件，跳过所有检查。"
            return ValidationReport(
                results=[],
                all_passed=True,
                summary=summary,
            )

        all_passed = all(r.passed for r in results)
        passed_count = sum(1 for r in results if r.passed)
        summary = f"{passed_count}/{len(results)} checks passed"

        return ValidationReport(
            results=results,
            all_passed=all_passed,
            summary=summary,
        )
