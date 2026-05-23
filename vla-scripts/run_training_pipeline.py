"""
run_training_pipeline.py — 训练流水线编排器

按顺序执行 5 个训练阶段：
  Stage 1: 数据采集（Data Collection）
  Stage 2: VLM Stage 1 微调（子任务完成验证）
  Stage 3: VLA LoRA 微调
  Stage 4: VLM Stage 2 微调（验证 + 扫描 + 终止判断）
  Stage 5: 端到端 PEV 系统评估

每个阶段前检查前置条件，不满足时抛出 PipelineError 并终止。
每个阶段完成后生成 StageReport，包含训练时长、最终 loss、验证指标和 checkpoint 路径。

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PipelineError(Exception):
    """前置条件检查失败时抛出。"""
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """训练流水线配置。"""

    # 数据根目录（包含 atomic_ops/、full_episodes/、vlm_data/ 子目录）
    data_root: str = "raw_demos"

    # VLA checkpoint 输出目录（由 train_fruit_vla.sh 写入）
    vla_checkpoint_dir: str = "runs/fruit_vla"

    # VLM Stage 1 checkpoint 输出目录
    vlm_stage1_checkpoint_dir: str = "checkpoints/vlm_stage1"

    # VLM Stage 2 checkpoint 输出目录
    vlm_stage2_checkpoint_dir: str = "checkpoints/vlm_stage2"

    # Stage 2 前置条件：VQA 样本数下限（Requirement 8.2）
    min_vqa_samples: int = 400

    # Stage 3 前置条件：Episode 数下限（Requirement 8.3）
    min_episodes: int = 300

    # 混合数据目录（Requirement 5.6）
    mixed_data_dir: str = "raw_demos/mixed"

    # atomic_ops 占比（3:1 = 75%）（Requirement 5.6）
    atomic_ops_ratio: float = 0.75


# ---------------------------------------------------------------------------
# Stage Report
# ---------------------------------------------------------------------------


@dataclass
class StageReport:
    """单个训练阶段的执行报告（Requirement 8.5）。"""

    stage_id: int
    training_duration_seconds: float
    final_loss: float
    validation_metrics: dict
    checkpoint_path: str
    success: bool
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TrainingPipeline:
    """
    训练流水线编排器（Requirement 8）。
    按顺序执行 5 个阶段，每个阶段前检查前置条件。
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_root = Path(config.data_root)
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Data mixing
    # ------------------------------------------------------------------

    def build_mixed_dataset(self) -> Path:
        """
        按 3:1 比例（atomic_ops : full_episodes）将 episode 文件
        符号链接到 raw_demos/mixed/ 目录，然后重新计算并保存 dataset_statistics.json。

        Requirement 5.6

        Returns:
            混合数据目录路径
        """
        import os
        import sys
        import json
        from pathlib import Path

        mixed_dir = Path(self.config.mixed_data_dir)
        mixed_dir.mkdir(parents=True, exist_ok=True)

        atomic_ops_dir = self.data_root / "atomic_ops"
        full_episodes_dir = self.data_root / "full_episodes"

        # 获取所有 npz 文件
        atomic_files = sorted(atomic_ops_dir.glob("episode_*.npz")) if atomic_ops_dir.exists() else []
        full_files = sorted(full_episodes_dir.glob("episode_*.npz")) if full_episodes_dir.exists() else []

        # 按 3:1 比例计算需要的文件数
        # 如果 atomic_files 有 N 个，则 full_episodes 取 N//3 个
        n_atomic = len(atomic_files)
        n_full = min(len(full_files), max(1, n_atomic // 3))

        # 清空混合目录中的旧符号链接
        for old_link in mixed_dir.glob("episode_*.npz"):
            old_link.unlink(missing_ok=True)

        # 创建符号链接（Windows 上使用 os.symlink 或直接复制）
        episode_id = 0
        for src in atomic_files:
            dst = mixed_dir / f"episode_{episode_id:04d}.npz"
            try:
                os.symlink(src.resolve(), dst)
            except (OSError, NotImplementedError):
                # Windows 可能需要管理员权限，改用硬链接或复制
                import shutil
                shutil.copy2(src, dst)
            episode_id += 1

        for src in full_files[:n_full]:
            dst = mixed_dir / f"episode_{episode_id:04d}.npz"
            try:
                os.symlink(src.resolve(), dst)
            except (OSError, NotImplementedError):
                import shutil
                shutil.copy2(src, dst)
            episode_id += 1

        # 重新计算并保存 dataset_statistics.json
        repo_root = Path(__file__).resolve().parent.parent
        vla_scripts_dir = repo_root / "vla-scripts"
        if str(vla_scripts_dir) not in sys.path:
            sys.path.insert(0, str(vla_scripts_dir))

        from npz_dataset import compute_dataset_statistics
        stats = compute_dataset_statistics(str(mixed_dir))
        stats_path = mixed_dir / "dataset_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        self.logger.info(
            "混合数据集构建完成：%d atomic + %d full = %d episodes，保存至 %s",
            n_atomic,
            n_full,
            episode_id,
            mixed_dir,
        )
        return mixed_dir

    # ------------------------------------------------------------------
    # Prerequisite checks
    # ------------------------------------------------------------------

    def check_stage2_prerequisites(self) -> None:
        """
        验证 Stage 2 前置条件（Requirement 8.2）：
        VQA 数据集存在且样本数 >= 400。

        检查路径：<data_root>/vlm_data/vqa_completion.jsonl
        统计方式：逐行计数（每行一个 JSON 样本）。

        Raises:
            PipelineError: 如果文件不存在或样本数不足。
        """
        vqa_path = self.data_root / "vlm_data" / "vqa_completion.jsonl"

        if not vqa_path.exists():
            raise PipelineError(
                f"Stage 2 前置条件不满足：VQA 数据集文件不存在。\n"
                f"  期望路径：{vqa_path}\n"
                f"  请先运行 Stage 1 数据采集（data/collect_fruit_demos.py）。"
            )

        # 统计非空行数（每行一个 JSON 样本）
        sample_count = 0
        with vqa_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample_count += 1

        if sample_count < self.config.min_vqa_samples:
            raise PipelineError(
                f"Stage 2 前置条件不满足：VQA 样本数不足。\n"
                f"  当前样本数：{sample_count}\n"
                f"  最低要求：{self.config.min_vqa_samples}\n"
                f"  缺少样本数：{self.config.min_vqa_samples - sample_count}\n"
                f"  请补充采集 VQA 数据（Requirement 3）。"
            )

        logger.info(
            "Stage 2 前置条件通过：VQA 样本数 %d >= %d",
            sample_count,
            self.config.min_vqa_samples,
        )

    def check_stage3_prerequisites(self) -> None:
        """
        验证 Stage 3 前置条件（Requirement 8.3）：
        npz 数据集存在且 Episode 数 >= 300。

        统计范围：
          - <data_root>/atomic_ops/*.npz
          - <data_root>/full_episodes/*.npz

        Raises:
            PipelineError: 如果两个目录均不存在，或 npz 文件总数不足。
        """
        atomic_ops_dir = self.data_root / "atomic_ops"
        full_episodes_dir = self.data_root / "full_episodes"

        # 至少一个目录需要存在
        if not atomic_ops_dir.exists() and not full_episodes_dir.exists():
            raise PipelineError(
                f"Stage 3 前置条件不满足：npz 数据目录不存在。\n"
                f"  期望目录（至少一个）：\n"
                f"    {atomic_ops_dir}\n"
                f"    {full_episodes_dir}\n"
                f"  请先运行 Stage 1 数据采集（data/collect_fruit_demos.py）。"
            )

        # 统计两个目录中的 npz 文件总数
        atomic_count = len(list(atomic_ops_dir.glob("*.npz"))) if atomic_ops_dir.exists() else 0
        full_count = len(list(full_episodes_dir.glob("*.npz"))) if full_episodes_dir.exists() else 0
        total_episodes = atomic_count + full_count

        if total_episodes < self.config.min_episodes:
            raise PipelineError(
                f"Stage 3 前置条件不满足：Episode 数不足。\n"
                f"  atomic_ops/ 中的 npz 文件数：{atomic_count}\n"
                f"  full_episodes/ 中的 npz 文件数：{full_count}\n"
                f"  合计：{total_episodes}\n"
                f"  最低要求：{self.config.min_episodes}\n"
                f"  缺少 Episode 数：{self.config.min_episodes - total_episodes}\n"
                f"  请补充采集演示数据（Requirements 1, 2）。"
            )

        logger.info(
            "Stage 3 前置条件通过：Episode 总数 %d (atomic=%d, full=%d) >= %d",
            total_episodes,
            atomic_count,
            full_count,
            self.config.min_episodes,
        )

    def check_stage5_prerequisites(self) -> None:
        """
        验证 Stage 5 前置条件（Requirement 8.4）：
        VLA checkpoint 和 VLM checkpoint 均存在。

        检查路径：
          - config.vla_checkpoint_dir
          - config.vlm_stage2_checkpoint_dir

        Raises:
            PipelineError: 如果任一 checkpoint 目录不存在。
        """
        vla_ckpt = Path(self.config.vla_checkpoint_dir)
        vlm_ckpt = Path(self.config.vlm_stage2_checkpoint_dir)

        missing: List[str] = []
        if not vla_ckpt.exists():
            missing.append(f"VLA checkpoint：{vla_ckpt}")
        if not vlm_ckpt.exists():
            missing.append(f"VLM Stage 2 checkpoint：{vlm_ckpt}")

        if missing:
            missing_str = "\n  ".join(missing)
            raise PipelineError(
                f"Stage 5 前置条件不满足：以下 checkpoint 不存在。\n"
                f"  {missing_str}\n"
                f"  请先完成 Stage 3（VLA 微调）和 Stage 4（VLM Stage 2 微调）。"
            )

        logger.info(
            "Stage 5 前置条件通过：VLA checkpoint=%s，VLM checkpoint=%s",
            vla_ckpt,
            vlm_ckpt,
        )

    # ------------------------------------------------------------------
    # Stage execution
    # ------------------------------------------------------------------

    def run_stage(self, stage_id: int) -> StageReport:
        """
        运行单个阶段，生成训练报告（Requirement 8.5）。

        Args:
            stage_id: 阶段编号（1–5）。

        Returns:
            StageReport，包含训练时长、最终 loss、验证指标和 checkpoint 路径。
        """
        stage_runners = {
            1: self._run_stage1,
            2: self._run_stage2,
            3: self._run_stage3,
            4: self._run_stage4,
            5: self._run_stage5,
        }

        if stage_id not in stage_runners:
            return StageReport(
                stage_id=stage_id,
                training_duration_seconds=0.0,
                final_loss=float("nan"),
                validation_metrics={},
                checkpoint_path="",
                success=False,
                error_message=f"未知阶段 ID：{stage_id}，有效范围为 1–5。",
            )

        start_time = time.time()
        try:
            report = stage_runners[stage_id]()
        except PipelineError as exc:
            duration = time.time() - start_time
            logger.error("Stage %d 前置条件检查失败：%s", stage_id, exc)
            return StageReport(
                stage_id=stage_id,
                training_duration_seconds=duration,
                final_loss=float("nan"),
                validation_metrics={},
                checkpoint_path="",
                success=False,
                error_message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - start_time
            logger.exception("Stage %d 执行时发生意外错误", stage_id)
            return StageReport(
                stage_id=stage_id,
                training_duration_seconds=duration,
                final_loss=float("nan"),
                validation_metrics={},
                checkpoint_path="",
                success=False,
                error_message=str(exc),
            )

        return report

    def run_all(self) -> List[StageReport]:
        """
        按顺序运行所有 5 个阶段（Requirement 8.1）。

        执行顺序：
          Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5

        如果某阶段失败（success=False），后续阶段仍会尝试执行，
        但前置条件检查会阻止依赖该阶段输出的后续阶段继续运行。

        Returns:
            包含 5 个 StageReport 的列表，按阶段顺序排列。
        """
        reports: List[StageReport] = []
        for stage_id in range(1, 6):
            logger.info("=" * 60)
            logger.info("开始执行 Stage %d", stage_id)
            report = self.run_stage(stage_id)
            reports.append(report)
            if report.success:
                logger.info(
                    "Stage %d 完成，耗时 %.1f 秒，最终 loss=%.4f",
                    stage_id,
                    report.training_duration_seconds,
                    report.final_loss,
                )
            else:
                logger.error(
                    "Stage %d 失败：%s",
                    stage_id,
                    report.error_message,
                )
                # 前置条件不满足时终止后续阶段
                if report.error_message and "前置条件不满足" in report.error_message:
                    logger.error("由于前置条件不满足，终止后续阶段执行。")
                    # 为剩余阶段填充失败报告
                    for remaining_id in range(stage_id + 1, 6):
                        reports.append(
                            StageReport(
                                stage_id=remaining_id,
                                training_duration_seconds=0.0,
                                final_loss=float("nan"),
                                validation_metrics={},
                                checkpoint_path="",
                                success=False,
                                error_message=f"Stage {stage_id} 失败，跳过此阶段。",
                            )
                        )
                    break

        return reports

    # ------------------------------------------------------------------
    # Internal stage implementations (stubs — to be completed in task 11.3)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Output parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_loss_from_output(output: str) -> float:
        """
        从脚本标准输出中提取最终 loss 值。

        支持以下格式（不区分大小写）：
          - "final loss: 0.0312"
          - "train/loss: 0.0312"
          - "loss=0.0312"
          - "loss: 0.0312"

        Returns:
            解析到的 loss 值；若无法解析则返回 float('nan')。
        """
        import re

        patterns = [
            r"final\s+loss[:\s=]+([0-9]+\.[0-9]+(?:e[+-]?[0-9]+)?)",
            r"train/loss[:\s=]+([0-9]+\.[0-9]+(?:e[+-]?[0-9]+)?)",
            r"loss[:\s=]+([0-9]+\.[0-9]+(?:e[+-]?[0-9]+)?)",
        ]
        for pat in patterns:
            m = re.search(pat, output, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        return float("nan")

    @staticmethod
    def _parse_metrics_from_output(output: str) -> dict:
        """
        从脚本标准输出中提取验证指标键值对。

        支持以下格式（不区分大小写）：
          - "val/action_l1_loss: 0.042"
          - "completion_accuracy: 0.91"
          - "termination_accuracy: 0.93"
          - "scan_count_accuracy: 0.85"

        Returns:
            包含解析到的指标的字典；若无法解析则返回空字典。
        """
        import re

        metrics: dict = {}
        pattern = r"([\w/]+)[:\s=]+([0-9]+\.[0-9]+(?:e[+-]?[0-9]+)?)"
        for m in re.finditer(pattern, output, re.IGNORECASE):
            key = m.group(1).lower()
            try:
                metrics[key] = float(m.group(2))
            except ValueError:
                pass
        return metrics

    @staticmethod
    def _find_latest_checkpoint(checkpoint_dir: str) -> str:
        """
        在 checkpoint_dir 下查找最新的 checkpoint 子目录。

        优先返回名为 "final" 的子目录；否则返回编号最大的
        "checkpoint-{N}" 子目录；若均不存在则返回 checkpoint_dir 本身。
        """
        ckpt_root = Path(checkpoint_dir)
        if not ckpt_root.exists():
            return checkpoint_dir

        # 优先返回 "final"
        final_dir = ckpt_root / "final"
        if final_dir.exists():
            return str(final_dir)

        # 查找 checkpoint-{N} 子目录
        import re

        numbered: List[tuple] = []
        for d in ckpt_root.iterdir():
            if d.is_dir():
                m = re.match(r"checkpoint-(\d+)$", d.name)
                if m:
                    numbered.append((int(m.group(1)), d))

        if numbered:
            numbered.sort(key=lambda x: x[0], reverse=True)
            return str(numbered[0][1])

        return checkpoint_dir

    def _run_subprocess(
        self,
        cmd: List[str],
        stage_id: int,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """
        运行子进程并捕获输出。

        Args:
            cmd: 命令及参数列表。
            stage_id: 阶段编号（用于日志）。
            timeout: 超时秒数（None 表示不限时）。

        Returns:
            subprocess.CompletedProcess 对象。

        Raises:
            PipelineError: 子进程返回非零退出码时抛出。
        """
        logger.info("Stage %d 执行命令：%s", stage_id, " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.stdout:
            logger.debug("Stage %d stdout:\n%s", stage_id, result.stdout[-4000:])
        if result.stderr:
            logger.debug("Stage %d stderr:\n%s", stage_id, result.stderr[-4000:])
        if result.returncode != 0:
            raise PipelineError(
                f"Stage {stage_id} 脚本退出码非零（{result.returncode}）。\n"
                f"stderr（最后 2000 字符）：\n{result.stderr[-2000:]}"
            )
        return result

    # ------------------------------------------------------------------
    # Internal stage implementations
    # ------------------------------------------------------------------

    def _run_stage1(self) -> StageReport:
        """
        Stage 1：数据采集。

        调用 data/collect_fruit_demos.py 脚本（若存在），
        完成后以 data_root 作为 checkpoint_path 返回报告。
        数据采集阶段无训练 loss，final_loss 固定为 0.0。
        """
        start = time.time()
        logger.info("Stage 1：启动数据采集脚本 data/collect_fruit_demos.py")

        # 定位脚本路径（相对于本文件所在目录的上级）
        script_path = Path(__file__).resolve().parent.parent / "data" / "collect_fruit_demos.py"

        combined_output = ""
        if script_path.exists():
            try:
                result = self._run_subprocess(
                    ["python", str(script_path), "--data_root", str(self.data_root)],
                    stage_id=1,
                )
                combined_output = result.stdout + result.stderr
            except PipelineError:
                raise
        else:
            logger.warning(
                "Stage 1：脚本 %s 不存在，跳过实际采集（仅记录报告）。",
                script_path,
            )

        duration = time.time() - start
        metrics = self._parse_metrics_from_output(combined_output)

        return StageReport(
            stage_id=1,
            training_duration_seconds=duration,
            final_loss=0.0,
            validation_metrics=metrics,
            checkpoint_path=str(self.data_root),
            success=True,
        )

    def _run_stage2(self) -> StageReport:
        """
        Stage 2：VLM Stage 1 微调（子任务完成验证）。

        前置条件：VQA 数据集存在且样本数 >= 400（Requirement 8.2）。
        调用 finetune_qwenvl.py --stage 1，解析输出中的 loss 和验证指标。
        checkpoint_path 指向 vlm_stage1_checkpoint_dir 下的最新 checkpoint。
        """
        self.check_stage2_prerequisites()
        start = time.time()
        logger.info("Stage 2：启动 VLM Stage 1 微调（finetune_qwenvl.py --stage 1）")

        script_path = Path(__file__).resolve().parent / "finetune_qwenvl.py"

        # 推断训练/验证数据路径
        vlm_data_dir = self.data_root / "vlm_data"
        train_data = str(vlm_data_dir / "stage1_train.jsonl")
        val_data = str(vlm_data_dir / "stage1_val.jsonl")

        cmd = [
            "python", str(script_path),
            "--stage", "1",
            "--train_data_path", train_data,
            "--val_data_path", val_data,
            "--run_root_dir", self.config.vlm_stage1_checkpoint_dir,
        ]

        result = self._run_subprocess(cmd, stage_id=2)
        combined_output = result.stdout + result.stderr
        duration = time.time() - start

        final_loss = self._parse_loss_from_output(combined_output)
        metrics = self._parse_metrics_from_output(combined_output)
        checkpoint_path = self._find_latest_checkpoint(self.config.vlm_stage1_checkpoint_dir)

        return StageReport(
            stage_id=2,
            training_duration_seconds=duration,
            final_loss=final_loss,
            validation_metrics=metrics,
            checkpoint_path=checkpoint_path,
            success=True,
        )

    def _run_stage3(self) -> StageReport:
        """
        Stage 3：VLA LoRA 微调。

        前置条件：npz 数据集存在且 Episode 数 >= 300（Requirement 8.3）。
        调用 train_fruit_vla.sh，解析输出中的 loss 和验证指标。
        checkpoint_path 指向 vla_checkpoint_dir 下的最新 checkpoint。
        """
        self.check_stage3_prerequisites()
        start = time.time()
        logger.info("Stage 3：启动 VLA LoRA 微调脚本 train_fruit_vla.sh")

        script_path = Path(__file__).resolve().parent / "train_fruit_vla.sh"

        # 混合数据目录（atomic_ops + full_episodes 按 3:1 混合）
        mixed_dir = str(self.data_root / "mixed")

        cmd = ["bash", str(script_path), mixed_dir]

        result = self._run_subprocess(cmd, stage_id=3)
        combined_output = result.stdout + result.stderr
        duration = time.time() - start

        final_loss = self._parse_loss_from_output(combined_output)
        metrics = self._parse_metrics_from_output(combined_output)
        checkpoint_path = self._find_latest_checkpoint(self.config.vla_checkpoint_dir)

        return StageReport(
            stage_id=3,
            training_duration_seconds=duration,
            final_loss=final_loss,
            validation_metrics=metrics,
            checkpoint_path=checkpoint_path,
            success=True,
        )

    def _run_stage4(self) -> StageReport:
        """
        Stage 4：VLM Stage 2 微调（验证 + 扫描 + 终止判断）。

        调用 finetune_qwenvl.py --stage 2，解析输出中的 loss 和验证指标。
        checkpoint_path 指向 vlm_stage2_checkpoint_dir 下的最新 checkpoint。
        """
        start = time.time()
        logger.info("Stage 4：启动 VLM Stage 2 微调（finetune_qwenvl.py --stage 2）")

        script_path = Path(__file__).resolve().parent / "finetune_qwenvl.py"

        vlm_data_dir = self.data_root / "vlm_data"
        train_data = str(vlm_data_dir / "stage2_train.jsonl")
        val_data = str(vlm_data_dir / "stage2_val.jsonl")

        cmd = [
            "python", str(script_path),
            "--stage", "2",
            "--train_data_path", train_data,
            "--val_data_path", val_data,
            "--run_root_dir", self.config.vlm_stage2_checkpoint_dir,
        ]

        result = self._run_subprocess(cmd, stage_id=4)
        combined_output = result.stdout + result.stderr
        duration = time.time() - start

        final_loss = self._parse_loss_from_output(combined_output)
        metrics = self._parse_metrics_from_output(combined_output)
        checkpoint_path = self._find_latest_checkpoint(self.config.vlm_stage2_checkpoint_dir)

        return StageReport(
            stage_id=4,
            training_duration_seconds=duration,
            final_loss=final_loss,
            validation_metrics=metrics,
            checkpoint_path=checkpoint_path,
            success=True,
        )

    def _run_stage5(self) -> StageReport:
        """
        Stage 5：端到端 PEV 系统评估。

        前置条件：VLA 和 VLM checkpoint 均存在（Requirement 8.4）。
        调用 evaluate_vlm.py，解析输出中的评估指标。
        checkpoint_path 为空（评估阶段不产生新 checkpoint）。
        """
        self.check_stage5_prerequisites()
        start = time.time()
        logger.info("Stage 5：启动端到端评估脚本 evaluate_vlm.py")

        script_path = Path(__file__).resolve().parent / "evaluate_vlm.py"

        vlm_ckpt = self._find_latest_checkpoint(self.config.vlm_stage2_checkpoint_dir)
        vlm_data_dir = self.data_root / "vlm_data"
        val_data = str(vlm_data_dir / "stage2_val.jsonl")

        cmd = [
            "python", str(script_path),
            "--model_id", vlm_ckpt,
            "--val_data_path", val_data,
        ]

        result = self._run_subprocess(cmd, stage_id=5)
        combined_output = result.stdout + result.stderr
        duration = time.time() - start

        metrics = self._parse_metrics_from_output(combined_output)

        return StageReport(
            stage_id=5,
            training_duration_seconds=duration,
            final_loss=float("nan"),
            validation_metrics=metrics,
            checkpoint_path="",
            success=True,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """命令行入口：运行完整训练流水线并保存报告。"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Multi-Fruit Table Clearing 训练流水线编排器"
    )
    parser.add_argument(
        "--data_root",
        default="raw_demos",
        help="数据根目录（默认：raw_demos）",
    )
    parser.add_argument(
        "--vla_checkpoint_dir",
        default="runs/fruit_vla",
        help="VLA checkpoint 输出目录",
    )
    parser.add_argument(
        "--vlm_stage1_checkpoint_dir",
        default="checkpoints/vlm_stage1",
        help="VLM Stage 1 checkpoint 输出目录",
    )
    parser.add_argument(
        "--vlm_stage2_checkpoint_dir",
        default="checkpoints/vlm_stage2",
        help="VLM Stage 2 checkpoint 输出目录",
    )
    parser.add_argument(
        "--min_vqa_samples",
        type=int,
        default=400,
        help="Stage 2 前置条件：最低 VQA 样本数（默认：400）",
    )
    parser.add_argument(
        "--min_episodes",
        type=int,
        default=300,
        help="Stage 3 前置条件：最低 Episode 数（默认：300）",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="仅运行指定阶段（1–5）；不指定则运行全部阶段",
    )
    parser.add_argument(
        "--report_path",
        default="training_report.json",
        help="训练报告输出路径（默认：training_report.json）",
    )
    args = parser.parse_args()

    config = PipelineConfig(
        data_root=args.data_root,
        vla_checkpoint_dir=args.vla_checkpoint_dir,
        vlm_stage1_checkpoint_dir=args.vlm_stage1_checkpoint_dir,
        vlm_stage2_checkpoint_dir=args.vlm_stage2_checkpoint_dir,
        min_vqa_samples=args.min_vqa_samples,
        min_episodes=args.min_episodes,
    )
    pipeline = TrainingPipeline(config)

    if args.stage is not None:
        reports = [pipeline.run_stage(args.stage)]
    else:
        reports = pipeline.run_all()

    # 序列化报告
    report_data = []
    for r in reports:
        report_data.append(
            {
                "stage_id": r.stage_id,
                "training_duration_seconds": r.training_duration_seconds,
                "final_loss": r.final_loss if r.final_loss == r.final_loss else None,  # NaN → null
                "validation_metrics": r.validation_metrics,
                "checkpoint_path": r.checkpoint_path,
                "success": r.success,
                "error_message": r.error_message,
            }
        )

    report_path = Path(args.report_path)
    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("训练报告已保存至：%s", report_path)

    # 打印摘要
    all_success = all(r.success for r in reports)
    print("\n" + "=" * 60)
    print("训练流水线执行摘要")
    print("=" * 60)
    for r in reports:
        status = "✓ 成功" if r.success else "✗ 失败"
        print(f"  Stage {r.stage_id}: {status}  ({r.training_duration_seconds:.1f}s)")
        if not r.success and r.error_message:
            # 只打印第一行错误信息
            first_line = r.error_message.splitlines()[0]
            print(f"           {first_line}")
    print("=" * 60)
    print(f"整体结果：{'全部成功' if all_success else '存在失败阶段'}")


if __name__ == "__main__":
    main()
