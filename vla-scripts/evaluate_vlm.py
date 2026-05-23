"""
evaluate_vlm.py — VLM 评估模块

在验证集上评估 Qwen2.5-VL 微调模型的各项指标，包括：
  - check_completion_with_qwen_vl 的 Precision / Recall / F1 / Accuracy（Requirement 9.6）
  - check_termination_with_qwen_vl 的 Accuracy（Requirement 6.5）
  - scan_targets_with_qwen_vl 的水果数量准确率（Requirement 6.6）

用法示例：
    evaluator = VLMEvaluator("Qwen/Qwen2.5-VL-7B-Instruct")
    metrics = evaluator.evaluate_completion(val_samples)
    print(metrics)
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Evaluation metric dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CompletionMetrics:
    """check_completion_with_qwen_vl 的评估指标（Requirement 9.6）

    Attributes
    ----------
    precision : float
        TP / (TP + FP)。要求 >= 0.80。
    recall : float
        TP / (TP + FN)。要求 >= 0.80。
    f1 : float
        2 * precision * recall / (precision + recall)。
    accuracy : float
        (TP + TN) / (TP + FP + TN + FN)。
    confusion_matrix : dict
        {"TP": int, "FP": int, "TN": int, "FN": int}
    """

    precision: float
    recall: float
    f1: float
    accuracy: float
    confusion_matrix: dict  # {"TP": int, "FP": int, "TN": int, "FN": int}


@dataclass
class TerminationMetrics:
    """check_termination_with_qwen_vl 的评估指标（Requirement 6.5）

    Attributes
    ----------
    accuracy : float
        (TP + TN) / total。要求 >= 0.90。
    confusion_matrix : dict
        {"TP": int, "FP": int, "TN": int, "FN": int}
    """

    accuracy: float
    confusion_matrix: dict  # {"TP": int, "FP": int, "TN": int, "FN": int}


@dataclass
class ScanMetrics:
    """scan_targets_with_qwen_vl 的评估指标（Requirement 6.6）

    Attributes
    ----------
    count_accuracy : float
        |predicted_count - annotated_count| <= 1 的样本比例。要求 >= 0.80。
    mean_abs_error : float
        |predicted_count - annotated_count| 的均值。
    """

    count_accuracy: float   # fraction with |pred - gt| <= 1
    mean_abs_error: float


# ---------------------------------------------------------------------------
# VLMEvaluator
# ---------------------------------------------------------------------------


class VLMEvaluator:
    """在验证集上评估 VLM 的各项指标（Requirement 9.6）

    模型采用延迟加载策略：首次调用评估方法时才加载模型，
    避免在没有 GPU 的环境下导入时报错。

    Parameters
    ----------
    model_id : str
        Hugging Face 模型 ID，例如 "Qwen/Qwen2.5-VL-7B-Instruct"。
    device : str
        设备映射策略，传给 ``from_pretrained`` 的 ``device_map`` 参数。
        默认 "auto"（自动分配到可用 GPU）。
    """

    def __init__(self, model_id: str, device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self.vlm_model = None
        self.vlm_processor = None
        # 延迟加载模型（避免在没有 GPU 的环境下报错）

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """延迟加载 Qwen2.5-VL 模型。

        首次调用时从 ``self.model_id`` 加载模型和 processor，
        后续调用为空操作（已加载则跳过）。
        使用 try/except 处理导入失败（例如缺少 transformers / GPU）。
        """
        if self.vlm_model is not None:
            return  # 已加载，跳过

        try:
            from experiments.robot.libero.qwenvl import load_qwen_vl_model

            self.vlm_model, self.vlm_processor = load_qwen_vl_model(
                self.model_id, self.device
            )
            if self.vlm_model is None:
                raise RuntimeError(
                    f"load_qwen_vl_model returned None for model_id={self.model_id!r}"
                )
        except Exception as exc:
            print(f"[VLMEvaluator] Failed to load model: {exc}")
            traceback.print_exc()
            self.vlm_model = None
            self.vlm_processor = None

    def _predict(self, image_paths: List[str], question: str) -> str:
        """调用 VLM 进行预测，返回答案字符串。

        Parameters
        ----------
        image_paths : List[str]
            图像文件路径列表（通常 1~2 张）。
        question : str
            发送给 VLM 的问题文本。

        Returns
        -------
        str
            VLM 的原始回答字符串（已 strip）。
            若模型未加载或推理失败，返回空字符串 ``""``。
        """
        self._load_model()
        if self.vlm_model is None or self.vlm_processor is None:
            return ""

        try:
            import collections

            from PIL import Image

            from experiments.robot.libero.qwenvl import _run_vlm_query

            images = [Image.open(p).convert("RGB") for p in image_paths]
            labels = [f"Image {i + 1}:" for i in range(len(images))]

            return _run_vlm_query(
                self.vlm_model,
                self.vlm_processor,
                images,
                labels,
                question,
                max_new_tokens=64,
            )
        except Exception as exc:
            print(f"[VLMEvaluator._predict] Exception: {exc}")
            traceback.print_exc()
            return ""

    # ------------------------------------------------------------------
    # Public evaluation methods
    # ------------------------------------------------------------------

    def evaluate_completion(
        self, val_samples: List[dict]
    ) -> CompletionMetrics:
        """计算 check_completion_with_qwen_vl 的 Precision 和 Recall（Requirement 9.6）。

        每个样本格式（与 VQASample JSONL 一致）：
            {
                "image_paths": ["main.png", "wrist.png"],
                "question": "...",
                "answer": "Yes" | "No",   # ground truth
                ...
            }

        Parameters
        ----------
        val_samples : List[dict]
            验证集样本列表。

        Returns
        -------
        CompletionMetrics
            包含 precision、recall、f1、accuracy 和混淆矩阵。
            若样本列表为空，返回全零指标。
        """
        if not val_samples:
            return CompletionMetrics(
                precision=0.0,
                recall=0.0,
                f1=0.0,
                accuracy=0.0,
                confusion_matrix={"TP": 0, "FP": 0, "TN": 0, "FN": 0},
            )

        self._load_model()

        tp = fp = tn = fn = 0

        for sample in val_samples:
            image_paths: List[str] = sample["image_paths"]
            question: str = sample["question"]
            gt_answer: str = sample["answer"].strip().lower()  # "yes" or "no"

            raw_pred = self._predict(image_paths, question)
            pred_positive = raw_pred.strip().lower().startswith("yes")
            gt_positive = gt_answer.startswith("yes")

            if pred_positive and gt_positive:
                tp += 1
            elif pred_positive and not gt_positive:
                fp += 1
            elif not pred_positive and not gt_positive:
                tn += 1
            else:  # not pred_positive and gt_positive
                fn += 1

        total = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / total if total > 0 else 0.0

        return CompletionMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            confusion_matrix={"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        )

    def evaluate_termination(
        self, val_samples: List[dict]
    ) -> TerminationMetrics:
        """计算 check_termination_with_qwen_vl 的准确率（Requirement 6.5）。

        每个样本格式（与终止判断 VQA JSONL 一致）：
            {
                "image_paths": ["main.png", "wrist.png"],
                "question": "...",
                "answer": "Yes" | "No",   # ground truth
                ...
            }

        Parameters
        ----------
        val_samples : List[dict]
            验证集样本列表。

        Returns
        -------
        TerminationMetrics
            包含 accuracy 和混淆矩阵。
            若样本列表为空，返回全零指标。
        """
        if not val_samples:
            return TerminationMetrics(
                accuracy=0.0,
                confusion_matrix={"TP": 0, "FP": 0, "TN": 0, "FN": 0},
            )

        self._load_model()

        tp = fp = tn = fn = 0

        for sample in val_samples:
            image_paths: List[str] = sample["image_paths"]
            question: str = sample["question"]
            gt_answer: str = sample["answer"].strip().lower()

            raw_pred = self._predict(image_paths, question)
            pred_positive = raw_pred.strip().lower().startswith("yes")
            gt_positive = gt_answer.startswith("yes")

            if pred_positive and gt_positive:
                tp += 1
            elif pred_positive and not gt_positive:
                fp += 1
            elif not pred_positive and not gt_positive:
                tn += 1
            else:
                fn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0

        return TerminationMetrics(
            accuracy=accuracy,
            confusion_matrix={"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        )

    def evaluate_scanning(
        self, val_samples: List[dict]
    ) -> ScanMetrics:
        """计算 scan_targets_with_qwen_vl 的水果数量准确率（Requirement 6.6）。

        每个样本格式（与 ScanSample JSONL 一致）：
            {
                "image_path": "main.png",
                "question": "...",          # 可选，若无则使用默认扫描问题
                "target_count": int,        # ground truth 水果数量
                "visible_targets": [...],   # ground truth 目标列表
                ...
            }

        Parameters
        ----------
        val_samples : List[dict]
            验证集样本列表。

        Returns
        -------
        ScanMetrics
            包含 count_accuracy（|pred - gt| <= 1 的比例）和 mean_abs_error。
            若样本列表为空，返回全零指标。
        """
        if not val_samples:
            return ScanMetrics(count_accuracy=0.0, mean_abs_error=0.0)

        self._load_model()

        within_one = 0
        abs_errors: List[float] = []

        _default_question = (
            "How many fruits are currently visible on the table? "
            "Output ONLY a single integer (0 if none)."
        )

        for sample in val_samples:
            image_path: str = sample["image_path"]
            gt_count: int = int(sample["target_count"])
            question: str = sample.get("question", _default_question)

            raw_pred = self._predict([image_path], question)

            # 尝试从回答中提取整数
            pred_count = _parse_count_from_response(raw_pred)

            abs_err = abs(pred_count - gt_count)
            abs_errors.append(float(abs_err))
            if abs_err <= 1:
                within_one += 1

        total = len(val_samples)
        count_accuracy = within_one / total if total > 0 else 0.0
        mean_abs_error = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0

        return ScanMetrics(
            count_accuracy=count_accuracy,
            mean_abs_error=mean_abs_error,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_count_from_response(response: str) -> int:
    """从 VLM 回答字符串中提取第一个非负整数。

    若无法解析，返回 0（保守估计：桌面已清空）。

    Parameters
    ----------
    response : str
        VLM 的原始回答，例如 "3" 或 "There are 2 fruits." 或 "NONE"。

    Returns
    -------
    int
        解析到的水果数量，解析失败时返回 0。
    """
    import re

    if not response:
        return 0

    # 先检查 NONE / empty 关键词
    if response.strip().upper() in {"NONE", "EMPTY", "0", "ZERO"}:
        return 0

    # 提取第一个数字
    match = re.search(r"\b(\d+)\b", response)
    if match:
        return int(match.group(1))

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json as _json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Qwen2.5-VL on fruit-clearing VLM tasks."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Hugging Face model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="",
        help="Path to validation JSONL file (VQA format).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device map strategy for model loading (default: auto).",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="eval_report.json",
        help="Output path for the evaluation report JSON.",
    )
    args = parser.parse_args()

    evaluator = VLMEvaluator(model_id=args.model_id, device=args.device)

    # Load validation samples
    val_samples: list = []
    if args.val_data_path:
        try:
            with open(args.val_data_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            val_samples.append(_json.loads(line))
                        except _json.JSONDecodeError:
                            pass
        except FileNotFoundError:
            print(f"[evaluate_vlm] Warning: val_data_path not found: {args.val_data_path}")

    # Run evaluations
    completion_metrics = evaluator.evaluate_completion(val_samples)
    termination_metrics = evaluator.evaluate_termination(val_samples)

    # Scan samples use a different format; filter those that have image_path
    scan_samples = [s for s in val_samples if "image_path" in s and "target_count" in s]
    scan_metrics = evaluator.evaluate_scanning(scan_samples)

    # Print metrics (parsed by run_training_pipeline._parse_metrics_from_output)
    print(f"completion_accuracy: {completion_metrics.accuracy:.4f}")
    print(f"completion_precision: {completion_metrics.precision:.4f}")
    print(f"completion_recall: {completion_metrics.recall:.4f}")
    print(f"completion_f1: {completion_metrics.f1:.4f}")
    print(f"termination_accuracy: {termination_metrics.accuracy:.4f}")
    print(f"scan_count_accuracy: {scan_metrics.count_accuracy:.4f}")
    print(f"scan_mean_abs_error: {scan_metrics.mean_abs_error:.4f}")

    # Save report
    report = {
        "completion": {
            "accuracy": completion_metrics.accuracy,
            "precision": completion_metrics.precision,
            "recall": completion_metrics.recall,
            "f1": completion_metrics.f1,
            "confusion_matrix": completion_metrics.confusion_matrix,
        },
        "termination": {
            "accuracy": termination_metrics.accuracy,
            "confusion_matrix": termination_metrics.confusion_matrix,
        },
        "scanning": {
            "count_accuracy": scan_metrics.count_accuracy,
            "mean_abs_error": scan_metrics.mean_abs_error,
        },
    }
    import logging as _logging
    _log = _logging.getLogger(__name__)
    try:
        with open(args.report_path, "w", encoding="utf-8") as fh:
            _json.dump(report, fh, indent=2, ensure_ascii=False)
        _log.info("Evaluation report saved to: %s", args.report_path)
    except Exception as exc:
        _log.warning("Could not save report: %s", exc)

    # Exit with non-zero if key metrics are below thresholds
    ok = (
        completion_metrics.precision >= 0.80
        and completion_metrics.recall >= 0.80
        and termination_metrics.accuracy >= 0.90
        and scan_metrics.count_accuracy >= 0.80
    )
    sys.exit(0 if ok else 1)
