# Requirements Document

## Introduction

本文档规划针对**桌面水果收纳长时序任务**（Multi-Fruit Table Clearing Task）的完整数据采集与模型训练方案。任务目标：机器人将桌上所有水果（种类不同、数量未知）逐一放入指定篮子，直到桌面清空。

系统基于已有的 PEV（Plan → Execute → Verify）三层架构：LRM（DeepSeek）负责任务分解并生成 REPEAT…UNTIL 计划、VLA（X-VLA）负责低层动作执行、VLM（Qwen2.5-VL）负责子任务验证与循环终止判断。

相比蓝莓采摘，本任务的优势在于：动作为标准桌面抓取（`pick up` + `place`），无需从枝条上拔取；水果种类不同使目标识别更容易；场景与已有 raw_demos 数据（`Pick up the chili on the table`）高度相似，可直接复用现有 `NpzEpisodeDataset` 和 `finetune_xvla.py` 训练管道。

## Glossary

- **PEV_System**：Plan → Execute → Verify 三层机器人控制系统
- **LRM**：Language Reasoning Module，基于 DeepSeek 的任务分解模块，将高层任务描述转化为结构化计划（含 REPEAT…UNTIL 块）
- **VLA**：Vision-Language-Action 模型（X-VLA），输入图像 + 语言指令，输出 7-DoF 动作向量
- **VLM**：Vision-Language Model（Qwen2.5-VL），用于子任务完成验证、目标扫描、循环终止判断
- **Episode**：一次完整的任务演示，从初始状态（桌上有若干水果）到桌面清空
- **Transition**：单步 (观测, 动作) 对，即 npz 数据中的一个时间步
- **Atomic_Op**：原子操作，本任务使用 `pick up [fruit]` 和 `place [fruit] in the basket`
- **REPEAT_Block**：LRM 计划中的循环结构，包含循环体（body）和终止条件（until_condition）
- **Scan_Step**：REPEAT 循环体内的感知步骤，由 VLM 执行（非 VLA），调用 `scan_targets_with_qwen_vl` 返回当前桌面可见水果列表
- **NpzEpisodeDataset**：已有的 PyTorch Dataset 类，从 `episode_*.npz` 文件加载 (观测, 动作) 转换对
- **ActionNormalizer**：将连续动作归一化到 [-1, 1] 的工具类，基于数据集统计量
- **TCP_Pose**：末端执行器位姿，6 维 [x, y, z, roll, pitch, yaw]
- **Domain_ID**：X-VLA 的具身域标识符，本任务使用独立 ID（推荐值：5）
- **LoRA**：Low-Rank Adaptation，参数高效微调方法，已在 `finetune_xvla.py` 中实现
- **VQA_Sample**：视觉问答样本，用于 VLM 微调，格式为 (图像列表, 问题, 答案)
- **Scan_Sample**：目标扫描标注样本，格式为 (场景图像, 可见水果有序列表)
- **Data_Collector**：数据采集模块，负责遥操作录制、标注和格式转换
- **Training_Pipeline**：训练流水线，负责调用微调脚本、管理检查点、输出评估报告
- **Data_Validator**：数据质量验证模块，负责运行正确性属性检查
- **Evaluator**：模型评估模块，负责在验证集上计算量化指标
- **Parser**：`parse_llm_plan` 函数，将 LRM 原始输出字符串解析为 `Plan` 对象

## Requirements

### Requirement 1: VLA 原子操作演示数据采集

**User Story:** As a robot engineer, I want to collect npz demonstration data for the two core atomic operations (`pick up [fruit]` and `place [fruit] in the basket`), so that I can fine-tune the VLA model to reliably execute each step of the fruit-clearing task.

#### Acceptance Criteria

1. THE Data_Collector SHALL collect no fewer than 50 Episode demonstrations for each of the two atomic operation types: `pick up [fruit]` and `place [fruit] in the basket`.
2. WHEN collecting `pick up [fruit]` demonstrations, THE Data_Collector SHALL cover at least 5 different fruit types: apple, orange, banana, pear, and tomato (or equivalent substitutes available in the physical/simulated environment).
3. WHEN collecting `pick up [fruit]` demonstrations, THE Data_Collector SHALL cover at least 3 different spatial positions on the table: left side, center, and right side.
4. THE Data_Collector SHALL store each Episode in the existing npz format with fields: `images (T, 256, 256, 3) uint8`, `images_wrist (T, 256, 256, 3) uint8`, `tcp_poses (T, 6) float64`, `joint_positions (T, 6) float64`, `gripper (T,) float64`, `instruction str`, `fps int`.
5. WHEN an Episode collection is complete, THE Data_Collector SHALL verify that `images.shape[0] == tcp_poses.shape[0] == gripper.shape[0]`. IF the check fails, THEN THE Data_Collector SHALL discard the Episode and write an error log entry.
6. THE Data_Collector SHALL use standardized instruction strings for each atomic operation, naming the specific fruit type (e.g., `"pick up the apple"`, `"place the apple in the basket"`), to ensure language instruction consistency across training samples.
7. WHEN the full atomic operation dataset collection is complete, THE Data_Collector SHALL invoke `compute_dataset_statistics()` to generate `dataset_statistics.json` containing `mean`, `std`, `min`, `max`, `q01`, and `q99` fields for both `action` and `state`.

---

### Requirement 2: 完整 Episode 演示数据采集

**User Story:** As a robot engineer, I want to collect complete multi-step fruit-clearing Episodes covering the full REPEAT loop workflow, so that the VLA model learns the context of sequential pick-and-place within a long-horizon task.

#### Acceptance Criteria

1. THE Data_Collector SHALL collect no fewer than 200 complete fruit-clearing Episodes. Episodes where zero loop iterations occur (e.g., empty initial table) SHALL still count toward the 200 required Episodes but SHALL be flagged in metadata with `loop_iterations: 0`.
2. WHEN collecting complete Episodes, THE Data_Collector SHALL ensure the fruit count per Episode is between 2 and 8, to cover varying loop iteration counts.
3. THE Data_Collector SHALL cover at least 3 different initial scene configurations: all fruits of different types (maximum variety), some fruits of the same type, and fruits arranged in different spatial layouts (clustered vs. spread out).
4. WHEN a complete Episode collection is finished, THE Data_Collector SHALL record metadata for each Episode including: total fruit count, fruit types present, successfully placed count, loop iteration count, and total timestep count.
5. THE Data_Collector SHALL store complete Episode data and atomic operation data in separate directories: complete Episodes in `raw_demos/full_episodes/` and atomic operations in `raw_demos/atomic_ops/`.

---

### Requirement 3: VLM 子任务完成验证数据采集

**User Story:** As a robot engineer, I want to collect visual question-answering data for fine-tuning the VLM on subtask completion verification, so that the VLM can accurately judge whether `pick up` and `place in basket` actions are complete.

#### Acceptance Criteria

1. THE Data_Collector SHALL collect no fewer than 200 VQA_Samples for each atomic operation type (`pick up` and `place in basket`). The positive-to-negative sample ratio SHALL be verified globally across both operation types combined, and SHALL be within [45%, 55%].
2. WHEN constructing VQA_Samples, THE Data_Collector SHALL use dual-view images (main camera + wrist camera), with each sample containing 2 images.
3. THE Data_Collector SHALL use a standardized question template for each VQA_Sample: `"The robot instruction is: '{instruction}'. Has this action been completed? Answer strictly 'Yes' or 'No'."`.
4. WHEN collecting negative samples for `pick up`, THE Data_Collector SHALL cover at least 3 failure modes: grasp failure (gripper closed on empty air), unstable grasp (fruit partially slipping), and position offset (end-effector not reaching the fruit).
5. WHEN collecting negative samples for `place in basket`, THE Data_Collector SHALL cover at least 2 failure modes: fruit placed outside the basket, and fruit dropped before reaching the basket.
6. THE Data_Collector SHALL store VQA_Samples in JSON Lines format, with each line containing fields: `image_paths: List[str]`, `question: str`, `answer: str` ("Yes" or "No"), `operation_type: str`, `fruit_type: str`, `failure_mode: Optional[str]`.
7. WHEN the VQA dataset construction is complete, THE Data_Collector SHALL compute the positive-to-negative sample ratio from actual positive and negative sample counts. IF the ratio is outside [45%, 55%], THEN THE Data_Collector SHALL report an imbalance warning.

---

### Requirement 4: VLM 目标扫描与循环终止判断数据采集

**User Story:** As a robot engineer, I want to collect data for training the VLM to identify remaining fruits on the table (`scan_targets_with_qwen_vl`) and judge loop termination (`check_termination_with_qwen_vl`), so that the REPEAT…UNTIL loop correctly iterates over all fruits and stops when the table is clear.

#### Acceptance Criteria

1. THE Data_Collector SHALL collect no fewer than 300 annotated Scan_Samples, each containing: a scene image (main camera view of the table) and an ordered list of visible fruit descriptions sorted by proximity/ease of grasping.
2. WHEN collecting Scan_Samples, THE Data_Collector SHALL cover fruit counts from 0 to 8, with no fewer than 50 samples containing 0 fruits (empty table), to train the VLM to recognize the cleared state.
3. WHEN collecting Scan_Samples with multiple fruits, THE Data_Collector SHALL include the fruit type in each target description (e.g., `"the apple on the left side of the table"`, `"the orange near the center"`).
4. THE Data_Collector SHALL collect no fewer than 200 VQA_Samples for loop termination judgment, with samples where the termination condition is true (table is clear) and false (fruits still on table) each comprising 50%.
5. WHEN constructing termination judgment samples, THE Data_Collector SHALL use the question template consistent with `check_termination_with_qwen_vl`: `"Termination condition: 'no fruits remain on the table'. Is this termination condition currently TRUE? Answer strictly 'Yes' or 'No'."`.
6. THE Data_Collector SHALL store Scan_Samples in JSON Lines format, with each line containing fields: `image_path: str`, `visible_targets: List[str]` (ordered list, empty list means table is clear), `target_count: int`, `fruit_types_present: List[str]`.

---

### Requirement 5: X-VLA LoRA 微调

**User Story:** As a robot engineer, I want to fine-tune X-VLA using LoRA on the collected fruit-clearing data, so that the VLA model can accurately execute `pick up [fruit]` and `place [fruit] in the basket` for diverse fruit types.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL use the existing `finetune_xvla.py` script with the `--npz_data_dir` parameter pointing to the fruit-clearing data directory.
2. THE Training_Pipeline SHALL assign `domain_id=5` for the fruit-clearing task, distinct from the existing LIBERO domain (`domain_id=3`).
3. WHEN fine-tuning X-VLA, THE Training_Pipeline SHALL use the following baseline hyperparameters: `lora_rank=32`, `learning_rate=2e-4`, `batch_size=8`, `grad_accumulation_steps=4`, `max_steps=20000`.
4. WHEN fine-tuning is in progress, THE Training_Pipeline SHALL save a checkpoint every 500 steps and log `train/loss`, `train/action_accuracy`, and `train/action_l1_loss` metrics to W&B.
5. WHEN fine-tuning is complete, THE Training_Pipeline SHALL evaluate the model on a held-out validation set (10% of total data), and the action L1 loss on the validation set SHALL be below 0.05.
6. THE Training_Pipeline SHALL mix atomic operation data and complete Episode data for training at a ratio of 3:1 (atomic ops : full episodes), using `NpzEpisodeDataset` with `image_aug=True`.

---

### Requirement 6: Qwen2.5-VL LoRA 微调

**User Story:** As a robot engineer, I want to fine-tune Qwen2.5-VL for fruit-clearing-specific tasks, so that the VLM achieves high accuracy on subtask verification, fruit scanning, and table-clear termination judgment.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL fine-tune Qwen2.5-VL-7B-Instruct using LoRA via the `transformers` + `peft` framework, with hyperparameters: `lora_rank=16`, `learning_rate=1e-4`, `batch_size=4`, `max_steps=5000`.
2. THE Training_Pipeline SHALL convert the VQA data from Requirements 3 and 4 into Qwen2.5-VL conversation format (messages list with image and text content blocks).
3. WHEN fine-tuning the VLM, THE Training_Pipeline SHALL train in two stages: Stage 1 trains only subtask completion verification (Requirement 3 data); Stage 2 jointly trains verification, scanning, and termination judgment (Requirements 3 + 4 data).
4. WHEN Stage 1 training is complete, THE Training_Pipeline SHALL evaluate `check_completion_with_qwen_vl` accuracy on the validation set, and the accuracy SHALL be no lower than 85%.
5. WHEN Stage 2 training is complete, THE Training_Pipeline SHALL evaluate `check_termination_with_qwen_vl` accuracy on the validation set, and the accuracy SHALL be no lower than 90%.
6. THE Training_Pipeline SHALL evaluate fruit-count accuracy on the scanning validation set (absolute difference between predicted and annotated fruit count ≤ 1), and the accuracy SHALL be no lower than 80%.

---

### Requirement 7: LRM 提示工程验证

**User Story:** As a robot engineer, I want to verify that the LRM prompt templates correctly handle the fruit-clearing task, so that the LRM generates well-formed REPEAT…UNTIL plans that enumerate no specific fruit indices and use VLM-verifiable termination conditions.

#### Acceptance Criteria

1. THE LRM SHALL automatically trigger the `_HARVEST_PROMPT` template for task descriptions containing keywords `"all"`, `"every"`, `"all fruits"`, `"clear the table"`, `"全部"`, or `"所有水果"` (via the existing `_is_harvest_task` heuristic function, extended with these keywords).
2. WHEN the LRM generates a fruit-clearing plan using the `_HARVEST_PROMPT` template, THE LRM SHALL output a plan containing exactly one REPEAT…UNTIL block, with a loop body containing `scan`, `pick up`, and `place` steps. IF the LRM generates a plan with zero or more than one REPEAT…UNTIL block, THE Parser SHALL reject it and return an empty Plan.
3. THE LRM SHALL use `"no fruits remain on the table"` or an equivalent visually verifiable phrase as the UNTIL condition.
4. WHEN `parse_llm_plan` parses LRM output, THE Parser SHALL correctly identify REPEAT…UNTIL blocks and return a `Plan` containing `RepeatBlock` objects.
5. FOR ALL valid `Plan` objects, after serializing to a string and re-parsing, THE Parser SHALL return a semantically equivalent `Plan` object with the same step count, step types, and step content (round-trip property).
6. THE LRM SHALL NOT generate plans that enumerate specific fruit indices or positions (e.g., `"pick up fruit 1"`, `"pick up the leftmost fruit"`), to ensure plan generalization across different scene configurations.

---

### Requirement 8: 训练顺序与依赖关系

**User Story:** As a robot engineer, I want to define the training order and dependency relationships between models, so that the entire training pipeline can be executed sequentially with each stage's outputs usable by downstream stages.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL execute training stages in the following order: Stage 1 (data collection) → Stage 2 (VLM Stage 1 fine-tuning) → Stage 3 (VLA LoRA fine-tuning) → Stage 4 (VLM Stage 2 fine-tuning) → Stage 5 (end-to-end PEV system evaluation).
2. WHEN Stage 2 begins, THE Training_Pipeline SHALL verify that the VQA dataset from Requirement 3 exists and contains no fewer than 400 samples total. IF the condition is not met, THEN THE Training_Pipeline SHALL abort and report a missing data error.
3. WHEN Stage 3 begins, THE Training_Pipeline SHALL verify that the npz datasets from Requirements 1 and 2 exist and contain no fewer than 300 Episodes total. IF the condition is not met, THEN THE Training_Pipeline SHALL abort and report a missing data error.
4. WHEN Stage 5 (end-to-end evaluation) begins, THE Training_Pipeline SHALL verify that both the VLA checkpoint and VLM checkpoint exist. IF either is missing, THEN THE Training_Pipeline SHALL abort and report a missing checkpoint error.
5. THE Training_Pipeline SHALL generate a training report for each stage, including: training duration, final loss, validation set metrics, and checkpoint path.

---

### Requirement 9: 数据与模型正确性验证属性

**User Story:** As a robot engineer, I want to establish quantifiable correctness verification metrics for collected data and trained models, so that automated tests can detect data quality issues and model degradation.

#### Acceptance Criteria

1. THE Data_Validator SHALL verify the temporal dimension consistency of each npz Episode: `images.shape[0] == images_wrist.shape[0] == tcp_poses.shape[0] == joint_positions.shape[0] == gripper.shape[0]`.
2. THE Data_Validator SHALL verify the round-trip property of `ActionNormalizer`: FOR ALL valid action vectors `a`, the L∞ error between `denormalize(normalize(a))` and `a` SHALL be less than a configurable threshold (default `1e-5`, adjustable based on normalizer implementation precision).
3. THE Data_Validator SHALL verify the index invariance of `NpzEpisodeDataset`: FOR ALL valid indices `i`, two consecutive calls to `dataset[i]` SHALL return tensors with identical shapes for `pixel_values`, `input_ids`, `labels`, and `state`.
4. THE Data_Validator SHALL verify the value range constraint of normalized actions: FOR ALL normalized actions `a_norm`, each dimension SHALL be within `[-1.0, 1.0]`.
5. THE Data_Validator SHALL verify the completeness of `dataset_statistics.json`: the file MUST contain fields `action.mean`, `action.std`, `action.q01`, `action.q99`, `state.mean`, `state.std`, `num_transitions`, and `num_episodes`.
6. WHEN evaluating the VLM, THE Evaluator SHALL compute Precision and Recall for `check_completion_with_qwen_vl` using a confusion matrix, and both Precision and Recall SHALL be no lower than 0.80.
7. THE Data_Validator SHALL verify the label distribution of the VQA dataset by computing the positive-to-negative sample ratio from actual sample counts. IF the ratio is outside `[0.45, 0.55]`, THEN THE Data_Validator SHALL output an imbalance warning and recommend an oversampling strategy. The validation check SHALL pass regardless of whether the warning output succeeds.
8. THE Data_Validator SHALL verify the idempotence of `parse_llm_plan`: FOR ALL valid plan strings `s`, `parse_llm_plan(format_plan(parse_llm_plan(s)))` SHALL be semantically equivalent to `parse_llm_plan(s)` (same step count, same step types, same step content).
