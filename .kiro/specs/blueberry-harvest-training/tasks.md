# Implementation Plan: Blueberry Harvest Training (Multi-Fruit Table Clearing)

## Overview

本实现计划将设计文档中的各组件逐步转化为可执行代码。实现顺序为：先修复现有数据采集脚本（`data/dataset.py`），再依次实现数据采集、数据验证、VLM 数据集构建、VLA 训练脚本、VLM 微调脚本、评估模块，最后完成流水线编排。每个阶段均包含属性测试子任务，确保核心数据变换的正确性。

## Tasks

- [x] 1. 修改 `data/dataset.py`：填充 TASK_LIST 并实现随机任务选择
  - [x] 1.1 填充 TASK_LIST 并替换硬编码 TASK_NAME
    - 在 `TASK_LIST` 中填入 10 条标准化指令字符串，覆盖 5 种水果（apple、orange、banana、pear、tomato）的两类原子操作（`pick up [fruit]` 和 `place [fruit] in the basket`）
    - 在 `main()` 函数的每条 Episode 录制循环开始处，使用 `random.choice(TASK_LIST)` 随机选取一条指令，赋值给局部变量 `task_instruction`
    - 将 `np.savez_compressed(...)` 调用中的 `instruction=TASK_NAME` 替换为 `instruction=task_instruction`
    - 在文件顶部添加 `import random`
    - _Requirements: 1.2, 1.6_

- [x] 2. 实现 `data/collect_fruit_demos.py` — 数据采集模块
  - [x] 2.1 实现 `EpisodeMetadata` 数据类和 `build_instruction()` 函数
    - 按设计文档定义 `EpisodeMetadata` dataclass，包含所有元数据字段
    - 实现 `build_instruction(task_type, fruit_type)` 函数，生成标准化指令字符串
    - _Requirements: 1.6, 2.4_

  - [x] 2.2 实现 `FruitDemoCollector` 类的核心方法
    - 实现 `__init__`，接受 `save_dir`、`task_type`、`fruit_type` 参数，复用 `data/dataset.py` 的 RealSense + RTDE 采集逻辑
    - 实现 `verify_temporal_consistency(episode_data)`，检查所有时序数组的第一维相等
    - 实现 `save_episode(episode_data, metadata)`，保存 npz 文件和 `episode_NNNN_meta.json` sidecar
    - 实现 `collect_episode()`，录制一条 Episode，验证时序一致性，失败时丢弃并写入错误日志
    - _Requirements: 1.4, 1.5, 2.4_

  - [ ]* 2.3 为 `verify_temporal_consistency` 编写属性测试
    - **Property 1: Episode Temporal Dimension Consistency**
    - **Validates: Requirements 1.5, 9.1**
    - _Requirements: 1.5, 9.1_

  - [x] 2.4 实现 `compute_and_save_statistics()` 方法
    - 调用 `vla_scripts.npz_dataset.compute_dataset_statistics()`，将结果保存为 `dataset_statistics.json`
    - 确保输出包含 `action.mean`、`action.std`、`action.q01`、`action.q99`、`state.mean`、`state.std`、`num_transitions`、`num_episodes` 字段
    - _Requirements: 1.7, 9.5_

  - [ ]* 2.5 为 `compute_dataset_statistics` 编写属性测试
    - **Property 5: dataset_statistics.json Completeness**
    - **Validates: Requirements 1.7, 9.5**
    - _Requirements: 1.7, 9.5_

  - [x] 2.6 实现 `VQASample`、`ScanSample` 数据类和 `collect_vqa_sample()` 函数
    - 按设计文档定义 `VQASample` 和 `ScanSample` dataclass
    - 实现 `collect_vqa_sample()`，使用标准化问题模板构造 VQA 样本，确保 `image_paths` 恰好包含 2 个路径
    - _Requirements: 3.2, 3.3, 3.6, 4.6_

  - [ ]* 2.7 为 `collect_vqa_sample` 编写属性测试
    - **Property 10: VQA Sample Structural Invariants**
    - **Validates: Requirements 3.2, 3.3, 3.6**
    - _Requirements: 3.2, 3.3, 3.6_

- [x] 3. Checkpoint — 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. 实现 `data/validate_dataset.py` — 数据验证模块
  - [x] 4.1 实现 `ValidationResult`、`ValidationReport` 数据类和 `format_plan()` 函数
    - 定义 `ValidationResult(passed, message, details)` 和 `ValidationReport(results, all_passed, summary)` dataclass
    - 实现 `format_plan(plan)`，将 `Plan` 对象序列化为字符串（线性步骤 → 编号列表，RepeatBlock → REPEAT:…UNTIL: 格式）
    - _Requirements: 7.5, 9.8_

  - [x] 4.2 实现 `DataValidator` 的各验证方法
    - 实现 `validate_temporal_consistency(npz_path)`（Requirement 9.1）
    - 实现 `validate_action_normalizer_roundtrip(stats, threshold=1e-5)`（Requirement 9.2）
    - 实现 `validate_dataset_index_invariance(dataset, num_samples=100)`（Requirement 9.3）
    - 实现 `validate_normalized_action_range(stats, num_samples=1000)`（Requirement 9.4）
    - 实现 `validate_statistics_completeness(stats_path)`（Requirement 9.5）
    - 实现 `validate_vqa_label_balance(jsonl_path, low=0.45, high=0.55)`（Requirement 9.7）
    - 实现 `validate_parse_llm_plan_idempotence(plan_strings)`（Requirement 9.8）
    - 实现 `run_all()`，汇总所有验证结果
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.7, 9.8_

  - [ ]* 4.3 为 `ActionNormalizer` 编写属性测试
    - **Property 2: ActionNormalizer Round-Trip**
    - **Validates: Requirements 9.2**
    - _Requirements: 9.2_

  - [ ]* 4.4 为归一化动作值域编写属性测试
    - **Property 3: Normalized Action Value Range**
    - **Validates: Requirements 9.4**
    - _Requirements: 9.4_

  - [ ]* 4.5 为 `NpzEpisodeDataset` 索引不变性编写属性测试
    - **Property 4: NpzEpisodeDataset Index Invariance**
    - **Validates: Requirements 9.3**
    - _Requirements: 9.3_

  - [ ]* 4.6 为 VQA 标签分布编写属性测试
    - **Property 6: VQA Label Balance**
    - **Validates: Requirements 3.7, 9.7**
    - _Requirements: 3.7, 9.7_

  - [ ]* 4.7 为 `parse_llm_plan` 幂等性编写属性测试
    - **Property 7: parse_llm_plan Idempotence**
    - **Validates: Requirements 7.5, 9.8**
    - _Requirements: 7.5, 9.8_

- [x] 5. 实现 `experiments/robot/libero/test_lrm.py` — LRM 提示工程验证测试
  - [x] 5.1 实现 `_is_harvest_task` 关键词路由测试
    - 编写单元测试，验证包含 harvest 关键词的任务描述返回 `True`，不含关键词的返回 `False`
    - 覆盖中英文关键词：`"all"`、`"every"`、`"clear the table"`、`"harvest"`、`"所有"`、`"全部"`、`"清空"` 等
    - _Requirements: 7.1_

  - [ ]* 5.2 为 `_is_harvest_task` 编写属性测试
    - **Property 8: _is_harvest_task Keyword Routing**
    - **Validates: Requirements 7.1**
    - _Requirements: 7.1_

  - [x] 5.3 实现 `parse_llm_plan` REPEAT 块解析测试
    - 编写单元测试，验证包含一个 REPEAT…UNTIL 块的计划字符串被正确解析为含一个 `RepeatBlock` 的 `Plan`
    - 验证 `body` 包含正确的子任务字符串，`until_condition` 与 UNTIL 行匹配
    - _Requirements: 7.2, 7.4_

  - [ ]* 5.4 为 `parse_llm_plan` REPEAT 块解析编写属性测试
    - **Property 9: parse_llm_plan REPEAT Block Parsing**
    - **Validates: Requirements 7.2, 7.4**
    - _Requirements: 7.2, 7.4_

- [x] 6. Checkpoint — 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. 实现 `data/build_vlm_dataset.py` — VQA 数据集构建模块
  - [x] 7.1 实现 `load_vqa_samples()`、`vqa_sample_to_conversation()`、`scan_sample_to_conversation()`
    - 实现从 JSONL 文件加载 VQA 样本的函数
    - 实现将 `VQASample` 转换为 Qwen2.5-VL 对话格式的函数，输出包含 `"messages"` 键的 dict，user 消息含 2 个 image 块和 1 个 text 块，assistant 消息含答案文本
    - 实现将 `ScanSample` 转换为对话格式的函数
    - _Requirements: 6.2_

  - [ ]* 7.2 为 VQA 到对话格式转换编写属性测试
    - **Property 13: VQA to Conversation Format Round-Trip**
    - **Validates: Requirements 6.2**
    - _Requirements: 6.2_

  - [x] 7.3 实现 `build_stage1_dataset()`、`build_stage2_dataset()`、`compute_label_balance()`
    - 实现 Stage 1 数据集构建（仅子任务完成验证数据），按 `val_split` 划分训练/验证集
    - 实现 Stage 2 数据集构建（验证 + 扫描 + 终止判断数据合并）
    - 实现 `compute_label_balance(conversations)`，计算正样本比例
    - _Requirements: 3.7, 6.3_

- [x] 8. 实现 `vla-scripts/train_fruit_vla.sh` — VLA 训练脚本
  - [x] 8.1 创建 `vla-scripts/train_fruit_vla.sh`
    - 按设计文档编写 Shell 脚本，调用 `finetune_xvla.py`，设置 `domain_id=5`、`lora_rank=32`、`learning_rate=2e-4`、`batch_size=8`、`grad_accumulation_steps=4`、`max_steps=20000`、`save_steps=500`、`image_aug=True`
    - 接受 `NPZ_DATA_DIR` 和 `NUM_GPUS` 两个位置参数，提供默认值
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 8.2 实现数据混合目录构建逻辑（在 `run_training_pipeline.py` 中）
    - 实现按 3:1 比例（atomic_ops : full_episodes）将 episode 文件符号链接到 `raw_demos/mixed/` 的函数
    - 混合后重新计算并保存 `dataset_statistics.json`
    - _Requirements: 5.6_

  - [ ]* 8.3 为数据混合比例编写属性测试
    - **Property 14: Data Mixing Ratio Invariant**
    - **Validates: Requirements 5.6**
    - _Requirements: 5.6_

- [x] 9. 实现 `vla-scripts/finetune_qwenvl.py` — Qwen2.5-VL LoRA 微调脚本
  - [x] 9.1 实现 `QwenVLFinetuneConfig` 数据类和 `QwenVLConversationDataset` 类
    - 按设计文档定义配置数据类，包含所有超参数字段（`lora_rank=16`、`learning_rate=1e-4`、`batch_size=4`、`max_steps=5000`）
    - 实现 `QwenVLConversationDataset`，从 JSONL 加载对话，应用 chat template，处理多图像输入，返回 `input_ids`、`attention_mask`、`pixel_values`、`labels`（prompt 位置 mask 为 -100）
    - _Requirements: 6.1, 6.2_

  - [x] 9.2 实现 `finetune_qwenvl()` 主训练循环
    - 加载 Qwen2.5-VL-7B-Instruct，应用 LoRA（`peft` 库），实现两阶段训练逻辑（Stage 1 仅完成验证数据，Stage 2 全部数据）
    - 每 `save_steps` 步保存 checkpoint，记录训练指标到 W&B
    - _Requirements: 6.1, 6.3, 6.4, 6.5_

- [x] 10. 实现 `vla-scripts/evaluate_vlm.py` — VLM 评估模块
  - [x] 10.1 实现 `CompletionMetrics`、`TerminationMetrics`、`ScanMetrics` 数据类
    - 按设计文档定义三个评估指标数据类
    - _Requirements: 6.4, 6.5, 6.6, 9.6_

  - [x] 10.2 实现 `VLMEvaluator` 类的三个评估方法
    - 实现 `evaluate_completion(val_samples)`，使用混淆矩阵计算 Precision 和 Recall，要求均 ≥ 0.80
    - 实现 `evaluate_termination(val_samples)`，计算准确率，要求 ≥ 0.90
    - 实现 `evaluate_scanning(val_samples)`，计算水果数量准确率（|预测 - 标注| ≤ 1），要求 ≥ 0.80
    - _Requirements: 6.4, 6.5, 6.6, 9.6_

- [x] 11. 实现 `vla-scripts/run_training_pipeline.py` — 流水线编排器
  - [x] 11.1 实现 `PipelineConfig`、`StageReport`、`PipelineError` 和前置条件检查方法
    - 定义配置和报告数据类
    - 实现 `check_stage2_prerequisites()`：验证 VQA 数据集存在且样本数 ≥ 400，否则抛出 `PipelineError`
    - 实现 `check_stage3_prerequisites()`：验证 npz 数据集存在且 Episode 数 ≥ 300，否则抛出 `PipelineError`
    - 实现 `check_stage5_prerequisites()`：验证 VLA 和 VLM checkpoint 均存在，否则抛出 `PipelineError`
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [ ]* 11.2 为流水线前置条件检查编写属性测试
    - **Property 12: Pipeline Prerequisite Enforcement**
    - **Validates: Requirements 8.2, 8.3, 8.4**
    - _Requirements: 8.2, 8.3, 8.4_

  - [x] 11.3 实现 `run_stage()` 和 `run_all()` 方法
    - 实现 `run_stage(stage_id)`，按阶段 ID 调用对应脚本，生成包含训练时长、最终 loss、验证指标、checkpoint 路径的 `StageReport`
    - 实现 `run_all()`，按顺序执行 5 个阶段（数据采集 → VLM Stage 1 → VLA → VLM Stage 2 → 端到端评估）
    - _Requirements: 8.1, 8.5_

- [x] 12. 实现 `tests/test_properties.py` — 属性测试汇总文件
  - [x] 12.1 将所有属性测试整合到 `tests/test_properties.py`
    - 使用 Hypothesis 框架，将 2.3、2.5、2.7、4.3、4.4、4.5、4.6、4.7、5.2、5.4、7.2、8.3、11.2 中的属性测试整合到统一测试文件
    - 确保每个属性测试均有对应的 `@given` 策略和 `@settings` 配置
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.7, 9.8, 7.1, 7.2, 7.4, 7.5, 3.2, 3.3, 3.6, 4.3, 4.6, 8.2, 8.3, 8.4, 5.6, 6.2_

  - [ ]* 12.2 为 `ScanSample` 结构不变性编写属性测试
    - **Property 11: Scan Sample Structural Invariants**
    - **Validates: Requirements 4.3, 4.6**
    - _Requirements: 4.3, 4.6_

- [x] 13. Final Checkpoint — 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- 任务 1.1 是最高优先级的代码修改，直接影响现有数据采集脚本的正确性
- 每个任务均引用具体需求条款，确保可追溯性
- 属性测试使用 Hypothesis 框架，与单元测试互补
- 检查点任务确保增量验证，避免后期集成问题
- `data/dataset.py` 修改后，`TASK_NAME` 变量可保留为注释或删除，以 `random.choice(TASK_LIST)` 替代

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "4.1", "5.1"] },
    { "id": 2, "tasks": ["2.2", "2.6", "4.2", "5.3"] },
    { "id": 3, "tasks": ["2.3", "2.4", "2.7", "4.3", "4.4", "4.5", "5.2", "5.4"] },
    { "id": 4, "tasks": ["2.5", "4.6", "4.7", "7.1", "8.1"] },
    { "id": 5, "tasks": ["7.2", "7.3", "8.2", "9.1"] },
    { "id": 6, "tasks": ["8.3", "9.2", "10.1", "11.1"] },
    { "id": 7, "tasks": ["10.2", "11.2", "11.3"] },
    { "id": 8, "tasks": ["12.1", "12.2"] }
  ]
}
```
