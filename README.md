<h1 align="center">Agentic Robot: A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents</h1>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-blue.svg" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/pytorch-orange.svg" alt="PyTorch"></a>
</p>


<p align="center"><a href="https://github.com/Agentic-Robot/agentic-robot">🏠 Project Page</a> • <a href="https://arxiv.org/abs/2505.23450">📄 Paper(Arxiv)</a> • <a href="https://agentic-robot.github.io/">🌐 Website</a> • 
</p>


<p align="center">Zhejian Yang, Yongchao Chen, Xueyang Zhou, Jiangyue Yan, Dingjie Song, Yinuo Liu, Yuting Li, Yu Zhang, Pan Zhou, Hechang Chen*, Lichao Sun</p>

---

![Agentic Robot](./figures/Flow.jpg)

Long-horizon robotic manipulation poses significant challenges for autonomous systems, requiring extended reasoning, precise execution, and robust error recovery across complex sequential tasks. Current approaches, whether based on static planning or end-to-end visuomotor policies, suffer from error accumulation and lack effective verification mechanisms during execution, limiting their reliability in real-world scenarios. We present Agentic Robot, a brain-inspired framework that addresses these limitations through Standardized Action Procedures (SAP)—a novel coordination protocol governing component interactions throughout manipulation tasks. Drawing inspiration from Standardized Operating Procedures (SOPs) in human organizations, SAP establishes structured workflows for planning, execution, and verification phases. Our architecture comprises three specialized components: (1) a large reasoning model that decomposes high-level instructions into semantically coherent subgoals, (2) a vision-language-action executor that generates continuous control commands from real-time visual inputs, and (3) a temporal verifier that enables autonomous progression and error recovery through introspective assessment. This SAP-driven closed-loop design supports dynamic self-verification without external supervision. On the LIBERO benchmark, Agentic Robot achieves state-of-the-art performance with an average success rate of 79.6%, outperforming SpatialVLA by 6.1% and OpenVLA by 7.4% on long-horizon tasks. These results demonstrate that SAP-driven coordination between specialized components enhances both performance and interpretability in sequential manipulation, suggesting significant potential for reliable autonomous systems.

---
## Daily Update

- **[2026/05/26]**
  - 精简了原子操作的内容，从 `Place X into the right/left basket` 修改为 **`Place it into the basket`**，去掉了具体蔬菜类型和篮子方向。
  - 优化了任务设计，不再是单次将全部水果放到篮子里，修改后挑选指定水果放到篮子里，即 **`Put both A and B into the basket`**。
  - 新增部分代码：
    - **`demo_real_robot.py`**：真实机器人推理演示脚本，集成 LRM 任务分解 + VLA 执行 + VLM 验证的完整闭环流程。
    - **`tune_mapping.py`**：动作空间映射参数调优脚本，用于校准 VLA 输出到真实机械臂关节/末端指令的缩放系数。
    - **`mapping.py`**：动作空间映射模块，定义 VLA 模型输出与 UR 机械臂控制指令之间的转换逻辑。
    - **`calibrate_mapping.py`**：映射标定工具，通过采集标定数据自动计算最优映射参数。

- **[2026/05/27]**
  - 优化了**dataset.py**文件,修改了其远程控制协议,从RCTD修改为XML，实现了通过键盘控制机械臂移动和夹爪的开闭,无需手动标记夹爪状态.
  - 优化了坐标系映射逻辑,目前基本实现输入指令与真实世界坐标系的对齐.

- **[2026/05/29]**
  - 完成了初步的数据采集，共获取约110条子任务数据
  - 修改部分代码:
    - **`finetune_xvla.py`**：微调底层控制模型VLA的脚本，设置了仅微调SoftPrompt(默认)、LoRA微调Backbone以及全量微调三种不同的微调方式。
    - **`xvla_npz_dataset.py`**: 将npz数据转为X-VLA原生的EE6D(20dim)格式。
  - 目前的数据输入是归一化的绝对TCP位姿以及旋转角，而非单步变化，后续需要对这方面进行修改，让模型去学习更容易学习的变化。
    - 值得一提的是，TCP位姿可以用下一步减当前步作为变化幅度，而旋转角不能做简单的减法


- **[2026/05/30]**
  - 解决了部分环境依赖冲突,跑通了微调X-VLA SoftPrompt的demo.
  - 解决了训练速度异常的问题,主要集中在数据读取方面,相关代码已在 **`xvla_npz_dataset.py`** 中进行修改 , 同时对微调代码进行了优化以适应上述修改.
  - 考虑是否需要对原始数据进行相关处理,删掉一些没有动作的帧.
  - 修改及新增部分代码
    - **`finetune_xvla.py`**:
      - 新增 evaluate():在验证集上跑(val_max_batches 限制条数),返回总 loss + 各分量。
      - 训练循环里每 val_every_steps 跑一次验证,记录到 wandb 的 val/*。
      - 早停:验证 loss 连续 early_stop_patience 次没提升(超过 early_stop_min_delta)就停。
      - 新参数:--val_ratio 0.15、--val_every_steps 200、--early_stop_patience 10、--val_max_batches 50、--split_seed。
      - 测试验证过:112 episode → 95 train / 17 val,无重叠、索引正确、batch 仍单 episode
      - 优化了训练完成后模型参数的保存及命名逻辑
    - **`visualize_predictions.py`**
      - 加载 checkpoint(HF 目录,或 基础模型 + --pth 把可训练参数贴上去)。
      - 取某条 episode 的某个起点,用 model.generate_actions 预测动作 chunk。
      - 把预测和真值的 EE6D 解码成 xyz 位移 + 夹爪,画成对比图(累积轨迹、逐步增量、夹爪开合),并打印位置 MAE 和夹爪匹配率。

## TODO
- 去除无效帧并对比效果
- 对比XVLA Soft Prompt参数微调,LoRA微调及全量微调的效果
- 网格搜索参数

## ✨ News ✨

- **[2025/06/22]** 🤖 We open-sourced Agentic Robot v1.0 — we’ll continue improving the project with new ideas and updates, so feel free to follow and ⭐️ Star us to stay in the loop! [Agentic Robot v1.0](https://github.com/Agentic-Robot/agentic-robot)

## Quick Setup
Before running Agentic Robot, please make sure the required environments are properly set up:

🧠 Our project is built on top of [OpenVLA](https://github.com/moojink/openvla-oft?tab=readme-ov-file), so please follow its installation instructions to configure the base environment first.

🧪 Experiments are conducted in the [LIBERO](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md) simulation environment. Make sure to install LIBERO and its dependencies as described in their official documentation.

## ## 🚀 Implementation

Our training framework consists of two stages:

1. Stage I: Decompose the complex task with LRM.  
2. Stage II: Evaluate Agentic Robot with VLA and VLM.

## Task Devision

Here, we take Deepseek-V3 as an example to decompose the complex task.

```python
python experiments/robot/libero/ds.py
```

**For convenience, we provide a test case that calls the LRM once, then generates a hard-coded plan and replaces it in the main.py file. For handling multiple tasks, we can add a function to call the LRM in main.py.**

## 📊 Evaluation

We evaluate the Agentic Robot on LIBERO benchmark with VLM (QwenVL-2.5) and VLA (OpenVLA).

Navigate to the evaluation directory:
```python
cd ./experiments/robot/libero/
```
# Launch XVLA finetune

```python
bash vla-scripts/train_fruit_vla.sh raw_demos_left_third 1 soft_prompt
```

# Launch LIBERO-Spatial evals

```python
python experiments/robot/libero/main.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True
```

# Launch LIBERO-Object evals

```python
python experiments/robot/libero/main.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True
```

# Launch LIBERO-Goal evals

```python
python experiments/robot/libero/main.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True
```

# Launch LIBERO-10 (LIBERO-Long) evals

```python
python experiments/robot/libero/main.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True
```

## Support

If you run into any issues, please open a new GitHub issue. If you do not receive a response within 2 business days, please email Zhejian Yang (JLU-Advisor@outlook.com) to bring the issue to his attention.

## Citation

If you use our code in your work, please cite [our paper](https://arxiv.org/abs/2505.23450):

```bibtex
@misc{yang2025agenticrobotbraininspiredframework,
      title={Agentic Robot: A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents},
      author={Zhejian Yang and Yongchao Chen and Xueyang Zhou and Jiangyue Yan and Dingjie Song and Yinuo Liu and Yuting Li and Yu Zhang and Pan Zhou and Hechang Chen and Lichao Sun},
      year={2025},
      eprint={2505.23450},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.23450},
}
```
