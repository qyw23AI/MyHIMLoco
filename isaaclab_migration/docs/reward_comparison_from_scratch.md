# 从随机初始化比较 Task-Gait 与迁移前奖励

本实验只比较 reward scales，不比较旧检查点上的微调能力。两个实验组都使用同一 task-gait 策略输入、phase/activation 观测、参考控制器、precision 命令分布、URDF、物理与随机化配置、PPO 参数、种子和训练预算；训练命令不传 `--checkpoint`，从随机初始化开始。

| 实验组 | 奖励配置 |
|---|---|
| `task_gait_current` | 当前 task-gait 全量奖励：迁移配置覆写以及五项 gait reward |
| `legacy_rewards` | 从 `legacy/mybot_v3/mybot_v3_config.py` 读取旧 Mybot V3 reward scales；task-gait 新增项统一置零 |

旧奖励由代码直接加载，不复制手写数值。其关键参数包括线速度 2.0、角速度 0.5、base height -1.0、foot clearance -0.02、rear thigh -0.2、dof position -0.1、joint position limits 0、collision 0。当前 task-gait 与旧配置的差异另见每次运行的 `config.json` 与 `run_manifest.json`。

首轮采用 seed 17、1024 环境、rollout 100、precision phase、每组 2000 updates。两组按 seed 配对、串行训练，不覆盖已有目录；每个最终模型使用相同的 32 环境固定 posture-scan 评估，并录制 4 环境、400 控制步、每 2 步采帧的短视频。输出根目录由操作者显式提供。

运行：

```bash
/home/qyw/miniconda3/envs/himloco/bin/python \
  isaaclab_migration/scripts/reward_comparison_experiments.py \
  --output /absolute/new/output/path
```

该对照能回答“在 task-gait 输入/控制结构相同的前提下，奖励变化是否导致差异”。它不能单独回答 phase/activation 输入或参考控制器本身是否造成步态问题；若差异明显，应使用额外的策略结构对照和另一 seed 复核。仿真扭矩不是实际电机电流或热安全证明。
