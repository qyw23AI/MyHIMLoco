# MyHIMLoco 项目架构与作用概览

## 项目作用
本仓库是 OpenRobotLab 的学习式四足机器人运动控制研究代码集合，当前包含两条研究线：HIMLoco（Hybrid Internal Model）与 H-Infinity Locomotion Control。项目基于 NVIDIA Isaac Gym 的并行仿真环境，结合修改版的 legged_gym（环境与任务）和 rsl_rl（强化学习算法）进行训练与评估。

## 总体结构（一图概念）
- 研究/项目层：projects/ 下存放两条研究线的论文介绍与演示。
- 仿真环境层：legged_gym/ 提供机器人环境、任务与资产。
- 强化学习算法层：rsl_rl/ 提供 PPO 及 HIM 的算法、模型与训练器。
- 资产与文档层：resources/ 与 assets/ 提供机器人模型与展示图片，根目录 README/quick_start 提供安装与运行说明。

## 目录与模块职责

### 1) 根目录
- README.md：项目总览、安装、训练与播放入口。
- quick_start.md：近期改动（主要是 mybot_v2 与 mybot_v2_1 的配置更新）。
- requirements.txt：基础 Python 依赖列表。
- assets/：论文与演示的图片资源。

### 2) legged_gym/（环境与任务）
- legged_gym/envs/：各机器人任务与配置。
  - base/：
    - base_task.py：Isaac Gym 任务基类与通用缓冲区。
    - legged_robot.py：核心环境实现，包含仿真步进、奖励、终止条件、观测与重置逻辑。
    - legged_robot_config.py：默认环境/地形/指令/控制等基础配置。
  - a1/, go1/, aliengo/, mybot/, mybot_v2/, mybot_v2_1/：各机器人特化配置。
  - __init__.py：任务注册（a1, go1, aliengo, mybot, mybot_v2, mybot_v2_1）。
- legged_gym/scripts/：
  - train.py：训练入口，调用 task_registry 创建环境与算法并开始学习。
  - play.py：加载并评估策略，同时支持导出 JIT 模型。
- legged_gym/utils/：
  - task_registry.py：任务注册与构建环境/算法的统一入口，绑定到 HIM 的训练器。
  - helpers.py：命令行参数、种子、仿真参数解析与策略导出工具。
  - terrain.py、math.py、logger.py：地形生成、数学工具与日志记录。
- resources/robots/：URDF/MJCF 与网格资源（含 mybot 系列）。

### 3) rsl_rl/（强化学习算法）
- algorithms/：
  - ppo.py：标准 PPO 实现。
  - him_ppo.py：HIM 版 PPO，在更新中引入估计器损失与交换损失。
- modules/：
  - actor_critic.py：标准 Actor-Critic 网络。
  - him_actor_critic.py：HIM 版本 Actor-Critic，包含 HIMEstimator 与历史观测处理。
  - him_estimator.py：HIM 的对比学习式估计器，输出速度与隐式嵌入。
- runners/：
  - him_on_policy_runner.py：训练主循环，收集 rollout、更新网络、记录日志与保存模型。
- storage/：
  - rollout_storage.py、him_rollout_storage.py：训练数据缓存与优势估计。

### 4) projects/
- projects/himloco/README.md：HIMLoco 论文与方法介绍。
- projects/h_infinity/README.md：H-Infinity 论文与方法介绍。

## 训练与推理主流程
1. 入口脚本 legged_gym/scripts/train.py 解析参数并调用 task_registry。
2. task_registry.make_env：根据任务名创建环境（LeggedRobot）。
3. task_registry.make_alg_runner：构建 HIMOnPolicyRunner 与 HIMPPO。
4. HIMOnPolicyRunner.learn：循环执行环境步进、收集数据、调用 HIMPPO.update 更新策略。
5. legged_gym/scripts/play.py：加载最新策略，执行评估并可导出 JIT 策略。

## HIMLoco 的核心数据流（简化描述）
- 环境输出的观测包含历史片段（obs_history）。
- HIMEstimator 从历史观测编码速度与隐式嵌入。
- HIMActorCritic 将当前观测片段 + 速度 + 隐式嵌入输入策略网络生成动作。
- HIMPPO 结合 PPO 损失与估计器损失进行联合优化。

## 关键配置与可扩展点
- 新增机器人：在 envs/ 下添加配置文件与资源目录，完成任务注册即可训练。
- 任务参数：主要在各 *_config.py 中定义（控制、奖励、地形、随机化等）。
- 训练参数：由 LeggedRobotCfgPPO 等训练配置类提供，并可通过命令行覆盖。

## Sim2Real 与部署（当前仓库已包含/缺失）
### 已包含的支持
- 域随机化与噪声：在 LeggedRobotCfg 中提供摩擦、负载、质心、关节增益、初始姿态、外力扰动、推搡、控制延迟等随机化，以及观测噪声配置，用于提升仿真到现实的鲁棒性。
- 策略导出：legged_gym/scripts/play.py 支持导出 JIT 策略；HIM 模型导出为 policy.pt，非 HIM 导出为 policy_1.pt，输出目录在 logs/<experiment_name>/exported/policies/。

### 目前缺失或未公开的部分
- README 的 TODO 明确说明“部署指导尚未发布”，仓库内没有完整的真实机器人部署流程、驱动接口或标定脚本。

### 需要自行对接与校验的事项（根据现有配置推断）
- 控制接口一致性：控制模式与尺度（如 control_type、action_scale、PD 参数、decimation）需与真实机器人控制环一致。
- 观测与历史窗口一致性：HIM 依赖历史观测与隐式嵌入，实际部署需保证观测拼接顺序与归一化一致。
- 动力学与接触参数对齐：随机化范围与真实机器人参数应做合理标定与调整。

## 近期改动摘要（quick_start）
- 新增 mybot_v2_1 任务与资源，调整奖励项与默认控制参数。
- mybot_v2 参数更新（高度、关节阻尼/刚度、奖励项与关节限制）。
- 全局默认训练参数与随机化范围调整。

## 依赖与环境
- NVIDIA Isaac Gym（Preview 4）
- PyTorch 1.10 + CUDA 11.3
- Python 3.7.x（README 提供建议环境）

---
如需更细粒度的模块解读（例如奖励函数、观测维度、具体网络结构），请告诉我你希望聚焦的部分。
