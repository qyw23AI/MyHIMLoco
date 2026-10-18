# 任务赛底层步态训练

独立任务：`mybot_v3_task_gait`。目标是机身坐标系速度 `vx/vy/yaw_rate` 的稳定跟踪、规律步态、低速响应与平滑启停。没有位置控制、目标 yaw 闭环、导航或绕杆规划。障碍赛和楼梯不属于这个任务。

## 快速运行

在项目根目录执行，脚本会使用本机 `himloco` 环境的 Python，无需提前激活 conda。

```bash
bash isaaclab_migration/scripts/task_gait.sh check
bash isaaclab_migration/scripts/task_gait.sh smoke

# 首次训练。默认 2048 环境、无界面，2000 次更新是预算，不表示收敛。
bash isaaclab_migration/scripts/task_gait.sh train --phase basic \
  --iterations 2000 --log_dir isaaclab_migration/logs/task_basic

# 后续课程必须显式传入自己的检查点，不自动寻找其他实验的模型。
bash isaaclab_migration/scripts/task_gait.sh train --phase precision \
  --checkpoint isaaclab_migration/logs/task_basic/model_2000.pt \
  --iterations 2000 --log_dir isaaclab_migration/logs/task_precision

bash isaaclab_migration/scripts/task_gait.sh train --phase robust \
  --checkpoint isaaclab_migration/logs/task_precision/model_4000.pt \
  --iterations 1000 --log_dir isaaclab_migration/logs/task_robust

# 更改并行数；它表示一个训练进程里的环境数，不是启动多个训练进程。
bash isaaclab_migration/scripts/task_gait.sh train --phase basic --num_envs 3072

# GUI、恢复同一课程、独立导出均可用。
bash isaaclab_migration/scripts/task_gait.sh train --gui --phase basic --iterations 100
bash isaaclab_migration/scripts/task_gait.sh train \
  --checkpoint /absolute/path/model.pt --iterations 100
python isaaclab_migration/scripts/export_policy.py --task mybot_v3_task_gait \
  --checkpoint /absolute/path/model.pt --output /absolute/path/policy.pt

bash isaaclab_migration/scripts/task_gait.sh evaluate --checkpoint /absolute/path/model.pt
bash isaaclab_migration/scripts/task_gait.sh play --checkpoint /absolute/path/model.pt \
  --command 0.03 0 0 --seconds 60
bash isaaclab_migration/scripts/task_gait.sh play --policy /absolute/path/policy.pt \
  --command 0 0 0 --standby
```

`TASK_GAIT_PYTHON=/path/to/python` 可覆盖解释器。训练参数沿用 `train.py`，课程参数用 `--phase`，消融用 `--gait_ablation baseline|timing|full`。每个新实验使用独立 `--log_dir`；不复用正在写入的目录。脚本不安装依赖、不改驱动、不自动覆盖/删除已有模型。

## 课程和初始参数

配置集中在 `himloco_lab/task_gait/config.py`，环境与奖励在 `core.py`，纯张量时钟和参考在 `reference.py`。原 `legged_gym`、迁移版 legacy 快照、PPO/HIM 网络算法均未改动。

| 课程 | 命令采样概率：保持/待命/低速/纯转向/正常前后/组合 | 地形 |
| --- | --- | --- |
| basic | 15/5/15/10/45/10% | 平地 |
| precision | 15/10/25/20/15/15% | 平地 |
| robust | 同 precision | 60% 平地、20% 缓坡、20% 轻微起伏 |

初始范围：前后 ±0.5 m/s，横移 ±0.2 m/s，偏航 ±0.6 rad/s；低速单轴抽样 0.015–0.10 m/s，小转向 0.03–0.20 rad/s，均覆盖正负方向。这些是训练覆盖范围，不是已经验证的运动能力。零命令显式采样；软件数值零容差为 `1e-6`，不再沿用 0.2 m/s 清零。连续外部命令只需在训练范围内，不会被抬升到最小采样速度。

25% 环境执行每段5秒的“正常移动→低速微调→保持→原地转向→待命踏步→保持”序列，正负方向交替覆盖；回合35秒，足以容纳完整30秒序列。其余75%使用上表概率独立采样。表中概率指随机采样部分，实际总体分布同时受序列和提前重置影响。

步频 2 Hz、支撑比例 0.65、抬脚 4.5 cm，均为可调起始值。腿序显式为 FL/FR/RL/RR，偏移 0/0.5/0.5/0。名义足位由当前 URDF/default angles 的正运动学计算，脚底半径由碰撞球读取（当前 2 cm），不复制 Go1 的站距。默认关节站姿只是起点，不宣称热最优。

速度经 0.15 s 时间常数平滑；步态激活约 0.4 s 过渡。停步先等待当前摆腿回到双组支撑区，再衰减激活；待命时即使身体目标速度为零也保持步态。待命和保持模式明确输入策略。摆腿高度不会随微小速度同比缩到零。

robust 坡度绝对值不超过 0.12（约 6.8°，实际随难度变化），起伏幅度上限 2 cm，无台阶/坑/跳石。域随机化先使用摩擦 0.7–1.2，robust 增加 ±10% PD 参数变化，避免继承楼梯任务的大扰动。平地训练默认带适量本体噪声，确定性评估关闭噪声和动力学随机化，评估配置写入报告。

## 观测、重置和恢复

Actor 每帧 49 维 = 原 45 维 + sin/cos 相位 + 激活量 + 待命模式。6 帧历史共 294 维；Critic 242 维 = Actor 当前帧 + 真实速度3 + 外力3 + 高度187。地形、真实速度和外力不进入 Actor。HIM 原有切片契约得到保留：速度标签位于 `[49:52]`，对比目标为 `[3:52]`。

只有重置环境的相位/历史被重置；新历史用其新传感帧填充。重置前的终止观测单独保存给 HIM，不能被新相位污染。相位与平滑命令在环境控制周期推进，导出的神经网络自身不包含这部分状态机。实机未来需要复现 `GaitState`、观测构造、历史顺序和重置逻辑，不能只拿 policy.pt 输入旧270维历史。

每个滚动/永久/最终检查点都嵌入 `task_config`，导出 JSON 也保存同一契约。相同课程恢复配置；显式切换课程时应用目标课程配置，同时恢复网络和优化器，变化写入新实验配置。旧270维 HIMloco 检查点拒绝加载，不能静默补零。不同课程学习曲线不应按奖励绝对值直接比较。

任务专用 Runner 另外修正周期检查点的恢复计数：文件中的 `iter` 表示已完成更新次数，避免沿用原 Runner 循环期间未更新的计数；最终文件名仍按累计更新次数生成。

## 奖励、负载和边界

速度误差尺度 = 固定下限 + 指令幅值乘相对尺度，分别处理平移/偏航，使小速度与完全不动有可分辨收益。新增接触、足位、摆腿奖励都是有界软约束。地面高度只用于训练奖励，不进入 Actor。此版本针对轻微地形，不用于高台阶。

负载惩罚使用每个物理子步的力矩平方均值，兼顾总负载与最高单关节相对负载；不强制左右完全相等。默认除以 URDF effort 仅作数值归一化，**不代表连续力矩**。`continuous_torque_nm` 暂为空，确认各电机型号/连续能力后可填写按关节名映射。没有热模型、没有预测温度、没有猜测电流/温度保护阈值，不能据此保证右后髋不过热。

零速踏步是待比较的行为，不能假设它比保持更凉。精确保持/待命踏步需要在相同实机负载与初温下比较电流、温升、位置漂移；遵守实际执行器保护限制。

## 诊断与评估

训练目录包含 `config.json`、TensorBoard、检查点、`validation.json`、`gait_metrics.json`。自动评估输出逐命令报告、各案例 `trace.csv` 和 `tracking.png`。评估包括正负速度扫描、零速两种模式、组合运动及启停/转向/模式切换序列；没有上层位置或目标 yaw 控制。

指标包括原始/平滑命令误差、速度 RMS/均值/标准差、支撑脚滑动、身体倾斜、对角接触一致性、接触周期/CV、各关节子步力矩 RMS/峰值/饱和比例。报告按关节名称标识 RR_hip，避免误认索引。接触使用 5N/2N 滞回并排除过短周期；仍需人工检查接触曲线。没有有效稳态/落脚样本时，对应跟踪/周期指标为 null，不能当作成功。跟踪统计排除命令改变/重置后一秒，负载统计包含全部瞬态。漂移仅对全程未重置环境统计，短于2秒的测试没有有效漂移结果。

`execution_validated_not_performance_approved` 只说明评估运行完成，**不是低速达标或步态收敛**。至少对充分训练模型做多次种子测试，再标定最低可靠速度、响应死区、稳态误差和启停漂移。

消融顺序：`baseline`（相同采样、维度、负载目标，无相位奖励）→ `timing`（接触+摆腿）→ `full`（加水平足位）。baseline 仍看到相位，属于受控消融，不等同于未经改动的原版 HIMloco。所有比较按相同环境交互步数与评估套件进行。

## 验证命令

```bash
python isaaclab_migration/tests/test_task_gait.py
python isaaclab_migration/tests/test_contract.py
python isaaclab_migration/tests/test_task_gait_sim.py --headless
python isaaclab_migration/tests/test_task_gait_sim.py --headless --phase robust
```

CPU 测试覆盖低速/转向/停止、参考连续性、监督切片、静止负载、配置隔离与模型导出；GPU 测试覆盖实际 Lab 局部重置、终止数据、低速命令、物理子步负载与轻地形。

## 方法来源

- HIMLoco：https://arxiv.org/abs/2312.11460
- Walk These Ways：https://arxiv.org/abs/2212.03238
- WTW 足位/接触奖励：https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/rewards/corl_rewards.py
- WTW 相位实现：https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/base/legged_robot.py

本实现使用上述思路，命令课程、完整平面刚体速度参考、占空比轨迹、启停/模式和负载目标属于本项目适配，不宣称是 WTW 原算法或已验证的新研究结论。
