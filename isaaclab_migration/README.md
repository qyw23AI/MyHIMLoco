# HIMLoco → Isaac Lab 独立迁移

迁移实现位于本目录。原 `legged_gym`、`rsl_rl` 和已有 Isaac Lab 安装保持原样；训练直接导入本仓库 `rsl_rl/rsl_rl`，不安装官方 RSL-RL 来替换 HIM 实现。

## 当前运行环境

本机已安装并在 RTX 5060 Ti 16GB 上运行验证：

| 组件 | 实测版本 |
| --- | --- |
| OS | Ubuntu 22.04 |
| NVIDIA Driver | 580.178.04 |
| GPU | RTX 5060 Ti，16311 MiB |
| Python | 3.11.17，`/home/qyw/miniconda3/envs/himloco/bin/python` |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab 源码 | `IsaacGymEnvs/IsaacLab`，VERSION 2.3.2 |
| PyTorch | 2.7.0+cu128 |
| NumPy | 1.26.0 |

Isaac Lab v2.3.x 的官方兼容表包含 Isaac Sim 5.1，官方 x86_64 安装指南使用 PyTorch 2.7/cu128。这是当前本机复用并实测的组合，不代表这些历史版本永久获得上游维护。官方 Isaac Sim 5.1 页面现已标为旧版。

- [Isaac Lab 官方版本依赖](https://github.com/isaac-sim/IsaacLab/tree/v2.3.2#isaac-sim-version-dependency)
- [Isaac Lab v2.3.2 官方安装指南](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/pip_installation.html)
- [Isaac Sim 5.1 系统要求与版本状态](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

本机无需安装依赖即可使用下面的脚本。运行前在普通桌面终端中执行：

```bash
cd /home/qyw/MyHIMLoco
conda activate himloco
python isaaclab_migration/scripts/preflight.py --output isaaclab_migration/outputs/preflight.json
```

如环境确实缺失，应建立新的 Python 3.11 环境，按照上述官方指南安装相同组合，然后使用已有 Lab 源码；不要执行根目录旧 `requirements.txt`（它锁定 cu121 与旧 Gym 依赖）。可选安装独立迁移包：

```bash
python -m pip install --no-deps -e /home/qyw/MyHIMLoco/isaaclab_migration
```

新环境参考命令（仅用于新建环境，本次未执行）：

```bash
conda create -n himloco_lab python=3.11 -y
conda activate himloco_lab
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e /home/qyw/MyHIMLoco/IsaacGymEnvs/IsaacLab/source/isaaclab
python -m pip install tensorboard 'numpy<2' scipy trimesh
python -m pip install --no-deps -e /home/qyw/MyHIMLoco/isaaclab_migration
```

## 四阶段运行

默认任务 `mybot_v3`，默认 1024 环境。`--stage 1/2/3` 使用平地，第四阶段恢复任务原地形配置。未指定 `--rollout_steps` 时保留原 100 步（桥梁任务 24 步），未指定 `--iterations` 时使用任务原最大迭代数。

```bash
# 第一阶段：PD 零动作站立，500 控制步 = 10 秒
python isaaclab_migration/scripts/train.py --stage 1 --steps 500 --headless

# 第二阶段：原普通 PPO/ActorCritic 验证
python isaaclab_migration/scripts/train.py --stage 2 --iterations 2 --headless

# 第三阶段：完整 HIM/PPO/Encoder/SwAV 平地训练
python isaaclab_migration/scripts/train.py --stage 3 --headless

# 第四阶段：原 mybot_v3 楼梯、课程学习与域随机化
python isaaclab_migration/scripts/train.py --stage 4 --headless

# 本仓库自定义桥梁
python isaaclab_migration/scripts/train.py --stage 4 --task mybot_v3_bridge_a --headless
python isaaclab_migration/scripts/train.py --stage 4 --task mybot_v3_bridge_b --headless

# 机械臂在资产中是固定负载，原策略仍为 12 维腿部动作
python isaaclab_migration/scripts/train.py --stage 4 --task mybot_v3_with_arm --headless
```

其他任务名：`a1`、`go1`、`aliengo`、`mybot`（MJCF）、`mybot_v2`、`mybot_v2_1`。各任务仍读取独立复制的实际配置，并加载原资产。具体已验证任务和限制见 `docs/MIGRATION_AUDIT.md`；支持注册不等于所有任务已经通过长期训练验证。

## 可视化训练与导出

不传 `--headless` 即开启桌面窗口。默认从第一个环境开始使用自由视角，鼠标旋转、平移和缩放不会被程序还原。加 `--follow_camera` 才会每次渲染自动跟随第一个机器人；跟随模式会覆盖鼠标修改的视角。运行中也可在 Isaac Lab 的 `Viewer Settings → Follow Mode` 中选 `Env` 或 `World` 关闭跟随，选 `Robot` 开启跟随。`--render_every 16` 只降低画面刷新频率，物理仍为 200 Hz，策略仍为 50 Hz。

当前 HIM 没有图像输入，显示清晰度不会改变训练观测。原地形高度采样仍直接来自数值高度场。`--viewer_resolution 1920 1080 --antialiasing FXAA` 可提高 GUI 纹理分辨率并改用单帧抗锯齿；`--render_every 4` 提高训练显示刷新频率。它们增加渲染开销，但不改变物理／控制频率、奖励或历史观测。

```bash
python isaaclab_migration/scripts/train.py \
  --task mybot_v3 --stage 3 --num_envs 1024 --iterations 100 \
  --render_every 16 \
  --log_dir isaaclab_migration/logs/visual_training \
  --capture_dir isaaclab_migration/outputs/visual_training \
  --capture_every 100 \
  --export_policy isaaclab_migration/outputs/visual_training/policy.pt \
  --preview_seconds 30
```

`model_100.pt` 是可恢复训练的检查点，包含 Actor、Critic、HIM Encoder、目标网络、原型与两个优化器状态。`policy.pt` 是独立 TorchScript 推理模型，包含 HIM Encoder 和 Actor；输入 `[batch,270]`，输出 `[batch,12]`。导出后会重新加载文件并与训练模型核对输出，结果和关节顺序写入旁边的 `policy.json`。

训练输出目录含 `config.json`、TensorBoard、滚动／永久检查点和 `validation.json`。视口目录含训练过程 PNG、`training.png` 和可选 `preview.png`。图像由实际 Isaac Sim 渲染生成。

从已有检查点恢复：

```bash
python isaaclab_migration/scripts/train.py --stage 3 --checkpoint /absolute/path/model_100.pt --iterations 100 --headless
```

单独导出已有 HIM 检查点（CPU 即可）：

```bash
python isaaclab_migration/scripts/export_policy.py \
  --task mybot_v3 \
  --checkpoint /absolute/path/model_100.pt \
  --output /absolute/path/policy.pt
```

使用导出的文件在实际 Lab 环境中可视化播放，并记录状态：

```bash
python isaaclab_migration/scripts/play.py \
  --policy /absolute/path/policy.pt --num_envs 4 --seconds 30 \
  --capture_dir isaaclab_migration/outputs/playback \
  --state_csv isaaclab_migration/outputs/playback/state.csv
```

加载推理文件：

```python
import torch
policy = torch.jit.load('/absolute/path/policy.pt', map_location='cpu').eval()
with torch.no_grad():
    actions = policy(torch.zeros(1, 270))
```

## 测试与审计

```bash
python isaaclab_migration/tests/test_contract.py
python isaaclab_migration/tests/test_sim.py --headless
python isaaclab_migration/tests/test_terrain.py
python isaaclab_migration/scripts/audit_sources.py
```

11 项 CPU 测试以原仓库代码为参照，比较观测、终止观测、全部基础奖励、P/V/T 控制、历史顺序、关节契约、桥梁网格，并实际执行 HIM 更新。GPU 测试核对部分环境重置、重置前终止目标、历史保留、超时及努力目标的关节映射。源文件清单检查原代码 SHA256 未变。

`test_terrain.py` 在实际楼梯碰撞网格上检查首次重置不误升级、按原距离规则升级／降级和正确的地形出生位置。这些受控探针结果与自然训练成功统计分开保存。第四阶段楼梯训练自动输出 `curriculum.json`：初末等级分布、实际等级增减次数、出生点迁移、重置／超时和指令范围变化；地形／指令课程保持原条件，不强制增加难度。

运行时显存报告中，`peak_gpu_used_mib` 是定时采样的整卡使用量，包含 Kit/PhysX 和桌面；`peak_torch_allocated_gib` 只统计 PyTorch。默认 1024 环境已验证；增加环境数应观察整卡使用量和训练吞吐。

短训练与站立测试验证了导入、步进、梯度更新、保存和恢复，并不能证明策略已收敛或获得原 Isaac Gym 的相同物理轨迹。PhysX、碰撞导入和接触求解器版本发生变化，长期运动质量仍需训练评估。

## 本次实际交付（2026-10-13）

1024 环境的可视化 HIM 训练已完成：从原先 `model_4.pt` 恢复，新增 100 次更新，每次 rollout 100 步，新增 10,240,000 个环境步。整卡采样峰值 6377 MiB，最终文件如下：

- `logs/visual_training/model_104.pt`：完整恢复训练检查点，约 6.3 MiB。
- `outputs/visual_training/policy.pt`：独立 Encoder + Actor TorchScript，约 969 KiB。
- `outputs/visual_training/policy.json`：关节／观测／控制周期与导出核对结果，CPU/GPU 输出最大绝对差约 `1.67e-6`，通过 `rtol=1e-6, atol=1e-6` 校验。
- `outputs/visual_training/training.png`、`preview.png`、`training_curve.png`：真实训练视口及原 TensorBoard 数据曲线。
- `outputs/playback_verified/playback.gif`：导出的独立策略在真实 Lab 环境中的回放，默认自由镜头；84 帧，80 ms/帧。
- `outputs/playback_verified/validation.json`、`state.csv`：4 环境回放 337 步，6.74 秒仿真时间，0 reset，观测／动作／奖励有限值。
- `outputs/free_camera_validation.json`：实际移动相机后渲染 20 控制步，相机变换没有被程序覆盖。

策略仍处于早期训练阶段。上述短回放中第一个机器人平均前向速度约 0.061 m/s（指令 0.2 m/s），不能据此宣称速度跟踪或复杂地形性能已收敛。

继续训练（保留任务原超参数，另建输出目录）：

```bash
conda activate himloco
cd /home/qyw/MyHIMLoco
python isaaclab_migration/scripts/train.py --stage 3 --num_envs 1024 \
  --checkpoint isaaclab_migration/logs/visual_training/model_104.pt \
  --iterations 1000 --render_every 16 \
  --log_dir isaaclab_migration/logs/continued_training \
  --export_policy isaaclab_migration/outputs/continued_training/policy.pt
```

自由视角查看已导出的策略，直到手动关闭窗口：

```bash
python isaaclab_migration/scripts/play.py \
  --policy isaaclab_migration/outputs/visual_training/policy.pt \
  --num_envs 4 --command 0.2 0 0 --seconds 0
```

生成动态图使用 `python isaaclab_migration/scripts/make_preview.py /absolute/path/capture_dir`。默认每 4 个控制步捕获一帧时，80 ms/帧对应实际仿真时间。GUI 训练默认自由视角；截图使用 Kit 官方接口，并等待异步 PNG 写入结束。生成的模型、日志和图像均留在本地，已加入 Git ignore，避免误提交大型运行产物。

## 楼梯与课程学习实测（2026-10-13）

上述 `visual_training` 是第三阶段平地模型。第四阶段已经另行实跑 `terrain_training`：复用实际 `mybot_v3` 双向楼梯配置，`trimesh`、10 行难度 × 20 列、200 块 8×8 m 地形，地形／指令课程开启。没有替换成平地，也没有强制提升等级。

从平地 model_104 继续 100 次完整 HIM 更新（1024 环境 × 100 rollout，新增 10,240,000 个环境步），最终为 `logs/terrain_training/model_204.pt` 和 `outputs/terrain_training/policy.pt`。整卡峰值 6668 MiB。自然训练统计在 `logs/terrain_training/curriculum.json`：

- 等级均值 1.0195 → 1.9102；初始只分配 0～2 级，训练快照曾达到 9 级，最终最高 8 级。
- 4172 次等级数值增大、3260 次减小，7432 次出生点迁移。
- 11298 次环境结束／重置，其中 9257 次超时。
- 指令范围保持 [-0.2,0.2]；指令课程已开启，但本轮原升级条件未满足，未人为扩大范围。

`outputs/terrain_training/curriculum_curve.png` 使用本次统计和原 TensorBoard 数据绘制，展示课程均值、等级分布、自然升降及奖励。受控规则探针单独保存在 `docs/terrain_contract_result.json`，没有加入上述自然训练统计。`test_terrain.py` 同时修复并覆盖首次 reset 可能因 Lab 克隆坐标而误升级的问题。

复现本次第四阶段训练：

```bash
python isaaclab_migration/scripts/train.py --task mybot_v3 --stage 4 \
  --num_envs 1024 --iterations 100 \
  --checkpoint isaaclab_migration/logs/visual_training/model_104.pt \
  --render_every 16 --log_dir isaaclab_migration/logs/terrain_training_new \
  --export_policy isaaclab_migration/outputs/terrain_training_new/policy.pt
```

高清地形回放（默认自由视角）：

```bash
python isaaclab_migration/scripts/play.py --stage 4 \
  --policy isaaclab_migration/outputs/terrain_training/policy.pt \
  --num_envs 4 --seconds 0 --command 0.6 0 0 \
  --viewer_resolution 1920 1080 --antialiasing FXAA \
  --terrain_color 0.18 0.24 0.30
```

`--terrain_color` 只绑定地形视觉材质，碰撞网格、物理材料和观测保持原样。录像可加 `--follow_camera` 保持机器人在画面中；自由镜头下行走机器人可能走出画面。`outputs/terrain_playback/validation.json` 记录高清自由镜头回放：15.92 秒仿真，0 reset，首个机器人平均 x 速度 0.531 m/s，指令 0.6 m/s。上述有限训练与短回放验证了地形课程实际生效，仍不代表所有难度和方向都已稳定收敛。

`outputs/terrain_video/playback.gif` 是实际跟随镜头录像：131 帧、960×540 动图；原 PNG 为 1920×1080。地形改为深灰蓝视觉材质以提高白色机器人与台阶的对比，使用 FXAA，4 环境回放 10.52 秒仿真，0 reset。物理与课程参数保持实际任务配置。
