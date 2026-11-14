`--headless--headless` 

# HIMLoco → Isaac Lab 独立迁移

任务赛底层规律步态、低速控制与独立训练入口见 [TASK_GAIT.md](docs/TASK_GAIT.md)。

自有机器人 WBLWBL Dog V1 的站姿、低趴配置和实测结果见 [WBLWBL_DOG_V1.md](docs/WBLWBL_DOG_V1.md)。

迁移实现位于本目录。原 `legged_gym`、`rsl_rl` 和已有 Isaac Lab 安装保持原样；训练直接导入本仓库 `rsl_rl/rsl_rl`，不安装官方 RSL-RL 来替换 HIM 实现。

## 当前运行环境

本机已安装并在 RTX 5060 Ti 16GB 上运行验证：

| 组件           | 实测版本                                                  |
| -------------- | --------------------------------------------------------- |
| OS             | Ubuntu 22.04                                              |
| NVIDIA Driver  | 580.178.04                                                |
| GPU            | RTX 5060 Ti，16311 MiB                                    |
| Python         | 3.11.17，`/home/qyw/miniconda3/envs/himloco/bin/python` |
| Isaac Sim      | 5.1.0.0                                                   |
| Isaac Lab 源码 | `IsaacGymEnvs/IsaacLab`，VERSION 2.3.2                  |
| PyTorch        | 2.7.0+cu128                                               |
| NumPy          | 1.26.0                                                    |

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

## 可视化训练与导出

### A1 论文配置快捷命令

针对 `logs/a1/upstream_paper_4096x100_5000_20261023` 的迁移版 HIMLoco 实验，可以直接使用快捷脚本。默认配置与论文源配置一致：A1、第四阶段 HIM、4096 个环境、每个环境 100 步 rollout、5000 次更新。训练会启动 Isaac Lab；加 `--headless` 可关闭窗口。

```bash
cd /home/qyw/MyHIMLoco
bash isaaclab_migration/scripts/a1_himloco.sh train --headless
```

训练完成后，导出 `model_5000.pt` 为包含 HIM Encoder 和 Actor 的 TorchScript：

```bash
bash isaaclab_migration/scripts/a1_himloco.sh export
```

默认输出为 `isaaclab_migration/outputs/a1/upstream_paper_4096x100_5000_20261023/policy.pt`，导出过程会重新加载模型并检查 TorchScript 与训练模型的输出误差。也可以指定其它检查点：

```bash
bash isaaclab_migration/scripts/a1_himloco.sh export \
  --checkpoint isaaclab_migration/logs/a1/upstream_paper_4096x100_5000_20261023/model_3000.pt
```

### 不录制视频，实时查看策略

如果只想快速观察策略效果，不需要 `ffmpeg`，使用 `view` 命令：

```bash
bash isaaclab_migration/scripts/a1_himloco.sh view --seconds 180
```

该命令会自动导出默认检查点，然后启动 Isaac Lab GUI，以实时速度运行 180 秒；默认使用自由视角，不会自动跟随机器人，可以用鼠标旋转、平移和缩放。不保存 PNG、不生成 MP4。省略 `--seconds` 或传入 `--seconds 0` 会一直运行，直到关闭窗口：

```bash
bash isaaclab_migration/scripts/a1_himloco.sh view
```

也可以修改环境数量和速度指令：

```bash
bash isaaclab_migration/scripts/a1_himloco.sh view \
  --num_envs 4 --seconds 60 --command 0.5 0 0
```

录制导出策略的真实 Isaac Lab GUI 回放并生成 MP4：

```bash
bash isaaclab_migration/scripts/a1_himloco.sh record --seconds 20
```

录像命令会读取 A1 配置中的地形列数（当前为 20 种地形），使用 128 个并行环境为每种地形选择一个机器人，按地形数量均分总控制步数，并分别跟随录制。默认 4000 个控制步会输出 20 个片段，每段约 200 步；每 10 步采一帧、960x540、FXAA，结果位于 `outputs/a1/.../video/terrain_00/playback.mp4` 至 `terrain_19/playback.mp4`。`--seconds N` 会按 50 Hz 换算总控制步数，`--steps N` 可直接指定总步数。需要先安装 `ffmpeg`。如需一次执行训练、导出和录像：

```bash
bash isaaclab_migration/scripts/a1_himloco.sh all
```

脚本支持通过 `TASK_PYTHON`、`A1_LOG_DIR`、`A1_OUTPUT_DIR`、`A1_CHECKPOINT` 和 `A1_POLICY` 覆盖 Python、日志、检查点和输出路径。

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
