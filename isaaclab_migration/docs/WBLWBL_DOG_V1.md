# WBLWBL Dog V1：正常行走与低高度训练

机器人参数定义在 [正常行走配置](../himloco_lab/legacy/wblwbl_dog_v1/wblwbl_dog_v1_config.py)。同目录的 [低趴配置](../himloco_lab/legacy/wblwbl_dog_v1/wblwbl_dog_v1_crouch_config.py)复用这些参数，只覆盖 `rewards.base_height_target` 和实验名称。

| 参数                                |     `wblwbl_dog_v1` | `wblwbl_dog_v1_crouch` |
| ----------------------------------- | --------------------: | -----------------------: |
| `init_state.pos`（m）             |  `[0.0, 0.0, 0.40]` |     `[0.0, 0.0, 0.40]` |
| `rewards.base_height_target`（m） | 0.33（沿用 Mybot V3） |                     0.15 |
| `rewards.scales.base_height`      |                  -1.0 |                     -1.0 |
| PD `stiffness` / `damping`      |                40 / 1 |                   40 / 1 |
| `control.action_scale`（rad）     |                  0.25 |                     0.25 |

控制器、指令、地形、奖励权重和 PPO 的起始数值与 Mybot V3 对齐，之后可独立调整。Stage 3 走平地；Stage 4 使用楼梯地形。两个 WBLWBL 任务都按标定站姿重置关节和根速度，观测与其余动力学随机化使用通用基类的默认值。

## 默认关节角怎么选

`default_joint_angles` 同时决定重置时的关节位置、关节位置观测的零点，以及零动作下 PD 控制器的目标角。角度单位为 rad，每行按 `hip, thigh, calf` 排列，两套任务共用：

| 腿      | 默认关节角            |
| ------- | --------------------- |
| FL 左前 | `(0.1, 0.8, -1.5)`  |
| FR 右前 | `(-0.1, 0.8, -1.5)` |
| RL 左后 | `(0.1, 1.0, -1.5)`  |
| RR 右后 | `(-0.1, 1.0, -1.5)` |

1. 上一版先指定四个脚球在地面上的落点，再用 URDF 的连杆长度、关节轴和足端偏移反求每条腿的角度。左右髋轴同为 +X，所以左右外展取反号；右前腿的几何偏移和左前略有差异，因此当时分别求解，得到了三位小数。
2. 这次以 A1 的一位小数站姿为候选，把角度带入 **WBLWBL 的 URDF** 正向计算。脚底高度取碰撞球球心的世界 Z 减球半径。在根高度 `0.40 m` 时，四个脚底约在地面以上 `1–11 mm`；关节角均在限位内，因此保留了这组一位小数角度。
3. 再用 Mybot V3 的 PD 参数在 GPU PhysX 中施加重力。两个任务各跑了 4 个环境、500 个控制步（10 秒），零动作、零重置；只有四脚触地。机身最低 `0.3424 m`，最终平均 `0.3575 m`，机身 +Z 与世界 +Z 的最小点积为 `0.9983`。两套任务的测试结果相同，因为奖励目标不会在零动作测试里改写 PD 目标。

训练侧动作顺序为 RR → RL → FR → FL，通过关节名称映射到 Isaac Lab。导入时解析本地 [URDF](../../legged_gym/resources/robots/wblwbl_dog_v1/urdf/wblwbl_dog.urdf) 的 `package://` 网格路径，保留四个足端碰撞体。导入副本和 USD 位于迁移目录下的 `.cache/wblwbl_dog`；修改 STL 后可清除对应的 USD 转换缓存再启动。

## 复测和训练

在仓库根目录运行：

```bash
conda activate himloco

# Stage 1：查看重置站姿、触地 link 和承重后的高度
python isaaclab_migration/scripts/train.py --task wblwbl_dog_v1 \
  --stage 1 --num_envs 4 --steps 500 --headless

# Stage 3：在平地训练降低机身高度
python isaaclab_migration/scripts/train.py --task wblwbl_dog_v1_crouch \
  --stage 3 --num_envs 512 --headless

python isaaclab_migration/scripts/train.py --task wblwbl_dog_v1_crouch \
  --stage 4 --headless
```

正常行走训练用同一条 Stage 3 命令，将任务名换成 `wblwbl_dog_v1`。导出和回放也使用训练时的任务名。

两个任务各完成了 32 环境、每轮 24 步、更新 3 次的 Stage 3 短训，验证了训练入口。

`pos[2]` 是出生和重置高度；`base_height_target` 是策略训练时希望达到的离地高度。当前低高度任务从 `0.40 m` 站姿重置，由策略学着下蹲和移动。零动作测试只验证站姿。训练后用回放检查行走中实际达到的高度；若长期维持站姿，再考虑调大高度惩罚权重的绝对值，或减弱默认姿态约束。

当前两个任务都在 `base_link` 触地时终止。水平机身的碰撞盒在根高度约 `0.0705 m` 时触地，倾斜时可能更早。若以后需要腹部贴地爬行，应单独设置允许接地的部位和翻倒终止条件。

## 查看训练、导出和回放

在仓库根目录打开 Bash，先执行 `conda activate himloco`。默认训练日志和检查点在 `isaaclab_migration/logs/<任务>/<run>/`；训练结束自动导出的策略在 `isaaclab_migration/OUTPUT/<任务>/<run>/policy.pt`（`OUTPUT` 大写）。

### TensorBoard

```bash
find isaaclab_migration/logs -type f -path '*/wblwbl_dog_v1*/*' \
  \( -name 'events.out.tfevents.*' -o -name 'model_*.pt' \) | sort -V

tensorboard --logdir isaaclab_migration/logs --host 127.0.0.1 --port 6006
```

浏览器打开 [TensorBoard](http://localhost:6006)。可以把 `--logdir` 改成某个任务或 run 的目录。在 Scalars 中查看 `Train/mean_reward`、`Train/mean_episode_length`、`Episode/rew_tracking_lin_vel` 和 `Episode/rew_base_height`。高度奖励不等于实际机身高度，后者可通过回放的 CSV 查看。

### 导出指定 checkpoint

以下以低趴任务的 Stage 4 检查点为例。换成正常行走时，把任务名改为 `wblwbl_dog_v1`，并选对应的 run 和 checkpoint。

```bash
task=wblwbl_dog_v1_crouch
stage=4
run=stage4_20261112_184620
checkpoint=model_3200.pt
out="isaaclab_migration/OUTPUT/$task/$run/${checkpoint%.pt}"

python isaaclab_migration/scripts/export_policy.py \
  --task "$task" \
  --checkpoint "isaaclab_migration/logs/$task/$run/$checkpoint" \
  --output "$out/policy.pt"
```

导出只需要 CPU。保留 `policy.pt`（TorchScript 推理模型）和同目录的 `policy.json`（观测、关节顺序和源 checkpoint）。恢复训练使用原来的 `model_*.pt`。若已有自动导出的策略，可直接把 `out` 设为其所在目录。

### 直接加载训练检查点回放

查看训练中保存的 Stage 3/4 HIM 检查点时，直接传 `--checkpoint` 即可，无需先导出 `policy.pt`：

```bash

python isaaclab_migration/scripts/play.py \
  --task wblwbl_dog_v1_crouch --stage 4 \
  --checkpoint isaaclab_migration/logs/wblwbl_dog_v1_crouch/stage4_20261112_184620/model_3200.pt \
  --num_envs 1 --seconds 60 --command 0.6 0 0 \
   --camera_env 0
```

换成其他检查点时，使用对应的任务名和回放 Stage（3 为平地，4 为楼梯）。`--checkpoint` 和 `--policy` 二选一；两种方式都支持 `--state_csv` 和跟拍参数。

### 测量正常行走和低趴策略的机身高度

下面用各自训练目录里的第 4700 次检查点，在 Stage 4 下回放。两次都给 `0.2 m/s` 的前向指令，这是这两个 run 保存的训练配置中前向速度范围的上限。`--steps 3000` 对应 60 秒仿真时间。`--state_all_envs` 将每个环境写入 CSV，`--camera_env 0` 只控制视角。

```bash
measure_dir=isaaclab_migration/OUTPUT/wblwbl_height_4700

python isaaclab_migration/scripts/play.py \
  --task wblwbl_dog_v1 --stage 4 \
  --checkpoint isaaclab_migration/logs/wblwbl_dog_v1/stage4_20261113_011548/model_latest_4700.pt \
  --num_envs 10 --steps 3000 --command 0.2 0 0 --follow_camera --camera_env 0 \
  --state_csv "$measure_dir/normal.csv" --state_all_envs

python isaaclab_migration/scripts/play.py \
  --task wblwbl_dog_v1_crouch --stage 4 \
  --checkpoint isaaclab_migration/logs/wblwbl_dog_v1_crouch/stage4_20261113_011157/model_latest_4700.pt \
  --num_envs 1 --steps 3000 --command 0.2 0 0 --follow_camera --camera_env 0 \
  --state_csv "$measure_dir/crouch.cs
```

`base_height_m` 与训练的高度奖励使用同一种测量方式，取机身根节点到附近地形采样点的平均垂直距离。`height_m` 是根节点的世界 Z。如果要对比平地步态，把两条命令的 `--stage` 都改为 3；如果要比较相同数量的楼梯环境，把两条命令的 `--num_envs` 设成同一个值。

下面按环境统计向前行走时的高度（厘米）。示例跳过启动和每次重置后的 2 秒，只纳入实际前向速度超过 `0.05 m/s` 的样本。如果没有样本符合条件，说明该环境下没有测到符合此口径的行走高度。

```bash
python - "$measure_dir/normal.csv" "$measure_dir/crouch.csv" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

for path in sys.argv[1:]:
    heights = defaultdict(list)
    last_reset = defaultdict(float)
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            env_id = int(row["env_id"])
            samples = heights[env_id]
            elapsed = float(row["time_s"])
            if row["reset"] == "True":
                last_reset[env_id] = elapsed
            elif elapsed - last_reset[env_id] >= 2 and float(row["velocity_x"]) > 0.05:
                samples.append(100 * float(row["base_height_m"]))
    print(path)
    for env_id, samples in sorted(heights.items()):
        if samples:
            print(f"  env {env_id}: mean={statistics.mean(samples):.2f} cm, "
                  f"median={statistics.median(samples):.2f} cm, samples={len(samples)}")
        else:
            print(f"  env {env_id}: no walking samples")
PY
```

### 回放已导出的策略

沿用上面的变量。Stage 3 是平地，Stage 4 是楼梯。 `--command` 依次是前向速度、侧向速度（m/s）和偏航角速度（rad/s）。

```bash
python isaaclab_migration/scripts/play.py \
  --task "$task" --stage "$stage" --policy "$out/policy.pt" \
  --num_envs 1 --seconds 30 --command 0.2 0 0 \
  --follow_camera --camera_env 0 --state_csv "$out/state.csv"
```

去掉 `--follow_camera` 可用鼠标自由查看；`--seconds 0` 会持续回放直到关闭窗口。CSV 包含 `height_m`、`velocity_x` 和 `reset`。楼梯上的 `height_m` 是根节点的世界 Z，会随台阶变化；对比两套策略的机身高度时，可统一用 Stage 3 平地和相同指令回放。

### 为每个策略录制和裁剪跟拍视频

录制需要 Isaac Sim 图形窗口和带 `libx264` 的 FFmpeg。函数依次接收任务名、回放 Stage、策略文件和一个新输出目录。每个策略录制 10 秒仿真，并另存第 2–8 秒的 6 秒片段。

```bash
record_clip() (
  set -e
  local task="$1" stage="$2" policy="$3" clip="$4"
  mkdir -p "$(dirname "$clip")"
  mkdir "$clip"  # 必须是新目录，避免混入旧帧

  python isaaclab_migration/scripts/play.py \
    --task "$task" --stage "$stage" --policy "$policy" \
    --num_envs 1 --command 0.2 0 0 --steps 500 --no-realtime \
    --follow_camera --camera_env 0 --viewer_resolution 1280 720 \
    --capture_dir "$clip/frames" --capture_every 2 \
    --state_csv "$clip/state.csv"

  shopt -s nullglob
  local frames=("$clip"/frames/frame_*.png)
  if [ "${#frames[@]}" -ne 250 ]; then
    printf '预期 250 帧，实际找到 %s 帧：%s\n' "${#frames[@]}" "$clip/frames" >&2
    exit 1
  fi

  ffmpeg -n -framerate 25 -pattern_type glob -i "$clip/frames/frame_*.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart "$clip/full.mp4"
  ffmpeg -n -i "$clip/full.mp4" -ss 2 -t 6 \
    -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart "$clip/follow_02s_08s.mp4"
)

record_clip "$task" "$stage" "$out/policy.pt" "$out/follow_clip"

# 换一个策略后，用对应任务、回放 Stage 和独立目录再执行一次：
# record_clip wblwbl_dog_v1 3 /path/to/normal/policy.pt /path/to/normal/follow_clip
```

控制频率为 50 Hz，每 2 步取一帧，所以按 25 fps 编码。调整录制时长时，同步修改 `--steps` 和预期帧数；要从完整视频另剪一段，只需改第二条 FFmpeg 命令的 `-ss`（起点）、`-t`（长度）和输出文件名。
