# 快速步态原因筛查

本实验以恢复已有 basic 模型为目标，不把短程微调等同于从零训练的因果证明。默认使用同一个 `basic/model_5000.pt`、precision 指令分布、seed=17、1024 环境、100 步 rollout，每组新增 300 次更新，即 30,720,000 条环境交互。各组独立运行，显存不会随组数叠加；它们不会启动原来的 robust 阶段。

## 八组对照

| 组名 | 相对原配置的修改 |
|---|---|
| baseline | 奖励保持原样，控制继续训练本身的影响 |
| height | 高度误差除以 0.04 m 后平方，权重 -0.25，替换原高度项 |
| hip | 髋关节相对默认角度的容许偏差：直行 0.15 rad，侧移/yaw 平滑放宽到 0.30 rad；超出后按 0.15 rad 归一化平方，权重 -0.3；距各关节限位小于 0.10 rad 开始惩罚，权重 -0.1 |
| gait | 空间误差使用归一化平方代替饱和成本；足端高度包括支撑期，权重 -0.5；XY 位置权重 -0.3；接触/滑动成本从 -0.3 提高为 -1 |
| combined | 合并 height、hip、gait |
| combined_tracking1 | 在 combined 基础上线速度跟踪权重从 2 改成 1 |
| combined_tracking1p5 | 在 combined 基础上线速度跟踪权重从 2 改成 1.5 |
| no_constraints | 将机身高度目标、新增 posture 项及五项 WTW 风格 gait 奖励全部关闭；保留速度跟踪、一般关节正则化和机械关节限位保护 |

`no_constraints` 从相同检查点开始微调，能检验这些奖励约束在后续更新中是否继续把策略推向异常姿态。为兼容该检查点的策略网络，它保留 gait phase/activation 的观测和参考控制器结构；因此不能解释成从策略中完全移除了 WTW 步态输入。检查点自身也已经包含此前训练学到的行为，300 次更新不能抹除这段历史。

hip 组同时改变髋关节姿态与限位余量，因此只定位这一约束组是否有效；若有效，再拆开这两个因素。gait 同时改变多个足端约束，也需在有效后细分。权重都是本次诊断候选值，未经过硬件标定。

固定评估覆盖 hold、standby、低速 0.03 m/s、yaw 0.06 rad/s、前后 0.5 m/s，以及起步/微调/停止/转向/原地踏步/停止序列。独立指令各 8 秒，序列保留 30 秒以容纳 6 个 5 秒片段。每个最终检查点生成 8 秒、4 环境、25 fps 的前进视频；录像设 5 分钟超时，防止回放长期占显存。平地策略不能只因 survival=100% 就通过验收。

```bash
bash isaaclab_migration/scripts/posture_experiments.sh \
  --checkpoint /home/qyw/MyHIMLoco/isaaclab_migration/logs/task_gait_full_20261019/basic/model_5000.pt \
  --output /home/qyw/MyHIMLoco/isaaclab_migration/logs/posture_screen_NEW_RUN \
  --num_envs 1024 --updates 300 --rollout_steps 100 --seed 17
```

输出目录必须是新的；已有计划会被拒绝覆盖。单组也可通过 `--arms height` 指定。运行期间创建 `STOP_REQUESTED` 文件，可在当前训练/评估/录像操作结束后停止；直接终止工作流进程会同时终止它当前启动的子进程。不会删除检查点。

## 日志与 TensorBoard

每组目录包含 `train/config.json`（完整生效配置）、`train/run_manifest.json`（实验名、权重、命令行、来源检查点与 URDF SHA256、Git 提交及差异摘要）、训练日志、检查点、验证结果；`evaluation/evaluation.json` 为固定评估结果。根目录含 `experiment_plan.json`、`status.json`、`comparison.json`、`analysis.json`（新值减对照值的逐命令差异）。

```bash
/home/qyw/miniconda3/envs/himloco/bin/tensorboard \
  --logdir /home/qyw/MyHIMLoco/isaaclab_migration/logs/posture_screen_NEW_RUN \
  --port 6006
```

TensorBoard 的 **Text** 查看 `Run/effective_configuration`、`Run/experiment_and_provenance`；**Scalars** 查看 `Config/reward_scale/*` 与 `Posture/*`。Posture 的步数从本次运行的零开始，每 10 次更新写一次；Loss/Episode 沿用检查点累计更新数。训练姿态是累计统计，包含多种指令，必须结合固定命令评估判断。

重点看：逐命令高度误差、髋关节均值、逐关节贴限位比例、逐腿摆动期触地率和参考顶点附近的实际离地高度；同时核对速度误差、生存率、滑动、倾斜与扭矩。没有摆动样本时返回 null，不能当成零误差。

## 如何判断与延长

300 次更新只筛查方向。高度组修复趴低、髋关节组减少贴限位、步态组改善摆腿，分别构成相应因素的证据；必须同时检查跟踪能力是否退化。`combined_tracking1`、`combined_tracking1p5` 与 `combined` 比较，分别验证速度权重 1.0、1.5、2.0，其他奖励配置相同。若合并组有效而单项无效，提示约束之间有交互，不应强行归因某一项。

筛查后按证据将对照与有希望的 1–2 组一起延长至新增 800–1000 次更新：从各自 300 次最终模型继续增加 500–700 次，省略 `--posture_experiment` 可精确保留已存配置。延长也必须使用新日志目录。单种子改善不是唯一主因的证明；无改善也可能是旧策略锁定，需要第二种子或从共同早期状态再验证。不要跨组用上一组结果初始化下一组。
