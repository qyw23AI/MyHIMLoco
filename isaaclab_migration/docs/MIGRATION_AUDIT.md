# 实际代码审计与迁移边界

## 代码与训练流程

本次审计以根目录 `legged_gym` 和 `rsl_rl` 为准。`GettingHimLoco` 是另一个代码副本；`IsaacGymEnvs/IsaacLab` 是已经安装使用的官方 Lab checkout，两者均未作为原版 HIMLoco 的替代依据。

- 原入口 `legged_gym/legged_gym/scripts/train.py` 通过 `task_registry` 构造任务和训练器。
- `envs/__init__.py` 注册 A1、Go1、AlienGo、mybot、v2、v2_1、v3、v3_with_arm、v3_bridge_a/b。机械臂任务有重复注册，但实际配置相同。
- `BaseTask` 持有 Gym sim/viewer 及张量缓冲；`LeggedRobot` 实现延迟动作、显式力矩控制、回调、奖励、终止、重置和历史观测。
- `LeggedRobotFieldMixin` 加入桥梁／BarrierTrack 终止与指标、目标指令和额外奖励。文件还包含未被已注册任务使用的可选功能；部分函数原本依赖基础类中不存在的方法，不能把这些声明视为已运行功能。
- `utils/terrain/terrain.py` 生成整张整数高度图和三角网格；BarrierTrack 组合桥梁、平台、侧墙与 Perlin 噪声，同时保存地形查询信息。
- `HIMOnPolicyRunner` 收集七返回值环境输出，将终止环境的 `next_critic_obs` 替换为重置前的终止观测，然后执行 HIMPPO。
- HIMPPO 保留 PPO clipped surrogate、clipped value loss、entropy、GAE、自适应 KL 学习率，以及独立 HIMEstimator 的更新。
- HIMEstimator 保留历史编码器、下一步目标编码器、速度回归、32 个原型、Sinkhorn 和 SwAV swapped loss，使用自己的 Adam 优化器。
- 原 Runner 的 TensorBoard 配置记录、滚动检查点、永久检查点及 estimator optimizer 保存／恢复直接复用。迁移没有改写任何原算法文件。

`isaacgym_api_inventory.txt` 是由 AST 生成的逐行 API 清单，包含导入、仿真/资产/状态/绘图调用以及直接导入的 Gym 张量数学函数。当前共 226 个位置。`source_manifest.json` 保存 40 个原源码文件的 SHA256，包含原算法全套文件。

## 独立实现组织

| 文件／目录 | 职责 |
| --- | --- |
| `himloco_lab/legacy` | 实际任务配置的独立副本，保留继承关系 |
| `logic.py` | 原 LeggedRobot 的张量公式，去除 Gym 仿真操作 |
| `field.py` | 原 Field 的数值逻辑，去除 Gym 绘图／资产构造 |
| `core.py` | 原生命周期与 Lab 资产写入的桥接 |
| `env_cfg.py` | 官方 DirectRLEnvCfg、ArticulationCfg、URDF/MJCF 转换、显式 effort actuator |
| `env.py` | 官方 DirectRLEnv 场景、物理步进、ContactSensor、状态与材料／质量随机化 |
| `adapter.py` | 原 RSL-RL 五／七返回值协议及原 reset 方式 |
| `terrain` | 原桥梁／Perlin 数值生成器与 Lab 网格生成、USD 碰撞创建 |
| `export.py` | 原 Encoder + Actor 的 TorchScript 导出及回读一致性验证 |
| `visualization.py` | 已安装 Isaac Sim 官方 viewport capture API |
| `scripts` | 环境检查、训练、独立导出、源码审计 |
| `tests` | CPU 原代码对照与真实 GPU 重置／关节映射测试 |

迁移运行时无 `isaacgym`、`legged_gym` 导入；原机器人资源按绝对路径读取。RSL-RL 通过明确插入根目录本地包路径复用，避免环境内其他版本覆盖。

## 不可变接口契约

针对实际 mybot_v3 配置：

| 项目 | 原代码与迁移实现 |
| --- | --- |
| 动作 | 12 个腿部关节 |
| 关节顺序 | FL hip/thigh/calf，FR hip/thigh/calf，RL hip/thigh/calf，RR hip/thigh/calf |
| Lab 原生顺序 | 四个 hip，四个 thigh，四个 calf |
| Policy → Lab | `[0,4,8,1,5,9,2,6,10,3,7,11]`，按名称运行时校验 |
| 单帧 actor | 45：command 3、angular velocity 3、gravity 3、joint position 12、joint velocity 12、action 12 |
| 历史 | 6 帧，270 维，最新帧在最前 |
| critic | 238：上述 45、线速度 3、外力 3、高度采样 187 |
| 物理周期 | 0.005 s，200 Hz |
| decimation | 4 |
| 策略周期 | 0.020 s，50 Hz |
| 动作缩放 | 0.25，hip_reduction 1.0 |
| PD | 每个物理步显式计算力矩，Lab actuator 的隐式刚度／阻尼为 0 |
| 奖励 | 原函数及权重，统一乘策略 dt；正奖励裁剪及终止奖励顺序保留 |
| 超时 | `episode_length > max_episode_length`，保留原严格大于条件 |
| 四元数 | 原公式 xyzw，Lab 边界显式转换 wxyz |
| 原 full rollout | 默认任务 100 步，桥梁 24 步 |

机器人关节默认角、限位、armature、碰撞／终止 body pattern 均取实际任务配置与原资产。URDF 解析取得原深度优先 sibling 顺序，再按名称映射到 Lab；不能把 Lab 原生排列直接送入网络。

## 原代码中需要保留或明确解释的行为

1. 历史观测在 reset 中不清空。本实现保留，不强行补零或复制首帧。
2. reset 后即时观测保留原代码的动作与部分重置前缓存值；终止 critic 目标在 reset 之前捕获。GPU 测试单独验证。
3. `_resample_commands` 对大部分环境的 x 指令固定采样 `[-1,1]`，只有前 20% 环境使用配置中的 x 范围；继承该行为。
4. 原控制器采样 `motor_strength_factors`，但不用于 P/V/T 力矩公式。迁移继续保留采样和原公式。
5. 原关节 reset 固定采样 default position 的 `[0.5,1.5]` 倍，未使用声明的开关／范围。本实现保留；只有第一阶段站立验证明确使用确定姿态。
6. `mybot_v2_1` 实际加载 v2 URDF，本实现保留其实际资源引用。
7. `mybot_v3_with_arm` 的臂为固定负载，实际 URDF 仍只有 12 个活动腿关节，未新增机械臂动作。
8. 桥梁设置 `mesh_type=None`；原 `_parse_cfg` 会关闭地形课程。保留这一实际结果，不因任务注释写了 curriculum 而悄悄启用它。楼梯任务的地形课程正常启用。
9. 桥梁 `mu_activation='tanh'` 被原 HIMActorCritic 的 `**kwargs` 忽略。本次继续使用原网络，不插入新的 Tanh。
10. 原 `timeout_at_border` 没有被已注册 Field 终止逻辑执行；不把该声明当成已有功能加入。
11. 原 HIM PPO 返回的第四项是最后一个 mini-batch 的 swap_loss，尽管计算了 mean_swap_loss。本次保留算法文件原样。
12. 原 Runner 的部分中间检查点 iteration／恢复后 ETA 记录存在偏差；最终检查点 iteration 可正常恢复，本次没有顺手修改算法或日志代码。

迁移中的必要修复：

- 将 height query 的 `if env_ids` 改为 `if env_ids is not None`，避免多元素 Tensor 的布尔歧义。
- 每步正确传递 Lab 当前超时状态，防止上一轮的 timeout 留在 infos 影响 PPO bootstrap。
- 桥梁日志延续原 extras 生命周期，满足 Runner 每轮 episode 字段集合一致的约定。
- 每次实例化 BarrierTrack 时复制默认参数字典，避免先建桥梁 A 后建 B 时类级字典相互污染。
- URDF importer 对 collision geometry 使用实例。先将源碰撞子树取消实例，再通过 Lab 公共接口设置原 contact/rest offset，然后进行环境克隆。
- 外力使用当前 Lab `instantaneous_wrench_composer`，一次物理步消费；不使用发出弃用警告的旧 setter。
- 可视化默认使用静态环境视角，自动跟随改为 `--follow_camera` 选项；将 Lab 场景原点元数据同步为原代码的出生／课程原点，避免自由视角对准 Lab 默认中心网格的错误位置。原 reset 位置和算法原点未更改。
- 地形专项验证发现首次 reset 前机器人仍位于 Lab 默认克隆网格，距离课程原点过远会误升级。初始化时先按原 actor 创建规则放置到真实地形原点附近，再执行原 reset；`tests/test_terrain.py` 验证首次等级不变，以及受控升级、降级、出生位置。升级条件和原课程函数未修改。
- 第四阶段增加观测用 `CurriculumMonitor`，统计自然训练的等级增减、出生点迁移、重置／超时、指令范围变化及等级分布。统计不改变动作、奖励、课程决策或随机数。
- Kit 捕获 future 可能早于 PNG 异步写入完成，结束截图等待完整 IEND 数据；异常退出时先关闭环境，再关闭 App。
- MJCF 导入显式启用已安装的官方 importer，将缓存网格目录保留为相对路径。Sim 5.1 还会在空 `worldBody` 上额外创建 articulation；确认其不含任何刚体后仅移除该空节点的 articulation schema。机器人本体的 articulation、17 个刚体和 12 个活动关节全部保留，ContactSensor 路径适配 `/Robot/base/.*` 嵌套层级。

## 地形与物理等价性边界

楼梯、斜坡、离散障碍的整数高度图规则和原 difficulty/列分配保留。三角网格转换调用 Lab 官方 `convert_height_field_to_mesh`；`heightfield` 输入也通过 Lab 支持的静态三角网格导入。高度观测仍使用原整数网格采样规则，不更换为分辨率不同的 ray grid。桥梁 A/B 的原网格顶点／三角形与 CPU 原代码比较完全相同。

材料每 episode 随机化、启动时 payload/其他 link mass/COM 随机化、Kp/Kd、延迟、推搡、外力保留原分布与生效时间。质量改变后按 Lab 官方事件中使用的质量比例调整惯量；新旧导入器的几何惯量重算和接触求解不能承诺数值逐轨迹相同。

MJCF 没有显式声明关节速度限幅，迁移读取导入器结果；原 Gym 默认值的严格数值等价仍需要旧仿真器环境验证。可视化 Gym wireframe debug overlays 未照搬旧接口；实际 GUI 提供 Lab 渲染场景与视口截图。当前已注册桥梁任务使用 procedural 模式，仓库中未包含默认 STL 课程资产，不应声称缺失的 STL 已恢复。

## 已完成运行验证

以下为本机实际结果，证据文件在 `logs` 中。短训练用于接受迁移，不代表策略已收敛。

| 验证 | 结果 |
| --- | --- |
| CPU 对照 | 11 tests passed；原源文件哈希一致 |
| GPU 重置／终止观测／历史／超时／关节映射 | `docs/sim_contract_result.json` passed |
| 阶段一 | 1024 环境 × 500 控制步，0 reset，最低高度约 0.296 m |
| 阶段二 PPO | 32 环境 × 16 步 rollout × 1 update passed |
| 阶段三完整 HIM | 1024 环境 × 100 步 rollout × 2 updates passed |
| 阶段四楼梯 | 1024 环境 × 100 步 rollout × 2 updates passed |
| 桥梁 A | 1024 环境 × 24 步 rollout × 2 updates passed |
| 桥梁 B | 1024 环境 × 24 步 rollout × 2 updates passed；整卡采样峰值 5450 MiB |
| 机械臂固定负载 | 32 环境 × 16 步 rollout × 1 update passed |
| 恢复 HIM 检查点 | 从 model_2 恢复，再完成 2 updates，最终 model_4；整卡峰值 5524 MiB |
| TorchScript 导出 | model_4 → outputs/restored_policy.pt，回读输出最大误差 0 |
| 可视化完整 HIM 训练 | 1024 环境 × 100 rollout × 100 新增 updates；model_4 → model_104，新增 10,240,000 个环境步；整卡峰值 6377 MiB |
| 可视化策略导出 | `outputs/visual_training/policy.pt`，CPU/GPU 最大绝对差约 1.67e-6，通过 rtol=1e-6/atol=1e-6 |
| 独立 TorchScript GUI 回放 | 4 环境，337 步／6.74 秒仿真，0 reset，有限值校验通过；`outputs/playback_verified/validation.json` |
| 自由视角 | 实际相机移动后连续渲染 20 控制步，变换最大变化 0；`outputs/free_camera_validation.json` |
| MJCF 旧 mybot | 官方 importer 导入后 4 环境 × 100 控制步站立，0 reset，最低高度 0.340 m；`logs/mjcf_standing/validation.json` |
| MJCF 完整 HIM | 32 环境 × 16 rollout × 1 update，PPO/估计/SwAV 更新及 model_1 保存通过；`logs/mjcf_him/validation.json` |
| 地形课程专项 | 真实楼梯网格；首次 reset 等级不变、受控升级／降级及更新后的出生点正确；`docs/terrain_contract_result.json` |
| 楼梯自然课程训练 | 1024 环境 × 100 rollout × 100 新增 updates，model_104 → model_204，整卡峰值 6668 MiB；均值 1.0195 → 1.9102，4172 次等级增大／3260 次减小，最终最高 8，历史快照曾达到 9 |
| 指令课程本轮结果 | 原规则开启，范围保持 [-0.2,0.2]，本轮未触发扩大，未人为放宽升级条件 |
| 楼梯高清独立回放 | 4 环境 × 796 步，15.92 秒仿真，0 reset；1920×1080 / FXAA；第一个机器人平均 x 速度 0.531 m/s，指令 0.6 m/s |

A1/Go1/AlienGo/v2/v2_1 已通过配置、维度、资源存在性校验，尚未各自执行长期 GPU 训练。MJCF 原 Gym 默认关节速度上限的严格物理等价性仍有上述限制。
