# 快速开始：Mybot 近期改动（近一月）

## 范围

- 时间窗口：最近 1 个月（提交：630e7a6、8851226、f2a04a4、190ae37）
- 聚焦：mybot_v2 与 mybot_v2_1 两个机器人版本，以及它们使用的共享训练默认值
- 说明：以下条目均附可跳转的代码链接，链接指向当前代码位置，便于审查现状

## 新任务：mybot_v2_1

- 在环境注册中新增任务名 `mybot_v2_1`（见 [legged_gym/legged_gym/envs/__init__.py](legged_gym/legged_gym/envs/__init__.py#L31-L51)）。
- 初始高度 0.43 与后腿大腿默认角 1.0 rad（见 [legged_gym/legged_gym/envs/mybot_v2_1/mybot_v2_1_config.py](legged_gym/legged_gym/envs/mybot_v2_1/mybot_v2_1_config.py#L33-L51)）。
- 指令范围：lin_vel_x/lin_vel_y 为 +/-0.2 m/s，偏航 +/-3.14（见 [legged_gym/legged_gym/envs/mybot_v2_1/mybot_v2_1_config.py](legged_gym/legged_gym/envs/mybot_v2_1/mybot_v2_1_config.py#L72-L82)）。
- 奖励权重与目标高度（tracking_lin_vel、base_height、smoothness、feet_air_time、hip_pos、rear_thigh_pos、base_height_target）（见 [legged_gym/legged_gym/envs/mybot_v2_1/mybot_v2_1_config.py](legged_gym/legged_gym/envs/mybot_v2_1/mybot_v2_1_config.py#L99-L132)）。
- 实验名 rough_mybot_v2_1（见 [legged_gym/legged_gym/envs/mybot_v2_1/mybot_v2_1_config.py](legged_gym/legged_gym/envs/mybot_v2_1/mybot_v2_1_config.py#L134-L139)）。

## mybot_v2 参数更新

- 初始高度 0.43 与后腿大腿默认角 1.0 rad（见 [legged_gym/legged_gym/envs/mybot_v2/mybot_v2_config.py](legged_gym/legged_gym/envs/mybot_v2/mybot_v2_config.py#L34-L50)）。
- 小腿刚度/阻尼 40/1（见 [legged_gym/legged_gym/envs/mybot_v2/mybot_v2_config.py](legged_gym/legged_gym/envs/mybot_v2/mybot_v2_config.py#L53-L65)）。
- 小腿转动惯量 armature 0.01（见 [legged_gym/legged_gym/envs/mybot_v2/mybot_v2_config.py](legged_gym/legged_gym/envs/mybot_v2/mybot_v2_config.py#L93-L97)）。
- 奖励调整：foot_clearance -0.05，base_height_target 0.40（见 [legged_gym/legged_gym/envs/mybot_v2/mybot_v2_config.py](legged_gym/legged_gym/envs/mybot_v2/mybot_v2_config.py#L99-L130)）。

## 奖励项新增（共享）

- 在 LeggedRobot 中新增奖励函数 `hip_pos` 与 `rear_thigh_pos`（见 [legged_gym/legged_gym/envs/base/legged_robot.py](legged_gym/legged_gym/envs/base/legged_robot.py#L1235-L1245)）。
- 在 mybot_v2_1 奖励权重中启用 `hip_pos` 与 `rear_thigh_pos`（见 [legged_gym/legged_gym/envs/mybot_v2_1/mybot_v2_1_config.py](legged_gym/legged_gym/envs/mybot_v2_1/mybot_v2_1_config.py#L99-L123)）。

## 训练/配置默认值更新（全局）

- LeggedRobotCfg：num_envs 4096（见 [legged_gym/legged_gym/envs/base/legged_robot_config.py](legged_gym/legged_gym/envs/base/legged_robot_config.py#L33-L43)）。
- 资产合并：collapse_fixed_joints = false（见 [legged_gym/legged_gym/envs/base/legged_robot_config.py](legged_gym/legged_gym/envs/base/legged_robot_config.py#L102-L110)）。
- 域随机化范围扩展（payload/com/link/friction/restitution/kp/kd 等）（见 [legged_gym/legged_gym/envs/base/legged_robot_config.py](legged_gym/legged_gym/envs/base/legged_robot_config.py#L124-L160)）。
- PPO 默认值：learning_rate 1e-3，max_grad_norm 10（见 [legged_gym/legged_gym/envs/base/legged_robot_config.py](legged_gym/legged_gym/envs/base/legged_robot_config.py#L248-L261)）。
- Runner 默认值：num_steps_per_env 100，max_iterations 5000，save_interval 100（见 [legged_gym/legged_gym/envs/base/legged_robot_config.py](legged_gym/legged_gym/envs/base/legged_robot_config.py#L263-L270)）。
- HIMPPO：返回值使用 mean_estimation_loss（见 [rsl_rl/rsl_rl/algorithms/him_ppo.py](rsl_rl/rsl_rl/algorithms/him_ppo.py#L180-L192)）。
