# GettingHimLoco 最近三次提交同步记录

日期: 2026-05-16

## 源提交概览

- 71772db 参数比较好的一版
  - 调整 mybot_v3 奖励参数: base_height, foot_clearance, smoothness, feet_air_time, rear_thigh_pos, dof_pos 权重与目标高度
- b759d7b 记录训练参数到 tensorboard
  - 在训练开始时将 env/train 配置写入 TensorBoard 文本区
- 51985c1 add pos reward and weight
  - 增加关节位置惩罚奖励与权重配置

## 同步到 MyHIMLoco 的修改

- legged_gym
  - legged_gym/envs/base/legged_robot.py: 增加 dof_pos 权重计算与 `_reward_dof_pos`
  - legged_gym/envs/base/legged_robot_config.py: 增加 dof_pos 奖励项与权重参数
  - legged_gym/envs/mybot_v3/mybot_v3_config.py: 更新奖励系数与目标高度，加入 dof_pos 权重
- rsl_rl
  - rsl_rl/rsl_rl/runners/him_on_policy_runner.py: 训练时记录 env/train 配置到 TensorBoard
