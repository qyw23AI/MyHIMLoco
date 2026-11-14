"""Use HIMLoco paper rewards with the released A1 implementation elsewhere."""

PAPER_A1_REWARD_SCALES = {
    'termination': 0.0,
    'tracking_lin_vel': 1.0,
    'tracking_ang_vel': 0.5,
    'lin_vel_z': -2.0,
    'ang_vel_xy': -0.05,
    'orientation': -0.2,
    'dof_acc': -2.5e-7,
    'joint_power': -2e-5,
    'base_height': -1.0,
    'foot_clearance': -0.01,
    'action_rate': -0.01,
    'smoothness': -0.01,
    'feet_air_time': 0.0,
    'collision': 0.0,
    'feet_stumble': 0.0,
    'stand_still': 0.0,
    'torques': 0.0,
    'dof_vel': 0.0,
    'dof_pos_limits': 0.0,
    'dof_vel_limits': 0.0,
    'torque_limits': 0.0,
}


def apply_paper_a1(cfg, train, rollout_steps):
    if rollout_steps not in (100, 200):
        raise ValueError('A1 rollout length must be 100 or 200')
    cfg.paper_a1_rewards = True
    for name, value in PAPER_A1_REWARD_SCALES.items():
        setattr(cfg.rewards.scales, name, value)
    cfg.rewards.only_positive_rewards = False
    cfg.rewards.tracking_sigma = 0.25
    cfg.rewards.base_height_target = 0.30
    cfg.rewards.clearance_height_target = -0.2
    cfg.rewards.max_contact_force = 100.0
    train['runner']['num_steps_per_env'] = rollout_steps
    return cfg, train
