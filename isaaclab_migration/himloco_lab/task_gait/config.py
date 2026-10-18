"""Independent, serializable task profiles; legacy configuration classes are untouched."""
from types import SimpleNamespace
import xml.etree.ElementTree as ET

PHASES = ('basic', 'precision', 'robust')
TASK = 'mybot_v3_task_gait'
FRAME_SIZE = 49  # legacy 45 + sin/cos phase + activation + standby mode


def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [namespace(v) for v in value]
    return value


def restore_config(data):
    """Restore attribute sections while preserving name-to-number mapping fields."""
    cfg = namespace(data)
    cfg.init_state.default_joint_angles = dict(data['init_state']['default_joint_angles'])
    for key in ('stiffness', 'damping'):
        setattr(cfg.control, key, dict(data['control'][key]))
    if isinstance(data['asset']['armature'], dict):
        cfg.asset.armature = dict(data['asset']['armature'])
    return cfg


def build_config(num_envs=2048, phase='basic'):
    from ..config import load_config, class_to_dict
    if phase not in PHASES:
        raise ValueError(f'Unknown gait phase: {phase}')
    original, train = load_config('mybot_v3', num_envs, 3)
    cfg = restore_config(class_to_dict(original))
    cfg.env.num_one_step_observations = FRAME_SIZE
    cfg.env.num_observations = FRAME_SIZE * 6
    cfg.env.num_one_step_privileged_obs = FRAME_SIZE + 3 + 3 + 187
    cfg.env.num_privileged_obs = cfg.env.num_one_step_privileged_obs
    cfg.env.episode_length_s = 35.0  # six 5-second sequence segments must fit
    cfg.commands.heading_command = False
    cfg.commands.curriculum = False  # explicit command profiles, no legacy curriculum
    cfg.commands.resampling_time = 5.0
    cfg.commands.ranges.lin_vel_x = [-0.5, 0.5]
    cfg.commands.ranges.lin_vel_y = [-0.2, 0.2]
    cfg.commands.ranges.ang_vel_yaw = [-0.6, 0.6]
    cfg.normalization.clip_actions = 5.0
    cfg.gait = namespace(dict(
        version=1, phase=phase, frequency=2.0, duty_factor=0.65,
        foot_names=['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'],
        phase_offsets=[0.0, 0.5, 0.5, 0.0],
        swing_height=0.045, transition_seconds=0.4, command_tau=0.15,
        zero_epsilon=1e-6, contact_blend=0.025,
        sequence_fraction=0.25,
        low_linear=[0.015, 0.10], low_yaw=[0.03, 0.20],
        # stand, standby-step, low translation, pure yaw, normal, mixed
        sample_weights=([0.15, 0.05, 0.15, 0.10, 0.45, 0.10]
                        if phase == 'basic' else [0.15, 0.10, 0.25, 0.20, 0.15, 0.15]),
        linear_error_floor=0.035, linear_error_relative=0.35,
        yaw_error_floor=0.06, yaw_error_relative=0.35,
        position_sigma=0.06, clearance_sigma=0.025,
        force_scale=40.0, slip_scale=0.15,
        # URDF effort is only a numerical normalization, NOT continuous torque.
        continuous_torque_nm=None,
        terrain_kind='gentle' if phase == 'robust' else 'plane',
        ablation='full',
    ))
    from ..config import asset_contract
    tree = ET.parse(asset_contract(cfg)[0]).getroot()
    radii = [float(tree.find(f"link[@name='{name}']/collision/geometry/sphere").get('radius'))
             for name in cfg.gait.foot_names]
    if len(set(radii)) != 1:
        raise ValueError('Task gait expects equal spherical foot collision radii')
    cfg.gait.foot_radius = radii[0]
    cfg.terrain.mesh_type = 'trimesh' if phase == 'robust' else 'plane'
    cfg.terrain.selected = None
    cfg.terrain.curriculum = phase == 'robust'
    cfg.terrain.num_rows = 5
    cfg.terrain.num_cols = 10
    cfg.terrain.terrain_length = cfg.terrain.terrain_width = 6.0
    cfg.terrain.border_size = 3.0
    cfg.terrain.max_init_terrain_level = 1
    cfg.terrain.terrain_proportions = [0.6, 0.2, 0.2, 0.0, 0.0]
    # Moderate dynamics randomization; do not inherit the stair task's disturbances.
    for key in vars(cfg.domain_rand):
        if isinstance(getattr(cfg.domain_rand, key), bool):
            setattr(cfg.domain_rand, key, False)
    cfg.domain_rand.randomize_friction = True
    cfg.domain_rand.friction_range = [0.7, 1.2]
    cfg.domain_rand.randomize_kp = phase == 'robust'
    cfg.domain_rand.randomize_kd = phase == 'robust'
    cfg.domain_rand.kp_range = cfg.domain_rand.kd_range = [0.9, 1.1]
    cfg.noise.noise_level = 0.5
    # Avoid competing default-pose / rear-thigh objectives; retain mild regularization.
    cfg.rewards.scales.rear_thigh_pos = 0.0
    cfg.rewards.scales.dof_pos = -0.02
    cfg.rewards.scales.foot_clearance = 0.0
    cfg.rewards.scales.tracking_lin_vel = 2.0
    cfg.rewards.scales.tracking_ang_vel = 1.0
    cfg.rewards.scales.gait_contact = -0.3
    cfg.rewards.scales.gait_position = -0.15
    cfg.rewards.scales.gait_clearance = -0.2
    cfg.rewards.scales.gait_stand = -0.1
    cfg.rewards.scales.gait_load = -0.15
    cfg.rewards.scales.dof_pos_limits = -1.0
    cfg.rewards.scales.collision = -0.2
    train['runner']['experiment_name'] = TASK
    train['runner']['max_iterations'] = 2000  # budget, not a convergence claim
    train['runner']['save_interval'] = 200
    return cfg, train


def set_ablation(cfg, kind):
    """All variants share commands, dimensions, load objective and diagnostics."""
    if kind not in ('baseline', 'timing', 'full'):
        raise ValueError(kind)
    cfg.gait.ablation = kind
    cfg.rewards.scales.gait_contact = -0.3
    cfg.rewards.scales.gait_clearance = -0.2
    cfg.rewards.scales.gait_position = -0.15
    if kind == 'baseline':
        cfg.rewards.scales.gait_contact = 0.0
        cfg.rewards.scales.gait_clearance = 0.0
    if kind != 'full':
        cfg.rewards.scales.gait_position = 0.0
