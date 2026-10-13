import numpy as np
from himloco_lab.legacy.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from himloco_lab.legacy.mybot_v3.mybot_v3_config import MybotV3RoughCfg, MybotV3RoughCfgPPO


class MybotV3BridgeBCfg(MybotV3RoughCfg):
    """ Bridge-B only training config for HIMloco.

    Bridge B: 3 beams (20cm wide), 2 gaps (10cm wide), 20cm high.
    Beams run along-track (X direction), gaps along Y (across-track).
    """
    class env(MybotV3RoughCfg.env):
        num_envs = 4096

    class terrain(MybotV3RoughCfg.terrain):
        mesh_type = None
        num_rows = 10          # 10 difficulty levels for curriculum
        num_cols = 20
        selected = "BarrierTrack"
        max_init_terrain_level = 0
        border_size = 5
        slope_treshold = 20.

        curriculum = True
        horizontal_scale = 0.025
        pad_unavailable_info = True

        measure_heights = True
        measured_points_x = [-1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        BarrierTrack_kwargs = dict(
            options=[
                "bridge_b",
            ],
            n_obstacles_per_track=3,
            randomize_obstacle_order=False,  # only bridge_b, order doesn't matter
            track_width=2.4,
            track_block_length=2.4,
            wall_thickness=(0.04, 0.2),
            wall_height=0.3,
            bridge_b=dict(
                n_beams=3,
                beam_width=0.20,           # 20cm
                gap_width=0.10,            # 10cm
                bridge_height=0.20,        # 20cm
                bridge_mesh_source="procedural",
                rotate_90=False,           # beams run along-track (X)
                ramp_length=0.40,          # approach ramp length
            ),
            add_perlin_noise=True,
            border_perlin_noise=True,
            border_height=0.,
            virtual_terrain=False,
            draw_virtual_terrain=True,
            engaging_next_threshold=1.2,
            engaging_finish_threshold=0.,
            curriculum_perlin=True,
            no_perlin_threshold=0.1,
        )

        TerrainPerlin_kwargs = dict(
            zScale=[0.0, 0.03],
            frequency=10,
        )

    class commands(MybotV3RoughCfg.commands):
        heading_command = False
        resampling_time = 10
        lin_cmd_cutoff = 0.2
        class ranges(MybotV3RoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]   # forward speed range
            lin_vel_y = [0.0, 0.0]     # no lateral movement
            ang_vel_yaw = [0., 0.]     # no turning

    class control(MybotV3RoughCfg.control):
        stiffness = {'joint': 40.0}
        damping = {'joint': 1.0}
        action_scale = 0.25
        torque_limits = 25
        computer_clip_torque = True
        motor_clip_torque = False

    class asset(MybotV3RoughCfg.asset):
        penalize_contacts_on = ["thigh", "calf", "body"]
        terminate_after_contacts_on = ["body"]
        front_hip_names = ["FR_hip_joint", "FL_hip_joint"]
        rear_hip_names = ["RR_hip_joint", "RL_hip_joint"]

    class termination:
        termination_terms = [
            "roll",
            "pitch",
        ]

        roll_kwargs = dict(
            threshold=1.4,
        )
        pitch_kwargs = dict(
            threshold=1.6,
        )

        check_obstacle_conditioned_threshold = True
        timeout_at_border = True

    class domain_rand(MybotV3RoughCfg.domain_rand):
        randomize_com = True
        class com_range:
            x = [-0.05, 0.15]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]

        randomize_motor = True
        leg_motor_strength_range = [0.8, 1.2]

        randomize_base_mass = True
        added_mass_range = [1.0, 3.0]

        randomize_friction = True
        friction_range = [0., 2.]

        init_base_pos_range = dict(
            x=[0.2, 0.6],
            y=[-0.25, 0.25],
        )

        push_robots = True
        max_push_vel_xy = 0.5
        push_interval_s = 2

    class rewards(MybotV3RoughCfg.rewards):
        class scales(MybotV3RoughCfg.rewards.scales):
            # Inherit all scales from MybotV3RoughCfg, override/add bridge-specific ones:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            collision = -0.05

        soft_dof_pos_limit = 1.0
        base_height_target = 0.33

    class curriculum:
        penetrate_depth_threshold_harder = 100
        penetrate_depth_threshold_easier = 200
        no_moveup_when_fall = True


class MybotV3BridgeBCfgPPO(MybotV3RoughCfgPPO):
    class algorithm(MybotV3RoughCfgPPO.algorithm):
        entropy_coef = 0.0
        num_mini_batches = 4
        learning_rate = 1e-3

    class policy(MybotV3RoughCfgPPO.policy):
        mu_activation = "tanh"

    class runner(MybotV3RoughCfgPPO.runner):
        experiment_name = "bridge_b_mybot_v3"
        num_steps_per_env = 24
        resume = False
        run_name = "bridge_b"
        max_iterations = 20000
        save_interval = 1000
        log_interval = 100
