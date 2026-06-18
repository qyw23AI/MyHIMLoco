# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class MybotV3WithArmRoughCfg(LeggedRobotCfg):
    class env:
        num_envs = 4096
        num_one_step_observations = 45
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = 45 + 3 + 3 + 187 # additional: base_lin_vel, external_forces, scan_dots
        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = None # select a unique terrain type and pass all arguments ("BarrierTrack", etc.)
        terrain_kwargs = None # Dict of arguments for selected terrain
        BarrierTrack_kwargs = None # Dict of arguments for BarrierTrack terrain
        TerrainPerlin_kwargs = None # Dict of arguments for TerrainPerlin noise
        pad_unavailable_info = False # If True, pad unavailable terrain info with zeros
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.7, 0.1, 0.05, 0.05, 0.1]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.40]  # x,y,z [m]
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "FR_hip_joint": -0.0,
            "RR_hip_joint": -0.0,

            "FL_thigh_joint": 1.0,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,

            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
            "FL_hip": 40.0, "FR_hip": 40.0, "RL_hip": 40.0, "RR_hip": 40.0,
            "FL_thigh": 40.0, "FR_thigh": 40.0, "RL_thigh": 40.0, "RR_thigh": 40.0,
            "FL_calf": 40.0, "FR_calf": 40.0, "RL_calf": 40.0, "RR_calf": 40.0,
        }
        damping = {
            "FL_hip": 1.0, "FR_hip": 1.0, "RL_hip": 1.0, "RR_hip": 1.0,
            "FL_thigh": 1.0, "FR_thigh": 1.0, "RL_thigh": 1.0, "RR_thigh": 1.0,
            "FL_calf": 1.0, "FR_calf": 1.0, "RL_calf": 1.0, "RR_calf": 1.0,
        }
        action_scale = 0.25
        decimation = 4
        hip_reduction = 1.0

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 2.0
        num_commands = 4
        resampling_time = 10.0
        heading_command = True

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.2, 0.2]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-3.14, 3.14]
            heading = [-3.14, 3.14]

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mybot_v3_with_arm/urdf/mybot_v3_with_arm.urdf"
        name = "mybot_v3_with_arm"
        foot_name = "foot"
        penalize_contacts_on = ["calf"]
        terminate_after_contacts_on = ["base", "body"]
        privileged_contacts_on = ["base", "body", "thigh", "calf"]
        self_collisions = 1
        flip_visual_attachments = False
        armature = {
            "FL_hip": 0.01, "FR_hip": 0.01, "RL_hip": 0.01, "RR_hip": 0.01,
            "FL_thigh": 0.01, "FR_thigh": 0.01, "RL_thigh": 0.01, "RR_thigh": 0.01,
            "FL_calf": 0.01, "FR_calf": 0.01, "RL_calf": 0.01, "RR_calf": 0.01,
        }
        
    class domain_rand:
        randomize_payload_mass = True
        payload_mass_range = [-1, 4]

        randomize_com_displacement = True
        com_displacement_range = [-0.2, 0.2]

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        
        randomize_friction = True
        friction_range = [0.1, 3.2]
        
        randomize_restitution = True
        restitution_range = [0., 1.0]
        
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        
        randomize_kp = True
        kp_range = [0.7, 1.3]
        
        randomize_kd = True
        kd_range = [0.7, 1.3]
        
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]
        
        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 6
        
        push_robots = True
        push_interval_s = 12
        max_push_vel_xy = 1.2

        delay = True
        
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            termination = -0.0
            tracking_lin_vel = 2.5
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -1.0
            foot_clearance = -0.02
            action_rate = -0.01
            smoothness = -0.04
            feet_air_time = 0.00
            collision = -1.0
            feet_stumble = -0.0
            stand_still = -0.0
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0
            # hip_pos = -0.0
            rear_thigh_pos = -0.00
            dof_pos = -0.1

        dof_pos_hip_weight = 4.0
        dof_pos_thigh_weight = 2.0
        dof_pos_calf_weight = 1.0
        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.33   
        max_contact_force = 100.0
        clearance_height_target = -0.22


class MybotV3WithArmRoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        
        # training params
        value_loss_coef = 1.0   
        use_clipped_value_loss = True
        clip_param = 0.2
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 10.
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_mybot_v3_with_arm"
        num_steps_per_env =100 # per iteration