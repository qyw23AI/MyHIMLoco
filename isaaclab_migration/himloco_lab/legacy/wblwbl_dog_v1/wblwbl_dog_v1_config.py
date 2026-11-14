# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""Walking profile for the WBLWBL Dog V1."""

from ..base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class WblwblDogV1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.40]
        # A1-style reference pose in radians, checked against this URDF.
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RL_hip_joint": 0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }
        reset_at_nominal_pose = True
        standing_test_min_height = 0.25

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

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        # Smooth slope, rough slope, stairs up, stairs down, discrete.
        terrain_proportions = [0.3, 0.1, 0.3, 0.3, 0.0]
        max_init_terrain_level = 2
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 20

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/wblwbl_dog_v1/urdf/wblwbl_dog.urdf"
        name = "wblwbl_dog_v1"
        foot_name = "_foot"
        penalize_contacts_on = ["_thigh", "_calf", "base_link"]
        terminate_after_contacts_on = ["base_link"]
        privileged_contacts_on = ["base_link", "_thigh", "_calf"]
        collapse_fixed_joints = False  # Keep the four foot contact bodies separate.
        self_collisions = 1
        flip_visual_attachments = False
        armature = {
            "FL_hip": 0.01, "FR_hip": 0.01, "RL_hip": 0.01, "RR_hip": 0.01,
            "FL_thigh": 0.01, "FR_thigh": 0.01, "RL_thigh": 0.01, "RR_thigh": 0.01,
            "FL_calf": 0.01, "FR_calf": 0.01, "RL_calf": 0.01, "RR_calf": 0.01,
        }

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_initial_joint_pos = False

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.33
        clearance_height_target = -0.21
        dof_pos_hip_weight = 4.0
        dof_pos_thigh_weight = 2.0
        dof_pos_calf_weight = 0.0
        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        max_contact_force = 100.0

        class scales:
            termination = 0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            lin_vel_y = 0.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -1.0
            foot_clearance = -0.02
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            stand_still = 0.0
            torques = 0.0
            dof_vel = 0.0
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0
            torque_limits = 0.0
            rear_thigh_pos = -0.2
            dof_pos = -0.1


class WblwblDogV1CfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        
    class algorithm(LeggedRobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
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
        experiment_name = "wblwbl_dog_v1"
        num_steps_per_env = 100
