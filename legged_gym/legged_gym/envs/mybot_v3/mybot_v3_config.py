# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class MybotV3RoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.40]  # x,y,z [m]
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,

            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,

            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
           
            "FL_hip": 40.0,
            "FR_hip": 40.0,
            "RL_hip": 40.0,
            "RR_hip": 40.0,
    
            "FL_thigh": 40.0,
            "FR_thigh": 40.0,
            "RL_thigh": 40.0,
            "RR_thigh": 40.0,

            "FL_calf": 40.0,
            "FR_calf": 40.0,
            "RL_calf": 40.0,
            "RR_calf": 40.0,
        }
        damping = {
            "FL_hip": 1.0, "FR_hip": 1.0, "RL_hip": 1.0, "RR_hip": 1.0,
            "FL_thigh": 1.0, "FR_thigh": 1.0, "RL_thigh": 1.0, "RR_thigh": 1.0,
            "FL_calf": 1.0, "FR_calf": 1.0, "RL_calf": 1.0, "RR_calf": 1.0,
        }
        action_scale = 0.25  # 调整动作幅度，使小腿运动更平稳
        decimation = 4
        hip_reduction = 1.0

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 2.0
        num_commands = 4
        resampling_time = 10.0
        heading_command = True

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-3.14, 3.14]
            heading = [-3.14, 3.14]


    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mybot_v3/urdf/mybot_v3.urdf"
        name = "mybot_v3"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base", "body"]
        privileged_contacts_on = ["base", "body", "thigh", "calf"]
        self_collisions = 1
        flip_visual_attachments = False
        armature = {
            "FL_hip": 0.01, "FR_hip": 0.01, "RL_hip": 0.01, "RR_hip": 0.01,
            "FL_thigh": 0.01, "FR_thigh": 0.01, "RL_thigh": 0.01, "RR_thigh": 0.01,
            "FL_calf": 0.0544, "FR_calf": 0.0544, "RL_calf": 0.0544, "RR_calf": 0.0544,
        }

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            termination = -0.0
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
            feet_air_time = 0.00
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -0.0
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0
            # hip_pos = -0.0
            rear_thigh_pos = -0.2
            dof_pos = -0.1

        dof_pos_hip_weight = 4.0
        dof_pos_thigh_weight = 2.0
        dof_pos_calf_weight = 0.0
        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.33
        max_contact_force = 100.0
        clearance_height_target = -0.21


class MybotV3RoughCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):

        # 降低 exploration
        init_noise_std = 1.0

        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        # learning_rate = 3e-4
        # num_mini_batches = 4
        
        # # 关闭 adaptive PPO
        # schedule = 'fixed'
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_mybot_v3"
        num_steps_per_env = 100
