import numpy as np
import torch
from .math import *
from .observation import get_obs_slice

class FieldMixin:
    def check_BarrierTrack_terrain(self):
        if getattr(self.cfg.terrain, "pad_unavailable_info", False):
            return self.cfg.terrain.selected == "BarrierTrack"
        assert self.cfg.terrain.selected == "BarrierTrack", "This implementation is only for BarrierTrack terrain"
        return True

    def check_termination(self):
        return_ = super().check_termination()
        if not hasattr(self.cfg, "termination"): return return_
        
        r, p, y = get_euler_xyz(self.base_quat)
        r[r > np.pi] -= np.pi * 2 # to range (-pi, pi)
        p[p > np.pi] -= np.pi * 2 # to range (-pi, pi)
        z = self.root_states[:, 2] - self.env_origins[:, 2]

        if getattr(self.cfg.termination, "check_obstacle_conditioned_threshold", False) and self.check_BarrierTrack_terrain():
            if hasattr(self, "volume_sample_points"):
                self.refresh_volume_sample_points()
                stepping_obstacle_info = self.terrain.get_stepping_obstacle_info(self.volume_sample_points.view(-1, 3))
            else:
                stepping_obstacle_info = self.terrain.get_stepping_obstacle_info(self.root_states[:, :3])
            stepping_obstacle_info = stepping_obstacle_info.view(self.num_envs, -1, stepping_obstacle_info.shape[-1])
            # Assuming that each robot will only be in one obstacle or non obstacle.
            robot_stepping_obstacle_id = torch.max(stepping_obstacle_info[:, :, 0], dim= -1)[0]
        
        if "roll" in self.cfg.termination.termination_terms:
            if "robot_stepping_obstacle_id" in locals():
                r_term_buff = torch.abs(r[robot_stepping_obstacle_id == 0]) > \
                    self.cfg.termination.roll_kwargs["threshold"]
                self.reset_buf[robot_stepping_obstacle_id == 0] |= r_term_buff
                for obstacle_name, obstacle_id in self.terrain.track_options_id_dict.items():
                    if (obstacle_name + "_threshold") in self.cfg.termination.roll_kwargs:
                        env_selection_mask = robot_stepping_obstacle_id == obstacle_id
                        r_term_buff = torch.abs(r[env_selection_mask]) > \
                            self.cfg.termination.roll_kwargs[obstacle_name + "_threshold"]
                        self.reset_buf[env_selection_mask] |= r_term_buff
            else:
                r_term_buff = torch.abs(r) > self.cfg.termination.roll_kwargs["threshold"]
                self.reset_buf |= r_term_buff
        if "pitch" in self.cfg.termination.termination_terms:
            if "robot_stepping_obstacle_id" in locals():
                p_term_buff = torch.abs(p[robot_stepping_obstacle_id == 0]) > \
                    self.cfg.termination.pitch_kwargs["threshold"]
                self.reset_buf[robot_stepping_obstacle_id == 0] |= p_term_buff
                for obstacle_name, obstacle_id in self.terrain.track_options_id_dict.items():
                    if (obstacle_name + "_threshold") in self.cfg.termination.pitch_kwargs:
                        env_selection_mask = robot_stepping_obstacle_id == obstacle_id
                        p_term_buff = torch.abs(p[env_selection_mask]) > \
                            self.cfg.termination.pitch_kwargs[obstacle_name + "_threshold"]
                        self.reset_buf[env_selection_mask] |= p_term_buff
            else:
                p_term_buff = torch.abs(p) > self.cfg.termination.pitch_kwargs["threshold"]
                self.reset_buf |= p_term_buff
        if "z_low" in self.cfg.termination.termination_terms:
            if "robot_stepping_obstacle_id" in locals():
                z_low_term_buff = z[robot_stepping_obstacle_id == 0] < \
                    self.cfg.termination.z_low_kwargs["threshold"]
                self.reset_buf[robot_stepping_obstacle_id == 0] |= z_low_term_buff
                for obstacle_name, obstacle_id in self.terrain.track_options_id_dict.items():
                    if (obstacle_name + "_threshold") in self.cfg.termination.z_low_kwargs:
                        env_selection_mask = robot_stepping_obstacle_id == obstacle_id
                        z_low_term_buff = z[env_selection_mask] < \
                            self.cfg.termination.z_low_kwargs[obstacle_name + "_threshold"]
                        self.reset_buf[env_selection_mask] |= z_low_term_buff
            else:
                z_low_term_buff = z < self.cfg.termination.z_low_kwargs["threshold"]
                self.reset_buf |= z_low_term_buff
        if "z_high" in self.cfg.termination.termination_terms:
            z_high_term_buff = z > self.cfg.termination.z_high_kwargs["threshold"]
            self.reset_buf |= z_high_term_buff
        if getattr(self.cfg.termination, "timeout_at_finished", False) and self.check_BarrierTrack_terrain():
            x = self.root_states[:, 0] - self.env_origins[:, 0]
            finished_buffer = x > (self.terrain.env_block_length * self.terrain.n_blocks_per_track)
            self.time_out_buf |= finished_buffer
            self.reset_buf |= finished_buffer
        
        return return_

    def _fill_extras(self, env_ids):
        return_ = super()._fill_extras(env_ids)

        self.extras["episode"]["n_obstacle_passed"] = 0.
        with torch.no_grad():
            pos_x = self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]
            self.extras["episode"]["pos_x"] = pos_x
            if self.check_BarrierTrack_terrain():
                self.extras["episode"]["n_obstacle_passed"] = torch.mean(torch.clip(
                    torch.div(pos_x, self.terrain.env_block_length, rounding_mode= "floor") - 1,
                    min= 0.0,
                )).cpu()
                
                terrain_type_names = self.terrain.get_terrain_type_names(self.terrain_types[env_ids])
                if not terrain_type_names is None:
                    # record terrain level for each terrain types
                    for name in self.terrain.available_terrain_type_names:
                        terrain_levels = [
                            self.terrain_levels[env_ids[i]] \
                            for i in range(len(env_ids)) \
                            if terrain_type_names[i] == name
                        ]
                        timeouts_count = 0
                        for i in range(len(env_ids)):
                            if terrain_type_names[i] == name and self.time_out_buf[env_ids[i]]:
                                timeouts_count += 1
                        if len(terrain_levels) > 0:
                            terrain_levels = torch.stack(terrain_levels).float()
                            self.extras["episode"]["terrain_level_" + name] = torch.mean(terrain_levels).cpu()
                            self.extras["episode"]["terrain_level_" + name + "_max"] = torch.max(terrain_levels).cpu()
                            self.extras["episode"]["terrain_level_" + name + "_min"] = torch.min(terrain_levels).cpu()
                            self.extras["episode"]["num_terminated_" + name] = len(terrain_levels)
                            self.extras["episode"]["num_timeouts_" + name] = timeouts_count
                        else:
                            self.extras["episode"]["terrain_level_" + name] = torch.nan
                            self.extras["episode"]["terrain_level_" + name + "_max"] = torch.nan
                            self.extras["episode"]["terrain_level_" + name + "_min"] = torch.nan
                            self.extras["episode"]["num_terminated_" + name] = torch.nan
                            self.extras["episode"]["num_timeouts_" + name] = torch.nan
        
        return return_

    def _post_physics_step_callback(self):
        return_ = super()._post_physics_step_callback()

        with torch.no_grad():
            pos_x = self.root_states[:, 0] - self.env_origins[:, 0]
            pos_y = self.root_states[:, 1] - self.env_origins[:, 1]
            if self.check_BarrierTrack_terrain():
                self.extras["episode"]["n_obstacle_passed"] = torch.mean(torch.clip(
                    torch.div(pos_x, self.terrain.env_block_length, rounding_mode= "floor") - 1,
                    min= 0.0,
                )).cpu()

        if getattr(self.cfg.commands, "is_goal_based", False):
            self._update_command_by_terrain_goal()

        return return_

    def _resample_commands(self, env_ids):
        return_ = super()._resample_commands(env_ids)
        if getattr(self.cfg.commands, "is_goal_based", False):
            self.sampled_x_cmd_buffer[env_ids] = self.commands[env_ids, 0]
        return return_

    def _get_terrain_curriculum_move(self, env_ids):
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        moved = distance > (self.terrain.env_block_length * 1.5) # 0.1 is the guess of robot touching the obstacle block.
        less_moved = torch.logical_and(
            distance < self.terrain.env_block_length,
            torch.norm(self.commands[env_ids, :2], dim= 1) > self.cfg.commands.lin_cmd_cutoff,
        )
        if not (self.cfg.terrain.selected == "BarrierTrack" and hasattr(self, "body_sample_indices")):
            if getattr(self.cfg.curriculum, "no_moveup_when_fall", False):
                move_up = moved & self.time_out_buf[env_ids]
            else:
                move_up = moved
            move_down = less_moved
            return move_up, move_down
        
        passed_depths = self.terrain.get_passed_obstacle_depths(
            self.terrain_levels[env_ids],
            self.terrain_types[env_ids],
            self.volume_sample_points[env_ids, :, 0].max(-1)[0], # choose the sample points that goes the furthest
        ) + 1e-12

        p_v_ok = p_d_ok = 1
        p_v_too_much = p_d_too_much = 0
        # NOTE: only when penetrate_* reward is computed does this function check the penetration
        if "penetrate_volume" in self.episode_sums:
            p_v = self.episode_sums["penetrate_volume"][env_ids]
            p_v_normalized = p_v / passed_depths / self.reward_scales["penetrate_volume"]
            p_v_ok = p_v_normalized < self.cfg.curriculum.penetrate_volume_threshold_harder
            p_v_too_much = p_v_normalized > self.cfg.curriculum.penetrate_volume_threshold_easier
        if "penetrate_depth" in self.episode_sums:
            p_d = self.episode_sums["penetrate_depth"][env_ids]
            p_d_normalized = p_d / passed_depths / self.reward_scales["penetrate_depth"]
            p_d_ok = p_d_normalized < self.cfg.curriculum.penetrate_depth_threshold_harder
            p_d_too_much = p_d_normalized > self.cfg.curriculum.penetrate_depth_threshold_easier

        move_up = p_v_ok * p_d_ok * moved
        move_down = ((~moved) + p_v_too_much + p_d_too_much).to(bool)
        # print(
        #     # "p_v:", p_v_normalized,
        #     "p_d:", p_d_normalized,
        #     "move_up:", move_up,
        #     "move_down:", move_down,
        # )
        return move_up, move_down

    def _init_buffers(self):
        super()._init_buffers()

        # projected gravity bias (if needed)
        if getattr(self.cfg.domain_rand, "randomize_gravity_bias", False):
            print("Initializing gravity bias for domain randomization")
            # add cross trajectory domain randomization on projected gravity bias
            # uniform sample from range
            self.gravity_bias = torch.rand(self.num_envs, 3, dtype= torch.float, device= self.device, requires_grad= False)
            self.gravity_bias[:, 0] *= self.cfg.domain_rand.gravity_bias_range["x"][1] - self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[:, 0] += self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[:, 1] *= self.cfg.domain_rand.gravity_bias_range["y"][1] - self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[:, 1] += self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[:, 2] *= self.cfg.domain_rand.gravity_bias_range["z"][1] - self.cfg.domain_rand.gravity_bias_range["z"][0]
            self.gravity_bias[:, 2] += self.cfg.domain_rand.gravity_bias_range["z"][0]

        if getattr(self.cfg.commands, "is_goal_based", False):
            # store sampled commands for original x velocity
            self.sampled_x_cmd_buffer = torch.zeros(self.num_envs, dtype= torch.float, device= self.device, requires_grad= False)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            self.extras["episode"]["n_obstacle_passed"] = 0

    def _reset_buffers(self, env_ids):
        return_ = super()._reset_buffers(env_ids)

        if getattr(self.cfg.domain_rand, "randomize_gravity_bias", False):
            assert hasattr(self, "gravity_bias")
            self.gravity_bias[env_ids] = torch.rand_like(self.gravity_bias[env_ids])
            self.gravity_bias[env_ids, 0] *= self.cfg.domain_rand.gravity_bias_range["x"][1] - self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[env_ids, 0] += self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[env_ids, 1] *= self.cfg.domain_rand.gravity_bias_range["y"][1] - self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[env_ids, 1] += self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[env_ids, 2] *= self.cfg.domain_rand.gravity_bias_range["z"][1] - self.cfg.domain_rand.gravity_bias_range["z"][0]
            self.gravity_bias[env_ids, 2] += self.cfg.domain_rand.gravity_bias_range["z"][0]
        
        return return_

    def _update_command_by_terrain_goal(self):
        """ compute a locomotion command based on current goal for the robot. """
        if self.check_BarrierTrack_terrain():
            self.current_goal_global = self.terrain.get_goal_position(self.root_states[:, :3])
            current_goal_local = tf_apply(
                *tf_inverse(self.base_quat, self.root_states[:, :3]),
                self.current_goal_global,
            )
            current_goal_yaw = torch.atan2(current_goal_local[:, 1], current_goal_local[:, 0]) # between +- pi
            x_cmd = torch.clip(
                current_goal_local[:, 0] * (1 if getattr(self.cfg.commands.goal_based, "x_ratio", None) is None else self.cfg.commands.goal_based.x_ratio),
                min= self.cfg.commands.ranges.lin_vel_x[0],
                max= self.cfg.commands.ranges.lin_vel_x[1],
            )
            if getattr(self.cfg.commands.goal_based, "follow_cmd_cutoff", False):
                x_cmd[torch.abs(x_cmd) < self.cfg.commands.lin_cmd_cutoff] = 0.
            y_cmd = torch.clip(
                current_goal_local[:, 1] * self.cfg.commands.goal_based.y_ratio,
                min= self.cfg.commands.ranges.lin_vel_y[0],
                max= self.cfg.commands.ranges.lin_vel_y[1],
            )
            if getattr(self.cfg.commands.goal_based, "follow_cmd_cutoff", False):
                y_cmd[torch.abs(y_cmd) < self.cfg.commands.lin_cmd_cutoff] = 0.
            yaw_cmd = torch.clip(
                current_goal_yaw * self.cfg.commands.goal_based.yaw_ratio,
                min= self.cfg.commands.ranges.ang_vel_yaw[0],
                max= self.cfg.commands.ranges.ang_vel_yaw[1],
            )
            if getattr(self.cfg.commands.goal_based, "follow_cmd_cutoff", False):
                yaw_cmd[torch.abs(yaw_cmd) < self.cfg.commands.ang_cmd_cutoff] = 0.
            if getattr(self.cfg.commands.goal_based, "x_ratio", None) is not None:
                self.commands[:, 0] = x_cmd
            if getattr(self.cfg.commands, "conditioned_on_obstacle", False):
                engaging_block_types = self.terrain.get_engaging_block_types(self.root_states[:, :3])
                for obstacle_name in self.cfg.terrain.BarrierTrack_kwargs["options"]:
                    env_mask = engaging_block_types == self.terrain.track_options_id_dict[obstacle_name]
                    self.commands[env_mask, 0] = torch.clip(
                        self.commands[env_mask, 0],
                        *getattr(self.cfg.commands.obstacle_conditioned_ranges, "{}_x".format(obstacle_name), [-1.0, 1.0]),
                    )
            self.commands[:, 1] = y_cmd
            self.commands[:, 2] = yaw_cmd
            if hasattr(self.cfg.commands.goal_based, "x_stop_by_yaw_threshold"):
                # set x to zero if the robot is facing almost the opposite direction of the goal
                x_stop_threshold = self.cfg.commands.goal_based.x_stop_by_yaw_threshold
                self.commands[torch.abs(current_goal_yaw) > x_stop_threshold, 0] = 0.
                self.commands[torch.abs(current_goal_yaw) < x_stop_threshold, 0] = self.sampled_x_cmd_buffer[torch.abs(current_goal_yaw) < x_stop_threshold]
            # set y and yaw to zero if x is sampled as zero
            self.commands[self.sampled_x_cmd_buffer == 0] = 0.
            # print("commands", self.commands)
        else:
            raise NotImplementedError("Only BarrierTrack terrain is supported for now.")

    def _get_proprioception_obs(self, privileged= False):
        obs_buf = super()._get_proprioception_obs(privileged= privileged)
        
        if getattr(self.cfg.domain_rand, "randomize_gravity_bias", False) and (not privileged):
            assert hasattr(self, "gravity_bias")
            proprioception_slice = get_obs_slice(self.obs_segments, "proprioception")
            obs_buf[:, proprioception_slice[0].start + 6: proprioception_slice[0].start + 9] += self.gravity_bias
        
        return obs_buf

    def _get_engaging_block_obs(self, privileged= False):
        """ Compute the obstacle info for the robot """
        if not self.check_BarrierTrack_terrain():
            # This could be wrong, check BarrierTrack implementation to get the exact shape.
            raise NotImplementedError("Only BarrierTrack terrain is supported for now.")
        base_positions = self.root_states[:, 0:3] # (n_envs, 3)
        self.refresh_volume_sample_points()
        engaging_block_distance = self.terrain.get_engaging_block_distance(
            base_positions,
            self.volume_sample_points - base_positions.unsqueeze(-2), # (n_envs, n_points, 3)
        ) # (num_envs,)
        engaging_block_types = self.terrain.get_engaging_block_types(
            base_positions,
            self.volume_sample_points - base_positions.unsqueeze(-2), # (n_envs, n_points, 3)
        ) # (num_envs,)
        engaging_block_type_onehot = torch.zeros(
            (self.num_envs, self.terrain.max_track_options),
            device= self.sim_device,
            dtype= torch.float,
        )
        engaging_block_type_onehot.scatter_(1, engaging_block_types.unsqueeze(1), 1.)
        engaging_block_info = self.terrain.get_engaging_block_info(
            base_positions,
            self.volume_sample_points - base_positions.unsqueeze(-2), # (n_envs, n_points, 3)
        ) # (num_envs, obstacle_info_dim)
        engaging_block_obs = torch.cat([
            engaging_block_distance.unsqueeze(-1),
            engaging_block_type_onehot,
            engaging_block_info,
        ], dim= -1) # (num_envs, 1 + (max_obstacle_types + 1) + obstacle_info_dim)
        return engaging_block_obs

    def _get_sidewall_distance_obs(self, privileged= False):
        if not self.check_BarrierTrack_terrain():
            return torch.zeros((self.num_envs, 2), device= self.sim_device)
        base_positions = self.root_states[:, 0:3] # (n_envs, 3)
        return self.terrain.get_sidewall_distance(base_positions)

    def _write_engaging_block_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "engaging_block"):
            return
        noise_vec[:] = self.cfg.noise.noise_scales.engaging_block * self.cfg.noise.noise_level * self.obs_scales.engaging_block

    def _write_sidewall_distance_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "sidewall_distance"):
            return
        noise_vec[:] = self.cfg.noise.noise_scales.sidewall_distance * self.cfg.noise.noise_level * self.obs_scales.sidewall_distance

    def get_obs_segment_from_components(self, components):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        segments = super().get_obs_segment_from_components(components)
    
        if "engaging_block" in components:
            if not self.check_BarrierTrack_terrain():
                # This could be wrong, please check the implementation of BarrierTrack
                raise NotImplementedError("Only BarrierTrack terrain is supported for now.")
            else:
                segments["engaging_block"] = \
                    (1 + self.terrain.max_track_options + self.terrain.block_info_dim,)
        if "sidewall_distance" in components:
            self.check_BarrierTrack_terrain()
            segments["sidewall_distance"] = (2,)
        
        return segments

    def _reward_world_vel_l2norm(self):
        return torch.norm((self.commands[:, :2] - self.root_states[:, 7:9]), dim= 1)

    def _reward_tracking_world_vel(self):
        world_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.root_states[:, 7:9]), dim= 1)
        return torch.exp(-world_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_lin_cmd(self):
        """ This reward term does not depend on the policy, depends on the command """
        return torch.norm(self.commands[:, :2], dim= 1)

    def _reward_lin_vel_x(self):
        return self.root_states[:, 7]

    def _reward_lin_vel_y_abs(self):
        return torch.abs(self.root_states[:, 8])

    def _reward_lin_vel_y_square(self):
        return torch.square(self.root_states[:, 8])

    def _reward_lin_pos_y(self):
        return torch.abs((self.root_states[:, :3] - self.env_origins)[:, 1])

    def _reward_yaw_abs(self):
        """ Aiming for the robot yaw to be zero (pointing to the positive x-axis) """
        yaw = get_euler_xyz(self.root_states[:, 3:7])[2]
        yaw[yaw > np.pi] -= np.pi * 2 # to range (-pi, pi)
        yaw[yaw < -np.pi] += np.pi * 2 # to range (-pi, pi)
        return torch.abs(yaw)

    def _reward_penetrate_depth(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        penetration_depths = self.terrain.get_penetration_depths(self.volume_sample_points.view(-1, 3)).view(self.num_envs, -1)
        penetration_depths *= torch.norm(self.volume_sample_points_vel, dim= -1) + 1e-3
        return torch.sum(penetration_depths, dim= -1)

    def _reward_penetrate_volume(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        penetration_mask = self.terrain.get_penetration_mask(self.volume_sample_points.view(-1, 3)).view(self.num_envs, -1)
        penetration_mask *= torch.norm(self.volume_sample_points_vel, dim= -1) + 1e-3
        return torch.sum(penetration_mask, dim= -1)

    def _reward_tilt_cond(self):
        """ Conditioned reward term in terms of whether the robot is engaging the tilt obstacle
        Use positive factor to enable rolling angle when incountering tilt obstacle
        """
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
        pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
        roll[roll > pi] -= pi * 2 # to range (-pi, pi)
        roll[roll < -pi] += pi * 2 # to range (-pi, pi)
        if hasattr(self, "volume_sample_points"):
            self.refresh_volume_sample_points()
            stepping_obstacle_info = self.terrain.get_stepping_obstacle_info(self.volume_sample_points.view(-1, 3))
        else:
            stepping_obstacle_info = self.terrain.get_stepping_obstacle_info(self.root_states[:, :3])
        stepping_obstacle_info = stepping_obstacle_info.view(self.num_envs, -1, stepping_obstacle_info.shape[-1])
        # Assuming that each robot will only be in one obstacle or non obstacle.
        robot_stepping_obstacle_id = torch.max(stepping_obstacle_info[:, :, 0], dim= -1)[0]
        tilting_mask = robot_stepping_obstacle_id == self.terrain.track_options_id_dict["tilt"]
        return_ = torch.where(tilting_mask, torch.clip(torch.abs(roll), 0, torch.pi/2), -torch.clip(torch.abs(roll), 0, torch.pi/2))
        return return_

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_front_hip_pos(self):
        """ Reward the robot to stop moving its front hips """
        return torch.sum(torch.square(self.dof_pos[:, self.front_hip_indices] - self.default_dof_pos[:, self.front_hip_indices]), dim=1)

    def _reward_rear_hip_pos(self):
        """ Reward the robot to stop moving its rear hips """
        return torch.sum(torch.square(self.dof_pos[:, self.rear_hip_indices] - self.default_dof_pos[:, self.rear_hip_indices]), dim=1)

    def _reward_down_cond(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_types = self.terrain.get_engaging_block_types(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_obstacle_info = self.terrain.get_engaging_block_info(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
        pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
        pitch[pitch > pi] -= pi * 2 # to range (-pi, pi)
        pitch[pitch < -pi] += pi * 2 # to range (-pi, pi)
        engaging_mask = (engaging_obstacle_types == self.terrain.track_options_id_dict["down"]) \
            & (engaging_obstacle_info[:, -1] < 0.)
        pitch_err = torch.abs(pitch - 0.2)
        return torch.exp(-pitch_err/self.cfg.rewards.tracking_sigma) * engaging_mask # the higher positive factor, the more you want the robot to pitch down 0.2 rad

    def _reward_jump_x_vel_cond(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_types = self.terrain.get_engaging_block_types(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_hurdle_mask = (engaging_obstacle_types == self.terrain.track_options_id_dict.get("hurdle", 5))
        engaging_jump_mask = (engaging_obstacle_types == self.terrain.track_options_id_dict.get("jump", 3))
        engaging_mask = engaging_hurdle_mask | engaging_jump_mask
        roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
        pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
        pitch[pitch > pi] -= pi * 2 # to range (-pi, pi)
        pitch[pitch < -pi] += pi * 2 # to range (-pi, pi)
        pitch_up_mask = pitch < -0.75 # a hack value

        return torch.clip(self.base_lin_vel[:, 0], max= 1.5) * engaging_mask * pitch_up_mask

    def _reward_sync_legs_cond(self):
        """ A hack to force same actuation on both rear legs when jump. """
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_types = self.terrain.get_engaging_block_types(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_hurdle_mask = (engaging_obstacle_types == self.terrain.track_options_id_dict.get("hurdle", 5))
        engaging_jump_mask = (engaging_obstacle_types == self.terrain.track_options_id_dict.get("jump", 3))
        engaging_mask = engaging_hurdle_mask | engaging_jump_mask
        rr_legs = torch.clone(self.actions[:, 6:9]) # shoulder, thigh, calf
        rl_legs = torch.clone(self.actions[:, 9:12]) # shoulder, thigh, calf
        rl_legs[:, 0] *= -1 # flip the sign of shoulder action
        return torch.norm(rr_legs - rl_legs, dim= -1) * engaging_mask

    def _reward_sync_all_legs_cond(self):
        """ A hack to force same actuation on both front/rear legs when jump. """
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_types = self.terrain.get_engaging_block_types(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_hurdle_mask = (engaging_obstacle_types == self.terrain.track_options_id_dict.get("hurdle", 5))
        engaging_jump_mask = (engaging_obstacle_types == self.terrain.track_options_id_dict.get("jump", 3))
        engaging_mask = engaging_hurdle_mask | engaging_jump_mask
        right_legs = torch.clone(torch.cat([
            self.actions[:, 0:3],
            self.actions[:, 6:9],
        ], dim= -1)) # shoulder, thigh, calf
        left_legs = torch.clone(torch.cat([
            self.actions[:, 3:6],
            self.actions[:, 9:12],
        ], dim= -1)) # shoulder, thigh, calf
        left_legs[:, 0] *= -1 # flip the sign of shoulder action
        left_legs[:, 3] *= -1 # flip the sign of shoulder action
        return torch.norm(right_legs - left_legs, p= 1, dim= -1) * engaging_mask

    def _reward_sync_all_legs(self):
        right_legs = torch.clone(torch.cat([
            self.actions[:, 0:3],
            self.actions[:, 6:9],
        ], dim= -1)) # shoulder, thigh, calf
        left_legs = torch.clone(torch.cat([
            self.actions[:, 3:6],
            self.actions[:, 9:12],
        ], dim= -1)) # shoulder, thigh, calf
        left_legs[:, 0] *= -1 # flip the sign of shoulder action
        left_legs[:, 3] *= -1 # flip the sign of shoulder action
        return torch.norm(right_legs - left_legs, p= 1, dim= -1)

    def _reward_dof_error_cond(self):
        """ Force dof error when not engaging obstacle """
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_types = self.terrain.get_engaging_block_types(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        not_engaging_mask = (engaging_obstacle_types == 0)
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1) * not_engaging_mask

    def _reward_leap_bonous_cond(self):
        """ counteract the tracking reward loss during leap"""
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_distance = self.terrain.get_engaging_block_distance(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_obstacle_types = self.terrain.get_engaging_block_types(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_mask = (engaging_obstacle_types == self.terrain.track_options_id_dict["leap"]) \
            & (engaging_obstacle_distance > 0.)

        world_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.root_states[:, 7:9]), dim= 1)
        return (1 - torch.exp(-world_vel_error/self.cfg.rewards.tracking_sigma)) * engaging_mask # reverse version of tracking reward
