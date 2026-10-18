"""Task-only extension points. LegacyLogic and the PPO implementation stay intact."""
import torch
from ..core import HIMCore
from ..config import asset_contract
from ..math import quat_apply_yaw
from .geometry import nominal_feet
from .reference import GaitState, sample_commands


class TaskGaitCore(HIMCore):
    def _init_buffers(self):
        super()._init_buffers()
        self.gait = GaitState(self.num_envs, self.cfg.gait, self.device)
        indices = [self.body_names.index(n) for n in self.cfg.gait.foot_names]
        if set(indices) != set(self.feet_indices.tolist()):
            raise ValueError('Task gait requires exactly the four named foot bodies')
        self.gait_feet = torch.tensor(indices, device=self.device)
        feet = nominal_feet(asset_contract(self.cfg)[0], self.cfg.init_state.default_joint_angles,
                            self.cfg.gait.foot_names)
        self.nominal_feet = torch.as_tensor(feet, device=self.device, dtype=torch.float)
        self.external_commands = False
        self.history_fresh = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self._cached_frame = None
        self._torque_square = torch.zeros_like(self.torques)
        self._torque_peak = torch.zeros_like(self.torques)
        self._torque_saturation = torch.zeros_like(self.torques)
        self._torque_samples = 0
        self.begin_control_step()

    def begin_control_step(self):
        self._control_torque_square = torch.zeros_like(self.torques)
        self._control_torque_samples = 0

    def _resample_commands(self, ids):
        if not len(ids) or self.external_commands:
            return
        target, mode, groups = sample_commands(len(ids), self.cfg.gait,
                                               self.cfg.commands.ranges, self.device)
        # A subset follows explicit start/adjust/stop/turn/standby/hold sequences;
        # other environments keep independent randomized commands.
        sequential = ids < round(self.num_envs * self.cfg.gait.sequence_fraction)
        segment = (self.episode_length_buf[ids] // round(self.cfg.commands.resampling_time / self.dt)) % 6
        sequence = torch.tensor([[.2, 0., 0.], [.03, 0., 0.], [0., 0., 0.],
                                 [0., 0., .1], [0., 0., 0.], [0., 0., 0.]], device=self.device)
        target[sequential] = sequence[segment[sequential]]
        target[sequential] *= torch.where(ids[sequential, None] % 2 == 0, 1., -1.)
        mode[sequential] = (segment[sequential] == 4).float()
        labels = torch.tensor([4, 2, 0, 3, 1, 0], device=self.device)
        groups[sequential] = labels[segment[sequential]]
        self.gait.target[ids] = target
        self.gait.standby[ids] = mode
        self.gait.groups[ids] = groups

    def set_command(self, command, standby=False):
        """Latch a raw body-frame velocity; applied to observations on the next step.

        Use prepare_external_observation() once before the first policy call.
        Changing commands never adds an extra frame to the sensor history.
        """
        value = torch.as_tensor(command, device=self.device, dtype=torch.float)
        if value.shape not in ((3,), (self.num_envs, 3)) or not torch.isfinite(value).all():
            raise ValueError('Expected finite [3] or [num_envs,3] body-frame commands')
        if standby and (value.abs() > self.cfg.gait.zero_epsilon).any():
            raise ValueError('Standby stepping requires a zero body-velocity command')
        limits = self.cfg.commands.ranges
        for axis, bounds in enumerate((limits.lin_vel_x, limits.lin_vel_y, limits.ang_vel_yaw)):
            if (value[..., axis] < bounds[0]).any() or (value[..., axis] > bounds[1]).any():
                raise ValueError(f'Command axis {axis} outside trained range {bounds}')
        self.external_commands = True
        self.gait.target[:] = value
        self.gait.standby[:] = float(standby)
        target = self.gait.target
        zero = target.abs().amax(-1) <= self.cfg.gait.zero_epsilon
        yaw_only = (target[:, :2].abs().amax(-1) <= self.cfg.gait.zero_epsilon) & ~zero
        translation_only = target[:, 2].abs() <= self.cfg.gait.zero_epsilon
        labels = torch.full_like(self.gait.groups, 5)
        labels[translation_only] = 4
        labels[translation_only & (target[:, :2].abs().amax(-1) <= .1)] = 2
        labels[yaw_only] = 3
        labels[zero] = 1 if standby else 0
        self.gait.groups.copy_(labels)

    def prepare_external_observation(self):
        """Replace only command/mode fields of the newest frame, without advancing time."""
        self.obs_buf[:, :3] = self.commands[:, :3] * self.commands_scale
        self.obs_buf[:, 48] = self.gait.standby

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.gait.advance(self.dt)
        self.commands[:, :3] = self.gait.command
        self._cached_frame = None

    def _reset_dofs(self, ids):
        # Small, symmetric perturbations instead of multiplying angles by [0.5,1.5].
        q = self.default_dof_pos + (torch.rand(len(ids), self.num_dof, device=self.device) - 0.5) * 0.10
        self.dof_pos[ids] = q
        self.dof_vel[ids] = 0
        self.backend.robot.write_joint_state_to_sim(q, torch.zeros_like(q),
                                                   joint_ids=self.backend.joint_ids, env_ids=ids)

    def _reset_root_states(self, ids):
        super()._reset_root_states(ids)
        self.root_states[ids, 7:13] = 0
        self.backend.write_root(self.root_states[ids], ids)

    def reset_idx(self, ids):
        if not len(ids):
            return
        super().reset_idx(ids)
        self.gait.reset(ids)
        self.commands[ids] = 0
        self.actions[ids] = 0
        self.obs_buf[ids] = 0
        self.privileged_obs_buf[ids] = 0
        self.history_fresh[ids] = True
        # New task deliberately refreshes post-reset state; terminal targets were
        # already captured by the backend before this reset.
        self.backend.sync_state(self)
        self._cached_frame = None

    def _compute_torques(self, actions):
        torques = super()._compute_torques(actions)
        self._torque_square += torques.square()
        self._torque_peak = torch.maximum(self._torque_peak, torques.abs())
        self._torque_saturation += (torques.abs() >= self.torque_limits * 0.95).float()
        self._torque_samples += 1
        self._control_torque_square += torques.square()
        self._control_torque_samples += 1
        return torques

    def get_current_obs(self):
        if self._cached_frame is not None:
            return self._cached_frame
        original = torch.cat((self.commands[:, :3] * self.commands_scale,
                              self.base_ang_vel * self.obs_scales.ang_vel,
                              self.projected_gravity,
                              (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                              self.dof_vel * self.obs_scales.dof_vel, self.actions), -1)
        if self.add_noise:
            original += (2 * torch.rand_like(original) - 1) * self.noise_scale_vec[:45]
        actor = torch.cat((original, self.gait.features()), -1)
        # Estimator contract: actor prefix, then scaled true velocity, force, heights.
        privileged = [actor, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0]]
        heights = ((self.root_states[:, 2:3] - 0.5 - self.measured_heights).clamp(-1, 1)
                   * self.obs_scales.height_measurements)
        if self.add_noise:
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[45:232]
        privileged.append(heights)
        self._cached_frame = torch.cat(privileged, -1)
        return self._cached_frame

    def compute_observations(self):
        frame = self.get_current_obs()
        actor = frame[:, :self.num_one_step_obs]
        self.obs_buf = torch.cat((actor, self.obs_buf[:, :-self.num_one_step_obs]), -1)
        self.privileged_obs_buf = frame.clone()
        # Warm-start only reset environments with their actual new frame.
        self.obs_buf[self.history_fresh] = actor[self.history_fresh].repeat(1, self.history_length)
        self.history_fresh[:] = False

    def compute_termination_observations(self, ids):
        return self.get_current_obs()[ids]

    def _reward_tracking_lin_vel(self):
        cfg = self.cfg.gait
        scale = cfg.linear_error_floor + cfg.linear_error_relative * self.commands[:, :2].abs()
        return torch.exp(-((self.commands[:, :2] - self.base_lin_vel[:, :2]) / scale).square().mean(-1))

    def _reward_tracking_ang_vel(self):
        cfg = self.cfg.gait
        scale = cfg.yaw_error_floor + cfg.yaw_error_relative * self.commands[:, 2].abs()
        return torch.exp(-((self.commands[:, 2] - self.base_ang_vel[:, 2]) / scale).square())

    def _foot_frame(self):
        relative = self.rigid_body_states[:, self.gait_feet, :3] - self.root_states[:, None, :3]
        inverse = self.base_quat.clone()
        inverse[:, :3] *= -1
        rotation = inverse[:, None].expand(-1, 4, -1).reshape(-1, 4)
        return quat_apply_yaw(rotation, relative.reshape(-1, 3)).view(-1, 4, 3)

    def _reward_gait_contact(self):
        _, _, contact = self.gait.reference(self.nominal_feet)
        forces = self.contact_forces[:, self.gait_feet].norm(dim=-1)
        speed = self.rigid_body_states[:, self.gait_feet, 7:10].norm(dim=-1)
        swing_force = (1 - contact) * (1 - torch.exp(-(forces / self.cfg.gait.force_scale).square()))
        stance_slip = contact * (1 - torch.exp(-(speed / self.cfg.gait.slip_scale).square()))
        return (swing_force + stance_slip).mean(-1)

    def _reward_gait_position(self):
        reference, _, _ = self.gait.reference(self.nominal_feet)
        error = (self._foot_frame()[..., :2] - reference) / self.cfg.gait.position_sigma
        # Saturation prevents large perturbations from overwhelming velocity tracking.
        return (1 - torch.exp(-error.square().sum(-1))).mean(-1)

    def _reward_gait_clearance(self):
        _, lift, contact = self.gait.reference(self.nominal_feet)
        if self.backend.terrain is None:
            ground = torch.zeros_like(lift)
        else:
            # Training-only supervision; never included in actor observations.
            positions = self.rigid_body_states[:, self.gait_feet, :3]
            ground = self.backend.terrain.get_terrain_heights(positions.clone())
        foot_height = self.rigid_body_states[:, self.gait_feet, 2] - ground
        # URDF sphere radius, not an assumed world-z foot-centre offset.
        target = self.cfg.gait.foot_radius + lift
        error = (foot_height - target) / self.cfg.gait.clearance_sigma
        return ((1 - contact) * (1 - torch.exp(-error.square()))).mean(-1)

    def _reward_gait_stand(self):
        holding = 1 - self.gait.activation
        return holding * self.dof_vel.square().mean(-1)

    def _reward_gait_load(self):
        specified = self.cfg.gait.continuous_torque_nm
        scale = self.torque_limits if specified is None else torch.tensor(
            [getattr(specified, n) for n in self.dof_names], device=self.device)
        load = self._control_torque_square / max(self._control_torque_samples, 1) / scale.square()
        # Penalize both overall burden and concentration on any one joint.
        return load.mean(-1) + load.amax(-1)
