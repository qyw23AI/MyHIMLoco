"""Streaming tracking/contact/load diagnostics; no motor-temperature claims."""
import json
from pathlib import Path
import torch


class TaskGaitMonitor:
    def __init__(self, env, directory, every_steps=250):
        self.env = env
        self.directory = Path(directory)
        self.every_steps = every_steps
        self.reset()

    def reset(self):
        c = self.env.core
        from .posture import PostureStats
        self.posture = PostureStats(c)
        self.steps = self.resets = self.samples = 0
        self.error = torch.zeros(3, device=c.device)
        self.raw_error = torch.zeros_like(self.error)
        self.velocity_square = torch.zeros_like(self.error)
        self.velocity_sum = torch.zeros_like(self.error)
        self.command_sum = torch.zeros_like(self.error)
        self.slip = self.tilt = 0.0
        self.groups = torch.zeros(6, 4, device=c.device)  # samples, absolute xyz errors
        self.contacts = torch.zeros(c.num_envs, 4, dtype=torch.bool, device=c.device)
        self.last_touch = torch.full((c.num_envs, 4), -1.0, device=c.device)
        self.period_sum = torch.zeros(4, device=c.device)
        self.period_square = torch.zeros_like(self.period_sum)
        self.period_count = torch.zeros_like(self.period_sum)
        self.diagonal_agreement = 0.0
        self.rhythm_samples = 0
        self.previous_target = c.gait.target.clone()
        self.previous_mode = c.gait.standby.clone()
        self.steady_age = torch.zeros(c.num_envs, device=c.device)
        self.snapshots = []
        c._torque_square.zero_()
        c._torque_peak.zero_()
        c._torque_saturation.zero_()
        c._torque_samples = 0

    def before_step(self):
        pass

    def after_step(self, terminated, truncated):
        c = self.env.core
        done = terminated | truncated
        self.steps += 1
        self.resets += int(done.sum())
        changed = ((c.gait.target - self.previous_target).abs().amax(-1) > 1e-6) | (c.gait.standby != self.previous_mode)
        self.steady_age = torch.where(changed | done, 0.0, self.steady_age + c.dt)
        self.previous_target.copy_(c.gait.target)
        self.previous_mode.copy_(c.gait.standby)
        valid = ~done & (self.steady_age >= 1.0)
        self.posture.update(valid)
        actual = torch.cat((c.base_lin_vel[:, :2], c.base_ang_vel[:, 2:3]), -1)
        err = (actual - c.commands[:, :3]).abs()
        self.samples += int(valid.sum())
        self.error += err[valid].sum(0)
        self.raw_error += (actual - c.gait.target).abs()[valid].sum(0)
        self.velocity_square += actual[valid].square().sum(0)
        self.velocity_sum += actual[valid].sum(0)
        self.command_sum += c.commands[valid, :3].sum(0)
        for group in range(6):
            mask = valid & (c.gait.groups == group)
            self.groups[group, 0] += mask.sum()
            self.groups[group, 1:] += err[mask].sum(0)
        feet_velocity = c.rigid_body_states[:, c.gait_feet, 7:9].norm(dim=-1)
        forces = c.contact_forces[:, c.gait_feet].norm(dim=-1)
        contact = torch.where(self.contacts, forces > 2.0, forces > 5.0)
        touching = contact & ~self.contacts & ~done[:, None]
        now = self.steps * c.dt
        periods = now - self.last_touch
        touching &= (self.last_touch < 0) | (periods >= 0.15)
        rhythm = valid & (c.gait.activation > 0.9)
        usable = touching & (self.last_touch >= 0) & (periods >= 0.15) & rhythm[:, None]
        self.period_sum += torch.where(usable, periods, 0).sum(0)
        self.period_square += torch.where(usable, periods.square(), 0).sum(0)
        self.period_count += usable.sum(0)
        self.last_touch[touching] = now
        self.last_touch[done | changed] = -1
        self.contacts = contact
        self.contacts[done] = False
        if valid.any():
            self.slip += float((feet_velocity * contact)[valid].mean(-1).sum())
            self.tilt += float(c.projected_gravity[valid, :2].square().sum(-1).sum())
        if rhythm.any():
            agreement = ((contact[:, 0] == contact[:, 3]).float() +
                         (contact[:, 1] == contact[:, 2]).float()) / 2
            self.diagonal_agreement += float(agreement[rhythm].sum())
            self.rhythm_samples += int(rhythm.sum())
        if self.steps % self.every_steps == 0:
            self.snapshots.append(dict(step=self.steps, mae=self.report()['processed_command_mae']))
            self.save()

    def report(self):
        c = self.env.core
        count = max(self.samples, 1)
        n = max(c._torque_samples, 1)
        rms = (c._torque_square.sum(0) / (n * c.num_envs)).sqrt()
        worst = (c._torque_square / n).sqrt().amax(0)
        periods = self.period_sum / self.period_count.clamp(min=1)
        variance = (self.period_square / self.period_count.clamp(min=1) - periods.square()).clamp(min=0)
        group_rows = {}
        for name, row in zip(('hold', 'standby', 'low_translation', 'pure_yaw', 'normal', 'mixed'), self.groups.cpu().tolist()):
            group_rows[name] = dict(samples=int(row[0]), mae=[x / row[0] for x in row[1:]] if row[0] else None)
        def tracking(tensor):
            return tensor.cpu().tolist() if self.samples else None
        def contact_values(tensor, minimum=1):
            return [float(value) if n >= minimum else None for value, n in zip(tensor.cpu(), self.period_count.cpu())]
        return dict(
            status='execution_diagnostics_only', sim_seconds=self.steps * c.dt,
            environment_steps=self.steps * c.num_envs, resets=self.resets, steady_samples=self.samples,
            processed_command_mae=tracking(self.error / count),
            raw_command_mae=tracking(self.raw_error / count),
            velocity_rms=tracking((self.velocity_square / count).sqrt()),
            velocity_mean=tracking(self.velocity_sum / count),
            velocity_std=tracking((self.velocity_square / count - (self.velocity_sum / count).square()).clamp(min=0).sqrt()),
            processed_command_mean=tracking(self.command_sum / count),
            stance_slip_mean=self.slip / count, tilt_xy_square_mean=self.tilt / count,
            diagonal_contact_agreement=self.diagonal_agreement / max(self.rhythm_samples, 1),
            foot_names=c.cfg.gait.foot_names, contact_period_count=self.period_count.cpu().tolist(),
            contact_period_mean=contact_values(periods),
            contact_period_cv=contact_values(variance.sqrt() / periods.clamp(min=1e-6), minimum=2),
            joint_names=c.dof_names, torque_rms_nm=rms.cpu().tolist(),
            worst_environment_torque_rms_nm=worst.cpu().tolist(),
            torque_peak_nm=c._torque_peak.amax(0).cpu().tolist(),
            saturation_fraction=(c._torque_saturation.sum(0) / (n * c.num_envs)).cpu().tolist(),
            groups=group_rows, snapshots=self.snapshots,
            posture=self.posture.report(),
            notes=['Steady tracking excludes first second after a command change/reset.',
                   'Torque statistics sample every physics substep, including transients.',
                   'No calibrated thermal model; URDF effort is not a continuous rating.',
                   'Missing steady/period samples are null, never zero-error success.'])

    def save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        result = self.report()
        (self.directory / 'gait_metrics.json').write_text(json.dumps(result, indent=2))
        return result
