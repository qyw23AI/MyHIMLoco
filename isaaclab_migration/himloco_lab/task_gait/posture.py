"""Opt-in posture experiments; unchanged observations and legacy reward behavior."""
from types import SimpleNamespace
import torch


EXPERIMENTS = ('baseline', 'height', 'hip', 'gait', 'combined',
               'combined_tracking1', 'combined_tracking1p5')


def apply_experiment(cfg, name):
    """Each arm starts from the same reference config, never another arm's result.

    These are diagnostic scales, not a calibrated hardware-safe controller.
    The old task objectives remain available for historical checkpoints.
    """
    if name not in EXPERIMENTS:
        raise ValueError(name)
    cfg.gait.posture = SimpleNamespace(
        experiment=name, height_sigma=.04, hip_sigma=.15,
        hip_margin_straight=.15, hip_margin_turn=.30,
        joint_limit_margin=.10, joint_limit_sigma=.10,
        clearance_sigma=.025, position_sigma=.06,
    )
    scales = cfg.rewards.scales
    for term in ('height', 'hip', 'limit', 'clearance', 'position'):
        setattr(scales, 'posture_' + term, 0.)
    # Always restore the reference arm so sequential application is idempotent.
    scales.base_height = -1.
    scales.tracking_lin_vel = 2.
    scales.gait_contact, scales.gait_clearance, scales.gait_position = -.3, -.2, -.15
    combined = name.startswith('combined')
    if name == 'height' or combined:
        scales.base_height = 0.
        scales.posture_height = -.25
    if name == 'hip' or combined:
        scales.posture_hip, scales.posture_limit = -.3, -.1
    if name == 'gait' or combined:
        # Replace saturated spatial terms rather than double-count them.
        scales.gait_clearance = scales.gait_position = 0.
        scales.gait_contact = -1.
        scales.posture_clearance, scales.posture_position = -.5, -.3
    if name == 'combined_tracking1':
        scales.tracking_lin_vel = 1.
    elif name == 'combined_tracking1p5':
        scales.tracking_lin_vel = 1.5


class PostureRewards:
    """Task-only costs, invoked by the existing reward dispatcher when enabled."""

    def _reward_posture_height(self):
        return ((self._get_base_heights() - self.cfg.rewards.base_height_target)
                / self.cfg.gait.posture.height_sigma).square()

    def _reward_posture_hip(self):
        p = self.cfg.gait.posture
        # Straight forward/backward motion keeps a narrower operating range;
        # lateral/yaw commands smoothly relax it, with no command dead zone.
        turn = torch.maximum(self.commands[:, 1].abs() / .2,
                             self.commands[:, 2].abs() / .6).clamp(0, 1)
        margin = p.hip_margin_straight + turn * (p.hip_margin_turn - p.hip_margin_straight)
        error = (self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]).abs()
        return ((error - margin[:, None]).clamp(min=0) / p.hip_sigma).square().mean(-1)

    def _reward_posture_limit(self):
        p = self.cfg.gait.posture
        # Cost begins BEFORE mechanical limits, including calf extension stops.
        distance = torch.minimum(self.dof_pos - self.dof_pos_limits[:, 0],
                                 self.dof_pos_limits[:, 1] - self.dof_pos)
        return ((p.joint_limit_margin - distance).clamp(min=0)
                / p.joint_limit_sigma).square().mean(-1)

    def _reward_posture_clearance(self):
        _, lift, _ = self.gait.reference(self.nominal_feet)
        positions = self.rigid_body_states[:, self.gait_feet, :3]
        ground = (torch.zeros_like(lift) if self.backend.terrain is None else
                  self.backend.terrain.get_terrain_heights(positions.clone()))
        target = self.cfg.gait.foot_radius + lift
        # Includes stance: a foot held airborne throughout its cycle is wrong too.
        return ((positions[..., 2] - ground - target)
                / self.cfg.gait.posture.clearance_sigma).square().mean(-1)

    def _reward_posture_position(self):
        target, _, _ = self.gait.reference(self.nominal_feet)
        return ((self._foot_frame()[..., :2] - target)
                / self.cfg.gait.posture.position_sigma).square().sum(-1).mean(-1)


class PostureStats:
    """Streaming per-leg checks shared by training TensorBoard and evaluation."""

    def __init__(self, core):
        self.c = core
        self.count = 0
        self.height_sum = self.height_error = self.low_count = 0.
        self.hip_ids = [i for i, n in enumerate(core.dof_names) if 'hip' in n.lower()]
        def zeros(n):
            return torch.zeros(n, device=core.device)
        self.hip_sum, self.near_limit = zeros(len(self.hip_ids)), zeros(core.num_dof)
        self.swing_count, self.swing_touch, self.lift_error = zeros(4), zeros(4), zeros(4)
        self.apex_count, self.apex_height = zeros(4), zeros(4)

    def update(self, valid):
        c = self.c
        n = int(valid.sum())
        if not n:
            return
        self.count += n
        height = c._get_base_heights()[valid]
        self.height_sum += float(height.sum())
        self.height_error += float((height - c.cfg.rewards.base_height_target).abs().sum())
        self.low_count += float((height < c.cfg.rewards.base_height_target - .05).sum())
        self.hip_sum += c.dof_pos[valid][:, self.hip_ids].sum(0)
        distance = torch.minimum(c.dof_pos - c.dof_pos_limits[:, 0],
                                 c.dof_pos_limits[:, 1] - c.dof_pos)
        self.near_limit += (distance[valid] < .02).sum(0)
        _, lift, contact = c.gait.reference(c.nominal_feet)
        moving = c.gait.activation > .9
        swing = (contact < .1) & (valid & moving)[:, None]
        touch = c.contact_forces[:, c.gait_feet, 2] > 1.
        pos = c.rigid_body_states[:, c.gait_feet, :3]
        ground = (torch.zeros_like(lift) if c.backend.terrain is None else
                  c.backend.terrain.get_terrain_heights(pos.clone()))
        clearance = pos[..., 2] - ground - c.cfg.gait.foot_radius
        self.swing_count += swing.sum(0)
        self.swing_touch += (swing & touch).sum(0)
        self.lift_error += ((clearance - lift).abs() * swing).sum(0)
        apex = swing & (lift > .8 * c.cfg.gait.swing_height)
        self.apex_count += apex.sum(0)
        self.apex_height += (clearance * apex).sum(0)

    def report(self):
        def ratio(total, count):
            return [float(v / n) if n > 0 else None
                    for v, n in zip(total.cpu(), count.cpu())]
        n = max(self.count, 1)
        return dict(samples=self.count,
                    height_mean_m=self.height_sum / n if self.count else None,
                    height_mae_m=self.height_error / n if self.count else None,
                    height_below_target_minus_5cm_fraction=self.low_count / n if self.count else None,
                    hip_mean_rad=dict(zip((self.c.dof_names[i] for i in self.hip_ids),
                                         (self.hip_sum / n).cpu().tolist())) if self.count else None,
                    joint_near_limit_fraction=dict(zip(self.c.dof_names,
                                                      (self.near_limit / n).cpu().tolist())) if self.count else None,
                    swing_samples=self.swing_count.cpu().tolist(),
                    swing_contact_fraction=ratio(self.swing_touch, self.swing_count),
                    swing_clearance_mae_m=ratio(self.lift_error, self.swing_count),
                    apex_clearance_mean_m=ratio(self.apex_height, self.apex_count))
