"""Pure tensor gait/command state, also used by deterministic playback.

WTW supplies the phase/contact/foot-position idea. This reference uses the
full planar rigid-body velocity and explicit stance duration, not WTW constants.
"""
import math
import torch


def sample_commands(count, cfg, ranges, device):
    weights = torch.tensor(cfg.sample_weights, device=device)
    groups = torch.multinomial(weights, count, replacement=True)
    target = torch.zeros(count, 3, device=device)
    standby = (groups == 1).float()
    def signed(low, high, size):
        sign = torch.where(torch.rand(size, device=device) < 0.5, -1.0, 1.0)
        return sign * (low + (high - low) * torch.rand(size, device=device))
    low = groups == 2
    n = int(low.sum())
    axes = torch.randint(0, 2, (n,), device=device)
    target[low, axes] = signed(*cfg.low_linear, n)
    yaw = groups == 3
    target[yaw, 2] = signed(*cfg.low_yaw, int(yaw.sum()))
    normal = groups == 4
    target[normal, 0] = signed(0.12, ranges.lin_vel_x[1], int(normal.sum()))
    mixed = groups == 5
    for axis, limits in enumerate([ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_yaw]):
        target[mixed, axis] = limits[0] + (limits[1] - limits[0]) * torch.rand(int(mixed.sum()), device=device)
        target[:, axis].clamp_(limits[0], limits[1])
    return target, standby, groups


class GaitState:
    def __init__(self, count, cfg, device):
        self.cfg = cfg
        self.phase = torch.zeros(count, device=device)
        self.activation = torch.zeros(count, device=device)
        self.command = torch.zeros(count, 3, device=device)
        self.target = torch.zeros_like(self.command)
        self.standby = torch.zeros(count, device=device)
        self.groups = torch.zeros(count, device=device, dtype=torch.long)
        self.offsets = torch.tensor(cfg.phase_offsets, device=device)

    def reset(self, ids):
        # Both diagonal pairs are in stance near phase 0 and 0.5 (duty > 0.5).
        self.phase[ids] = torch.randint(0, 2, (len(ids),), device=self.phase.device) * 0.5
        self.activation[ids] = 0
        self.command[ids] = 0

    def advance(self, dt):
        alpha = 1 - math.exp(-dt / self.cfg.command_tau)
        self.command += alpha * (self.target - self.command)
        zero = self.target.abs().amax(-1) <= self.cfg.zero_epsilon
        settled = self.command.abs().amax(-1) < self.cfg.zero_epsilon
        self.command[zero & settled] = 0
        wanted = ~zero | (self.standby > 0.5)
        phase = self.foot_phase
        safe = (phase < self.cfg.duty_factor - self.cfg.contact_blend).all(-1)
        # Finish the current swing before fading to a stationary reference.
        continuing = wanted | (~safe & (self.activation > 0)) | (~settled & zero)
        delta = dt / self.cfg.transition_seconds
        self.activation = (self.activation + torch.where(continuing, delta, -delta)).clamp(0, 1)
        self.phase = torch.remainder(self.phase + dt * self.cfg.frequency * continuing, 1.0)

    @property
    def foot_phase(self):
        return torch.remainder(self.phase[:, None] + self.offsets, 1.0)

    def features(self):
        angle = 2 * math.pi * self.phase
        return torch.stack((angle.sin(), angle.cos(), self.activation, self.standby), -1)

    def reference(self, nominal):
        phi = self.foot_phase
        duty = self.cfg.duty_factor
        swing = ((phi - duty) / (1 - duty)).clamp(0, 1)
        smooth = swing.square() * (3 - 2 * swing)
        progress = torch.where(phi < duty, 0.5 - phi / duty, -0.5 + smooth)
        vx, vy, yaw = self.command.unbind(-1)
        velocity = torch.stack((vx[:, None] - yaw[:, None] * nominal[None, :, 1],
                                vy[:, None] + yaw[:, None] * nominal[None, :, 0]), -1)
        xy = nominal[None, :, :2] + progress[..., None] * velocity * (duty / self.cfg.frequency)
        height = (math.pi * swing).sin().square() * self.cfg.swing_height * self.activation[:, None]
        width = self.cfg.contact_blend
        # Continuous contact target across stance/swing AND cycle wrap.
        contact = ((duty - phi) / width).clamp(0, 1) * (phi / width).clamp(0, 1)
        contact = 1 - self.activation[:, None] * (1 - contact)
        return xy, height, contact
