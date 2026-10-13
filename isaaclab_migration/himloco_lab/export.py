"""Export the actual repository HIM encoder + actor as standalone TorchScript."""
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class HIMInferencePolicy(nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.encoder = copy.deepcopy(actor_critic.estimator.encoder)
        self.one_step_obs = int(actor_critic.num_one_step_obs)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        parts = self.encoder(obs_history)
        velocity = parts[..., :3]
        latent = F.normalize(parts[..., 3:], p=2.0, dim=-1)
        return self.actor(torch.cat((obs_history[..., :self.one_step_obs], velocity, latent), dim=-1))


def export_policy(actor_critic, output, cfg, checkpoint=None):
    from .config import asset_contract

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = HIMInferencePolicy(actor_critic).cpu().eval()
    scripted = torch.jit.script(policy)
    scripted.save(str(output))
    reloaded = torch.jit.load(str(output), map_location='cpu').eval()
    sample = torch.randn(8, actor_critic.num_actor_obs)
    # PPO's cached Normal distribution holds non-leaf tensors; do not deepcopy it.
    device = next(actor_critic.parameters()).device
    was_training = actor_critic.training
    actor_critic.eval()
    try:
        with torch.no_grad():
            expected = actor_critic.act_inference(sample.to(device)).cpu()
            actual = reloaded(sample)
    finally:
        actor_critic.train(was_training)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    if not torch.isfinite(actual).all():
        raise RuntimeError('Export produced nonfinite actions')
    _, joint_names, _, _ = asset_contract(cfg)
    meta = {
        'format': 'TorchScript',
        'input_shape': ['batch', actor_critic.num_actor_obs],
        'output_shape': ['batch', cfg.env.num_actions],
        'one_step_obs': cfg.env.num_one_step_observations,
        'history_length': cfg.env.num_observations // cfg.env.num_one_step_observations,
        'history_order': 'newest frame first; preserve history across reset as in original code',
        'frame_order': ['scaled commands xyz', 'scaled base angular velocity', 'projected gravity',
                        'scaled joint position offset', 'scaled joint velocity', 'actions'],
        'joint_order': joint_names,
        'action_scale': cfg.control.action_scale,
        'hip_reduction': cfg.control.hip_reduction,
        'clip_actions': cfg.normalization.clip_actions,
        'clip_observations': cfg.normalization.clip_observations,
        'physics_dt': cfg.sim.dt,
        'control_dt': cfg.sim.dt * cfg.control.decimation,
        'default_joint_angles': cfg.init_state.default_joint_angles,
        'checkpoint': str(Path(checkpoint).resolve()) if checkpoint else None,
        'round_trip_max_error': float((actual - expected).abs().max()),
    }
    output.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    return meta
