"""Actual Lab acceptance of task reset, terminal targets and gait command state."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'isaaclab_migration'))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['basic', 'robust'], default='basic')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
env = None
try:
    import torch
    from himloco_lab.config import load_config
    from himloco_lab.env_cfg import make_env_cfg
    from himloco_lab.env import HIMLabEnv
    from himloco_lab.adapter import LegacyRunnerAdapter
    from himloco_lab.validation import validate_runtime
    cfg, _ = load_config('mybot_v3_task_gait', 8, gait_phase=args.phase)
    cfg.noise.add_noise = False
    env = HIMLabEnv(make_env_cfg(cfg, 4 if args.phase == 'robust' else 3, args.device))
    c = env.core
    wrapped = LegacyRunnerAdapter(env)
    c.external_commands = True
    c.set_command([.03, 0., .06])
    wrapped.reset()
    c.prepare_external_observation()
    for _ in range(10):
        wrapped.step(torch.zeros(8, 12, device=env.device))
    assert (c.commands[:, 0] > 0).all(), 'Low command was swallowed'
    assert (c.commands[:, 2] > 0).all(), 'Small yaw was swallowed'
    previous = c.obs_buf.clone()
    phases = c.gait.phase.clone()
    env._reset_idx(torch.tensor([0], device=env.device))
    torch.testing.assert_close(c.gait.phase[1:], phases[1:])
    torch.testing.assert_close(c.obs_buf[1:], previous[1:])
    assert not c.obs_buf[0].any()
    captured = {}
    original = c.compute_termination_observations
    def capture(ids):
        target = original(ids)
        captured['ids'] = ids.clone()
        captured['target'] = target.clone()
        return target
    c.compute_termination_observations = capture
    env.episode_length_buf[1] = int(c.max_episode_length)
    result = wrapped.step(torch.zeros(8, 12, device=env.device))
    assert 1 in result[5].tolist()
    torch.testing.assert_close(result[6], captured['target'])
    torch.testing.assert_close(c.obs_buf[1], c.obs_buf[1, :49].repeat(6))
    assert c._torque_samples == 12 * cfg.control.decimation
    assert torch.isfinite(c._reward_gait_load()).all()
    if args.phase == 'robust':
        assert c.terrain.height_field_raw.max() > c.terrain.height_field_raw.min()
        assert type(c.terrain).__name__ == 'GentleTerrain'
    report = validate_runtime(env)
    report.update(status='passed', phase=args.phase,
                  checks=['small translation/yaw preserved', 'isolated partial reset',
                          'terminal target before reset', 'new-frame reset history',
                          'per-physics-step load sampling', 'finite task rewards'])
    output = ROOT / 'isaaclab_migration/outputs/task_gait_checks'
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{args.phase}_sim.json').write_text(json.dumps(report, indent=2))
    print('TASK_GAIT_SIM_PASSED', json.dumps(report), flush=True)
finally:
    if env is not None:
        env.close()
    app.close()
