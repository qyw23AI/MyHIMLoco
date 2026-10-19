#!/usr/bin/env python3
"""Bottom-level velocity scans and mode transitions, not navigation/yaw-position control."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'isaaclab_migration'))
sys.path.insert(0, str(ROOT / 'rsl_rl'))


def cases(short=False):
    result = [('hold', [0., 0., 0.], False), ('standby', [0., 0., 0.], True)]
    if short:
        return result + [('low_x', [.03, 0., 0.], False), ('yaw', [0., 0., .06], False),
                         ('transitions', None, False)]
    for axis, values in enumerate(([.015, .03, .08, .2], [.015, .03, .08], [.03, .06, .15, .4])):
        for value in values:
            for sign in (-1, 1):
                command = [0., 0., 0.]
                command[axis] = sign * value
                result.append((f'axis{axis}_{sign * value:+.3f}', command, False))
    result.append(('mixed', [.15, .05, .15], False))
    result.append(('transitions', None, False))
    return result


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--policy', type=Path)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--posture_scan', action='store_true',
                        help='Short suite plus normal forward/backward; preserves 5-second transitions')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--command', nargs=3, type=float, default=None)
    parser.add_argument('--standby', action='store_true')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.command is None:
        args.command = [0., 0., 0.] if args.standby else [.2, 0., 0.]
    if args.num_envs < 1 or args.seconds <= 0:
        parser.error('num_envs and seconds must be positive')
    app = AppLauncher(args).app
    env = None
    try:
        import csv
        import json
        import time
        import numpy as np
        import torch
        from himloco_lab.config import load_config, class_to_dict
        from himloco_lab.env_cfg import make_env_cfg
        from himloco_lab.env import HIMLabEnv
        from himloco_lab.adapter import LegacyRunnerAdapter
        from himloco_lab.task_gait.checkpoint import saved_task_config, config_from_policy
        from himloco_lab.task_gait.metrics import TaskGaitMonitor
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if args.checkpoint:
            from rsl_rl.modules import HIMActorCritic
            from himloco_lab.export import export_policy
            cfg = saved_task_config(args.checkpoint)
            cfg.env.num_envs = args.num_envs
            _, train = load_config('mybot_v3_task_gait')
            state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
            train = state['infos'].get('training_config', train)
            model = HIMActorCritic(cfg.env.num_observations, cfg.env.num_privileged_obs,
                                  cfg.env.num_one_step_observations, cfg.env.num_actions,
                                  **train['policy']).to(args.device)
            model.load_state_dict(state['model_state_dict'])
            args.output.mkdir(parents=True, exist_ok=True)
            export_policy(model, args.output / 'policy.pt', cfg, args.checkpoint)
            policy = torch.jit.load(str(args.output / 'policy.pt'), map_location=args.device).eval()
        else:
            cfg, meta = config_from_policy(args.policy, args.num_envs)
            from himloco_lab.config import asset_contract
            if meta['joint_order'] != asset_contract(cfg)[1] or meta['input_shape'][-1] != cfg.env.num_observations:
                raise ValueError('Exported policy contract mismatch')
            policy = torch.jit.load(str(args.policy), map_location=args.device).eval()
        cfg.noise.add_noise = False
        for name in vars(cfg.domain_rand):
            if isinstance(getattr(cfg.domain_rand, name), bool):
                setattr(cfg.domain_rand, name, False)
        # Evaluate the selected terrain profile at fixed, reproducible spawn levels.
        cfg.terrain.curriculum = False
        lab_cfg = make_env_cfg(cfg, 4 if cfg.gait.phase == 'robust' else 3, args.device)
        lab_cfg.seed = args.seed
        env = HIMLabEnv(lab_cfg)
        wrapped = LegacyRunnerAdapter(env)
        c = env.core
        c.external_commands = True
        suite = [('play', args.command, args.standby)] if args.play else cases(args.short)
        if args.posture_scan:
            if args.play:
                parser.error('--posture_scan cannot be combined with --play')
            suite = cases(True) + [('forward', [.5, 0., 0.], False),
                                   ('backward', [-.5, 0., 0.], False)]
        args.output.mkdir(parents=True, exist_ok=True)
        reports, curves = {}, []
        with torch.inference_mode():
            for name, command, standby in suite:
                folder = args.output / name
                monitor = TaskGaitMonitor(env, folder, every_steps=100000000)
                wrapped.diagnostics = None
                c.set_command(command or [0., 0., 0.], standby)
                wrapped.reset()
                monitor.reset()
                wrapped.diagnostics = monitor
                start = c.root_states[:, :2].clone()
                baseline = None
                drift_max = torch.zeros(c.num_envs, device=c.device)
                survived = torch.ones(c.num_envs, device=c.device, dtype=torch.bool)
                rows = []
                case_seconds = max(30., args.seconds) if args.posture_scan and command is None else args.seconds
                for step in range(round(case_seconds / c.dt)):
                    if not app.is_running():
                        raise RuntimeError('Simulation closed before evaluation completed')
                    tick = time.monotonic()
                    if command is None:
                        index = min(int(step * c.dt / (case_seconds / 6)), 5)
                        sequence = [([.2, 0, 0], False), ([.03, 0, 0], False),
                                    ([0, 0, 0], False), ([0, 0, .1], False),
                                    ([0, 0, 0], True), ([0, 0, 0], False)]
                        c.set_command(*sequence[index])
                    c.prepare_external_observation()
                    action = policy(wrapped.get_observations())
                    result = wrapped.step(action)
                    if not all(torch.isfinite(t).all() for t in (action, result[0], result[2])):
                        raise RuntimeError('Nonfinite evaluation state')
                    survived &= ~result[3]
                    if baseline is None and step * c.dt >= 2.0:
                        baseline = c.root_states[:, :2].clone()
                    if baseline is not None:
                        drift = (c.root_states[:, :2] - baseline).norm(dim=-1)
                        drift_max = torch.maximum(drift_max, drift)
                    velocity = torch.cat((c.base_lin_vel[:, :2], c.base_ang_vel[:, 2:3]), -1)
                    rows.append([step * c.dt, *c.gait.target[0].tolist(), *c.commands[0, :3].tolist(),
                                 *velocity.mean(0).tolist(), float(c.gait.activation[0]),
                                 int(result[3].sum()), *c.contact_forces[0, c.gait_feet, 2].tolist()])
                    if args.play and not args.headless:
                        time.sleep(max(0., c.dt - (time.monotonic() - tick)))
                report = monitor.save()
                report['survival_fraction'] = float(survived.float().mean())
                report['command'] = command
                report['standby'] = standby
                report['mean_net_displacement_m'] = float((c.root_states[:, :2] - start).norm(dim=-1)[survived].mean()) if survived.any() else None
                valid_drift = survived & (baseline is not None)
                report['max_drift_after_2s_m'] = float(drift_max[valid_drift].max()) if valid_drift.any() and args.seconds > 2 else None
                reports[name] = report
                with (folder / 'trace.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['seconds', 'raw_vx', 'raw_vy', 'raw_yaw', 'cmd_vx', 'cmd_vy', 'cmd_yaw',
                                     'mean_vx', 'mean_vy', 'mean_yaw', 'activation', 'resets', 'FL_fz', 'FR_fz', 'RL_fz', 'RR_fz'])
                    writer.writerows(rows)
                curves.append((name, rows))
                print('CASE_COMPLETE', name, 'MAE', report['processed_command_mae'], flush=True)
        output = dict(status='execution_validated_not_performance_approved', seed=args.seed,
                      task_config=class_to_dict(cfg), cases=reports)
        (args.output / 'evaluation.json').write_text(json.dumps(output, indent=2))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        # One transition trace is more readable than dozens of overlapping scans.
        name, rows = next(((n, r) for n, r in curves if n == 'transitions'), curves[-1])
        values = np.asarray(rows)
        for axis, label in enumerate(('vx (m/s)', 'vy (m/s)', 'yaw rate (rad/s)')):
            axes[axis].plot(values[:, 0], values[:, 4 + axis], label='processed command')
            axes[axis].plot(values[:, 0], values[:, 7 + axis], label='mean measured')
            axes[axis].set_ylabel(label)
            axes[axis].grid(alpha=.3)
        axes[0].legend()
        axes[0].set_title(name)
        axes[-1].set_xlabel('simulation seconds')
        fig.tight_layout()
        fig.savefig(args.output / 'tracking.png', dpi=150)
        plt.close(fig)
        print('EVALUATION_SAVED', args.output.resolve(), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
