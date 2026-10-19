#!/usr/bin/env python3
"""Sequential, bounded diagnostic arms from ONE checkpoint; no phase progression.

Creates new directories, retains all checkpoints, and stops on any failed stage.
Use STOP_REQUESTED in the output root to stop safely BETWEEN operations.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'isaaclab_migration'))
from himloco_lab.task_gait.posture import EXPERIMENTS


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def compact(evaluation):
    """Keep per-command posture; never average away a dragging leg."""
    result = {}
    for name, case in evaluation['cases'].items():
        p = case['posture']
        result[name] = dict(survival=case['survival_fraction'],
                            command_mae=case['processed_command_mae'],
                            height_m=p['height_mean_m'], height_mae_m=p['height_mae_m'],
                            low_height_fraction=p['height_below_target_minus_5cm_fraction'],
                            hip_mean_rad=p['hip_mean_rad'], near_limit=p['joint_near_limit_fraction'],
                            swing_touch=p['swing_contact_fraction'],
                            apex_clearance_m=p['apex_clearance_mean_m'],
                            slip=case['stance_slip_mean'], tilt=case['tilt_xy_square_mean'],
                            torque_peak_nm=case['torque_peak_nm'])
    return result


def comparisons(results):
    """Descriptive paired differences; one seed cannot establish a unique cause."""
    control = results.get('baseline')
    if control is None:
        return dict(status='baseline_not_available', evidence={})
    evidence = {}
    for arm, cases in results.items():
        if arm == 'baseline':
            continue
        reference = results.get('combined') if arm == 'combined_tracking1' else control
        if reference is None:
            continue
        paired = {}
        for name, case in cases.items():
            old = reference[name]
            differences = {}
            for key in ('height_mae_m', 'low_height_fraction', 'slip', 'tilt'):
                differences[key] = case[key] - old[key] if case[key] is not None and old[key] is not None else None
            for key in ('command_mae', 'swing_touch', 'apex_clearance_m'):
                differences[key] = [x-y if x is not None and y is not None else None
                                    for x, y in zip(case[key], old[key])]
            differences['survival'] = case['survival'] - old['survival']
            paired[name] = differences
        evidence[arm] = dict(reference='combined' if arm == 'combined_tracking1' else 'baseline',
                             differences_new_minus_reference=paired)
    return dict(status='short_single_seed_adaptation_evidence_only', evidence=evidence,
                follow_up='Compare normal forward/backward and low speed separately. Repeat control and promising arms to 800-1000 updates and another seed before attributing a main cause.')


def log_evaluation(training, evaluation, updates):
    from torch.utils.tensorboard import SummaryWriter
    from himloco_lab.task_gait.run_log import write_posture_scalars
    # Separate writer closes immediately; training's writer/process has exited.
    with SummaryWriter(str(training)) as writer:
        for name, report in evaluation['cases'].items():
            class CaseWriter:
                def add_scalar(self, tag, value, step):
                    writer.add_scalar('Evaluation/' + name + '/' + tag, value, step)
            write_posture_scalars(CaseWriter(), report, updates)
            writer.add_scalar('Evaluation/' + name + '/survival', report['survival_fraction'], updates)
            for axis, error in zip(('vx', 'vy', 'yaw'), report['processed_command_mae'] or []):
                writer.add_scalar('Evaluation/' + name + '/command_mae/' + axis, error, updates)
        writer.add_text('Run/fixed_evaluation', '```json\n' + json.dumps(compact(evaluation), indent=2) + '\n```', updates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_envs', type=int, default=1024)
    parser.add_argument('--updates', type=int, default=300)
    parser.add_argument('--rollout_steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--arms', nargs='+', choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    if min(args.num_envs, args.updates, args.rollout_steps) < 1:
        parser.error('envs, updates and rollout must be positive')
    if len(set(args.arms)) != len(args.arms):
        parser.error('Duplicate arms would overwrite outputs')
    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.is_file():
        parser.error('Checkpoint not found')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'experiment_plan.json').exists():
        parser.error('Use a new output directory; existing experiment preserved')
    write_json(args.output / 'experiment_plan.json', dict(
        source_checkpoint=str(args.checkpoint), source_is_shared=True,
        phase='precision', arms=args.arms, seed=args.seed, updates=args.updates,
        num_envs=args.num_envs, rollout_steps=args.rollout_steps,
        transitions_per_arm=args.num_envs * args.updates * args.rollout_steps,
        comparison='One seed, short adaptation screen; no causal proof or convergence claim.',
        follow_up='Review individual posture + tracking. Extend control and promising arm together to 800-1000 updates; do not select on reward alone.'))
    env = dict(os.environ)
    env['PYTHONPATH'] = f'{ROOT / "isaaclab_migration"}:{ROOT / "rsl_rl"}' + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PYTHONUNBUFFERED'] = '1'
    env['TASK_GAIT_PYTHON'] = sys.executable
    env['MPLCONFIGDIR'] = str(args.output / 'matplotlib_cache')
    status = dict(state='running', current_arm=None, operation=None, completed=[], results={})

    def update():
        status['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(args.output / 'status.json', status)

    def run(command, log, timeout):
        if (args.output / 'STOP_REQUESTED').exists():
            raise InterruptedError('STOP_REQUESTED: stopped between operations')
        with log.open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            status['child_pid'] = process.pid
            update()
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        status.pop('child_pid', None)
        if code:
            raise RuntimeError(f'Command exited {code}; see {log}')

    def stop(_signal, _frame):
        raise InterruptedError('Supervisor interrupted; active operation stopped')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for arm in args.arms:
            status.update(current_arm=arm, operation='train')
            update()
            folder = args.output / arm
            folder.mkdir()
            training = folder / 'train'
            command = ['bash', 'isaaclab_migration/scripts/task_gait.sh', 'train', '--no-videos',
                       '--phase', 'precision', '--posture_experiment', arm,
                       '--num_envs', str(args.num_envs), '--rollout_steps', str(args.rollout_steps),
                       '--iterations', str(args.updates), '--seed', str(args.seed),
                       '--checkpoint', str(args.checkpoint), '--log_dir', str(training)]
            run(command, folder / 'train_launcher.log', timeout=3600)
            validation = json.loads((training / 'validation.json').read_text())
            if validation.get('status') != 'passed':
                raise RuntimeError(f'Training validation failed: {training}')
            checkpoint = training / f'model_{validation["iterations"]}.pt'
            if not checkpoint.is_file():
                raise RuntimeError(f'Final checkpoint absent: {checkpoint}')
            status['operation'] = 'evaluate'
            update()
            evaluation = folder / 'evaluation'
            run(['bash', 'isaaclab_migration/scripts/task_gait.sh', 'evaluate',
                 '--checkpoint', str(checkpoint), '--posture_scan', '--seconds', '8',
                 '--num_envs', '32', '--seed', str(args.seed), '--output', str(evaluation)],
                folder / 'evaluation_launcher.log', timeout=600)
            report = json.loads((evaluation / 'evaluation.json').read_text())
            status['results'][arm] = compact(report)
            log_evaluation(training, report, args.updates)
            write_json(args.output / 'comparison.json', status['results'])
            write_json(args.output / 'analysis.json', comparisons(status['results']))
            if not args.no_video:
                status['operation'] = 'record_final_checkpoint'
                update()
                # Only the final model, 8 seconds / 4 env / 25 fps.
                run(['bash', 'isaaclab_migration/scripts/task_gait.sh', 'record', str(training),
                     '--only', str(validation['iterations']), '--play_steps', '400',
                     '--capture_every_steps', '2', '--num_envs', '4'],
                    folder / 'record_launcher.log', timeout=300)
            status['completed'].append(arm)
            update()
        status.update(state='screen_complete_review_required', operation=None, current_arm=None)
        update()
        print('POSTURE_SCREEN_COMPLETE', args.output, flush=True)
    except BaseException as error:
        status.update(state='stopped' if isinstance(error, InterruptedError) else 'failed', error=str(error))
        update()
        raise


if __name__ == '__main__':
    main()
