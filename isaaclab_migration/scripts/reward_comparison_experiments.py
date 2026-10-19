#!/usr/bin/env python3
"""Paired from-scratch comparison of current task-gait vs legacy Mybot V3 rewards."""
import argparse
import datetime
import json
import os
import select
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable).resolve()
ARMS = ('task_gait_current', 'task_gait_no_gait_rewards', 'legacy_rewards')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def compact(evaluation):
    output = {}
    for name, case in evaluation['cases'].items():
        posture = case['posture']
        output[name] = dict(
            survival=case['survival_fraction'],
            command_mae=case['processed_command_mae'],
            height_m=posture['height_mean_m'],
            height_mae_m=posture['height_mae_m'],
            low_height_fraction=posture['height_below_target_minus_5cm_fraction'],
            hip_mean_rad=posture['hip_mean_rad'],
            near_limit=posture['joint_near_limit_fraction'],
            swing_touch=posture['swing_contact_fraction'],
            apex_clearance_m=posture['apex_clearance_mean_m'],
            slip=case['stance_slip_mean'],
            tilt=case['tilt_xy_square_mean'],
            torque_rms_nm=case.get('torque_rms_nm'),
            torque_peak_nm=case['torque_peak_nm'])
    return output


def paired_differences(results):
    paired = {arm: {} for arm in ARMS if arm != 'task_gait_current'}
    for seed in results['task_gait_current']:
        current = results['task_gait_current'][seed]
        for arm in paired:
            variant = results[arm].get(seed)
            if variant is None:
                continue
            per_case = {}
            for name, new in variant.items():
                old = current[name]
                delta = {}
                for key in ('height_mae_m', 'low_height_fraction', 'slip', 'tilt', 'survival'):
                    a, b = new[key], old[key]
                    delta[key] = None if a is None or b is None else a - b
                for key in ('command_mae', 'swing_touch', 'apex_clearance_m'):
                    a, b = new[key], old[key]
                    delta[key] = [None if x is None or y is None else x - y
                                  for x, y in zip(a, b)]
                per_case[name] = delta
            paired[arm][seed] = per_case
    return paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_envs', type=int, default=1024)
    parser.add_argument('--rollout_steps', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17])
    parser.add_argument('--eval_envs', type=int, default=32)
    parser.add_argument('--eval_seconds', type=float, default=8.)
    parser.add_argument('--train_timeout', type=int, default=3600)
    parser.add_argument('--eval_timeout', type=int, default=600)
    parser.add_argument('--record_timeout', type=int, default=300)
    args = parser.parse_args()
    if min(args.num_envs, args.rollout_steps, args.iterations, args.eval_envs) < 1:
        parser.error('environment, rollout, iteration and evaluation counts must be positive')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be unique and nonempty')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f'Output directory is not empty; refusing to overwrite: {output}')
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(
        comparison='Three-arm from-scratch paired comparison; reward configuration is the only intended difference.',
        task='mybot_v3_task_gait', phase='precision', source_checkpoint=None,
        initialization='random actor/critic initialization; no --checkpoint supplied',
        legacy_reward_source=str(ROOT / 'isaaclab_migration/himloco_lab/legacy/mybot_v3/mybot_v3_config.py'),
        arms=list(ARMS), seeds=args.seeds, num_envs=args.num_envs,
        rollout_steps=args.rollout_steps, iterations=args.iterations,
        environment_steps_per_run=args.num_envs * args.rollout_steps * args.iterations,
        eval_envs=args.eval_envs, eval_seconds=args.eval_seconds,
        evaluation='task_gait_eval.py --posture_scan with fixed per-seed randomization; reward is not a ranking metric',
        preserved_between_arms=['URDF/assets', 'phase/activation observations', 'gait reference/controller',
                                'commands', 'domain randomization', 'PPO parameters', 'training budget'],
        interpretation_limit='This isolates reward scales within task-gait structure. It does not isolate phase/reference-controller structure from the legacy reward source.')
    write_json(output / 'experiment_plan.json', plan)
    env = dict(os.environ)
    env['PYTHONPATH'] = f'{ROOT / "isaaclab_migration"}:{ROOT / "rsl_rl"}' + (
        ':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PYTHONUNBUFFERED'] = '1'
    env['TASK_GAIT_PYTHON'] = str(PYTHON)
    env['MPLCONFIGDIR'] = str(output / 'matplotlib_cache')
    status = dict(state='running', current_arm=None, current_seed=None,
                  operation=None, completed=[], results={})

    def update():
        status['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(output / 'status.json', status)

    def run(command, log, timeout, operation):
        status['operation'] = operation
        status['command'] = command
        print(f'\n[{operation}] ' + ' '.join(command), flush=True)
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True, bufsize=0)
            status['child_pid'] = child.pid
            update()
            deadline = time.monotonic() + timeout
            fd = child.stdout.fileno()
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    readable, _, _ = select.select([fd], [], [], min(1., remaining))
                    if readable:
                        chunk = os.read(fd, 65536)
                        if not chunk:
                            break
                        stream.buffer.write(chunk)
                        stream.flush()
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.flush()
                    if child.poll() is not None and not readable:
                        chunk = os.read(fd, 65536)
                        if chunk:
                            stream.buffer.write(chunk)
                            stream.flush()
                            sys.stdout.buffer.write(chunk)
                            sys.stdout.flush()
                            continue
                        break
                code = child.wait()
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
            finally:
                child.stdout.close()
        status.pop('child_pid', None)
        if code:
            raise RuntimeError(f'{operation} exited {code}; inspect {log}')
        print(f'[{operation}] verified exit 0; log={log}', flush=True)

    try:
        for arm in ARMS:
            status['results'][arm] = {}
        for seed in args.seeds:
            for arm in ARMS:
                status['current_arm'], status['current_seed'] = arm, seed
                folder = output / arm / f'seed_{seed}'
                training = folder / 'train'
                evaluation = folder / 'evaluation'
                training.mkdir(parents=True, exist_ok=False)
                status['operation'] = 'training_from_random_initialization'
                update()
                command = ['bash', 'isaaclab_migration/scripts/task_gait.sh', 'train',
                           '--no-videos', '--phase', 'precision',
                           '--posture_experiment', arm, '--num_envs', str(args.num_envs),
                           '--rollout_steps', str(args.rollout_steps), '--iterations', str(args.iterations),
                           '--seed', str(seed), '--log_dir', str(training)]
                # task_gait.sh creates the training directory; remove this empty placeholder only
                # after reserving the path, then refuse any unexpected pre-existing output.
                training.rmdir()
                run(command, folder / 'train_outer.log', args.train_timeout, 'training')
                console = (training / 'console.log').read_text()
                validation = json.loads((training / 'validation.json').read_text())
                if 'RUN_COMPLETED ' not in console or validation.get('status') != 'passed':
                    raise RuntimeError(f'Training verification failed for {arm}, seed {seed}; see {training}')
                checkpoint = training / f'model_{args.iterations}.pt'
                if not checkpoint.is_file():
                    raise RuntimeError(f'Expected final checkpoint missing: {checkpoint}')

                evaluation.mkdir(parents=True, exist_ok=False)
                command = [str(PYTHON), 'isaaclab_migration/scripts/task_gait_eval.py',
                           '--checkpoint', str(checkpoint), '--posture_scan',
                           '--seconds', str(args.eval_seconds), '--num_envs', str(args.eval_envs),
                           '--seed', str(seed), '--output', str(evaluation), '--headless']
                run(command, folder / 'evaluation_outer.log', args.eval_timeout, 'fixed_evaluation')
                eval_console = (evaluation / 'console.log').read_text()
                evaluation_json = json.loads((evaluation / 'evaluation.json').read_text())
                if 'EVALUATION_SAVED ' not in eval_console:
                    raise RuntimeError(f'Evaluation verification failed: {evaluation}')
                compact_metrics = compact(evaluation_json)
                status['results'][arm][str(seed)] = compact_metrics
                status['operation'] = 'recording_final_checkpoint'
                update()
                record_dir = training / 'checkpoint_videos'
                record_dir.mkdir(exist_ok=True)
                command = [str(PYTHON), 'isaaclab_migration/scripts/record_task_gait_checkpoints.py',
                           str(training), '--only', str(args.iterations), '--play_steps', '400',
                           '--num_envs', '4', '--capture_every_steps', '2']
                run(command, folder / 'record_outer.log', args.record_timeout, 'record_final_checkpoint')
                video = record_dir / checkpoint.stem / 'preview.mp4'
                if not video.is_file() or video.stat().st_size == 0:
                    raise RuntimeError(f'Final checkpoint video absent: {video}')
                status['completed'].append(dict(arm=arm, seed=seed,
                    training=str(training), checkpoint=str(checkpoint),
                    evaluation=str(evaluation / 'evaluation.json'), video=str(video)))
                status['current_arm'], status['current_seed'], status['operation'] = None, None, None
                update()
        comparison = dict(plan=plan, metrics_by_arm_and_seed=status['results'],
                          differences_vs_task_gait_current_by_arm_and_seed=paired_differences(status['results']),
                          completed=status['completed'],
        limitations=['Only reward scales differ by design; all arms retain task-gait phase/activation inputs and reference controller.',
                                        'This first pass uses one paired seed; replicate a clear difference before assigning a general cause.',
                                        'Simulation torque metrics do not establish real motor thermal safety.'])
        write_json(output / 'comparison.json', comparison)
        status['state'] = 'screen_complete_review_required'
        status['current_arm'], status['current_seed'] = None, None
        status['operation'] = None
        update()
        print('SCREEN_COMPLETE', output, flush=True)
    except BaseException as exc:
        status['state'] = 'failed_or_interrupted'
        status['failure'] = f'{type(exc).__name__}: {exc}'
        status['operation'] = 'failure_diagnostic_required'
        update()
        raise


if __name__ == '__main__':
    main()
