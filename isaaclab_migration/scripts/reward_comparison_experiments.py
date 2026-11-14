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
    parser.add_argument('--train_timeout', type=int, default=0,
                        help='Training timeout in seconds; 0 disables the training timeout.')
    parser.add_argument('--eval_timeout', type=int, default=600)
    parser.add_argument('--record_timeout', type=int, default=300)
    parser.add_argument('--resume-after', choices=ARMS,
                        help='Resume a non-empty output after this already-run arm; never reruns earlier arms.')
    parser.add_argument('--resume-partial-arm', choices=ARMS,
                        help='Continue a timed-out arm from its newest model_N.pt, then run later arms.')
    args = parser.parse_args()
    if args.resume_after and args.resume_partial_arm:
        parser.error('Use only one of --resume-after or --resume-partial-arm')
    if min(args.num_envs, args.rollout_steps, args.iterations, args.eval_envs) < 1:
        parser.error('environment, rollout, iteration and evaluation counts must be positive')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be unique and nonempty')
    output = args.output.resolve()
    resume_mode = args.resume_after or args.resume_partial_arm
    if output.exists() and any(output.iterdir()) and not resume_mode:
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
    plan_path = output / 'experiment_plan.json'
    if resume_mode:
        if not plan_path.is_file():
            parser.error(f'Cannot resume without existing experiment plan: {plan_path}')
        existing_plan = json.loads(plan_path.read_text())
        for key in ('comparison', 'task', 'phase', 'source_checkpoint', 'arms', 'seeds',
                    'num_envs', 'rollout_steps', 'iterations', 'eval_envs', 'eval_seconds'):
            if existing_plan.get(key) != plan.get(key):
                parser.error(f'Resume plan mismatch for {key}: {existing_plan.get(key)!r} != {plan.get(key)!r}')
    else:
        write_json(plan_path, plan)
    env = dict(os.environ)
    env['PYTHONPATH'] = f'{ROOT / "isaaclab_migration"}:{ROOT / "rsl_rl"}' + (
        ':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PYTHONUNBUFFERED'] = '1'
    env['TASK_GAIT_PYTHON'] = str(PYTHON)
    env['MPLCONFIGDIR'] = str(output / 'matplotlib_cache')
    status = dict(state='running', current_arm=None, current_seed=None,
                  operation=None, completed=[], results={})

    start_index = 0
    partial_checkpoint = None
    partial_iteration = None
    if resume_mode:
        boundary_arm = args.resume_after or ARMS[ARMS.index(args.resume_partial_arm) - 1]
        start_index = (ARMS.index(args.resume_partial_arm) if args.resume_partial_arm
                       else ARMS.index(args.resume_after) + 1)
        if start_index >= len(ARMS):
            parser.error('Selected resume point has no remaining arms to run')
        # Recover the completed baseline's metrics directly from its preserved output.
        # The previous driver failed only because task_gait_eval writes its console to
        # evaluation_outer.log, not evaluation/console.log.
        prior_status_path = output / 'status.json'
        if prior_status_path.is_file():
            previous_status = json.loads(prior_status_path.read_text())
            expected_current_arm = args.resume_partial_arm or args.resume_after
            if previous_status.get('current_arm') != expected_current_arm:
                parser.error(f"Existing status does not identify {expected_current_arm} as the interrupted arm")
        for seed in args.seeds:
            prior_folder = output / boundary_arm / f'seed_{seed}'
            prior_training = prior_folder / 'train'
            prior_eval = prior_folder / 'evaluation'
            prior_checkpoint = prior_training / f'model_{args.iterations}.pt'
            prior_validation = prior_training / 'validation.json'
            prior_console = prior_training / 'console.log'
            prior_evaluation = prior_eval / 'evaluation.json'
            prior_eval_log = prior_folder / 'evaluation_outer.log'
            if not all(p.is_file() for p in (prior_checkpoint, prior_validation, prior_console,
                                              prior_evaluation, prior_eval_log)):
                parser.error(f'Cannot resume: preserved outputs are incomplete under {prior_folder}')
            if 'RUN_COMPLETED ' not in prior_console.read_text(errors='replace'):
                parser.error(f'Cannot resume: prior training completion marker is missing: {prior_console}')
            if json.loads(prior_validation.read_text()).get('status') != 'passed':
                parser.error(f'Cannot resume: prior training validation did not pass: {prior_validation}')
            if 'EVALUATION_SAVED ' not in prior_eval_log.read_text(errors='replace'):
                parser.error(f'Cannot resume: prior evaluation completion marker is missing: {prior_eval_log}')
            prior_eval_json = json.loads(prior_evaluation.read_text())
            if prior_eval_json.get('seed') != seed or prior_eval_json.get('status') != 'execution_validated_not_performance_approved':
                parser.error(f'Cannot resume: evaluation seed mismatch: {prior_evaluation}')
            status['results'].setdefault(boundary_arm, {})[str(seed)] = compact(prior_eval_json)
            status['completed'].append(dict(
                arm=boundary_arm, seed=seed, training=str(prior_training),
                checkpoint=str(prior_checkpoint), evaluation=str(prior_evaluation),
                video=None, video_note='not recorded before the prior workflow stopped at log verification'))
        if args.resume_partial_arm:
            partial_folder = output / args.resume_partial_arm / f'seed_{args.seeds[0]}'
            partial_train = partial_folder / 'train'
            if len(args.seeds) != 1:
                parser.error('Partial-arm resume currently requires exactly one seed')
            partial_status = json.loads((output / 'status.json').read_text())
            previous_pid = partial_status.get('child_pid')
            active_child = bool(previous_pid and Path(f'/proc/{previous_pid}').exists())
            stale_running_state = partial_status.get('state') == 'running' and not active_child
            if not str(partial_status.get('failure', '')).startswith(('TimeoutExpired:', 'RuntimeError:')) and not stale_running_state:
                parser.error('Partial-arm resume requires a recorded failure or a stale running status with no live child process')
            if active_child:
                parser.error(f'Refusing to resume while recorded child PID {previous_pid} is still alive')
            candidates = []
            for item in partial_train.glob('model_*.pt'):
                if item.stem.startswith('model_latest_'):
                    continue
                try:
                    candidates.append((int(item.stem.removeprefix('model_')), item))
                except ValueError:
                    continue
            if not candidates:
                parser.error(f'No permanent model_N.pt checkpoint found in {partial_train}')
            partial_iteration, partial_checkpoint = max(candidates, key=lambda pair: pair[0])
            import torch
            checkpoint_payload = torch.load(partial_checkpoint, map_location='cpu', weights_only=True)
            checkpoint_iteration = int(checkpoint_payload['iter'])
            checkpoint_payload.clear()
            if checkpoint_iteration >= args.iterations:
                parser.error(f'Latest partial checkpoint is already at iteration {checkpoint_iteration}')
            partial_iteration = checkpoint_iteration
            partial_console = partial_train / 'console.log'
            if not partial_console.is_file() or 'TRAINING_COMPLETE ' in partial_console.read_text(errors='replace'):
                parser.error(f'Partial training log is missing or already completed: {partial_console}')
            status['partial_training'] = dict(
                arm=args.resume_partial_arm, seed=args.seeds[0],
                original_training=str(partial_train), checkpoint=str(partial_checkpoint),
                checkpoint_iteration=partial_iteration,
                additional_iterations=args.iterations - partial_iteration,
                checkpoint_file_iteration=int(partial_checkpoint.stem.removeprefix('model_')))
        status['resumed_after'] = boundary_arm
        status['resumed_partial_arm'] = args.resume_partial_arm
        status['operation'] = 'resuming_from_preserved_checkpoint' if args.resume_partial_arm else 'resuming_after_preserved_completed_arm'

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
            deadline = None if timeout is None else time.monotonic() + timeout
            fd = child.stdout.fileno()
            try:
                while True:
                    remaining = None if timeout is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    readable, _, _ = select.select([fd], [], [], 1. if remaining is None else min(1., remaining))
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
            status['results'].setdefault(arm, {})
        for seed in args.seeds:
            for arm in ARMS[start_index:]:
                status['current_arm'], status['current_seed'] = arm, seed
                folder = output / arm / f'seed_{seed}'
                continuing_partial = args.resume_partial_arm == arm
                continuation_suffix = ''
                training_name = f'train_from_{partial_iteration}'
                if continuing_partial:
                    retry = 0
                    while ((folder / training_name).exists() or
                           (folder / f'train_from_checkpoint_outer{continuation_suffix}.log').exists()):
                        retry += 1
                        continuation_suffix = f'_retry{retry}'
                        training_name = f'train_from_{partial_iteration}{continuation_suffix}'
                training = folder / (training_name if continuing_partial else 'train')
                evaluation = folder / 'evaluation'
                training.mkdir(parents=True, exist_ok=False)
                status['operation'] = 'continuing_partial_training' if continuing_partial else 'training_from_random_initialization'
                update()
                command = ['bash', 'isaaclab_migration/scripts/task_gait.sh', 'train',
                           '--no-videos', '--phase', 'precision',
                           '--posture_experiment', arm, '--num_envs', str(args.num_envs),
                           '--rollout_steps', str(args.rollout_steps), '--iterations',
                           str(args.iterations - partial_iteration if continuing_partial else args.iterations),
                           '--seed', str(seed)]
                if continuing_partial:
                    command.extend(['--checkpoint', str(partial_checkpoint)])
                command.extend(['--log_dir', str(training)])
                # task_gait.sh creates the training directory; remove this empty placeholder only
                # after reserving the path, then refuse any unexpected pre-existing output.
                training.rmdir()
                run(command, folder / (f'train_from_checkpoint_outer{continuation_suffix}.log' if continuing_partial else 'train_outer.log'),
                    args.train_timeout if args.train_timeout > 0 else None, 'training')
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
                eval_log = folder / 'evaluation_outer.log'
                eval_console = eval_log.read_text(errors='replace')
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
