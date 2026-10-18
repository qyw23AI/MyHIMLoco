#!/usr/bin/env python3
"""Replay permanent task-gait checkpoints with the WMP clip settings."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoint_dir', type=Path)
parser.add_argument('--output_dir', type=Path)
parser.add_argument('--play_steps', type=int, default=1000)
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--capture_every_steps', type=int, default=10)
parser.add_argument('--forward_speed', type=float, default=0.5)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--only', type=int, nargs='+')
args = parser.parse_args()
if min(args.play_steps, args.num_envs, args.capture_every_steps) < 1:
    parser.error('play_steps, num_envs and capture_every_steps must be positive')
if not 0 <= args.forward_speed <= 0.5:
    parser.error('forward_speed must be within the task-gait range [0, 0.5] m/s')
directory = args.checkpoint_dir.resolve()
checkpoints = sorted((p for p in directory.glob('model_*.pt')
                      if p.stem.removeprefix('model_').isdigit()),
                     key=lambda p: int(p.stem.removeprefix('model_')))
if args.only:
    checkpoints = [p for p in checkpoints if int(p.stem.removeprefix('model_')) in args.only]
if not checkpoints:
    parser.error(f'No permanent model_N.pt checkpoints found in {directory}')

output = (args.output_dir or directory / 'checkpoint_videos').resolve()
output.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONPATH=f'{ROOT / "isaaclab_migration"}:{ROOT / "rsl_rl"}'
           + (':' + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''))

def run(command, log):
    with log.open('a') as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)

for index, checkpoint in enumerate(checkpoints, 1):
    clip = output / checkpoint.stem
    video = clip / 'preview.mp4'
    if video.is_file() and video.stat().st_size > 0 and not args.overwrite:
        print(f'[{index}/{len(checkpoints)}] Already recorded: {checkpoint.name}', flush=True)
        continue
    clip.mkdir(parents=True, exist_ok=True)
    for pattern in ('frame_*.png', 'preview.mp4', 'playback.png', 'validation.json', 'state.csv'):
        for old in clip.glob(pattern):
            old.unlink()
    log = clip / 'run.log'
    log.write_text('')
    policy = clip / 'policy.pt'
    print(f'[{index}/{len(checkpoints)}] Recording {checkpoint.name}', flush=True)
    try:
        run([sys.executable, str(ROOT / 'isaaclab_migration/scripts/export_policy.py'),
             '--task', 'mybot_v3_task_gait', '--checkpoint', str(checkpoint),
             '--output', str(policy)], log)
        run([sys.executable, str(ROOT / 'isaaclab_migration/scripts/play.py'),
             '--task', 'mybot_v3_task_gait', '--policy', str(policy),
             '--num_envs', str(args.num_envs), '--steps', str(args.play_steps),
             '--command', str(args.forward_speed), '0', '0', '--capture_dir', str(clip),
             '--capture_every', str(args.capture_every_steps), '--follow_camera',
             '--no-realtime', '--state_csv', str(clip / 'state.csv')], log)
        recorded = json.loads((clip / 'validation.json').read_text())
        if recorded['status'] != 'passed' or recorded['sim_steps'] != args.play_steps:
            raise RuntimeError(f'Playback validation failed for {checkpoint.name}')
        from PIL import Image
        import imageio.v2 as imageio
        import numpy as np
        frames = sorted(clip.glob('frame_*.png'))
        expected = args.play_steps // args.capture_every_steps
        if len(frames) != expected:
            raise RuntimeError(f'Expected {expected} frames, found {len(frames)}')
        fps = 1 / (recorded['sim_seconds'] / args.play_steps * args.capture_every_steps)
        with imageio.get_writer(video, fps=fps, codec='libx264', quality=8,
                                macro_block_size=2) as writer:
            for frame in frames:
                with Image.open(frame) as image:
                    image = image.convert('RGB')
                    image.thumbnail((960, 540))
                    writer.append_data(np.asarray(image))
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError('MP4 encoder produced no video')
        print('VIDEO_SAVED', video, len(frames), f'{fps:g} fps', flush=True)
    except Exception as exc:
        print(f'VIDEO_FAILED {checkpoint.name}: {exc}; see {log}', file=sys.stderr, flush=True)
        raise
print('CHECKPOINT_VIDEOS_COMPLETE', output, flush=True)
