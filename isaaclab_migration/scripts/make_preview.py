#!/usr/bin/env python3
"""Assemble real viewport captures into a compact, looping playback GIF."""
import argparse
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
parser.add_argument('--output', type=Path)
parser.add_argument('--duration_ms', type=int, default=80,
                    help='Simulation time between captured playback frames')
parser.add_argument('--width', type=int, default=640)
args = parser.parse_args()
paths = sorted(args.directory.glob('frame_*.png'))
if not paths:
    parser.error('No captured frame_*.png files found')
frames = []
for path in paths:
    with Image.open(path) as source:
        frame = source.convert('RGB')
        frame.thumbnail((args.width, args.width), Image.Resampling.LANCZOS)
        frames.append(frame)
output = args.output or args.directory / 'playback.gif'
output.parent.mkdir(parents=True, exist_ok=True)
frames[0].save(output, save_all=True, append_images=frames[1:],
               duration=args.duration_ms, loop=0, optimize=True)
print(f'PREVIEW_CREATED {output.resolve()} frames={len(frames)}')
