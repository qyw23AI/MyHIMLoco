#!/usr/bin/env python3
"""CPU-only export of a trained HIM checkpoint; Isaac Sim is not required."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'isaaclab_migration'))
sys.path.insert(0, str(ROOT / 'rsl_rl'))

from himloco_lab.config import TASKS, load_config
from himloco_lab.export import export_policy
from rsl_rl.modules import HIMActorCritic
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--task', choices=TASKS, default='mybot_v3')
args = parser.parse_args()
cfg, train = load_config(args.task)
model = HIMActorCritic(cfg.env.num_observations, cfg.env.num_privileged_obs,
                      cfg.env.num_one_step_observations, cfg.env.num_actions, **train['policy'])
checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
metadata = export_policy(model, args.output, cfg, args.checkpoint)
print('EXPORTED', args.output.resolve(), 'max_error=', metadata['round_trip_max_error'])
