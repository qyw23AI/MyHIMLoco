"""Versioned task contract saved with EVERY checkpoint, not inferred from filenames."""
import json
from pathlib import Path
import torch
from .config import TASK, build_config, restore_config


def saved_task_config(checkpoint):
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    info = data.get('infos') or {}
    if info.get('task') != TASK or 'task_config' not in info:
        raise ValueError('This is not a task-gait checkpoint. Legacy 270-input weights cannot be resumed as 294-input gait weights.')
    cfg = restore_config(info['task_config'])
    if cfg.gait.version != 1:
        raise ValueError('Unsupported gait observation/controller version')
    return cfg


def resolve_training_config(num_envs, phase=None, checkpoint=None, ablation=None):
    from .config import set_ablation
    if checkpoint:
        saved = saved_task_config(checkpoint)
        phase = phase or saved.gait.phase
        cfg, train = build_config(num_envs, phase)
        saved_train = torch.load(checkpoint, map_location='cpu', weights_only=True)['infos'].get('training_config')
        if saved_train is not None:
            train = saved_train
        if phase == saved.gait.phase:
            cfg = saved
            cfg.env.num_envs = num_envs
            # Resume exact scales, including opt-in posture experiments.
            if ablation is not None:
                set_ablation(cfg, ablation)
        else:
            set_ablation(cfg, ablation or saved.gait.ablation)
    else:
        cfg, train = build_config(num_envs, phase or 'basic')
        set_ablation(cfg, ablation or 'full')
    return cfg, train


def config_from_policy(policy, num_envs):
    meta = json.loads(Path(policy).with_suffix('.json').read_text())
    if meta.get('task') != TASK:
        raise ValueError('Expected exported task-gait policy and its JSON sidecar')
    cfg = restore_config(meta['task_config'])
    cfg.env.num_envs = num_envs
    return cfg, meta
