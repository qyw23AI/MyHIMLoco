"""Effective run provenance in JSON and TensorBoard Text; no external services."""
import hashlib
import json
from pathlib import Path
import subprocess

from ..config import ROOT, asset_contract, class_to_dict


def record_run(directory, cfg, train, args):
    directory = Path(directory)
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path else None
    def git(*command):
        return subprocess.check_output(['git', *command], cwd=ROOT, text=True).strip()
    asset = asset_contract(cfg)[0]
    manifest = dict(
        experiment=getattr(getattr(cfg.gait, 'posture', None), 'experiment', 'historical'),
        phase=cfg.gait.phase, seed=args.seed, num_envs=cfg.env.num_envs,
        rollout_steps=train['runner']['num_steps_per_env'],
        requested_additional_updates=args.iterations if args.iterations is not None else train['runner']['max_iterations'],
        source_checkpoint=str(checkpoint) if checkpoint else None,
        source_checkpoint_sha256=digest(checkpoint), asset_path=str(asset), asset_sha256=digest(asset),
        git_commit=git('rev-parse', 'HEAD'), git_worktree_status=git('status', '--short'),
        git_diff_sha256=hashlib.sha256(git('diff', 'HEAD').encode()).hexdigest(),
        tracking_weight=cfg.rewards.scales.tracking_lin_vel,
        reward_scales=class_to_dict(cfg.rewards.scales),
        posture_config=class_to_dict(getattr(cfg.gait, 'posture', {})),
        notes=['Training posture statistics are cumulative; compare fixed-command evaluation separately.',
               'Short checkpoint adaptation diagnoses recoverability, not from-scratch convergence.'])
    (directory / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def write_tensorboard_config(writer, directory):
    """Called after the runner creates its writer; keep complete serializable data."""
    directory = Path(directory)
    for filename, tag in [('config.json', 'Run/effective_configuration'),
                          ('run_manifest.json', 'Run/experiment_and_provenance')]:
        path = directory / filename
        if path.exists():
            writer.add_text(tag, '```json\n' + path.read_text() + '\n```', 0)
    manifest = json.loads((directory / 'run_manifest.json').read_text())
    for key, value in manifest['reward_scales'].items():
        writer.add_scalar('Config/reward_scale/' + key, value, 0)
    for key in ('seed', 'num_envs', 'rollout_steps', 'requested_additional_updates'):
        writer.add_scalar('Config/' + key, manifest[key], 0)
    writer.flush()


def write_posture_scalars(writer, report, step):
    p = report['posture']
    for name in ('height_mean_m', 'height_mae_m', 'height_below_target_minus_5cm_fraction'):
        if p[name] is not None:
            writer.add_scalar('Posture/' + name, p[name], step)
    for field in ('hip_mean_rad', 'joint_near_limit_fraction'):
        for name, value in (p[field] or {}).items():
            writer.add_scalar('Posture/' + field + '/' + name, value, step)
    for field in ('swing_contact_fraction', 'swing_clearance_mae_m', 'apex_clearance_mean_m'):
        for name, value in zip(report['foot_names'], p[field]):
            if value is not None:
                writer.add_scalar('Posture/' + field + '/' + name, value, step)
