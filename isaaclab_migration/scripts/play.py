#!/usr/bin/env python3
"""Replay a HIM training checkpoint or exported TorchScript in Isaac Lab."""
import argparse
import csv
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'isaaclab_migration'))
sys.path.insert(0,str(ROOT/'rsl_rl'))
from isaaclab.app import AppLauncher
from himloco_lab.config import TASKS,load_config

parser=argparse.ArgumentParser()
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument('--policy', type=Path, help='Exported TorchScript policy')
source.add_argument('--checkpoint', type=Path, help='Stage 3/4 HIM training checkpoint')
parser.add_argument('--task',choices=TASKS,default='mybot_v3')
parser.add_argument('--num_envs',type=int,default=4)
parser.add_argument('--stage',type=int,choices=[3,4],default=3)
parser.add_argument('--seconds',type=float,default=30.,help='0 keeps the viewer running until closed')
parser.add_argument('--steps',type=int,default=None,help='Exact control steps for reproducible checkpoint videos')
parser.add_argument('--command',type=float,nargs=3,default=[1.,0.,0.],metavar=('VX','VY','YAW'))
parser.add_argument('--capture_dir',type=Path)
parser.add_argument('--capture_every',type=int,default=4)
parser.add_argument('--state_csv',type=Path)
parser.add_argument('--state_all_envs', action='store_true', help='Record all environments in --state_csv')
parser.add_argument('--realtime',action=argparse.BooleanOptionalAction,default=True)
parser.add_argument('--follow_camera',action='store_true',help='Track robot every render; default allows free mouse navigation')
parser.add_argument('--camera_env',type=int,default=0,help='Environment index used by the follow camera')
parser.add_argument('--viewer_resolution',type=int,nargs=2,metavar=('WIDTH','HEIGHT'))
parser.add_argument('--antialiasing',choices=['Off','FXAA','DLSS','DLAA','TAA'])
parser.add_argument('--terrain_color',type=float,nargs=3,metavar=('R','G','B'),help='Optional ground visual shader RGB (0 to 1)')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.state_all_envs and not args.state_csv:
    parser.error('--state_all_envs requires --state_csv')
if args.capture_dir and args.headless:parser.error('viewport capture requires GUI')
if args.steps is not None and args.steps < 1:parser.error('--steps must be positive')
if args.viewer_resolution and (args.headless or min(args.viewer_resolution)<=0):
    parser.error('--viewer_resolution requires GUI and positive width/height')
if args.terrain_color and (min(args.terrain_color)<0 or max(args.terrain_color)>1):
    parser.error('--terrain_color components must be between 0 and 1')
if args.camera_env < 0 or args.camera_env >= args.num_envs:
    parser.error('--camera_env must be within [0, num_envs)')
app=AppLauncher(args).app
try:
    import json
    import torch
    from himloco_lab.env_cfg import make_env_cfg
    from himloco_lab.env import HIMLabEnv
    from himloco_lab.adapter import LegacyRunnerAdapter
    from himloco_lab.config import asset_contract

    cfg,train=load_config(args.task,args.num_envs,args.stage)
    playback_path = (args.policy or args.checkpoint).resolve()
    if args.checkpoint:
        checkpoint = torch.load(playback_path, map_location='cpu', weights_only=True)
    if args.task=='mybot_v3_task_gait':
        from himloco_lab.task_gait.checkpoint import config_from_policy, saved_task_config
        if args.checkpoint:
            cfg = saved_task_config(playback_path)
            cfg.env.num_envs = args.num_envs
            train = checkpoint['infos'].get('training_config') or train
        else:
            cfg,_=config_from_policy(args.policy,args.num_envs)
        args.stage=4 if cfg.gait.phase=='robust' else 3
    # Same playback overrides as the repository's original play.py.
    cfg.noise.add_noise=False
    for flag in ['randomize_friction','push_robots','disturbance','randomize_payload_mass']:
        setattr(cfg.domain_rand,flag,False)
    cfg.commands.heading_command=False
    lab_cfg=make_env_cfg(cfg,args.stage,args.device)
    if args.antialiasing:lab_cfg.sim.render.antialiasing_mode=args.antialiasing
    if args.viewer_resolution:lab_cfg.viewer.resolution=tuple(args.viewer_resolution)
    lab_cfg.viewer.origin_type='asset_root' if args.follow_camera else 'env'
    lab_cfg.viewer.env_index=args.camera_env
    lab_cfg.viewer.asset_name='robot'
    lab_cfg.viewer.eye=(2.3,2.3,1.7);lab_cfg.viewer.lookat=(0.,0.,0.)
    env=HIMLabEnv(lab_cfg);wrapped=LegacyRunnerAdapter(env)
    if args.terrain_color:
        from himloco_lab.visualization import color_terrain
        color_terrain(args.terrain_color)
    if args.viewer_resolution:
        from himloco_lab.visualization import configure_viewport
        configure_viewport(args.viewer_resolution)
    if args.checkpoint:
        from rsl_rl.modules import HIMActorCritic

        model = HIMActorCritic(
            cfg.env.num_observations, cfg.env.num_privileged_obs,
            cfg.env.num_one_step_observations, cfg.env.num_actions, **train['policy'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        policy = model.to(env.device).eval().act_inference
        del checkpoint
    else:
        policy=torch.jit.load(str(playback_path),map_location=args.device).eval()
        metadata=args.policy.with_suffix('.json')
        if metadata.exists():
            meta=json.loads(metadata.read_text())
            if meta['joint_order']!=asset_contract(cfg)[1]:raise ValueError('Exported policy joint order does not match this task')
            if meta['input_shape'][-1]!=wrapped.num_obs:raise ValueError('Policy observation dimension mismatch')
    if args.capture_dir:
        from himloco_lab.visualization import ViewportCapture
        wrapped.capture=ViewportCapture(args.capture_dir,args.capture_every)
        if args.stage==4 and cfg.terrain.curriculum:
            from himloco_lab.curriculum import CurriculumMonitor
            wrapped.diagnostics=CurriculumMonitor(env,args.capture_dir)
    wrapped.reset()
    started=time.monotonic()
    deadline=started+args.seconds if args.seconds else float('inf')
    rows=[];step=0;resets=0
    print('PLAYBACK_STARTED',playback_path,
          'camera='+lab_cfg.viewer.origin_type,flush=True)
    with torch.inference_mode():
        while app.is_running() and (step < args.steps if args.steps is not None else time.monotonic()<deadline):
            tick=time.monotonic()
            if hasattr(cfg,'gait'):
                env.core.set_command(args.command)
                env.core.prepare_external_observation()
            else:
                env.core.commands[:,:3]=torch.tensor(args.command,device=env.device)
            actions=policy(wrapped.get_observations())
            if not torch.isfinite(actions).all():raise RuntimeError('Nonfinite playback actions')
            result=wrapped.step(actions)
            if not torch.isfinite(result[0]).all() or not torch.isfinite(result[2]).all():
                raise RuntimeError('Nonfinite playback observations or rewards')
            resets+=int(result[3].sum())
            if args.state_csv:
                c=env.core
                selection = slice(None, None if args.state_all_envs else 1)
                heights = c._get_base_heights()[selection].tolist()
                velocities = c.base_lin_vel[selection].tolist()
                world_heights = c.root_states[selection, 2].tolist()
                rewards = result[2][selection].tolist()
                reset_flags = result[3][selection].tolist()
                for env_id, (velocity, world_z, reward, reset, height) in enumerate(
                    zip(velocities, world_heights, rewards, reset_flags, heights)
                ):
                    rows.append([(step + 1) * env.step_dt, *args.command, *velocity,
                                 world_z, reward, bool(reset), height, env_id])
            step+=1
            if args.realtime:time.sleep(max(0.,env.step_dt-(time.monotonic()-tick)))
    if args.steps is not None and step != args.steps:
        raise RuntimeError(f'Playback stopped at {step}/{args.steps} control steps')
    if wrapped.capture is not None:
        print('PLAYBACK_CAPTURED',wrapped.capture.capture_final(env.sim,'playback.png'),flush=True)
    if args.state_csv:
        args.state_csv.parent.mkdir(parents=True,exist_ok=True)
        with args.state_csv.open('w',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['time_s','command_x','command_y','command_yaw','velocity_x','velocity_y',
                             'velocity_z','height_m','reward','reset','base_height_m','env_id'])
            writer.writerows(rows)
    from himloco_lab.validation import validate_runtime
    report=validate_runtime(env)
    if wrapped.diagnostics is not None:
        report['terrain_curriculum']=wrapped.diagnostics.save()['final']
    report.update(status='passed',task=args.task,
                  num_envs=args.num_envs,sim_steps=step,sim_seconds=step*env.step_dt,
                  wall_seconds=time.monotonic()-started,total_resets=resets,
                  command=args.command)
    report['checkpoint' if args.checkpoint else 'policy'] = str(playback_path)
    report['viewer_resolution']=args.viewer_resolution
    report['antialiasing']=args.antialiasing
    report['terrain_color']=args.terrain_color
    if args.capture_dir:
        (args.capture_dir/'validation.json').write_text(json.dumps(report,indent=2))
    print('PLAYBACK_VALIDATED',json.dumps(report),flush=True)
    print('PLAYBACK_COMPLETE',step,flush=True)
    wrapped.close()
except BaseException:
    import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();raise
finally:
    if 'env' in globals():
        env.close()
    app.close()
