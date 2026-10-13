#!/usr/bin/env python3
"""Visual playback of exported HIM TorchScript, using the real Lab environment."""
import argparse
import csv
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'isaaclab_migration'))
from isaaclab.app import AppLauncher
from himloco_lab.config import TASKS,load_config

parser=argparse.ArgumentParser()
parser.add_argument('--policy',type=Path,required=True)
parser.add_argument('--task',choices=TASKS,default='mybot_v3')
parser.add_argument('--num_envs',type=int,default=4)
parser.add_argument('--stage',type=int,choices=[3,4],default=3)
parser.add_argument('--seconds',type=float,default=30.,help='0 keeps the viewer running until closed')
parser.add_argument('--command',type=float,nargs=3,default=[1.,0.,0.],metavar=('VX','VY','YAW'))
parser.add_argument('--capture_dir',type=Path)
parser.add_argument('--capture_every',type=int,default=4)
parser.add_argument('--state_csv',type=Path)
parser.add_argument('--realtime',action=argparse.BooleanOptionalAction,default=True)
parser.add_argument('--follow_camera',action='store_true',help='Track robot every render; default allows free mouse navigation')
parser.add_argument('--viewer_resolution',type=int,nargs=2,metavar=('WIDTH','HEIGHT'))
parser.add_argument('--antialiasing',choices=['Off','FXAA','DLSS','DLAA','TAA'])
parser.add_argument('--terrain_color',type=float,nargs=3,metavar=('R','G','B'),help='Optional ground visual shader RGB (0 to 1)')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.capture_dir and args.headless:parser.error('viewport capture requires GUI')
if args.viewer_resolution and (args.headless or min(args.viewer_resolution)<=0):
    parser.error('--viewer_resolution requires GUI and positive width/height')
if args.terrain_color and (min(args.terrain_color)<0 or max(args.terrain_color)>1):
    parser.error('--terrain_color components must be between 0 and 1')
app=AppLauncher(args).app
try:
    import json
    import torch
    from himloco_lab.env_cfg import make_env_cfg
    from himloco_lab.env import HIMLabEnv
    from himloco_lab.adapter import LegacyRunnerAdapter
    from himloco_lab.config import asset_contract

    cfg,_=load_config(args.task,args.num_envs,args.stage)
    # Same playback overrides as the repository's original play.py.
    cfg.noise.add_noise=False
    for flag in ['randomize_friction','push_robots','disturbance','randomize_payload_mass']:
        setattr(cfg.domain_rand,flag,False)
    cfg.commands.heading_command=False
    lab_cfg=make_env_cfg(cfg,args.stage,args.device)
    if args.antialiasing:lab_cfg.sim.render.antialiasing_mode=args.antialiasing
    if args.viewer_resolution:lab_cfg.viewer.resolution=tuple(args.viewer_resolution)
    lab_cfg.viewer.origin_type='asset_root' if args.follow_camera else 'env'
    lab_cfg.viewer.asset_name='robot'
    lab_cfg.viewer.eye=(2.3,2.3,1.7);lab_cfg.viewer.lookat=(0.,0.,0.)
    env=HIMLabEnv(lab_cfg);wrapped=LegacyRunnerAdapter(env)
    if args.terrain_color:
        from himloco_lab.visualization import color_terrain
        color_terrain(args.terrain_color)
    if args.viewer_resolution:
        from himloco_lab.visualization import configure_viewport
        configure_viewport(args.viewer_resolution)
    policy=torch.jit.load(str(args.policy.resolve()),map_location=args.device).eval()
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
    print('PLAYBACK_STARTED',args.policy.resolve(),
          'camera='+lab_cfg.viewer.origin_type,flush=True)
    with torch.inference_mode():
        while app.is_running() and time.monotonic()<deadline:
            tick=time.monotonic()
            env.core.commands[:,:3]=torch.tensor(args.command,device=env.device)
            actions=policy(wrapped.get_observations())
            if not torch.isfinite(actions).all():raise RuntimeError('Nonfinite playback actions')
            result=wrapped.step(actions)
            if not torch.isfinite(result[0]).all() or not torch.isfinite(result[2]).all():
                raise RuntimeError('Nonfinite playback observations or rewards')
            resets+=int(result[3].sum())
            if args.state_csv:
                c=env.core
                rows.append([step*env.step_dt,*args.command,*c.base_lin_vel[0].tolist(),
                             float(c.root_states[0,2]),float(result[2][0]),bool(result[3][0])])
            step+=1
            if args.realtime:time.sleep(max(0.,env.step_dt-(time.monotonic()-tick)))
    if wrapped.capture is not None:
        print('PLAYBACK_CAPTURED',wrapped.capture.capture_final(env.sim,'playback.png'),flush=True)
    if args.state_csv:
        args.state_csv.parent.mkdir(parents=True,exist_ok=True)
        with args.state_csv.open('w',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['time_s','command_x','command_y','command_yaw','velocity_x','velocity_y','velocity_z','height_m','reward','reset'])
            writer.writerows(rows)
    from himloco_lab.validation import validate_runtime
    report=validate_runtime(env)
    if wrapped.diagnostics is not None:
        report['terrain_curriculum']=wrapped.diagnostics.save()['final']
    report.update(status='passed',policy=str(args.policy.resolve()),task=args.task,
                  num_envs=args.num_envs,sim_steps=step,sim_seconds=step*env.step_dt,
                  wall_seconds=time.monotonic()-started,total_resets=resets,
                  command=args.command)
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
