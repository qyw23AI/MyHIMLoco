#!/usr/bin/env python3
"""Four-stage launch using the installed Isaac Lab runtime and local RSL-RL."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'isaaclab_migration'))
sys.path.insert(0,str(ROOT/'rsl_rl'))
from isaaclab.app import AppLauncher
from himloco_lab.config import TASKS,load_config
parser=argparse.ArgumentParser()
parser.add_argument('--task',choices=TASKS,default='mybot_v3')
parser.add_argument('--stage',type=int,choices=[1,2,3,4],default=3)
parser.add_argument('--num_envs',type=int,default=1024)
parser.add_argument('--iterations',type=int,default=None)
parser.add_argument('--steps',type=int,default=500,help='Standing test policy steps')
parser.add_argument('--rollout_steps',type=int,default=None,help='Explicit smoke-test override; default retains original 100/24')
parser.add_argument('--seed',type=int,default=1)
parser.add_argument('--checkpoint',type=Path)
parser.add_argument('--log_dir',type=Path)
parser.add_argument('--export_policy',type=Path,help='Export validated TorchScript encoder+actor after HIM training')
parser.add_argument('--capture_dir',type=Path,help='Save actual viewport frames; requires GUI')
parser.add_argument('--capture_every',type=int,default=100)
parser.add_argument('--render_every',type=int,default=None,help='Render interval in physics steps; does not change control dt')
parser.add_argument('--preview_seconds',type=float,default=0.,help='Run trained deterministic policy in GUI after training')
parser.add_argument('--follow_camera',action='store_true',help='Track robot every render; default allows free mouse navigation')
parser.add_argument('--viewer_resolution',type=int,nargs=2,metavar=('WIDTH','HEIGHT'))
parser.add_argument('--antialiasing',choices=['Off','FXAA','DLSS','DLAA','TAA'])
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.capture_dir and args.headless:parser.error('--capture_dir requires a GUI launch without --headless')
if args.export_policy and args.stage in (1,2):parser.error('--export_policy requires HIM stage 3 or 4')
if args.viewer_resolution and (args.headless or min(args.viewer_resolution)<=0):
    parser.error('--viewer_resolution requires GUI and positive width/height')
from himloco_lab.validation import MemoryMonitor
monitor=MemoryMonitor(int(args.device.split(':')[-1]) if ':' in args.device else 0)
launcher=AppLauncher(args)
app=launcher.app
try:
    import json,time
    import numpy as np
    import torch
    from himloco_lab.env_cfg import make_env_cfg
    from himloco_lab.env import HIMLabEnv
    from himloco_lab.adapter import LegacyRunnerAdapter
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg,train=load_config(args.task,args.num_envs,args.stage)
    lab_cfg=make_env_cfg(cfg,args.stage,args.device); lab_cfg.seed=args.seed
    if args.antialiasing:lab_cfg.sim.render.antialiasing_mode=args.antialiasing
    if args.viewer_resolution:lab_cfg.viewer.resolution=tuple(args.viewer_resolution)
    if args.render_every:
        lab_cfg.sim.render_interval=args.render_every
    lab_cfg.viewer.origin_type='asset_root' if args.follow_camera else 'env'
    lab_cfg.viewer.asset_name='robot'
    lab_cfg.viewer.eye=(3.,3.,2.); lab_cfg.viewer.lookat=(0.,0.,0.)
    env=HIMLabEnv(lab_cfg)
    if args.viewer_resolution:
        from himloco_lab.visualization import configure_viewport
        configure_viewport(args.viewer_resolution)
    wrapped=LegacyRunnerAdapter(env,plain_ppo=args.stage==2)
    log=args.log_dir or ROOT/'isaaclab_migration/logs'/args.task/f'stage{args.stage}_{time.strftime("%Y%m%d_%H%M%S")}'
    log.mkdir(parents=True,exist_ok=True)
    from himloco_lab.config import class_to_dict
    (log/'config.json').write_text(json.dumps({'env':class_to_dict(cfg),'train':train,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},indent=2))
    if args.capture_dir:
        from himloco_lab.visualization import ViewportCapture
        wrapped.capture=ViewportCapture(args.capture_dir,args.capture_every)
    if args.stage==4 and cfg.terrain.curriculum and env.terrain is not None:
        from himloco_lab.curriculum import CurriculumMonitor
        wrapped.diagnostics=CurriculumMonitor(env,log)
    if args.stage==1:
        wrapped.reset(); failures=0; minimum_height=100.; max_joint_error=0.
        for _ in range(args.steps):
            result=wrapped.step(torch.zeros(args.num_envs,wrapped.num_actions,device=wrapped.device))
            if not torch.isfinite(result[0]).all():raise RuntimeError('Nonfinite standing observation')
            failures+=int(result[3].sum())
            minimum_height=min(minimum_height,float(env.core.root_states[:,2].min()))
            max_joint_error=max(max_joint_error,float((env.core.dof_pos-env.core.default_dof_pos).abs().max()))
        report={'stage':1,'steps':args.steps,'num_envs':args.num_envs,'resets':failures,'minimum_height':minimum_height,'max_joint_error':max_joint_error}
        (log/'standing.json').write_text(json.dumps(report,indent=2)); print(report)
        if failures or minimum_height<.15:raise RuntimeError('Standing validation failed; see standing.json')
    else:
        if args.stage==2:
            from rsl_rl.runners import OnPolicyRunner as Runner
            train['runner']['policy_class_name']='ActorCritic';train['runner']['algorithm_class_name']='PPO'
        else:
            from rsl_rl.runners import HIMOnPolicyRunner as Runner
        if args.rollout_steps:train['runner']['num_steps_per_env']=args.rollout_steps
        # Persist the effective training config after stage/rollout overrides.
        (log/'config.json').write_text(json.dumps({'env':class_to_dict(cfg),'train':train,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},indent=2))
        runner=Runner(wrapped,train,log_dir=str(log),device=args.device)
        if args.checkpoint:runner.load(str(args.checkpoint))
        runner.learn(args.iterations if args.iterations is not None else train['runner']['max_iterations'],init_at_random_ep_len=True)
        print('TRAINING_COMPLETE',log)
    from himloco_lab.validation import validate_runtime
    report=validate_runtime(env)
    if wrapped.diagnostics is not None:
        curriculum=wrapped.diagnostics.save()
        report['terrain_curriculum']=curriculum['final']
    report.update(stage=args.stage,task=args.task,num_envs=args.num_envs,status='passed',
        peak_gpu_used_mib=monitor.stop(),peak_torch_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0.)
    if args.stage!=1:
        if not all(torch.isfinite(p).all() for p in runner.alg.actor_critic.parameters()):
            raise RuntimeError('Nonfinite trained parameter')
        report['iterations']=runner.current_learning_iteration
        report['rollout_steps']=runner.num_steps_per_env
    (log/'validation.json').write_text(json.dumps(report,indent=2))
    print('VALIDATION_PASSED',json.dumps(report),flush=True)
    if args.export_policy:
        from himloco_lab.export import export_policy
        checkpoint=log/f'model_{runner.current_learning_iteration}.pt'
        metadata=export_policy(runner.alg.actor_critic,args.export_policy,cfg,checkpoint)
        print('POLICY_EXPORTED',args.export_policy.resolve(),metadata['round_trip_max_error'],flush=True)
    if wrapped.capture is not None:
        print('VIEWPORT_CAPTURED',wrapped.capture.capture_final(env.sim),flush=True)
    if args.preview_seconds>0 and args.stage in (3,4):
        deadline=time.monotonic()+args.preview_seconds
        with torch.inference_mode():
            while app.is_running() and time.monotonic()<deadline:
                tick=time.monotonic()
                wrapped.step(runner.alg.actor_critic.act_inference(wrapped.get_observations()))
                time.sleep(max(0.,env.step_dt-(time.monotonic()-tick)))
        if wrapped.capture is not None:
            print('PREVIEW_CAPTURED',wrapped.capture.capture_final(env.sim,'preview.png'),flush=True)
    wrapped.close()
except BaseException:
    import traceback
    traceback.print_exc()
    monitor.stop()
    sys.stdout.flush(); sys.stderr.flush()
    raise
finally:
    if 'env' in globals():
        env.close()
    app.close()
