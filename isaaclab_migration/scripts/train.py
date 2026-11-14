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
from himloco_lab.task_gait.experiments import EXPERIMENTS as POSTURE_EXPERIMENTS
parser=argparse.ArgumentParser()

# 日常训练示例（在仓库根目录运行）：
# python isaaclab_migration/scripts/train.py --task wblwbl_dog_v1_crouch --stage 3 --num_envs 512 --headless
#
# --task 是 himloco_lab/config.py 中 TASKS 的键，用来选择环境配置和训练配置。
# 环境配置包括机器人资产、初始姿态、奖励等；同模块的 *CfgPPO 提供训练参数。
# wblwbl_dog_v1 选正常行走配置，wblwbl_dog_v1_crouch 选同一机器人低趴配置。
# 省略 --task 会选 mybot_v3，因此训练自己的机器人时要显式指定。
parser.add_argument('--task',choices=TASKS,default='mybot_v3')

# --stage 选运行方式：1 零动作站立测试，2 平地普通 PPO，3 平地 HIM，
# 4 HIM 并使用任务配置里的地形。WBLWBL 常规训练用 3。
parser.add_argument('--stage',type=int,choices=[1,2,3,4],default=4)
# 并行仿真环境数。环境越多，每轮采样越多，显存开销也越大。
parser.add_argument('--num_envs',type=int,default=1024)
# 策略更新轮数。省略时用任务的 runner.max_iterations。
parser.add_argument('--iterations',type=int,default=None)
# 只在 Stage 1 使用，指定零动作测试的控制步数。
parser.add_argument('--steps',type=int,default=500,help='Standing test policy steps')
# 每轮更新前，每个环境采集的控制步数。省略时用 runner.num_steps_per_env。
parser.add_argument('--rollout_steps',type=int,default=None,help='Explicit smoke-test override; default retains original 100/24')
# 随机种子，用于重复同一组实验。
parser.add_argument('--seed',type=int,default=1)

# 专项实验参数。A1 论文配置只用于 --task a1 --stage 4 --num_envs 4096。
parser.add_argument('--paper_a1_rollout',type=int,choices=[100,200],default=None,
                    help='Use paper A1 reward parameters with released-source behavior; choose 100 or 200 rollout steps')
# 以下三项仅用于 mybot_v3_task_gait，分别选择步态训练阶段、消融方案和姿态诊断方案。
parser.add_argument('--gait_phase', choices=['basic','precision','robust'], default=None)
parser.add_argument('--gait_ablation', choices=['baseline','timing','full'], default=None)
parser.add_argument('--posture_experiment', choices=POSTURE_EXPERIMENTS, default=None,
                    help='Opt-in controlled posture diagnostic arm; saved in checkpoints')

# 加载已有的 model_*.pt 继续训练。省略时从头训练。
parser.add_argument('--checkpoint',type=Path)
# 指定输出目录。默认按 task、stage 和时间归档，config.json 记录生效配置。
parser.add_argument('--log_dir',type=Path)
# 覆盖 Stage 3/4 的 TorchScript 导出路径。默认写入 OUTPUT/<task>/<run>/policy.pt。
parser.add_argument('--export_policy',type=Path,help='Override the Stage 3/4 policy export path (default: OUTPUT/<task>/<run>/policy.pt)')

# 界面和录制参数。capture_dir 保存视口图片，capture_every 控制截图间隔。
parser.add_argument('--capture_dir',type=Path,help='Save actual viewport frames; requires GUI')
parser.add_argument('--capture_every',type=int,default=100)
# 每隔多少个物理步渲染一帧，与策略的控制频率无关。
parser.add_argument('--render_every',type=int,default=None,help='Render interval in physics steps; does not change control dt')
# 训练后用当前策略预览多少秒。0 跳过预览，仅在 Stage 3/4 执行。
parser.add_argument('--preview_seconds',type=float,default=0.,help='Run trained deterministic policy in GUI after training')
# 跟随相机、窗口分辨率和抗锯齿模式用于调整视口。
parser.add_argument('--follow_camera',action='store_true',help='Track robot every render; default allows free mouse navigation')
parser.add_argument('--viewer_resolution',type=int,nargs=2,metavar=('WIDTH','HEIGHT'))
parser.add_argument('--antialiasing',choices=['Off','FXAA','DLSS','DLAA','TAA'])

# Isaac Lab 在这里添加运行参数，例如 --headless 关闭窗口，--device cuda:0 选择设备。
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.paper_a1_rollout is not None and (args.task!='a1' or args.stage!=4 or args.num_envs!=4096):
    parser.error('--paper_a1_rollout requires --task a1 --stage 4 --num_envs 4096')
if args.task=='mybot_v3_task_gait':
    if args.stage not in (3,4):parser.error('Task gait requires HIM stage 3/4')
elif args.gait_phase or args.gait_ablation or args.posture_experiment:
    parser.error('Gait options require --task mybot_v3_task_gait')
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
    # 根据 task 载入环境配置 cfg 和训练参数 train，再应用阶段和命令行覆盖项。
    cfg,train=load_config(args.task,args.num_envs,args.stage)
    if args.paper_a1_rollout is not None:
        from himloco_lab.paper_a1 import apply_paper_a1
        cfg,train=apply_paper_a1(cfg,train,args.paper_a1_rollout)
    if args.task=='mybot_v3_task_gait':
        from himloco_lab.task_gait.checkpoint import resolve_training_config
        cfg,train=resolve_training_config(args.num_envs,args.gait_phase,args.checkpoint,args.gait_ablation)
        if args.posture_experiment:
            from himloco_lab.task_gait.posture import apply_experiment
            apply_experiment(cfg, args.posture_experiment)
        args.stage=4 if cfg.gait.phase=='robust' else 3
    train['runner']['experiment_name'] = args.task
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
    run_name = f'stage{args.stage}_{time.strftime("%Y%m%d_%H%M%S")}'
    migration_dir = ROOT / 'isaaclab_migration'
    log = args.log_dir or migration_dir / 'logs' / args.task / run_name
    if args.stage in (3, 4) and args.export_policy is None:
        args.export_policy = migration_dir / 'OUTPUT' / args.task / run_name / 'policy.pt'
    log.mkdir(parents=True,exist_ok=True)
    from himloco_lab.config import class_to_dict
    (log/'config.json').write_text(json.dumps({'env':class_to_dict(cfg),'train':train,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},indent=2))
    if args.capture_dir:
        from himloco_lab.visualization import ViewportCapture
        wrapped.capture=ViewportCapture(args.capture_dir,args.capture_every)
    if args.stage==4 and cfg.terrain.curriculum and env.terrain is not None:
        from himloco_lab.curriculum import CurriculumMonitor
        wrapped.diagnostics=CurriculumMonitor(env,log)
    if hasattr(cfg,'gait'):
        from himloco_lab.task_gait.metrics import TaskGaitMonitor
        wrapped.diagnostics=TaskGaitMonitor(env,log)
    if args.stage==1:
        wrapped.reset(); failures=0; minimum_height=100.; max_joint_error=0.
        contact_bodies = torch.zeros(env.robot.num_bodies, dtype=torch.bool, device=wrapped.device)
        min_up_dot = 1.0
        max_effort = 0.0
        min_allowed_height = getattr(cfg.init_state, 'standing_test_min_height', 0.15)
        for _ in range(args.steps):
            result=wrapped.step(torch.zeros(args.num_envs,wrapped.num_actions,device=wrapped.device))
            if not torch.isfinite(result[0]).all():raise RuntimeError('Nonfinite standing observation')
            failures+=int(result[3].sum())
            minimum_height=min(minimum_height,float(env.core.root_states[:,2].min()))
            max_joint_error=max(max_joint_error,float((env.core.dof_pos-env.core.default_dof_pos).abs().max()))
            min_up_dot = min(min_up_dot, float(-env.core.projected_gravity[:, 2].max()))
            max_effort = max(max_effort, float(env.core.torques.abs().max()))
            contact_bodies |= (env.core.contact_forces.norm(dim=-1) > 1.0).any(dim=0)
        report={'stage':1,'steps':args.steps,'num_envs':args.num_envs,'resets':failures,
            'minimum_height':minimum_height,'minimum_allowed_height':min_allowed_height,
            'final_height_mean':float(env.core.root_states[:,2].mean()),
            'min_up_dot':min_up_dot,'peak_joint_effort_nm':max_effort,'max_joint_error':max_joint_error,
            'bodies_with_contact': [name for name, touched in zip(env.robot.body_names, contact_bodies.tolist()) if touched]}
        (log/'standing.json').write_text(json.dumps(report,indent=2)); print(report)
        if failures or minimum_height<min_allowed_height:raise RuntimeError('Standing validation failed; see standing.json')
    else:
        if args.stage==2:
            from rsl_rl.runners import OnPolicyRunner as Runner
            train['runner']['policy_class_name']='ActorCritic';train['runner']['algorithm_class_name']='PPO'
        else:
            from rsl_rl.runners import HIMOnPolicyRunner as Runner
            if hasattr(cfg,'gait'):
                from himloco_lab.task_gait.runner import TaskGaitRunner as Runner
        if args.rollout_steps:train['runner']['num_steps_per_env']=args.rollout_steps
        if args.iterations is not None:train['runner']['max_iterations']=args.iterations
        # Persist the effective training config after stage/rollout overrides.
        (log/'config.json').write_text(json.dumps({'env':class_to_dict(cfg),'train':train,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},indent=2))
        if hasattr(cfg, 'gait'):
            from himloco_lab.task_gait.run_log import record_run
            record_run(log, cfg, train, args)
        runner=Runner(wrapped,train,log_dir=str(log),device=args.device)
        if args.checkpoint:runner.load(str(args.checkpoint))
        runner.learn(args.iterations if args.iterations is not None else train['runner']['max_iterations'],init_at_random_ep_len=True)
        print('TRAINING_COMPLETE',log)
    from himloco_lab.validation import validate_runtime
    report=validate_runtime(env)
    if wrapped.diagnostics is not None:
        curriculum=wrapped.diagnostics.save()
        report['gait_metrics' if hasattr(cfg,'gait') else 'terrain_curriculum']=(
            curriculum if hasattr(cfg,'gait') else curriculum['final'])
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
    print('RUN_COMPLETED',log.resolve(),flush=True)
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
