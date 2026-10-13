"""GPU acceptance: terminal targets before reset, history continuity and partial reset."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'isaaclab_migration'))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
app=AppLauncher(args).app
try:
    import torch
    from himloco_lab.config import load_config
    from himloco_lab.env_cfg import make_env_cfg
    from himloco_lab.env import HIMLabEnv
    from himloco_lab.adapter import LegacyRunnerAdapter
    cfg,_=load_config('mybot_v3',8,1)
    env=HIMLabEnv(make_env_cfg(cfg,1,args.device));wrapped=LegacyRunnerAdapter(env)
    wrapped.reset();c=env.core
    c.obs_buf.copy_(torch.arange(270,device=env.device)[None,:].expand(8,-1)/10)
    before=c.obs_buf.clone()
    env.episode_length_buf[0]=int(c.max_episode_length)
    captured={};original=c.compute_termination_observations
    def capture(ids):
        result=original(ids)
        captured['ids']=ids.clone();captured['target']=result.clone();captured['q']=c.dof_pos[ids].clone()
        return result
    c.compute_termination_observations=capture
    result=wrapped.step(torch.zeros(8,12,device=env.device))
    obs,priv,reward,dones,info,ids,terminal=result
    assert ids.tolist()==[0],ids
    assert info['time_outs'].tolist()==[True]+[False]*7
    torch.testing.assert_close(terminal,captured['target'],rtol=0,atol=0)
    torch.testing.assert_close(terminal[:,9:21],captured['q']-c.default_dof_pos,rtol=0,atol=0)
    torch.testing.assert_close(obs[:,45:],before[:,:225],rtol=0,atol=0)
    assert env.episode_length_buf[0]==0
    assert torch.equal(c.last_actions,c.actions)
    # The following step must not inherit a stale timeout flag.
    result=wrapped.step(torch.zeros(8,12,device=env.device))
    assert not result[4]['time_outs'].any()
    # Actions must map to the simulator by joint name, in both directions.
    target=torch.linspace(-1.,1.,12,device=env.device).repeat(8,1)
    env.robot.set_joint_effort_target(target,joint_ids=env.joint_ids)
    torch.testing.assert_close(env.robot.data.joint_effort_target[:,env.joint_ids],target)
    report={'status':'passed','num_envs':8,'checks':['terminal observation before reset','unchanged six-frame history across reset','strict timeout threshold','no stale timeout','name-based effort permutation']}
    out=ROOT/'isaaclab_migration/docs/sim_contract_result.json';out.write_text(json.dumps(report,indent=2))
    print('SIM_CONTRACT_PASSED',report,flush=True);wrapped.close()
except BaseException:
    import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();raise
finally:app.close()
