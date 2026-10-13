"""Real terrain startup, curriculum transitions, and reset placement acceptance."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'isaaclab_migration'))
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
try:
    import torch
    from himloco_lab.config import load_config
    from himloco_lab.env_cfg import make_env_cfg
    from himloco_lab.env import HIMLabEnv
    from himloco_lab.adapter import LegacyRunnerAdapter
    cfg,_=load_config('mybot_v3',8,4)
    env=HIMLabEnv(make_env_cfg(cfg,4));c=env.core
    assert cfg.terrain.curriculum and cfg.commands.curriculum
    assert c.terrain.height_field_raw.max()>c.terrain.height_field_raw.min()
    original=c.terrain_levels.clone()
    env._reset_idx(torch.arange(8,device=env.device))
    assert torch.equal(c.terrain_levels,original),'Initial reset falsely changed levels'
    # Controlled acceptance probes, separate from natural training statistics.
    ids=torch.tensor([0,1],device=env.device)
    c.terrain_levels[ids]=2
    c.env_origins[ids]=c.terrain_origins[c.terrain_levels[ids],c.terrain_types[ids]]
    c.root_states[ids,:2]=c.env_origins[ids,:2]
    c.root_states[0,0]+=c.terrain.env_length*.6
    c.commands[ids,:2]=torch.tensor([1.,0.],device=env.device)
    env._reset_idx(ids)
    assert c.terrain_levels[ids].tolist()==[3,1],'Promotion/demotion failed'
    torch.testing.assert_close(c.env_origins[ids],c.terrain_origins[c.terrain_levels[ids],c.terrain_types[ids]])
    torch.testing.assert_close(env.scene.env_origins[ids],c.env_origins[ids])
    actual=env.robot.data.root_pos_w[ids]
    assert ((actual[:,:2]-c.env_origins[ids,:2]).abs()<=1.001).all()
    torch.testing.assert_close(actual[:,2],c.env_origins[ids,2]+cfg.init_state.pos[2])
    wrapped=LegacyRunnerAdapter(env)
    for _ in range(10):
        result=wrapped.step(torch.zeros(8,12,device=env.device))
        assert torch.isfinite(result[0]).all() and torch.isfinite(result[2]).all()
    report={'status':'passed','mesh_type':cfg.terrain.mesh_type,'terrain_curriculum':True,
            'command_curriculum':True,'checks':['nonflat collision terrain',
            'initial reset leaves levels unchanged','distance-based promotion 2 to 3',
            'insufficient-distance demotion 2 to 1','reset uses updated terrain origin',
            'finite observations and rewards on actual terrain'],
            'note':'Controlled transition probes; not training success statistics'}
    (ROOT/'isaaclab_migration/docs/terrain_contract_result.json').write_text(json.dumps(report,indent=2))
    print('TERRAIN_CONTRACT_PASSED',report,flush=True)
except BaseException:
    import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();raise
finally:
    if 'env' in globals():env.close()
    app.close()
