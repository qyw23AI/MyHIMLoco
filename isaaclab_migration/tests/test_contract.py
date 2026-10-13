"""CPU differential tests against actual local Gym source, without importing Gym."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'isaaclab_migration'))
sys.path.insert(0,str(ROOT/'rsl_rl'))
from himloco_lab.config import load_config,asset_contract,TASKS,class_to_dict
from himloco_lab.logic import LegacyLogic
from himloco_lab import math as math_utils
from himloco_lab.terrain.heightfield import SubTerrain,pyramid_stairs_terrain

def original_logic():
    """Compile only independent original tensor methods, not simulator imports."""
    p=ROOT/'legged_gym/legged_gym/envs/base/legged_robot.py'
    tree=ast.parse(p.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef))
    names=[n for n in LegacyLogic.__dict__ if not n.startswith('__') and n!='_init_buffers']
    cls.bases=[];cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    module=ast.Module(body=[cls],type_ignores=[])
    scope=dict(vars(math_utils),np=np,torch=torch,class_to_dict=class_to_dict)
    exec(compile(ast.fix_missing_locations(module),str(p),'exec'),scope)
    return scope['LeggedRobot']

class ContractTests(unittest.TestCase):
    def fixture(self,klass=LegacyLogic):
        c=klass(); c.cfg,_=load_config(); c.device='cpu'; c.num_envs=8
        c.num_actions=c.num_dof=12; c.num_obs=270;c.num_privileged_obs=238
        c.num_one_step_obs=45;c.num_one_step_privileged_obs=238;c.num_bodies=5
        c.sim_params=SimpleNamespace(dt=.005); c._parse_cfg(c.cfg)
        c.obs_buf=torch.randn(8,270);c.privileged_obs_buf=torch.randn(8,238)
        c.commands=torch.randn(8,4);c.commands_scale=torch.tensor([2.,2.,.25])
        c.base_ang_vel=torch.randn(8,3);c.base_lin_vel=torch.randn(8,3)
        c.projected_gravity=torch.randn(8,3);c.dof_pos=torch.randn(8,12);c.dof_vel=torch.randn(8,12)
        c.default_dof_pos=torch.randn(1,12);c.actions=torch.randn(8,12)
        c.last_actions=torch.randn(8,12);c.last_last_actions=torch.randn(8,12)
        c.last_dof_vel=torch.randn(8,12); c.p_gains=torch.full((12,),40.);c.d_gains=torch.ones(12)
        c.Kp_factors=torch.rand(8,1);c.Kd_factors=torch.rand(8,1);c.torque_limits=torch.full((12,),25.)
        c.motor_strength_factors=torch.full((8,1),100.) # deliberately unused by original controller
        c.torques=torch.randn(8,12);c.disturbance=torch.randn(8,5,3);c.root_states=torch.randn(8,13)
        c.base_quat=math_utils.normalize(torch.randn(8,4));c.root_states[:,3:7]=c.base_quat
        c.measured_heights=torch.randn(8,187);c.noise_scale_vec=c._get_noise_scale_vec(c.cfg)
        c.contact_forces=torch.randn(8,5,3);c.feet_indices=torch.tensor([1,2,3,4])
        c.penalised_contact_indices=torch.tensor([0,1]);c.termination_contact_indices=torch.tensor([0])
        c.feet_pos=torch.randn(8,4,3);c.feet_vel=torch.randn(8,4,3)
        c.dof_pos_weights=torch.tensor([4.,2.,0.]*4)
        c.dof_vel_limits=torch.full((12,),30.);c.dof_pos_limits=torch.tensor([[-2.,2.]]*12)
        c.feet_air_time=torch.rand(8,4);c.last_contacts=torch.zeros(8,4,dtype=torch.bool)
        c.episode_length_buf=torch.arange(997,1005);c.reset_buf=torch.zeros(8,dtype=torch.bool)
        c.time_out_buf=torch.zeros(8,dtype=torch.bool);c.rew_buf=torch.zeros(8)
        c.dof_names=asset_contract(c.cfg)[1]
        return c

    def compare(self,name,*args):
        a=self.fixture();b=original_logic()();b.__dict__=copy.deepcopy(a.__dict__)
        torch.manual_seed(42);ra=getattr(a,name)(*args)
        torch.manual_seed(42);rb=getattr(b,name)(*args)
        if isinstance(ra,torch.Tensor):torch.testing.assert_close(ra,rb,rtol=0,atol=0)
        for key in ['obs_buf','privileged_obs_buf','reset_buf','time_out_buf']:
            torch.testing.assert_close(getattr(a,key),getattr(b,key),rtol=0,atol=0)

    def test_observation_and_terminal_differential(self):
        self.compare('compute_observations');self.compare('get_current_obs')
        self.compare('compute_termination_observations',torch.tensor([1,4]))

    def test_pd_and_rewards_differential(self):
        for mode in ['P','V','T']:
            a=self.fixture();b=original_logic()();b.__dict__=copy.deepcopy(a.__dict__)
            a.cfg.control.control_type=b.cfg.control.control_type=mode
            actions=torch.randn(8,12)
            torch.testing.assert_close(a._compute_torques(actions),b._compute_torques(actions),rtol=0,atol=0)
        for name in LegacyLogic.__dict__:
            if name.startswith('_reward_'):self.compare(name)
        self.compare('check_termination')

    def test_current_first_history_and_velocity_target(self):
        c=self.fixture();old=c.obs_buf.clone();c.cfg.noise.noise_level=0.;c.add_noise=False
        c.noise_scale_vec.zero_();c.compute_observations()
        torch.testing.assert_close(c.obs_buf[:,45:],old[:,:225])
        torch.testing.assert_close(c.privileged_obs_buf[:,45:48],c.base_lin_vel*2.)
        torch.testing.assert_close(c.privileged_obs_buf[:,48:51],c.disturbance[:,0])
        torch.testing.assert_close(c.obs_buf[:,:3],c.commands[:,:3]*c.commands_scale)

    def test_timeout_strictly_greater(self):
        c=self.fixture();c.contact_forces.zero_();c.check_termination()
        self.assertFalse(c.time_out_buf[3]);self.assertTrue(c.time_out_buf[4])

    def test_dimensions_and_joint_contracts(self):
        for task in TASKS:
            c,_=load_config(task);_,names,_,_=asset_contract(c)
            self.assertEqual(len(names),c.env.num_actions)
            self.assertEqual(c.env.num_one_step_observations,9+3*len(names))
            self.assertEqual(c.env.num_observations,6*c.env.num_one_step_observations)
            self.assertEqual(c.env.num_privileged_obs,c.env.num_one_step_observations+6+187)
        c,_=load_config('mybot_v3');self.assertEqual(asset_contract(c)[1][0:3],['FL_hip_joint','FL_thigh_joint','FL_calf_joint'])

    def test_stair_quantization_and_center(self):
        t=SubTerrain('test',80,80,.005,.1)
        pyramid_stairs_terrain(t,.31,.05,3.)
        self.assertEqual(t.height_field_raw[0,0],0)
        self.assertEqual(t.height_field_raw[3,3],10)
        self.assertEqual(t.height_field_raw[40,40],90)

    def test_no_gym_runtime_imports(self):
        for path in (ROOT/'isaaclab_migration/himloco_lab').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith(('isaacgym','legged_gym')),str(path))
                elif isinstance(node,ast.Import):
                    self.assertFalse(any(n.name.startswith(('isaacgym','legged_gym')) for n in node.names),str(path))

    def test_bridge_meshes_unchanged(self):
        import importlib.util
        from himloco_lab.terrain import bridge_mesh as migrated
        spec=importlib.util.spec_from_file_location('original_bridge',ROOT/'legged_gym/legged_gym/utils/terrain/bridge_mesh.py')
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        for rotated in [False,True]:
            kwargs=dict(track_width=2.4,track_block_length=2.4,wall_thickness=.04,n_beams=5,
                beam_width=.4,gap_width=.15,bridge_height=.15,bridge_length=2.4,rotate_90=rotated,ramp_length=.4)
            a=migrated.create_bridge_mesh(**kwargs);b=original.create_bridge_mesh(**kwargs)
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

    def test_sources_not_modified(self):
        manifest=json.loads((ROOT/'isaaclab_migration/docs/source_manifest.json').read_text())
        for name,digest in manifest.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)

    def test_torchscript_export_after_cached_distribution(self):
        import tempfile
        from rsl_rl.modules import HIMActorCritic
        from himloco_lab.export import export_policy
        cfg,_=load_config()
        model=HIMActorCritic(270,238,45,12)
        # Cached PPO distribution deliberately contains non-leaf actor outputs.
        model.act(torch.randn(8,270))
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'policy.pt'
            meta=export_policy(model,output,cfg)
            scripted=torch.jit.load(str(output))
            self.assertEqual(meta['round_trip_max_error'],0.)
            for batch in [1,8,128]:
                sample=torch.randn(batch,270)
                with torch.no_grad():
                    torch.testing.assert_close(scripted(sample),model.act_inference(sample),rtol=1e-6,atol=1e-6)

    def test_him_update_preserves_all_losses(self):
        from rsl_rl.modules import HIMActorCritic
        from rsl_rl.algorithms import HIMPPO
        torch.manual_seed(1)
        model=HIMActorCritic(270,238,45,12)
        alg=HIMPPO(model,num_learning_epochs=1,num_mini_batches=1)
        alg.init_storage(8,4,[270],[238],[12])
        before=model.estimator.proto.weight.detach().clone()
        with torch.inference_mode():
            for _ in range(4):
                obs=torch.randn(8,270);critic=torch.randn(8,238)
                alg.act(obs,critic)
                alg.process_env_step(torch.rand(8),torch.zeros(8,dtype=torch.bool),{},torch.randn(8,238))
            alg.compute_returns(critic)
        losses=alg.update()
        self.assertEqual(len(losses),4);self.assertTrue(np.isfinite(losses).all())
        self.assertGreater(losses[2],0);self.assertGreater(losses[3],0)
        self.assertFalse(torch.equal(before,model.estimator.proto.weight))

if __name__=='__main__':unittest.main()
