"""Legacy tensor lifecycle, with simulator operations delegated to the Lab backend."""
from types import SimpleNamespace
import torch
from .logic import LegacyLogic
from .field import FieldMixin
from .math import torch_rand_float
from .config import asset_contract

class HIMCore(LegacyLogic):
    def __init__(self,backend):
        self.backend=backend; self.cfg=backend.cfg.legacy
        self.device=backend.device; self.sim_device=self.device; self.sim_params=SimpleNamespace(dt=backend.physics_dt)
        self.num_envs=backend.num_envs; self.num_actions=self.cfg.env.num_actions
        self.num_obs=self.cfg.env.num_observations; self.num_privileged_obs=self.cfg.env.num_privileged_obs
        self.num_one_step_obs=self.cfg.env.num_one_step_observations
        self.num_one_step_privileged_obs=self.cfg.env.num_one_step_privileged_obs
        self.history_length=self.num_obs//self.num_one_step_obs
        self.num_dof=self.num_dofs=self.num_actions
        self.num_bodies=backend.robot.num_bodies; self.up_axis_idx=2; self.init_done=False
        _,self.dof_names,limits,root=asset_contract(self.cfg)
        self.body_names=backend.robot.body_names
        self.base_body_id=self.body_names.index(root)
        self.feet_indices=self.body_indices([self.cfg.asset.foot_name])
        self.penalised_contact_indices=self.body_indices(self.cfg.asset.penalize_contacts_on)
        self.termination_contact_indices=self.body_indices(self.cfg.asset.terminate_after_contacts_on)
        self.terrain=backend.terrain
        self._parse_cfg(self.cfg)
        # The original parser disables selected-terrain curriculum because mesh_type=None.
        # Preserve that behavior; do not silently activate previously inactive curriculum.
        self._get_env_origins()
        self.base_init_state=torch.tensor(self.cfg.init_state.pos+self.cfg.init_state.rot+self.cfg.init_state.lin_vel+self.cfg.init_state.ang_vel,device=self.device)
        self.obs_buf=torch.zeros(self.num_envs,self.num_obs,device=self.device)
        self.privileged_obs_buf=torch.zeros(self.num_envs,self.num_privileged_obs,device=self.device)
        self.rew_buf=torch.zeros(self.num_envs,device=self.device)
        self.reset_buf=torch.ones(self.num_envs,device=self.device,dtype=torch.bool)
        self.time_out_buf=torch.zeros_like(self.reset_buf)
        self.episode_length_buf=backend.episode_length_buf
        self.dof_pos_limits=torch.tensor([[limits[n]['lower'],limits[n]['upper']] for n in self.dof_names],device=self.device)
        mid=self.dof_pos_limits.mean(-1); width=self.dof_pos_limits[:,1]-self.dof_pos_limits[:,0]
        self.dof_pos_limits[:,0]=mid-width*.5*self.cfg.rewards.soft_dof_pos_limit
        self.dof_pos_limits[:,1]=mid+width*.5*self.cfg.rewards.soft_dof_pos_limit
        self.dof_vel_limits=backend.robot.data.joint_vel_limits[0,backend.joint_ids].clone() if any(limits[n]['velocity'] is None for n in self.dof_names) else torch.tensor([limits[n]['velocity'] for n in self.dof_names],device=self.device)
        self.torque_limits=torch.tensor([limits[n]['effort'] for n in self.dof_names],device=self.device)
        self._init_buffers()
        self._prepare_reward_function()
        self.friction_coeffs=torch.full((self.num_envs,1),self.cfg.terrain.dynamic_friction,device=self.device)
        self.restitution_coeffs=torch.full((self.num_envs,1),self.cfg.terrain.restitution,device=self.device)
        for prefix in ['front','rear']:
            names=getattr(self.cfg.asset,prefix+'_hip_names',[])
            setattr(self,prefix+'_hip_indices',[self.dof_names.index(n) for n in names])
        self.hip_indices=[i for i,n in enumerate(self.dof_names) if 'hip' in n.lower()]
        backend.randomize_body_properties(self)
        self.init_done=True

    def body_indices(self,patterns):
        return torch.tensor([i for p in patterns for i,n in enumerate(self.body_names) if p in n],device=self.device,dtype=torch.long)

    def _reset_dofs(self,ids):
        # Legacy reset ignores randomize_initial_joint_pos and its configured range.
        q=self.default_dof_pos*torch_rand_float(.5,1.5,(len(ids),self.num_dof),self.device)
        if self.backend.cfg.stage==1: q=self.default_dof_pos.expand(len(ids),-1).clone()
        self.dof_pos[ids]=q; self.dof_vel[ids]=0.
        self.backend.robot.write_joint_state_to_sim(q,torch.zeros_like(q),joint_ids=self.backend.joint_ids,env_ids=ids)

    def _reset_root_states(self,ids):
        self.root_states[ids]=self.base_init_state
        self.root_states[ids,:3]+=self.env_origins[ids]
        if self.custom_origins:
            self.root_states[ids,:2]+=torch_rand_float(-1.,1.,(len(ids),2),self.device)
        self.root_states[ids,7:13]=torch_rand_float(-.5,.5,(len(ids),6),self.device)
        if self.backend.cfg.stage==1:self.root_states[ids,7:13]=0.
        self.backend.write_root(self.root_states[ids],ids)

    def refresh_actor_rigid_shape_props(self,ids):
        self.backend.randomize_material(self,ids)

    def _push_robots(self):
        max_vel=self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:,7:9]=torch_rand_float(-max_vel,max_vel,(self.num_envs,2),self.device)
        self.backend.robot.write_root_velocity_to_sim(self.root_states[:,7:13])

    def _disturbance_robots(self):
        # Legacy force is scheduled after physics; consume it on the next first substep.
        force=torch_rand_float(*self.cfg.domain_rand.disturbance_range,(self.num_envs,3),self.device)
        self.disturbance[:,0,:]=force
        self.backend.pending_force[:,self.base_body_id,:]=force

    def _resample_commands(self,ids):
        super()._resample_commands(ids)
        if self.backend.cfg.stage==1:self.commands[ids]=0.

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.backend.cfg.stage==1:self.commands[:]=0.

class HIMFieldCore(FieldMixin,HIMCore):
    pass
