"""DirectRLEnv backend. Original PPO sees only LegacyRunnerAdapter."""
import numpy as np
import torch
import trimesh
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.terrains.utils import create_prim_from_mesh
from .core import HIMCore, HIMFieldCore
from .config import asset_contract
from .math import quat_rotate_inverse, torch_rand_float

class HIMLabEnv(DirectRLEnv):
    def __init__(self,cfg,**kwargs):
        self.core=None
        super().__init__(cfg,**kwargs)
        _,names,_,_=asset_contract(cfg.legacy)
        self.joint_ids,_=self.robot.find_joints(names,preserve_order=True)
        if [self.robot.joint_names[i] for i in self.joint_ids]!=names:
            raise RuntimeError('Joint permutation mismatch')
        self.contact_ids=[self.contacts.body_names.index(n) for n in self.robot.body_names]
        self.pending_force=torch.zeros(self.num_envs,self.robot.num_bodies,3,device=self.device)
        core_cls=HIMFieldCore if cfg.legacy.terrain.selected=='BarrierTrack' else HIMCore
        self.core=core_cls(self)
        # Gym creates actors at legacy terrain origins before the first reset.
        # Lab clones start on its centered grid; leaving those poses here would
        # falsely promote terrain levels on the runner's initial reset.
        initial=self.core.base_init_state.repeat(self.num_envs,1)
        initial[:,:3]+=self.core.env_origins
        initial[:,:2]+=torch_rand_float(-1.,1.,(self.num_envs,2),self.device)
        self.write_root(initial,torch.arange(self.num_envs,device=self.device))
        self.sync_state(self.core)
        # Lab's centered clone grid differs from the legacy spawn/terrain origins.
        # Synchronize scene metadata so free-view and UI environment selection
        # start at the actual robot location. Robot reset positions remain legacy.
        self.scene.env_origins.copy_(self.core.env_origins)
        if self.viewport_camera_controller is not None and cfg.viewer.origin_type=='env':
            self.viewport_camera_controller.update_view_to_env()
        self.terminal_ids=torch.empty(0,device=self.device,dtype=torch.long)
        self.terminal_obs=torch.empty(0,self.core.num_privileged_obs,device=self.device)
        print('Legacy joint order:',names)
        print('Lab joint order:',self.robot.joint_names)
        print('Policy->Lab indices:',self.joint_ids)

    def _setup_scene(self):
        c=self.cfg.legacy
        self.robot=Articulation(self.cfg.robot)
        # URDF importer instances collision subtrees even with make_instanceable=False.
        # Deinstance the source before cloning so legacy contact/rest offsets are applied.
        from pxr import Usd,UsdPhysics,PhysxSchema
        prim=self.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')
        if asset_contract(c)[0].suffix=='.xml':
            # Sim 5.1 MJCF importer authors an extra articulation on the empty
            # worldBody even after ground/lights are excluded from the XML.
            # Keep the robot articulation and all links/joints unchanged.
            world=self.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/worldBody')
            if world and world.HasAPI(UsdPhysics.ArticulationRootAPI):
                if any(node.HasAPI(UsdPhysics.RigidBodyAPI) for node in Usd.PrimRange(world)):
                    raise RuntimeError('Unexpected MJCF worldBody contains robot bodies')
                world.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                world.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        for child in list(Usd.PrimRange(prim)):
            if child.IsInstance(): child.SetInstanceable(False)
        sim_utils.modify_collision_properties('/World/envs/env_0/Robot',
            sim_utils.CollisionPropertiesCfg(contact_offset=c.sim.physx.contact_offset,rest_offset=c.sim.physx.rest_offset))
        self.scene.articulations['robot']=self.robot
        self.contacts=ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors['contacts']=self.contacts
        self.terrain=None
        material=sim_utils.RigidBodyMaterialCfg(static_friction=c.terrain.static_friction,
            dynamic_friction=c.terrain.dynamic_friction,restitution=c.terrain.restitution,
            friction_combine_mode='average',restitution_combine_mode='average')
        if c.terrain.selected=='BarrierTrack':
            from .terrain.barrier_track import BarrierTrack
            self.terrain=BarrierTrack(c.terrain,self.num_envs)
            meshes=[]
            def collect(mesh,origin):
                meshes.append(trimesh.Trimesh(vertices=mesh[0]+np.asarray(origin),faces=mesh[1],process=False))
            self.terrain.build(collect,self.device)
            # Concatenate once, preserving every original bridge mesh and heightfield.
            mesh=trimesh.util.concatenate(meshes)
            create_prim_from_mesh('/World/ground',mesh,physics_material=material)
        elif c.terrain.mesh_type in ['trimesh','heightfield']:
            from .terrain.rough import Terrain
            from .terrain.heightfield import convert_heightfield_to_trimesh
            self.terrain=Terrain(c.terrain,self.num_envs)
            if not hasattr(self.terrain,'vertices'):
                self.terrain.vertices,self.terrain.triangles=convert_heightfield_to_trimesh(self.terrain.heightsamples,c.terrain.horizontal_scale,c.terrain.vertical_scale,c.terrain.slope_treshold)
            vertices=self.terrain.vertices.copy(); vertices[:,:2]-=c.terrain.border_size
            create_prim_from_mesh('/World/ground',trimesh.Trimesh(vertices=vertices,faces=self.terrain.triangles,process=False),physics_material=material)
        elif c.terrain.mesh_type=='plane':
            ground=sim_utils.GroundPlaneCfg(physics_material=material)
            ground.func('/World/ground',ground)
        else:
            raise ValueError(f'Unsupported terrain: {c.terrain.mesh_type}/{c.terrain.selected}')
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=['/World/ground'])
        light=sim_utils.DomeLightCfg(intensity=1500.,color=(.75,.75,.75))
        light.func('/World/Light',light)

    def sync_state(self,c,joints_only=False):
        d=self.robot.data
        c.dof_pos=d.joint_pos[:,self.joint_ids].clone()
        c.dof_vel=d.joint_vel[:,self.joint_ids].clone()
        if joints_only:return
        # Gym xyzw, Lab wxyz. Never pass policy-ordered values to Lab without explicit IDs.
        state=d.root_state_w.clone()
        state[:,3:7]=state[:,[4,5,6,3]]
        c.root_states=state; c.base_quat=state[:,3:7]
        bodies=d.body_state_w.clone()
        bodies[:,:,3:7]=bodies[:,:,[4,5,6,3]]
        c.rigid_body_states=bodies
        c.contact_forces=self.contacts.data.net_forces_w[:,self.contact_ids].clone()
        c.feet_pos=bodies[:,c.feet_indices,:3]
        c.feet_vel=bodies[:,c.feet_indices,7:10]
        if hasattr(c,'gravity_vec'):
            c.base_lin_vel=quat_rotate_inverse(c.base_quat,state[:,7:10])
            c.base_ang_vel=quat_rotate_inverse(c.base_quat,state[:,10:13])
            c.projected_gravity=quat_rotate_inverse(c.base_quat,c.gravity_vec)

    def write_root(self,state,ids):
        pose=state[:,:7].clone(); pose[:,3:7]=state[:,[6,3,4,5]]
        self.robot.write_root_pose_to_sim(pose,env_ids=ids)
        self.robot.write_root_velocity_to_sim(state[:,7:13],env_ids=ids)

    def randomize_body_properties(self,c):
        view=self.robot.root_physx_view
        ids=torch.arange(self.num_envs,dtype=torch.int32,device='cpu')
        masses=view.get_masses().clone(); original=masses.clone()
        base=c.base_body_id
        if c.cfg.domain_rand.randomize_payload_mass:masses[:,base]+=c.payload[:,0].cpu()
        if c.cfg.domain_rand.randomize_link_mass:
            others=[i for i in range(c.num_bodies) if i!=base]
            factors=np.random.uniform(*c.cfg.domain_rand.link_mass_range,(c.num_envs,len(others)))
            masses[:,others]*=torch.tensor(factors,dtype=masses.dtype)
        if (masses<=0).any():raise ValueError('Domain randomization produced nonpositive link mass')
        view.set_masses(masses,ids)
        inertia=view.get_inertias().clone()
        inertia *= (masses/original).unsqueeze(-1)
        view.set_inertias(inertia,ids)
        if c.cfg.domain_rand.randomize_com_displacement:
            com=view.get_coms().clone(); com[:,base,:3]=c.com_displacement.cpu()
            view.set_coms(com,ids)

    def randomize_material(self,c,ids):
        if c.cfg.domain_rand.randomize_friction:
            c.friction_coeffs[ids]=torch_rand_float(*c.cfg.domain_rand.friction_range,(len(ids),1),self.device)
        if c.cfg.domain_rand.randomize_restitution:
            c.restitution_coeffs[ids]=torch_rand_float(*c.cfg.domain_rand.restitution_range,(len(ids),1),self.device)
        cpu_ids=ids.to(device='cpu',dtype=torch.int32)
        mat=self.robot.root_physx_view.get_material_properties()
        mat[cpu_ids,:,0]=c.friction_coeffs[ids].cpu()
        mat[cpu_ids,:,1]=c.friction_coeffs[ids].cpu()
        mat[cpu_ids,:,2]=c.restitution_coeffs[ids].cpu()
        self.robot.root_physx_view.set_material_properties(mat,cpu_ids)

    def _pre_physics_step(self,actions):
        c=self.core
        c.actions=actions.clamp(-c.cfg.normalization.clip_actions,c.cfg.normalization.clip_actions)
        delay=torch.randint(0,self.cfg.decimation,(self.num_envs,1),device=self.device)
        c.delayed_actions=c.actions[:,None,:].repeat(1,self.cfg.decimation,1)
        if c.cfg.domain_rand.delay:
            for i in range(self.cfg.decimation):
                c.delayed_actions[:,i]=c.last_actions+(c.actions-c.last_actions)*(i>=delay)
        self.substep=0

    def _apply_action(self):
        c=self.core
        self.sync_state(c,joints_only=True)
        c.torques=c._compute_torques(c.delayed_actions[:,self.substep])
        self.robot.set_joint_effort_target(c.torques,joint_ids=self.joint_ids)
        self.robot.instantaneous_wrench_composer.set_forces_and_torques(forces=self.pending_force,torques=torch.zeros_like(self.pending_force),is_global=False)
        # Gym external force API applies one simulation step only.
        self.pending_force.zero_()
        self.substep+=1

    def _get_dones(self):
        c=self.core
        self.sync_state(c)
        c.common_step_counter=self.common_step_counter
        c._post_physics_step_callback()
        c.check_termination()
        return c.reset_buf & ~c.time_out_buf, c.time_out_buf

    def _get_rewards(self):
        c=self.core
        c.compute_reward()
        self.terminal_ids=c.reset_buf.nonzero(as_tuple=False).flatten()
        self.terminal_obs=c.compute_termination_observations(self.terminal_ids).clone()
        return c.rew_buf

    def _reset_idx(self,ids):
        if self.core is None:return
        # Preserve legacy pre-reset base velocity/gravity and action-history behavior.
        self.core.reset_idx(ids)
        self.scene.env_origins[ids]=self.core.env_origins[ids]
        self.robot.reset(ids)
        self.contacts.reset(ids)
        self.episode_length_buf[ids]=0

    def _get_observations(self):
        c=self.core
        c.compute_observations()
        clip=c.cfg.normalization.clip_observations
        c.obs_buf=c.obs_buf.clamp(-clip,clip)
        c.privileged_obs_buf=c.privileged_obs_buf.clamp(-clip,clip)
        c.disturbance.zero_()
        c.last_last_actions.copy_(c.last_actions)
        c.last_actions.copy_(c.actions)
        c.last_dof_vel.copy_(c.dof_vel)
        c.last_root_vel.copy_(c.root_states[:,7:13])
        self.extras=c.extras
        self.extras['time_outs']=c.time_out_buf.clone()
        return {'policy':c.obs_buf,'critic':c.privileged_obs_buf}
