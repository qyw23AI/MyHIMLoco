"""Runtime acceptance checks and measured GPU memory (includes PhysX/Kit)."""
import subprocess
import threading
import torch

class MemoryMonitor:
    def __init__(self,index=0):
        self.peak_mib=0.;self.index=index;self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def _run(self):
        while not self.stop_event.is_set():
            try:
                out=subprocess.check_output(['nvidia-smi',f'--id={self.index}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=3)
                self.peak_mib=max(self.peak_mib,float(out.strip()))
            except (OSError,ValueError,subprocess.SubprocessError):pass
            self.stop_event.wait(.5)
    def stop(self):
        self.stop_event.set();self.thread.join(timeout=4)
        return self.peak_mib

def validate_runtime(env):
    c=env.core
    assert c.obs_buf.shape==(c.num_envs,c.num_obs)
    assert c.privileged_obs_buf.shape==(c.num_envs,c.num_privileged_obs)
    assert abs(c.dt-env.step_dt)<1e-10
    assert len(set(env.joint_ids))==c.num_actions
    assert len(c.feet_indices)==4
    for tensor in [c.obs_buf,c.privileged_obs_buf,c.dof_pos,c.dof_vel,c.rew_buf]:
        if not torch.isfinite(tensor).all():raise RuntimeError('Nonfinite environment tensor')
    return {'actor_obs':c.num_obs,'critic_obs':c.num_privileged_obs,'one_step_obs':c.num_one_step_obs,
        'history_length':c.history_length,'dt':c.dt,'physics_dt':env.physics_dt,
        'joint_order':c.dof_names,'lab_joint_order':env.robot.joint_names,'policy_to_lab':env.joint_ids}
