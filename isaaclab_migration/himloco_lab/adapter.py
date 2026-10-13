"""Keep the repository's RSL-RL runner and terminal-observation contract unchanged."""
import torch
class LegacyRunnerAdapter:
    def __init__(self,env,plain_ppo=False):
        self.env=env; self.plain_ppo=plain_ppo
        self.capture=None
        self.diagnostics=None
    def __getattr__(self,name):return getattr(self.env.core,name)
    @property
    def episode_length_buf(self):return self.env.episode_length_buf
    @episode_length_buf.setter
    def episode_length_buf(self,value):
        self.env.episode_length_buf.copy_(value)
    def get_observations(self):return self.env.core.obs_buf
    def get_privileged_observations(self):return self.env.core.privileged_obs_buf
    def reset(self):
        self.env._reset_idx(torch.arange(self.num_envs,device=self.device))
        result=self.step(torch.zeros(self.num_envs,self.num_actions,device=self.device))
        return result[0],result[1]
    def step(self,actions):
        if self.diagnostics is not None:self.diagnostics.before_step()
        obs,reward,terminated,truncated,extras=self.env.step(actions)
        if self.diagnostics is not None:self.diagnostics.after_step(terminated,truncated)
        if self.capture is not None:self.capture.on_step()
        result=(obs['policy'],obs['critic'],reward,terminated|truncated,extras)
        if self.plain_ppo:return result
        return result+(self.env.terminal_ids,self.env.terminal_obs)
    def close(self):self.env.close()
