"""Attach the task contract to rolling and final checkpoints without changing PPO."""
from rsl_rl.runners import HIMOnPolicyRunner
import json
from ..config import class_to_dict
from .config import TASK


class TaskGaitRunner(HIMOnPolicyRunner):
    def log(self, locs, *args, **kwargs):
        # The legacy loop updates its counter only after learn(). Periodic
        # checkpoints must resume AFTER the update whose weights they contain.
        self._completed_iterations = locs['it'] + 1
        return super().log(locs, *args, **kwargs)

    def save(self, path, infos=None):
        info = dict(infos or {})
        # Legacy parsing introduces numpy.float64 intervals. A JSON round-trip
        # removes their pickle globals so torch.load(weights_only=True) stays safe.
        def plain(value):
            return json.loads(json.dumps(value, default=lambda scalar: scalar.item()))
        info.update(task=TASK, task_config=plain(class_to_dict(self.env.cfg)))
        info['training_config'] = plain(dict(runner=self.cfg, algorithm=self.alg_cfg, policy=self.policy_cfg))
        counter = self.current_learning_iteration
        try:
            self.current_learning_iteration = getattr(self, '_completed_iterations', counter)
            super().save(path, info)
        finally:
            self.current_learning_iteration = counter
