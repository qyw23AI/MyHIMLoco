"""Meaningful reward counterexamples, resume provenance and TensorBoard contracts."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'isaaclab_migration'))
sys.path.insert(0, str(ROOT / 'rsl_rl'))
import torch
from himloco_lab.config import load_config, class_to_dict, asset_contract
from himloco_lab.task_gait.core import TaskGaitCore
from himloco_lab.task_gait.posture import apply_experiment, PostureStats
from himloco_lab.task_gait.reference import GaitState


class PostureExperimentTests(unittest.TestCase):
    def fixture(self):
        c = TaskGaitCore.__new__(TaskGaitCore)
        c.cfg, self.train = load_config('mybot_v3_task_gait', 2, gait_phase='precision')
        apply_experiment(c.cfg, 'combined')
        c.device, c.num_envs, c.num_dof = 'cpu', 2, 12
        c.dof_names = asset_contract(c.cfg)[1]
        c.hip_indices = [i for i, n in enumerate(c.dof_names) if 'hip' in n]
        c.default_dof_pos = torch.tensor([[c.cfg.init_state.default_joint_angles[n] for n in c.dof_names]])
        c.dof_pos = c.default_dof_pos.repeat(2, 1)
        limits = asset_contract(c.cfg)[2]
        c.dof_pos_limits = torch.tensor([[limits[n]['lower'], limits[n]['upper']] for n in c.dof_names])
        c.root_states = torch.zeros(2, 13)
        c.root_states[:, 2] = c.cfg.rewards.base_height_target
        c.commands = torch.zeros(2, 4)
        c.gait = GaitState(2, c.cfg.gait, 'cpu')
        c.gait.activation[:] = 1
        c.gait.phase[:] = .825
        c.nominal_feet = torch.zeros(4, 3)
        c.gait_feet = torch.arange(4)
        c.rigid_body_states = torch.zeros(2, 4, 13)
        _, lift, _ = c.gait.reference(c.nominal_feet)
        c.rigid_body_states[:, :, 2] = lift + c.cfg.gait.foot_radius
        c.contact_forces = torch.zeros(2, 4, 3)
        c.backend = SimpleNamespace(terrain=None)
        return c

    def test_height_penalty_rejects_low_and_high_pose(self):
        c = self.fixture()
        self.assertEqual(float(c._reward_posture_height().sum()), 0)
        c.root_states[:, 2] = torch.tensor([.18, .48])
        cost = c._reward_posture_height() * abs(c.cfg.rewards.scales.posture_height)
        self.assertTrue((cost > 3.).all())
        torch.testing.assert_close(cost[0], cost[1])

    def test_hip_allows_balance_but_rejects_limit_and_relaxes_turns(self):
        c = self.fixture()
        c.dof_pos[:, c.hip_indices] += .1
        self.assertEqual(float(c._reward_posture_hip().sum()), 0)
        c.dof_pos[:, c.hip_indices] += .4
        straight = c._reward_posture_hip()
        c.commands[:, 2] = .6
        self.assertTrue((c._reward_posture_hip() < straight).all())
        c.dof_pos[:] = c.dof_pos_limits[:, 1]
        self.assertTrue((c._reward_posture_limit() > 0).all())

    def test_clearance_distinguishes_reference_dragging_and_worse_error(self):
        c = self.fixture()
        torch.testing.assert_close(c._reward_posture_clearance(), torch.zeros(2), atol=1e-10, rtol=0)
        c.rigid_body_states[:, :, 2] = .02
        dragging = c._reward_posture_clearance()
        self.assertTrue((dragging > 1.).all())
        c.rigid_body_states[:, :, 2] = -.03
        self.assertTrue((c._reward_posture_clearance() > dragging).all())

    def test_metrics_detect_drag_and_do_not_treat_no_swing_as_success(self):
        c = self.fixture()
        stats = PostureStats(c)
        self.assertIsNone(stats.report()['height_mean_m'])
        self.assertEqual(stats.report()['swing_contact_fraction'], [None] * 4)
        c.rigid_body_states[:, :, 2] = .02
        c.contact_forces[:, :, 2] = 30.
        stats.update(torch.tensor([True, True]))
        p = stats.report()
        self.assertEqual(p['swing_contact_fraction'][0], 1.)
        self.assertEqual(p['apex_clearance_mean_m'][0], 0.)
        self.assertIsNone(p['swing_contact_fraction'][1])

    def test_exact_resume_preserves_arm_and_scales(self):
        from himloco_lab.task_gait.checkpoint import resolve_training_config
        c = self.fixture()
        expected = class_to_dict(c.cfg.rewards)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'model.pt'
            torch.save(dict(infos=dict(task='mybot_v3_task_gait',
                                      task_config=class_to_dict(c.cfg), training_config=self.train)), checkpoint)
            cfg, _ = resolve_training_config(1024, checkpoint=checkpoint)
            self.assertEqual(class_to_dict(cfg.rewards), expected)
            self.assertEqual(cfg.gait.posture.experiment, 'combined')

    def test_baseline_matches_reference_rewards(self):
        cfg, _ = load_config('mybot_v3_task_gait')
        old = class_to_dict(cfg.rewards.scales)
        apply_experiment(cfg, 'baseline')
        new = class_to_dict(cfg.rewards.scales)
        for key, value in old.items():
            self.assertEqual(new[key], value, key)

    def test_speed_variants_only_change_linear_tracking_weight(self):
        reference, _ = load_config('mybot_v3_task_gait')
        apply_experiment(reference, 'combined')
        scales = class_to_dict(reference.rewards.scales)
        for name, weight in [('combined_tracking1', 1.), ('combined_tracking1p5', 1.5)]:
            cfg, _ = load_config('mybot_v3_task_gait')
            apply_experiment(cfg, name)
            expected = dict(scales, tracking_lin_vel=weight)
            self.assertEqual(class_to_dict(cfg.rewards.scales), expected)

    def test_effective_configuration_in_tensorboard(self):
        from himloco_lab.task_gait.run_log import record_run, write_tensorboard_config, write_posture_scalars
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        c = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            args = SimpleNamespace(checkpoint=None, seed=17, iterations=300)
            config = dict(env=class_to_dict(c.cfg), train=self.train,
                          args=dict(posture_experiment='combined'))
            (folder / 'config.json').write_text(json.dumps(config))
            record_run(folder, c.cfg, self.train, args)
            p = PostureStats(c)
            p.update(torch.ones(2, dtype=torch.bool))
            with SummaryWriter(str(folder)) as writer:
                write_tensorboard_config(writer, folder)
                write_posture_scalars(writer, dict(posture=p.report(), foot_names=c.cfg.gait.foot_names), 0)
            events = EventAccumulator(str(folder)).Reload()
            self.assertIn('Run/effective_configuration/text_summary', events.Tags()['tensors'])
            self.assertIn('Run/experiment_and_provenance/text_summary', events.Tags()['tensors'])
            self.assertIn('Posture/height_mean_m', events.Tags()['scalars'])
            self.assertEqual(events.Scalars('Config/reward_scale/posture_height')[0].value, -.25)


if __name__ == '__main__':
    unittest.main()
