"""CPU contracts for small commands, stop/turn behavior, supervision and export."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'isaaclab_migration'))
sys.path.insert(0, str(ROOT / 'rsl_rl'))
from himloco_lab.config import load_config, class_to_dict, asset_contract
from himloco_lab.task_gait.config import restore_config, set_ablation
from himloco_lab.task_gait.core import TaskGaitCore
from himloco_lab.task_gait.geometry import nominal_feet
from himloco_lab.task_gait.reference import GaitState, sample_commands


class TaskGaitTests(unittest.TestCase):
    def test_profiles_are_isolated_and_round_trip(self):
        legacy, _ = load_config('mybot_v3')
        before = class_to_dict(legacy)
        for phase in ('basic', 'precision', 'robust'):
            cfg, _ = load_config('mybot_v3_task_gait', 8, gait_phase=phase)
            self.assertEqual((cfg.env.num_observations, cfg.env.num_privileged_obs), (294, 242))
            self.assertFalse(cfg.commands.heading_command)
            self.assertEqual(class_to_dict(restore_config(class_to_dict(cfg))), class_to_dict(cfg))
            cfg.control.stiffness['FL_hip'] = 12
        self.assertEqual(class_to_dict(load_config('mybot_v3')[0]), before)

    def test_sampling_preserves_low_commands_pure_yaw_and_ranges(self):
        torch.manual_seed(42)
        cfg, _ = load_config('mybot_v3_task_gait', gait_phase='precision')
        target, standby, group = sample_commands(12000, cfg.gait, cfg.commands.ranges, 'cpu')
        self.assertTrue((target[group < 2] == 0).all())
        self.assertTrue((standby[group == 1] == 1).all())
        low = target[group == 2, :2].norm(dim=-1)
        self.assertTrue((low >= .015 - 1e-6).all())
        self.assertTrue((low <= .10 + 1e-6).all())
        self.assertTrue((target[group == 3, :2] == 0).all())
        self.assertTrue((target[group == 3, 2].abs() >= .03).all())
        self.assertTrue((target.abs() <= torch.tensor([.5, .2, .6])).all())

    def test_tiny_motion_yaw_standby_and_stop(self):
        cfg, _ = load_config('mybot_v3_task_gait')
        state = GaitState(4, cfg.gait, 'cpu')
        state.target[:] = torch.tensor([[.001, 0, 0], [0, 0, .003], [0, 0, 0], [0, 0, 0]])
        state.standby[2] = 1
        for _ in range(100):
            state.advance(.02)
        self.assertTrue((state.activation[:3] > .99).all())
        self.assertEqual(float(state.activation[3]), 0)
        self.assertGreater(float(state.command[0, 0]), .00099)
        state.target.zero_()
        state.standby.zero_()
        for _ in range(200):
            state.advance(.02)
        self.assertTrue((state.activation == 0).all())
        phase = state.phase.clone()
        state.advance(.02)
        torch.testing.assert_close(state.phase, phase)
        self.assertTrue((state.command == 0).all())

    def test_sequence_contains_adjust_stop_turn_and_mode_changes(self):
        c = self.fixture()
        c.external_commands = False
        self.assertGreater(c.cfg.env.episode_length_s, 6 * c.cfg.commands.resampling_time)
        c.dt = .02
        c.episode_length_buf = torch.zeros(4, dtype=torch.long)
        ids = torch.arange(4)
        for segment, expected, mode in [(0, [.2, 0, 0], 0), (1, [.03, 0, 0], 0),
                                        (2, [0, 0, 0], 0), (3, [0, 0, .1], 0),
                                        (4, [0, 0, 0], 1), (5, [0, 0, 0], 0)]:
            c.episode_length_buf[:] = segment * 250
            c._resample_commands(ids)
            torch.testing.assert_close(c.gait.target[0], torch.tensor(expected, dtype=torch.float))
            self.assertEqual(float(c.gait.standby[0]), mode)

    def test_reference_turn_sign_continuity_and_swing_height(self):
        cfg, _ = load_config('mybot_v3_task_gait')
        nominal = torch.tensor(nominal_feet(asset_contract(cfg)[0], cfg.init_state.default_joint_angles, cfg.gait.foot_names), dtype=torch.float)
        self.assertTrue((nominal[[0, 2], 1] > 0).all())
        self.assertTrue((nominal[[1, 3], 1] < 0).all())
        state = GaitState(1, cfg.gait, 'cpu')
        state.activation[:] = 1
        state.command[0, 2] = .1
        state.phase[:] = .1
        xy, height, contact = state.reference(nominal)
        self.assertLess(float(xy[0, 0, 0] - nominal[0, 0]), 0)
        self.assertGreater(float(xy[0, 3, 0] - nominal[3, 0]), 0)
        for boundary in (0., cfg.gait.duty_factor):
            state.phase[:] = (boundary - 1e-5) % 1
            a = state.reference(nominal)
            state.phase[:] = (boundary + 1e-5) % 1
            b = state.reference(nominal)
            for x, y in zip(a, b):
                self.assertLess(float((x - y).abs().max()), .002)
        state.phase[:] = (1 + cfg.gait.duty_factor) / 2
        _, height, _ = state.reference(nominal)
        self.assertAlmostEqual(float(height[0, 0]), cfg.gait.swing_height, places=6)

    def fixture(self):
        c = TaskGaitCore.__new__(TaskGaitCore)
        c.cfg, _ = load_config('mybot_v3_task_gait', 4)
        c.device = 'cpu'
        c.num_envs, c.num_actions, c.num_dof = 4, 12, 12
        c.num_one_step_obs, c.num_obs, c.num_privileged_obs = 49, 294, 242
        c.history_length = 6
        c.commands = torch.zeros(4, 4)
        c.commands_scale = torch.tensor([2., 2., .25])
        c.base_ang_vel = torch.zeros(4, 3)
        c.base_lin_vel = torch.randn(4, 3)
        c.projected_gravity = torch.tensor([[0., 0., -1.]]).repeat(4, 1)
        c.dof_pos = torch.zeros(4, 12)
        c.dof_vel = torch.zeros(4, 12)
        c.default_dof_pos = torch.zeros(1, 12)
        c.actions = torch.zeros(4, 12)
        c.obs_scales = c.cfg.normalization.obs_scales
        c.root_states = torch.zeros(4, 13)
        c.measured_heights = torch.zeros(4, 187)
        c.disturbance = torch.zeros(4, 1, 3)
        c.add_noise = False
        c.gait = GaitState(4, c.cfg.gait, 'cpu')
        c._cached_frame = None
        c.history_fresh = torch.tensor([True, False, False, False])
        c.obs_buf = torch.randn(4, 294)
        c.privileged_obs_buf = torch.randn(4, 242)
        return c

    def test_observation_history_and_him_supervision(self):
        from rsl_rl.modules.him_estimator import HIMEstimator
        c = self.fixture()
        before = c.obs_buf.clone()
        terminal = c.compute_termination_observations(torch.tensor([2]))
        c.compute_observations()
        torch.testing.assert_close(c.obs_buf[1:, 49:], before[1:, :-49])
        torch.testing.assert_close(c.obs_buf[0], c.obs_buf[0, :49].repeat(6))
        torch.testing.assert_close(terminal[0], c.privileged_obs_buf[2])
        torch.testing.assert_close(c.privileged_obs_buf[:, 49:52], c.base_lin_vel * 2)
        model = HIMEstimator(6, 49)
        captured = []
        handle = model.target.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
        losses = model.update(c.obs_buf, c.privileged_obs_buf)
        handle.remove()
        torch.testing.assert_close(captured[0], c.privileged_obs_buf[:, 3:52])
        self.assertTrue(all(np.isfinite(float(v)) for v in losses))

    def test_external_commands_do_not_add_history_or_hide_small_yaw(self):
        c = self.fixture()
        history = c.obs_buf.clone()
        c.set_command([0., 0., .001])
        self.assertAlmostEqual(float(c.gait.target[0, 2]), .001)
        torch.testing.assert_close(c.obs_buf, history)
        with self.assertRaisesRegex(ValueError, 'zero body-velocity'):
            c.set_command([.1, 0., 0.], standby=True)
        with self.assertRaisesRegex(ValueError, 'outside trained range'):
            c.set_command([1., 0., 0.])
        c.set_command([0., 0., 0.], standby=True)
        c.prepare_external_observation()
        torch.testing.assert_close(c.obs_buf[:, 49:], history[:, 49:])
        self.assertTrue((c.obs_buf[:, 48] == 1).all())

    def test_load_penalty_detects_stationary_support(self):
        c = self.fixture()
        c.torques = torch.full((4, 12), 5.)
        c.torque_limits = torch.full((12,), 33.5)
        c._control_torque_square = c.torques.square() * 4
        c._control_torque_samples = 4
        self.assertEqual(float(c._reward_joint_power().max()), 0)
        self.assertTrue((c._reward_gait_load() > 0).all())
        c.commands[:, 0] = .03
        stopped = c._reward_tracking_lin_vel().clone()
        c.base_lin_vel[:, :2] = c.commands[:, :2]
        self.assertTrue((c._reward_tracking_lin_vel() > stopped).all())

    def test_checkpoint_export_and_resume_contract(self):
        from rsl_rl.modules import HIMActorCritic
        from himloco_lab.export import export_policy
        from himloco_lab.task_gait.checkpoint import saved_task_config, config_from_policy, resolve_training_config
        cfg, train = load_config('mybot_v3_task_gait', 4, gait_phase='precision')
        model = HIMActorCritic(294, 242, 49, 12, **train['policy'])
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'model.pt'
            torch.save(dict(infos=dict(task='mybot_v3_task_gait', task_config=class_to_dict(cfg))), checkpoint)
            export_policy(model, Path(folder) / 'policy.pt', cfg, checkpoint)
            restored, metadata = config_from_policy(Path(folder) / 'policy.pt', 8)
            self.assertEqual(restored.env.num_envs, 8)
            self.assertEqual(metadata['input_shape'][-1], 294)
            cfg2, _ = resolve_training_config(8, 'robust', checkpoint, 'timing')
            self.assertEqual(cfg2.gait.phase, 'robust')
            self.assertEqual(cfg2.rewards.scales.gait_position, 0)
            set_ablation(cfg2, 'full')
            self.assertLess(cfg2.rewards.scales.gait_position, 0)
            torch.save(dict(infos=None), checkpoint)
            with self.assertRaisesRegex(ValueError, 'not a task-gait checkpoint'):
                saved_task_config(checkpoint)

    def test_runner_saves_numpy_runtime_values_as_safe_primitives(self):
        from himloco_lab.task_gait.runner import TaskGaitRunner
        cfg, train = load_config('mybot_v3_task_gait', 4)
        cfg.domain_rand.push_interval = np.float64(100.)
        model = torch.nn.Linear(3, 2)
        model.estimator = SimpleNamespace(optimizer=torch.optim.Adam(model.parameters()))
        runner = TaskGaitRunner.__new__(TaskGaitRunner)
        runner.env = SimpleNamespace(cfg=cfg)
        runner.cfg, runner.alg_cfg, runner.policy_cfg = train['runner'], train['algorithm'], train['policy']
        runner.alg = SimpleNamespace(actor_critic=model, optimizer=torch.optim.Adam(model.parameters()))
        runner.current_learning_iteration = 2
        runner._completed_iterations = 3
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'safe.pt'
            runner.save(path)
            data = torch.load(path, weights_only=True)
            self.assertIsInstance(data['infos']['task_config']['domain_rand']['push_interval'], float)
            self.assertEqual(data['iter'], 3)
            self.assertEqual(runner.current_learning_iteration, 2)

    def test_metrics_do_not_claim_perfect_tracking_without_samples(self):
        from himloco_lab.task_gait.metrics import TaskGaitMonitor
        c = self.fixture()
        c.dt = .02
        c.torques = torch.zeros(4, 12)
        c._torque_square = torch.zeros_like(c.torques)
        c._torque_peak = torch.zeros_like(c.torques)
        c._torque_saturation = torch.zeros_like(c.torques)
        c.dof_names = asset_contract(c.cfg)[1]
        with tempfile.TemporaryDirectory() as folder:
            monitor = TaskGaitMonitor(SimpleNamespace(core=c), folder)
            report = monitor.report()
            self.assertIsNone(report['processed_command_mae'])
            self.assertEqual(report['contact_period_cv'], [None] * 4)


if __name__ == '__main__':
    unittest.main()
