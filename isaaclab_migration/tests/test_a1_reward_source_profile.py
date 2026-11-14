"""A1 paper rewards must leave released training behavior intact."""
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from himloco_lab.config import class_to_dict, load_config
from himloco_lab.logic import LegacyLogic
from himloco_lab.paper_a1 import PAPER_A1_REWARD_SCALES, apply_paper_a1


def test_paper_reward_profile_preserves_source_configuration():
    baseline, baseline_train = load_config('a1', 4096, 4)
    cfg, train = load_config('a1', 4096, 4)
    cfg, train = apply_paper_a1(cfg, train, 100)
    actual = class_to_dict(cfg)
    expected = class_to_dict(baseline)
    actual.pop('rewards')
    actual.pop('paper_a1_rewards')
    expected.pop('rewards')
    assert actual == expected
    assert train == baseline_train
    assert {name: getattr(cfg.rewards.scales, name) for name in PAPER_A1_REWARD_SCALES} == PAPER_A1_REWARD_SCALES


def test_terrain_curriculum_uses_released_distance_rule():
    core = SimpleNamespace(
        init_done=True,
        root_states=torch.zeros(3, 13),
        env_origins=torch.zeros(3, 3),
        terrain=SimpleNamespace(env_length=8.0),
        commands=torch.zeros(3, 4),
        max_episode_length_s=20.0,
        terrain_levels=torch.tensor([2, 2, 2]),
        terrain_types=torch.zeros(3, dtype=torch.long),
        max_terrain_level=5,
        terrain_origins=torch.zeros(5, 1, 3),
    )
    core.root_states[:, 0] = torch.tensor([5.0, 0.0, 3.0])
    core.commands[:, 0] = torch.tensor([1.0, 1.0, 0.1])
    LegacyLogic._update_terrain_curriculum(core, torch.arange(3))
    assert core.terrain_levels.tolist() == [3, 1, 2]
