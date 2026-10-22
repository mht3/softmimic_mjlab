from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tasks.compliant_tracking.rl.health_metrics import ModeHealthMetrics


def test_step_reward_includes_unfinished_envs_and_initial_completions_are_censored():
    metrics = ModeHealthMetrics(['forcefield', 'forcefield', 'zero-wrench'], 'cpu')
    rewards = torch.tensor([1., 3., 8.])
    errors = torch.tensor([.1, .3, .01])
    original = rewards.clone()
    metrics.update(rewards, torch.tensor([1, 1, 1]), torch.ones(3, dtype=torch.bool), errors)
    first = metrics.flush()
    assert first['Compliance/forcefield/reward_per_step'] == 2.
    assert first['Compliance/zero-wrench/reward_per_step'] == 8.
    assert first['Compliance/forcefield/full_episodes_since_resume'] == 0
    metrics.update(rewards, torch.tensor([1, 1, 0]), torch.tensor([1, 0, 0]), errors)
    metrics.update(rewards, torch.tensor([0, 0, 1]), torch.tensor([0, 0, 1]), errors)
    values = metrics.flush()
    assert values['Compliance/forcefield/full_episodes_since_resume'] == 2
    assert values['Compliance/forcefield/timeout_fraction_since_resume'] == .5
    assert values['Compliance/zero-wrench/timeout_fraction_since_resume'] == 1.
    torch.testing.assert_close(rewards, original)
    assert metrics.flush() == {}
