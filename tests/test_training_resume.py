from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.training_resume import prepare_training_iterations


def test_resume_preserves_adam_lr_and_finishes_at_requested_total():
    parameter = torch.nn.Parameter(torch.ones(1))
    original = torch.optim.Adam([parameter], lr=7.59375e-5)
    parameter.sum().backward()
    original.step()
    restored = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=.001)
    restored.load_state_dict(original.state_dict())
    runner = NS(current_learning_iteration=2000,
                alg=NS(optimizer=restored, learning_rate=.001))
    remaining = prepare_training_iterations(runner, 20000, resumed=True)
    assert runner.alg.learning_rate == 7.59375e-5
    assert runner.current_learning_iteration == 2001
    assert remaining == 17999
    assert list(range(runner.current_learning_iteration,
                      runner.current_learning_iteration + remaining))[-1] == 19999
    assert next(iter(restored.state.values()))['step'] == 1


def test_fresh_run_budget_is_unchanged():
    runner = NS(current_learning_iteration=0)
    assert prepare_training_iterations(runner, 20000, resumed=False) == 20000
    assert runner.current_learning_iteration == 0


def test_completed_run_is_not_silently_extended():
    runner = NS(current_learning_iteration=19999)
    with pytest.raises(ValueError, match='already completed'):
        prepare_training_iterations(runner, 20000, resumed=True)
    assert runner.current_learning_iteration == 19999
