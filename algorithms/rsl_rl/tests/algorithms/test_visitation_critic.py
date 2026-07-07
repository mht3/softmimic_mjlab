# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the new state-action VisitationCritic (SB3 VCPPO port)."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.extensions.visitation_critic import (
    AFlow,
    SAFlow,
    VisitationCritic,
    _local_topmcq_mask,
    resolve_alpha_schedule,
)


STATE_DIM = 4
ACT_DIM = 2
NUM_ENVS = 3
DEVICE = "cpu"


def _make_obs_td(state: torch.Tensor) -> TensorDict:
    return TensorDict({"policy": state}, batch_size=[state.shape[0]])


def _base_cfg(**overrides) -> dict:
    cfg = dict(
        sample_method="sdedit",
        alpha=0.5,
        warmup_iters=0,
        num_samples=8,
        tau=0.9,
        policy_trust_std=100.0,
        model_train_steps=10,
        model_train_every=1,
        model_batch_size=16,
        model_lambda_steps=4,
        model_net=(16, 16),
        buffer_size=512,
        q_top_fraction=0.5,
        q_filter_k=4,
        q_mode="off",
        gamma_mcq=0.99,
        seed=0,
    )
    cfg.update(overrides)
    return cfg


def _make_vc(cfg: dict | None = None) -> VisitationCritic:
    return VisitationCritic(
        cfg=cfg or _base_cfg(),
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        num_envs=NUM_ENVS,
        device=DEVICE,
    )


def test_alpha_schedule_constant():
    sched = resolve_alpha_schedule(0.3)
    assert sched(0) == pytest.approx(0.3)
    assert sched(10000) == pytest.approx(0.3)


def test_alpha_schedule_linear_decay():
    sched = resolve_alpha_schedule({"alpha": 1.0, "mode": "linear_decay", "stop_iter": 100})
    assert sched(0) == pytest.approx(1.0)
    assert sched(50) == pytest.approx(0.5)
    assert sched(100) == 0.0
    assert sched(200) == 0.0


def test_vc_alpha_envelope_holds_constant_after_warmup_by_default():
    """Default behavior: linear ramp during warmup, then HOLD at base_alpha forever.
    """
    cfg = _base_cfg(alpha=0.6, warmup_iters=100, stop_iter=None)
    vc = _make_vc(cfg)
    # Ramp from 0 to base_alpha.
    assert vc.alpha(0) == pytest.approx(0.0)
    assert vc.alpha(50) == pytest.approx(0.3)
    assert vc.alpha(99) == pytest.approx(0.6 * 0.99)
    # After warmup: hold at base_alpha forever.
    assert vc.alpha(100) == pytest.approx(0.6)
    assert vc.alpha(1_000) == pytest.approx(0.6)
    assert vc.alpha(100_000) == pytest.approx(0.6)


def test_vc_alpha_envelope_warmup_and_decay():
    cfg = _base_cfg(alpha=1.0, warmup_iters=10, stop_iter=100, decay_start_iter=50)
    vc = _make_vc(cfg)
    assert vc.alpha(0) == pytest.approx(0.0)
    assert vc.alpha(5) == pytest.approx(0.5)
    assert vc.alpha(10) == pytest.approx(1.0)
    assert vc.alpha(50) == pytest.approx(1.0)
    assert vc.alpha(75) == pytest.approx(0.5)
    assert vc.alpha(100) == pytest.approx(0.0)


def test_local_topmcq_mask_keeps_top_percentile():
    # 6 points laid out on a line; high MCQ rows should survive a 50% top filter.
    states = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    mcq = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    mask = _local_topmcq_mask(states, mcq, states, mcq, k_nn=3, keep_frac=0.5)
    # The highest-MCQ rows in each local neighborhood must survive.
    assert bool(mask[-1]) is True
    assert bool(mask[0]) is False


def test_saflow_buffer_admits_and_trains():
    flow = SAFlow(state_dim=STATE_DIM, act_dim=ACT_DIM, device=torch.device(DEVICE), hidden_sizes=(16, 16))
    rows = torch.randn(64, STATE_DIM + ACT_DIM)
    mcq = torch.randn(64)
    flow.add(rows, mcq)
    assert flow.buffer_size == 64
    loss = flow.train_steps(n_steps=5, batch_size=16)
    assert loss is not None and loss > 0.0
    cands = flow.sample_sdedit(
        torch.randn(2, STATE_DIM),
        torch.randn(2, ACT_DIM),
        tau=0.9,
        num_samples=4,
        n_steps=3,
    )
    assert cands.shape == (2, 4, STATE_DIM + ACT_DIM)


def test_saflow_buffer_eviction_caps_at_max():
    flow = SAFlow(state_dim=2, act_dim=1, device=torch.device(DEVICE), max_buffer=32, hidden_sizes=(8,))
    # Push many rows; eviction filters by MCQ when capacity exceeded.
    for _ in range(20):
        rows = torch.randn(8, 3)
        mcq = torch.randn(8)
        flow.add(rows, mcq)
    assert flow.buffer_size <= 32


def test_vc_action_falls_back_to_ppo_before_training():
    vc = _make_vc()
    state = torch.randn(NUM_ENVS, STATE_DIM)
    obs_td = _make_obs_td(state)
    a_ppo = torch.randn(NUM_ENVS, ACT_DIM)
    a_mean = torch.zeros(NUM_ENVS, ACT_DIM)
    a_std = torch.ones(NUM_ENVS, ACT_DIM)
    out = vc.vc_action(obs_td, a_ppo, a_mean, a_std)
    assert torch.allclose(out, a_ppo)


def test_record_step_pushes_completed_episodes_into_buffer():
    vc = _make_vc()
    # Roll for 5 steps with done on the 5th to flush an episode for env 0.
    obs_t = torch.randn(NUM_ENVS, STATE_DIM)
    for t in range(5):
        action = torch.randn(NUM_ENVS, ACT_DIM)
        reward = torch.randn(NUM_ENVS)
        next_obs_t = torch.randn(NUM_ENVS, STATE_DIM)
        done = torch.zeros(NUM_ENVS)
        if t == 4:
            done[0] = 1.0
        vc.record_step(
            _make_obs_td(obs_t),
            action,
            reward,
            _make_obs_td(next_obs_t),
            done,
        )
        obs_t = next_obs_t
    # 5 transitions from env 0 should now sit in the visitation buffer.
    assert vc.flow.buffer_size == 5


def test_vc_action_sdedit_returns_blendable_shape_after_training():
    vc = _make_vc()
    # Force-feed the buffer so the flow has data to train on.
    rows = torch.randn(128, STATE_DIM + ACT_DIM)
    mcq = torch.randn(128)
    vc.flow.add(rows, mcq)
    vc.flow.train_steps(n_steps=5, batch_size=16)
    assert vc.is_ready()
    state = torch.randn(NUM_ENVS, STATE_DIM)
    a_ppo = torch.randn(NUM_ENVS, ACT_DIM)
    a_mean = torch.zeros(NUM_ENVS, ACT_DIM)
    a_std = torch.ones(NUM_ENVS, ACT_DIM) * 10.0  # wide trust band
    out = vc.vc_action(_make_obs_td(state), a_ppo, a_mean, a_std)
    assert out.shape == a_ppo.shape


def test_vc_action_cfg_path_runs_with_aflow():
    cfg = _base_cfg(sample_method="cfg", num_samples=6)
    vc = _make_vc(cfg)
    assert isinstance(vc.flow, AFlow)
    rows = torch.randn(64, STATE_DIM + ACT_DIM)
    mcq = torch.randn(64)
    vc.flow.add(rows, mcq)
    vc.flow.train_steps(n_steps=3, batch_size=8)
    state = torch.randn(NUM_ENVS, STATE_DIM)
    out = vc.vc_action(_make_obs_td(state), torch.randn(NUM_ENVS, ACT_DIM), torch.zeros(NUM_ENVS, ACT_DIM), torch.ones(NUM_ENVS, ACT_DIM) * 10.0)
    assert out.shape == (NUM_ENVS, ACT_DIM)


def test_vc_action_inpainting_falls_back_to_a_ppo_outside_trust_band():
    # policy_trust_std=0 makes the trust band an empty set unless the candidate
    # exactly equals the mean — practically: no candidate is in trust → use a_ppo.
    cfg = _base_cfg(sample_method="inpainting", num_samples=8, policy_trust_std=0.0)
    vc = _make_vc(cfg)
    rows = torch.randn(64, STATE_DIM + ACT_DIM)
    mcq = torch.randn(64)
    vc.flow.add(rows, mcq)
    vc.flow.train_steps(n_steps=3, batch_size=8)
    state = torch.randn(NUM_ENVS, STATE_DIM)
    a_ppo = torch.randn(NUM_ENVS, ACT_DIM)
    a_mean = torch.zeros(NUM_ENVS, ACT_DIM)
    a_std = torch.ones(NUM_ENVS, ACT_DIM)
    out = vc.vc_action(_make_obs_td(state), a_ppo, a_mean, a_std)
    assert torch.allclose(out, a_ppo)
    assert vc.last_vc_active_frac == 0.0


def test_vc_action_cfg_falls_back_to_a_ppo_outside_trust_band():
    cfg = _base_cfg(sample_method="cfg", num_samples=8, policy_trust_std=0.0)
    vc = _make_vc(cfg)
    rows = torch.randn(64, STATE_DIM + ACT_DIM)
    mcq = torch.randn(64)
    vc.flow.add(rows, mcq)
    vc.flow.train_steps(n_steps=3, batch_size=8)
    state = torch.randn(NUM_ENVS, STATE_DIM)
    a_ppo = torch.randn(NUM_ENVS, ACT_DIM)
    a_mean = torch.zeros(NUM_ENVS, ACT_DIM)
    a_std = torch.ones(NUM_ENVS, ACT_DIM)
    out = vc.vc_action(_make_obs_td(state), a_ppo, a_mean, a_std)
    assert torch.allclose(out, a_ppo)


def test_vc_action_sdedit_falls_back_to_a_ppo_outside_trust_band():
    cfg = _base_cfg(sample_method="sdedit", num_samples=8, policy_trust_std=0.0)
    vc = _make_vc(cfg)
    rows = torch.randn(64, STATE_DIM + ACT_DIM)
    mcq = torch.randn(64)
    vc.flow.add(rows, mcq)
    vc.flow.train_steps(n_steps=3, batch_size=8)
    state = torch.randn(NUM_ENVS, STATE_DIM)
    a_ppo = torch.randn(NUM_ENVS, ACT_DIM)
    a_mean = torch.zeros(NUM_ENVS, ACT_DIM)
    a_std = torch.ones(NUM_ENVS, ACT_DIM)
    out = vc.vc_action(_make_obs_td(state), a_ppo, a_mean, a_std)
    assert torch.allclose(out, a_ppo)


def test_vc_train_returns_loss_dict_with_buffer_metrics():
    vc = _make_vc()
    rows = torch.randn(64, STATE_DIM + ACT_DIM)
    mcq = torch.randn(64)
    vc.flow.add(rows, mcq)
    losses = vc.train()
    assert "vc/flow_loss" in losses
    assert "vc/buffer_size" in losses
    assert losses["vc/buffer_size"] == pytest.approx(64.0)
