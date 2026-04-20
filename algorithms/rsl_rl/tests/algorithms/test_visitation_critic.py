# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Visitation Critic (CFM) extension and PPO integration."""

from __future__ import annotations

import copy

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import VisitationCritic
from rsl_rl.extensions.visitation_critic import TrajectoryBuffer, resolve_visitation_critic_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
REL_STATE_DIM = 8
NUM_ACTIONS = 4


def _make_actor(obs: TensorDict, obs_groups: dict, num_actions: int = 4, **kwargs: object) -> MLPModel:
    """Create an MLPModel actor with a Gaussian distribution."""
    defaults: dict[str, object] = {
        "hidden_dims": [32, 32],
        "activation": "elu",
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    }
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "actor", num_actions, **defaults)


def _make_critic(obs: TensorDict, obs_groups: dict, **kwargs: object) -> MLPModel:
    """Create an MLPModel critic (no distribution)."""
    defaults: dict[str, object] = {"hidden_dims": [32, 32], "activation": "elu"}
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "critic", 1, **defaults)


def _vc_algorithm_cfg() -> dict:
    """Minimal visitation-critic subsection for tests (tiny CFM inner loop)."""
    return {
        "class_name": "PPO",
        "num_learning_epochs": 1,
        "num_mini_batches": 1,
        "visitation_critic_cfg": {
            "enabled": True,
            "train_every_n_iters": 1,
            "num_warmup_iterations": 0,
            "num_train_steps": 2,
            "warmup_steps": 1,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "label_mode": "l2_ball",
            "l2_radius": 10.0,
            "conditioning_type": "discrete",
            "num_classes": 2,
            "null_label": 2,
            "hidden_dims": [16, 16],
            "class_dim": 4,
            "max_trajectories": 100,
            "num_collect_trajectories": 4,
            "guidance_scale": 1.0,
            "num_euler_steps": 2,
            "cfg_dropout_prob": 0.0,
            "min_scatter_states": 10,
            "generated_states_per_class": 10,
        },
    }


class VCTestDummyEnv(VecEnv):
    """Minimal VecEnv exposing ``policy`` and dedicated ``relative_state`` groups."""

    def __init__(self, device: str = "cpu") -> None:
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = 50
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {
                "policy": torch.randn(self.num_envs, OBS_DIM, device=self.device),
                "relative_state": torch.randn(self.num_envs, REL_STATE_DIM, device=self.device),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        obs = self.get_observations()
        rewards = torch.randn(self.num_envs, device=self.device)
        extras: dict = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return obs, rewards, dones, extras

    def set_reset_states(self, env_ids: torch.Tensor, states: torch.Tensor) -> torch.Tensor | None:  # noqa: ARG002
        return None


def _make_train_cfg_with_vc() -> dict:
    """Return a minimal training configuration with VC enabled (copy-safe for construct_algorithm)."""
    return {
        "num_steps_per_env": NUM_STEPS,
        "save_interval": 100,
        "multi_gpu": None,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
            "relative_state": ["relative_state"],
        },
        "algorithm": _vc_algorithm_cfg(),
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
        },
    }


def _build_ppo_with_vc() -> tuple[PPO, TensorDict, VCTestDummyEnv]:
    """Construct PPO + visitation critic via ``construct_algorithm`` (mirrors training path)."""
    cfg = copy.deepcopy(_make_train_cfg_with_vc())
    env = VCTestDummyEnv(device="cpu")
    obs = env.get_observations()
    ppo = PPO.construct_algorithm(obs, env, cfg, device="cpu")
    return ppo, obs, env


def _seed_vc_buffer(ppo: PPO, num_traj: int = 16, state_dim: int = REL_STATE_DIM) -> None:
    """Populate VC trajectory lists with synthetic data (public buffer fields used at runtime)."""
    vc = ppo.visitation_critic
    assert vc is not None
    buf = vc.buffer
    for _ in range(num_traj):
        buf.start_states.append(torch.randn(state_dim))
        buf.end_states.append(torch.randn(state_dim) * 0.01)
        buf.cumulative_rewards.append(0.0)
        buf.trajectory_lengths.append(3)


class TestTrajectoryBuffer:
    """Episode segmentation rules used for CFM labels."""

    def test_done_uses_previous_step_obs_as_end_state(self) -> None:
        """On terminal step, recorded end state is ``_last_obs``, not the post-reset observation."""
        buffer = TrajectoryBuffer(max_trajectories=100, state_dim=4, device="cpu")
        obs_a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        obs_b = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
        obs_c = torch.tensor([[99.0, 99.0, 99.0, 99.0]])
        r = torch.zeros(1)
        buffer.add_step(obs_a, r, torch.zeros(1))
        buffer.add_step(obs_b, r, torch.zeros(1))
        buffer.add_step(obs_c, r, torch.ones(1))

        assert buffer.num_trajectories == 1
        assert torch.allclose(buffer.start_states[0], obs_a.squeeze(0))
        assert torch.allclose(buffer.end_states[0], obs_b.squeeze(0))


class TestVisitationCriticLabeling:
    """``l2_ball`` label assignment on end-state norms."""

    def test_l2_ball_labels_end_state_norm(self) -> None:
        vc_cfg = {
            "label_mode": "l2_ball",
            "l2_radius": 2.0,
            "conditioning_type": "discrete",
            "num_classes": 2,
            "train_every_n_iters": 1,
            "num_warmup_iterations": 0,
            "num_train_steps": 1,
            "batch_size": 4,
        }
        vc = VisitationCritic(
            vc_cfg,
            state_dim=4,
            obs_groups={"relative_state": ["relative_state"]},
            device="cpu",
        )
        data = {
            "start_states": torch.zeros(3, 4),
            "end_states": torch.tensor(
                [
                    [3.0, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "cumulative_rewards": torch.zeros(3),
        }
        _starts, labels = vc._label_trajectories(data)
        assert labels.tolist() == [0, 1, 1]


class TestVisitationCriticSchedule:
    """``should_collect`` / ``should_train`` gating."""

    def test_should_train_requires_buffer_and_schedule(self) -> None:
        vc_cfg = {
            "label_mode": "l2_ball",
            "conditioning_type": "discrete",
            "num_classes": 2,
            "train_every_n_iters": 5,
            "num_warmup_iterations": 10,
            "num_train_steps": 1,
            "batch_size": 4,
        }
        vc = VisitationCritic(
            vc_cfg,
            state_dim=4,
            obs_groups={"relative_state": ["relative_state"]},
            device="cpu",
        )
        assert not vc.should_train(10)
        vc.buffer.start_states.append(torch.randn(4))
        vc.buffer.end_states.append(torch.randn(4))
        vc.buffer.cumulative_rewards.append(0.0)
        vc.buffer.trajectory_lengths.append(2)
        assert vc.should_train(10)
        assert not vc.should_train(9)


class TestPPOWithVisitationCritic:
    """PPO.update triggers CFM training when ``should_train`` is true."""

    def test_construct_algorithm_creates_visitation_critic(self) -> None:
        ppo, _obs, _env = _build_ppo_with_vc()
        assert ppo.visitation_critic is not None
        assert ppo.visitation_critic.state_dim == REL_STATE_DIM

    def test_update_runs_cfm_when_buffer_populated(self) -> None:
        ppo, obs, _env = _build_ppo_with_vc()
        _seed_vc_buffer(ppo, num_traj=16)

        for _ in range(NUM_STEPS):
            ppo.act(obs)
            stored_values = ppo.transition.values.clone()
            raw_reward = torch.ones(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS)
            time_outs = torch.zeros(NUM_ENVS)
            ppo.process_env_step(obs, raw_reward, dones, {"time_outs": time_outs})

        ppo.compute_returns(obs)
        loss_dict = ppo.update(iteration=0)

        assert "visitation_critic/cfm_loss" in loss_dict
        assert ppo.visitation_critic is not None
        assert ppo.visitation_critic.is_trained


class TestResolveVisitationCriticConfig:
    """Guardrails for observation groups."""

    def test_rejects_policy_fallback_for_relative_state(self) -> None:
        obs = make_obs(NUM_ENVS, OBS_DIM)
        bad_groups = {"relative_state": ["policy"]}
        vc_cfg = {"enabled": True}
        with pytest.raises(ValueError, match="relative_state"):
            resolve_visitation_critic_config(vc_cfg, obs, bad_groups, VCTestDummyEnv())
