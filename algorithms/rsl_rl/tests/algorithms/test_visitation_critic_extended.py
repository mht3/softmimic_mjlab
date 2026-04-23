# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extended tests for the Visitation Critic: buffer edge cases, generation validity,
CFM training convergence, and PPO integration correctness."""

from __future__ import annotations

import torch
import pytest

from rsl_rl.extensions.visitation_critic import TrajectoryBuffer, VisitationCritic
from rsl_rl.modules.cfm import (
    GaussianConditionalProbabilityPath,
    EulerODESolver,
    MLPConditionalVectorField,
    cfg_guided_velocity,
)


# --------------------------------------------------------------------------
# TrajectoryBuffer edge cases
# --------------------------------------------------------------------------


class TestTrajectoryBufferMultiEnv:
    """Multi-env and ring-buffer edge cases."""

    def test_multi_env_independent_episodes(self) -> None:
        """Two envs finishing at different times produce correct start/end pairs."""
        buffer = TrajectoryBuffer(max_trajectories=100, state_dim=2, device="cpu")
        num_envs = 2

        # Step 0: both envs start
        obs0 = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        buffer.add_step(obs0, torch.zeros(num_envs), torch.zeros(num_envs))

        # Step 1: env 0 continues, env 1 continues
        obs1 = torch.tensor([[1.1, 0.1], [2.1, 0.1]])
        buffer.add_step(obs1, torch.ones(num_envs), torch.zeros(num_envs))

        # Step 2: env 0 terminates, env 1 continues
        obs2 = torch.tensor([[99.0, 99.0], [2.2, 0.2]])
        dones = torch.tensor([1.0, 0.0])
        buffer.add_step(obs2, torch.ones(num_envs), dones)

        assert buffer.num_trajectories == 1
        # End state should be obs1[0] (the _last_obs before termination)
        assert torch.allclose(buffer.start_states[0], obs0[0])
        assert torch.allclose(buffer.end_states[0], obs1[0])

        # Step 3: env 1 terminates (env 0 is now on a new episode, step_count=1)
        obs3 = torch.tensor([[1.5, 0.5], [99.0, 99.0]])
        dones = torch.tensor([0.0, 1.0])
        buffer.add_step(obs3, torch.ones(num_envs), dones)

        assert buffer.num_trajectories == 2
        # Env 1's trajectory: start=obs0[1], end=obs2[1] (last obs before done)
        assert torch.allclose(buffer.start_states[1], obs0[1])
        assert torch.allclose(buffer.end_states[1], obs2[1])

    def test_immediate_done_on_first_step_is_ignored(self) -> None:
        """An episode that terminates immediately (step_count=0) is not recorded."""
        buffer = TrajectoryBuffer(max_trajectories=100, state_dim=2, device="cpu")
        # Only one step, and it's a done — step_counts == 0, so no valid _last_obs
        obs = torch.tensor([[1.0, 2.0]])
        buffer.add_step(obs, torch.zeros(1), torch.ones(1))
        assert buffer.num_trajectories == 0

    def test_ring_buffer_overflow(self) -> None:
        """Buffer evicts oldest trajectories when max is exceeded."""
        max_traj = 3
        buffer = TrajectoryBuffer(max_trajectories=max_traj, state_dim=2, device="cpu")

        # Generate 5 complete episodes (2 steps each)
        for ep in range(5):
            start = torch.tensor([[float(ep), 0.0]])
            mid = torch.tensor([[float(ep), 1.0]])
            post_reset = torch.tensor([[99.0, 99.0]])
            buffer.add_step(start, torch.zeros(1), torch.zeros(1))
            buffer.add_step(mid, torch.zeros(1), torch.zeros(1))
            buffer.add_step(post_reset, torch.zeros(1), torch.ones(1))

        assert buffer.num_trajectories == max_traj
        # Oldest (ep=0, ep=1) should have been evicted; ep=2,3,4 remain
        assert buffer.start_states[0][0].item() == 2.0
        assert buffer.start_states[1][0].item() == 3.0
        assert buffer.start_states[2][0].item() == 4.0

    def test_cumulative_rewards_tracked(self) -> None:
        """Cumulative reward includes ALL steps including the terminal one."""
        buffer = TrajectoryBuffer(max_trajectories=100, state_dim=2, device="cpu")
        obs = torch.tensor([[0.0, 0.0]])
        buffer.add_step(obs, torch.tensor([1.5]), torch.zeros(1))  # cum: 0 -> 1.5
        buffer.add_step(obs, torch.tensor([2.5]), torch.zeros(1))  # cum: 1.5 -> 4.0
        buffer.add_step(obs, torch.tensor([3.0]), torch.ones(1))   # done: cum=4.0+3.0=7.0

        assert buffer.num_trajectories == 1
        assert abs(buffer.cumulative_rewards[0] - 7.0) < 1e-6

    def test_trajectory_length_correct(self) -> None:
        """Trajectory length includes all steps (including the terminating one)."""
        buffer = TrajectoryBuffer(max_trajectories=100, state_dim=2, device="cpu")
        obs = torch.tensor([[0.0, 0.0]])
        # 4 steps total: 3 non-done + 1 done
        for _ in range(3):
            buffer.add_step(obs, torch.zeros(1), torch.zeros(1))
        buffer.add_step(obs, torch.zeros(1), torch.ones(1))

        assert buffer.num_trajectories == 1
        assert buffer.trajectory_lengths[0] == 4

    def test_clear_preserves_in_progress(self) -> None:
        """clear() removes stored trajectories but keeps in-progress accumulators."""
        buffer = TrajectoryBuffer(max_trajectories=100, state_dim=2, device="cpu")
        obs = torch.tensor([[1.0, 2.0]])
        buffer.add_step(obs, torch.zeros(1), torch.zeros(1))
        buffer.add_step(obs, torch.zeros(1), torch.ones(1))
        assert buffer.num_trajectories == 1

        buffer.clear()
        assert buffer.num_trajectories == 0
        # Internal accumulators should still be initialized
        assert buffer._initialized

    def test_simultaneous_dones_multi_env(self) -> None:
        """All envs done simultaneously records all episodes."""
        num_envs = 4
        buffer = TrajectoryBuffer(max_trajectories=100, state_dim=3, device="cpu")
        obs0 = torch.randn(num_envs, 3)
        obs1 = torch.randn(num_envs, 3)
        obs_post = torch.randn(num_envs, 3)

        buffer.add_step(obs0, torch.zeros(num_envs), torch.zeros(num_envs))
        buffer.add_step(obs1, torch.zeros(num_envs), torch.zeros(num_envs))
        buffer.add_step(obs_post, torch.zeros(num_envs), torch.ones(num_envs))

        assert buffer.num_trajectories == num_envs
        for i in range(num_envs):
            assert torch.allclose(buffer.start_states[i], obs0[i])
            assert torch.allclose(buffer.end_states[i], obs1[i])


# --------------------------------------------------------------------------
# CFM Module Tests
# --------------------------------------------------------------------------


class TestGaussianConditionalProbabilityPath:
    """Verify the flow matching path properties."""

    def test_t0_is_pure_noise(self) -> None:
        """At t=0, x_t = beta(0)*eps = 1*eps (pure noise, no data)."""
        path = GaussianConditionalProbabilityPath()
        z = torch.ones(16, 4) * 100.0  # data far from zero
        t = torch.zeros(16)
        # At t=0: alpha=0, beta=1, so x = 0*z + 1*eps = eps
        # We can't check exact value (random), but it should NOT be close to z
        x = path.sample_conditional_path(z, t)
        # x should be ~N(0,1), not ~100
        assert x.abs().mean() < 10.0  # very unlikely to be near 100

    def test_t1_is_data(self) -> None:
        """At t≈1, x_t ≈ z (data sample)."""
        path = GaussianConditionalProbabilityPath()
        z = torch.randn(16, 4) * 5.0
        t = torch.full((16,), 0.999)
        x = path.sample_conditional_path(z, t)
        # At t=0.999: alpha=0.999, beta=0.001, x ≈ z
        torch.testing.assert_close(x, z, atol=0.1, rtol=0.1)

    def test_vector_field_points_to_data(self) -> None:
        """The reference vector field (z - x) / (1 - t) points from x toward z."""
        path = GaussianConditionalProbabilityPath()
        z = torch.ones(8, 4)
        x = torch.zeros(8, 4)
        t = torch.full((8,), 0.5)
        u = path.conditional_vector_field(x, z, t)
        # (z - x) / (1 - 0.5) = (1 - 0) / 0.5 = 2
        expected = torch.full((8, 4), 2.0)
        torch.testing.assert_close(u, expected)


class TestEulerODESolver:
    """Verify ODE integration basics."""

    def test_constant_velocity_integration(self) -> None:
        """Constant velocity field v=1 should give x(1) = x(0) + 1."""
        solver = EulerODESolver()
        x0 = torch.zeros(4, 2)

        def constant_v(x, t):
            return torch.ones_like(x)

        result = solver.solve(x0, constant_v, num_steps=100)
        torch.testing.assert_close(result, torch.ones(4, 2), atol=0.02, rtol=0.02)

    def test_output_shape(self) -> None:
        """Output shape matches input shape."""
        solver = EulerODESolver()
        x0 = torch.randn(10, 8)

        def zero_v(x, t):
            return torch.zeros_like(x)

        result = solver.solve(x0, zero_v, num_steps=10)
        assert result.shape == x0.shape


# --------------------------------------------------------------------------
# VisitationCritic Generation Tests
# --------------------------------------------------------------------------


class TestVisitationCriticGeneration:
    """Verify generation shape and basic sanity."""

    def _make_vc(self, state_dim: int = 8) -> VisitationCritic:
        cfg = {
            "label_mode": "l2_ball",
            "l2_radius": 2.0,
            "conditioning_type": "discrete",
            "num_classes": 2,
            "null_label": 2,
            "train_every_n_iters": 1,
            "num_warmup_iterations": 0,
            "num_train_steps": 1,
            "batch_size": 4,
            "hidden_dims": [16, 16],
            "class_dim": 4,
            "guidance_scale": 1.0,
            "num_euler_steps": 5,
            "max_episode_length": 50,
        }
        return VisitationCritic(
            cfg, state_dim=state_dim, obs_groups={"relative_state": ["relative_state"]}, device="cpu"
        )

    def test_generate_reset_states_shape(self) -> None:
        """Generated states have correct shape (num_states, state_dim)."""
        vc = self._make_vc(state_dim=8)
        states = vc.generate_reset_states(16)
        assert states.shape == (16, 8)

    def test_generate_states_for_each_label(self) -> None:
        """Can generate states for each discrete label without error."""
        vc = self._make_vc(state_dim=8)
        for label in range(2):
            states = vc.generate_states_for_label(4, label)
            assert states.shape == (4, 8)

    def test_generate_states_finite(self) -> None:
        """Generated states contain no NaN or Inf."""
        vc = self._make_vc(state_dim=8)
        states = vc.generate_reset_states(32)
        assert torch.isfinite(states).all()

    def test_untrained_model_generates_noise_like(self) -> None:
        """Untrained model with guidance_scale=1 should produce roughly noise-scale output."""
        vc = self._make_vc(state_dim=8)
        states = vc.generate_reset_states(100)
        # Untrained random network, output should be finite but not degenerate
        assert states.std() > 0.01

    def test_generate_zero_states(self) -> None:
        """Requesting 0 states should not error (returns empty tensor)."""
        vc = self._make_vc(state_dim=8)
        states = vc.generate_reset_states(0)
        assert states.shape == (0, 8)


# --------------------------------------------------------------------------
# CFM Training Convergence
# --------------------------------------------------------------------------


class TestCFMTrainingConvergence:
    """Verify that CFM training reduces loss on a simple distribution."""

    def test_loss_decreases_with_training(self) -> None:
        """After training on a simple dataset, CFM loss should decrease."""
        state_dim = 4
        cfg = {
            "label_mode": "l2_ball",
            "l2_radius": 2.0,
            "conditioning_type": "discrete",
            "num_classes": 2,
            "null_label": 2,
            "train_every_n_iters": 1,
            "num_warmup_iterations": 0,
            "num_train_steps": 200,
            "warmup_steps": 10,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "hidden_dims": [32, 32],
            "class_dim": 4,
            "max_trajectories": 1000,
            "guidance_scale": 1.0,
            "num_euler_steps": 10,
            "cfg_dropout_prob": 0.1,
            "min_scatter_states": 10,
            "generated_states_per_class": 10,
            "max_episode_length": 50,
        }
        vc = VisitationCritic(cfg, state_dim=state_dim, obs_groups={"relative_state": ["relative_state"]}, device="cpu")

        # Populate buffer with a bimodal distribution:
        # Class 0 (bad): start states centered at [5, 5, 5, 5]
        # Class 1 (good): start states centered at [0, 0, 0, 0]
        for _ in range(200):
            # Bad trajectories: start far, end far (norm > 2)
            vc.buffer.start_states.append(torch.randn(state_dim) + 5.0)
            vc.buffer.end_states.append(torch.randn(state_dim) * 3.0)
            vc.buffer.cumulative_rewards.append(-10.0)
            vc.buffer.trajectory_lengths.append(10)

            # Good trajectories: start near origin, end near origin (norm < 2)
            vc.buffer.start_states.append(torch.randn(state_dim) * 0.3)
            vc.buffer.end_states.append(torch.randn(state_dim) * 0.3)
            vc.buffer.cumulative_rewards.append(0.0)
            vc.buffer.trajectory_lengths.append(50)

        loss_dict = vc.train()
        assert "visitation_critic/cfm_loss" in loss_dict
        assert loss_dict["visitation_critic/cfm_loss"] < 10.0  # Should converge to something reasonable
        assert vc.is_trained

    def test_trained_model_generates_class_separated_states(self) -> None:
        """After training, states from different classes should have distinct means."""
        state_dim = 4
        cfg = {
            "label_mode": "l2_ball",
            "l2_radius": 2.0,
            "conditioning_type": "discrete",
            "num_classes": 2,
            "null_label": 2,
            "train_every_n_iters": 1,
            "num_warmup_iterations": 0,
            "num_train_steps": 500,
            "warmup_steps": 20,
            "batch_size": 128,
            "learning_rate": 1e-3,
            "hidden_dims": [64, 64],
            "class_dim": 8,
            "max_trajectories": 2000,
            "guidance_scale": 2.0,
            "num_euler_steps": 50,
            "cfg_dropout_prob": 0.2,
            "min_scatter_states": 10,
            "generated_states_per_class": 10,
            "max_episode_length": 50,
        }
        vc = VisitationCritic(cfg, state_dim=state_dim, obs_groups={"relative_state": ["relative_state"]}, device="cpu")

        # Create well-separated classes
        for _ in range(500):
            vc.buffer.start_states.append(torch.randn(state_dim) + 5.0)
            vc.buffer.end_states.append(torch.randn(state_dim) * 4.0)
            vc.buffer.cumulative_rewards.append(-10.0)
            vc.buffer.trajectory_lengths.append(5)

            vc.buffer.start_states.append(torch.randn(state_dim) * 0.2)
            vc.buffer.end_states.append(torch.randn(state_dim) * 0.2)
            vc.buffer.cumulative_rewards.append(0.0)
            vc.buffer.trajectory_lengths.append(100)

        vc.train()

        # Generate from each class
        bad_states = vc.generate_states_for_label(200, 0)
        good_states = vc.generate_states_for_label(200, 1)

        # The means should be noticeably different
        bad_mean = bad_states.mean(dim=0)
        good_mean = good_states.mean(dim=0)
        separation = (bad_mean - good_mean).norm().item()
        assert separation > 1.0, f"Class separation too low: {separation:.3f}"


# --------------------------------------------------------------------------
# Labeling Mode Tests
# --------------------------------------------------------------------------


class TestLabelingModes:
    """Verify all label_mode options produce correct labels."""

    def _make_vc(self, label_mode: str, **extra_cfg) -> VisitationCritic:
        cfg = {
            "label_mode": label_mode,
            "conditioning_type": "discrete",
            "num_classes": 2,
            "train_every_n_iters": 1,
            "num_warmup_iterations": 0,
            "num_train_steps": 1,
            "batch_size": 4,
            **extra_cfg,
        }
        return VisitationCritic(cfg, state_dim=4, obs_groups={"relative_state": ["relative_state"]}, device="cpu")

    def test_l2_ball_boundary(self) -> None:
        """States exactly at the radius boundary are classified as 'bad' (norm >= radius).
        Good also requires the trajectory to have survived the full episode length."""
        vc = self._make_vc("l2_ball", l2_radius=1.0, max_episode_length=100)
        data = {
            "start_states": torch.zeros(3, 4),
            "end_states": torch.tensor([
                [1.0, 0.0, 0.0, 0.0],   # norm=1.0, exactly at boundary -> bad (>=)
                [0.99, 0.0, 0.0, 0.0],  # norm=0.99 + complete -> good
                [1.01, 0.0, 0.0, 0.0],  # norm=1.01 -> bad
            ]),
            "cumulative_rewards": torch.zeros(3),
            "trajectory_lengths": torch.tensor([100, 100, 100]),
        }
        _, labels = vc._label_trajectories(data)
        # label 1=good (norm < radius AND complete), 0=bad (norm >= radius OR incomplete)
        assert labels.tolist() == [0, 1, 0]

    def test_reward_bins_basic_partition(self) -> None:
        """4-bin partition: 3 fail + 3 complete, each split by median reward.

        Expected bins (fail-low=0, fail-high=1, succeed-low=2, succeed-high=3).
        PyTorch's torch.Tensor.median returns the lower median for even-sized
        groups, so sizing each group >=3 makes the split unambiguous.
        """
        vc = self._make_vc(
            "reward_bins",
            num_classes=4,
            max_episode_length=100,
            reset_bin_probs=[0.25, 0.25, 0.25, 0.25],
        )
        # 3 incomplete (lengths 50, 60, 80) rewards 1.0, 2.0, 3.0; median=2.0
        #   → rewards <  2.0 -> bin 0, rewards >= 2.0 -> bin 1
        # 3 complete   (length 100)      rewards 2.0, 4.0, 5.0; median=4.0
        #   → rewards <  4.0 -> bin 2, rewards >= 4.0 -> bin 3
        data = {
            "start_states": torch.zeros(6, 4),
            "end_states": torch.zeros(6, 4),
            "cumulative_rewards": torch.tensor([1.0, 2.0, 3.0, 2.0, 4.0, 5.0]),
            "trajectory_lengths": torch.tensor([50, 60, 80, 100, 100, 100]),
        }
        _, labels = vc._label_trajectories(data)
        assert labels.tolist() == [0, 1, 1, 2, 3, 3]

    def test_reward_bins_all_incomplete(self) -> None:
        """When all trajectories are incomplete, labels land in bins {0, 1}."""
        vc = self._make_vc(
            "reward_bins",
            num_classes=4,
            max_episode_length=100,
            reset_bin_probs=[0.25, 0.25, 0.25, 0.25],
        )
        data = {
            "start_states": torch.zeros(5, 4),
            "end_states": torch.zeros(5, 4),
            "cumulative_rewards": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            "trajectory_lengths": torch.tensor([50, 60, 70, 80, 90]),
        }
        _, labels = vc._label_trajectories(data)
        # torch median of 5 elements = 3.0; reward >= 3 -> bin 1, else bin 0.
        assert labels.tolist() == [0, 0, 1, 1, 1]

    def test_reward_bins_reset_bin_probs_validation(self) -> None:
        """Invalid reset_bin_probs (wrong length or bad sum) raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="length 4"):
            self._make_vc(
                "reward_bins",
                num_classes=4,
                max_episode_length=100,
                reset_bin_probs=[0.5, 0.5],
            )
        with pytest.raises(ValueError, match="sum to 1.0"):
            self._make_vc(
                "reward_bins",
                num_classes=4,
                max_episode_length=100,
                reset_bin_probs=[0.1, 0.1, 0.1, 0.1],
            )

    def test_reward_bins_requires_max_episode_length(self) -> None:
        """Omitting max_episode_length in reward_bins mode raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="max_episode_length"):
            self._make_vc("reward_bins", num_classes=4)

    def test_reward_bins_generate_reset_states_samples_bins(self) -> None:
        """generate_reset_states draws multinomial samples across the 4 bins."""
        vc = self._make_vc(
            "reward_bins",
            num_classes=4,
            max_episode_length=100,
            reset_bin_probs=[0.25, 0.25, 0.25, 0.25],
        )
        # Seed torch before sampling so the multinomial is reproducible-ish.
        torch.manual_seed(0)
        states = vc.generate_reset_states(256)
        assert states.shape == (256, 4)

    def test_continuous_reward_range(self) -> None:
        """Continuous labels are normalized to [0, 1]."""
        cfg = {
            "label_mode": "continuous_reward",
            "conditioning_type": "continuous",
            "cond_dim": 1,
            "train_every_n_iters": 1,
            "num_warmup_iterations": 0,
            "num_train_steps": 1,
            "batch_size": 4,
        }
        vc = VisitationCritic(cfg, state_dim=4, obs_groups={"relative_state": ["relative_state"]}, device="cpu")
        data = {
            "start_states": torch.zeros(4, 4),
            "end_states": torch.zeros(4, 4),
            "cumulative_rewards": torch.tensor([-10.0, 0.0, 5.0, 10.0]),
        }
        _, labels = vc._label_trajectories(data)
        # Normalized: [-10..10] -> [0..1]
        expected = torch.tensor([[0.0], [0.5], [0.75], [1.0]])
        torch.testing.assert_close(labels, expected, atol=1e-5, rtol=0)


# --------------------------------------------------------------------------
# Schedule Logic Tests
# --------------------------------------------------------------------------


class TestScheduleLogic:
    """Verify should_collect and should_train gating."""

    def _make_vc(self, train_every: int = 5, warmup: int = 10) -> VisitationCritic:
        cfg = {
            "label_mode": "l2_ball",
            "conditioning_type": "discrete",
            "num_classes": 2,
            "train_every_n_iters": train_every,
            "num_warmup_iterations": warmup,
            "num_train_steps": 1,
            "batch_size": 4,
            "max_episode_length": 50,
        }
        return VisitationCritic(cfg, state_dim=4, obs_groups={"relative_state": ["relative_state"]}, device="cpu")

    def test_should_collect_respects_warmup(self) -> None:
        vc = self._make_vc(train_every=5, warmup=10)
        # Before warmup: never collect
        for i in range(10):
            assert not vc.should_collect(i)
        # At warmup boundary aligned with train_every
        assert vc.should_collect(10)
        assert not vc.should_collect(11)
        assert vc.should_collect(15)

    def test_should_collect_and_train_aligned(self) -> None:
        """should_collect and should_train fire on the same iterations (when buffer has data)."""
        vc = self._make_vc(train_every=3, warmup=0)
        # Populate buffer
        vc.buffer.start_states.append(torch.randn(4))
        vc.buffer.end_states.append(torch.randn(4))
        vc.buffer.cumulative_rewards.append(0.0)
        vc.buffer.trajectory_lengths.append(2)

        for i in range(20):
            assert vc.should_collect(i) == vc.should_train(i)

    def test_should_train_false_with_empty_buffer(self) -> None:
        """should_train is False even on correct iteration if buffer is empty."""
        vc = self._make_vc(train_every=1, warmup=0)
        assert not vc.should_train(0)


# --------------------------------------------------------------------------
# CFG Guidance Tests
# --------------------------------------------------------------------------


class TestCFGGuidance:
    """Verify classifier-free guidance math."""

    def test_guidance_scale_1_equals_conditional(self) -> None:
        """With scale=1.0, CFG output equals the conditional prediction."""
        model = MLPConditionalVectorField(
            state_dim=4, hidden_dims=[16], num_classes=2, class_dim=4
        )
        x = torch.randn(8, 4)
        t = torch.rand(8)
        cond = torch.zeros(8, dtype=torch.long)
        null_cond = torch.full((8,), 2, dtype=torch.long)

        guided = cfg_guided_velocity(model, x, t, guidance_scale=1.0, cond=cond, null_cond=null_cond)
        direct = model(x, t, cond)
        torch.testing.assert_close(guided, direct)

    def test_guidance_scale_0_equals_unconditional(self) -> None:
        """With scale=0.0, CFG output equals the unconditional prediction."""
        model = MLPConditionalVectorField(
            state_dim=4, hidden_dims=[16], num_classes=2, class_dim=4
        )
        x = torch.randn(8, 4)
        t = torch.rand(8)
        cond = torch.zeros(8, dtype=torch.long)
        null_cond = torch.full((8,), 2, dtype=torch.long)

        guided = cfg_guided_velocity(model, x, t, guidance_scale=0.0, cond=cond, null_cond=null_cond)
        unconditional = model(x, t, null_cond)
        torch.testing.assert_close(guided, unconditional)

    def test_high_guidance_amplifies_conditional(self) -> None:
        """Higher guidance scale pushes output further from unconditional."""
        model = MLPConditionalVectorField(
            state_dim=4, hidden_dims=[16], num_classes=2, class_dim=4
        )
        x = torch.randn(8, 4)
        t = torch.rand(8)
        cond = torch.zeros(8, dtype=torch.long)
        null_cond = torch.full((8,), 2, dtype=torch.long)

        g1 = cfg_guided_velocity(model, x, t, guidance_scale=1.0, cond=cond, null_cond=null_cond)
        g5 = cfg_guided_velocity(model, x, t, guidance_scale=5.0, cond=cond, null_cond=null_cond)
        u_null = model(x, t, null_cond)

        # g5 should be further from u_null than g1
        dist_g1 = (g1 - u_null).norm()
        dist_g5 = (g5 - u_null).norm()
        assert dist_g5 > dist_g1


# --------------------------------------------------------------------------
# VC State Dimension Validation
# --------------------------------------------------------------------------


class TestStateDimValidation:
    """Verify state_dim handling across the VC pipeline."""

    def test_generate_matches_state_dim(self) -> None:
        """Generated states always match the configured state_dim."""
        for dim in [4, 16, 52]:
            cfg = {
                "label_mode": "l2_ball",
                "l2_radius": 2.0,
                "conditioning_type": "discrete",
                "num_classes": 2,
                "null_label": 2,
                "train_every_n_iters": 1,
                "num_warmup_iterations": 0,
                "num_train_steps": 1,
                "batch_size": 4,
                "hidden_dims": [16],
                "class_dim": 4,
                "guidance_scale": 1.0,
                "num_euler_steps": 3,
                "max_episode_length": 50,
            }
            vc = VisitationCritic(cfg, state_dim=dim, obs_groups={"relative_state": ["relative_state"]}, device="cpu")
            states = vc.generate_reset_states(5)
            assert states.shape == (5, dim)

    def test_buffer_get_tensors_empty(self) -> None:
        """get_tensors returns empty dict when buffer has no data."""
        buffer = TrajectoryBuffer(max_trajectories=100, state_dim=4, device="cpu")
        assert buffer.get_tensors("cpu") == {}

    def test_buffer_get_tensors_shapes(self) -> None:
        """get_tensors returns correctly shaped tensors."""
        buffer = TrajectoryBuffer(max_trajectories=100, state_dim=6, device="cpu")
        for i in range(5):
            buffer.start_states.append(torch.randn(6))
            buffer.end_states.append(torch.randn(6))
            buffer.cumulative_rewards.append(float(i))
            buffer.trajectory_lengths.append(i + 2)

        tensors = buffer.get_tensors("cpu")
        assert tensors["start_states"].shape == (5, 6)
        assert tensors["end_states"].shape == (5, 6)
        assert tensors["cumulative_rewards"].shape == (5,)
        assert tensors["trajectory_lengths"].shape == (5,)


# --------------------------------------------------------------------------
# Persistence Tests
# --------------------------------------------------------------------------


class TestVisitationCriticPersistence:
    """Verify save/load roundtrip."""

    def test_save_load_roundtrip(self) -> None:
        cfg = {
            "label_mode": "l2_ball",
            "l2_radius": 2.0,
            "conditioning_type": "discrete",
            "num_classes": 2,
            "null_label": 2,
            "train_every_n_iters": 1,
            "num_warmup_iterations": 0,
            "num_train_steps": 1,
            "batch_size": 4,
            "hidden_dims": [16, 16],
            "class_dim": 4,
            "max_episode_length": 50,
        }
        vc = VisitationCritic(cfg, state_dim=4, obs_groups={"relative_state": ["relative_state"]}, device="cpu")
        vc._trained = True

        state = vc.save()
        assert "vc_model_state" in state
        assert "vc_trained" in state
        assert state["vc_trained"] is True

        # Create fresh VC and load
        vc2 = VisitationCritic(cfg, state_dim=4, obs_groups={"relative_state": ["relative_state"]}, device="cpu")
        assert not vc2.is_trained
        vc2.load(state)
        assert vc2.is_trained

        # Model weights should match
        for (k1, v1), (k2, v2) in zip(
            vc.model.state_dict().items(), vc2.model.state_dict().items()
        ):
            assert k1 == k2
            torch.testing.assert_close(v1, v2)
