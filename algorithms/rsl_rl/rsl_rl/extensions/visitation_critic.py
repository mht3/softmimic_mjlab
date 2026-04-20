# @2026 Matthew Taylor

"""Visitation Critic extension using Conditional Flow Matching (CFM).

Collects whole trajectories, labels them (good/bad), trains a CFM model to learn the
distribution of start states conditioned on trajectory quality, and generates targeted
reset states from the "not good" distribution.
"""

from __future__ import annotations

import contextlib
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.env import VecEnv
from rsl_rl.modules.cfm import (
    MLPConditionalVectorField,
    MLPContinuousConditionalVectorField,
    GaussianConditionalProbabilityPath,
    EulerODESolver,
    cfg_guided_velocity,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrajectoryBuffer:
    """Ring buffer storing complete trajectories for CFM training.

    Accumulates per-step data from multiple parallel environments and segments them
    into complete episodes using done flags. All operations are vectorized over envs
    to avoid Python-level per-env loops.
    """

    def __init__(self, max_trajectories: int, state_dim: int, device: str = "cpu") -> None:
        self.max_trajectories = max_trajectories
        self.state_dim = state_dim
        self.device = device
        self._initialized = False

        # Storage for completed trajectories
        self.start_states: list[torch.Tensor] = []  # (state_dim,) each
        self.end_states: list[torch.Tensor] = []  # (state_dim,) each
        self.cumulative_rewards: list[float] = []
        self.trajectory_lengths: list[int] = []

    def _lazy_init(self, num_envs: int, obs: torch.Tensor) -> None:
        """Initialize vectorized per-env accumulators on first call."""
        if self._initialized:
            return
        cpu = torch.device("cpu")
        self._start_obs = torch.zeros(num_envs, self.state_dim, device=cpu)
        self._last_obs = torch.zeros(num_envs, self.state_dim, device=cpu)
        self._cum_rewards = torch.zeros(num_envs, device=cpu)
        self._step_counts = torch.zeros(num_envs, dtype=torch.long, device=cpu)
        self._initialized = True

    @property
    def num_trajectories(self) -> int:
        return len(self.start_states)

    def add_step(self, obs: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """Record one timestep of data from all parallel envs (vectorized).

        IMPORTANT: when ``dones[i]=True``, the environment has already auto-reset and
        ``obs[i]`` is the *post-reset* observation, not the terminal observation. To
        record the actual pre-failure state as the trajectory's end state, we use
        ``_last_obs`` (the obs from the *previous* step, before this terminating step)
        and process completions BEFORE overwriting ``_last_obs`` with the current obs.

        Args:
            obs: Observations, shape (num_envs, state_dim).
            rewards: Rewards, shape (num_envs,).
            dones: Done flags, shape (num_envs,).
        """
        num_envs = obs.shape[0]
        self._lazy_init(num_envs, obs)

        obs_cpu = obs.detach().cpu()
        rewards_cpu = rewards.detach().cpu()
        dones_cpu = dones.detach().cpu().bool()

        # 1. Process completed episodes FIRST, using stale `_last_obs` which holds
        #    the obs from the previous step (the pre-failure state).
        #    Require step_counts >= 1 so that `_last_obs` actually contains a real
        #    obs from this episode (i.e. the episode had at least 2 steps total).
        done_complete = dones_cpu & (self._step_counts >= 1)
        if done_complete.any():
            done_indices = done_complete.nonzero(as_tuple=True)[0]
            for idx in done_indices:
                i = idx.item()
                self.start_states.append(self._start_obs[i].clone())
                self.end_states.append(self._last_obs[i].clone())
                self.cumulative_rewards.append(self._cum_rewards[i].item())
                # +1 because the current (terminating) step is also part of the episode
                self.trajectory_lengths.append(int(self._step_counts[i].item()) + 1)

            # Enforce ring buffer limit
            overflow = len(self.start_states) - self.max_trajectories
            if overflow > 0:
                del self.start_states[:overflow]
                del self.end_states[:overflow]
                del self.cumulative_rewards[:overflow]
                del self.trajectory_lengths[:overflow]

        # 2. Record start obs for envs on their first step (after a reset or fresh).
        first_step_mask = self._step_counts == 0
        if first_step_mask.any():
            self._start_obs[first_step_mask] = obs_cpu[first_step_mask]

        # 3. Update running state for all envs.
        self._last_obs.copy_(obs_cpu)
        self._cum_rewards += rewards_cpu
        self._step_counts += 1

        # 4. Reset accumulators for done envs so the next episode starts clean.
        if dones_cpu.any():
            self._cum_rewards[dones_cpu] = 0.0
            self._step_counts[dones_cpu] = 0

    def get_tensors(self, device: str) -> dict[str, torch.Tensor]:
        """Return all stored data as tensors on the specified device."""
        if self.num_trajectories == 0:
            return {}
        return {
            "start_states": torch.stack(self.start_states).to(device),
            "end_states": torch.stack(self.end_states).to(device),
            "cumulative_rewards": torch.tensor(self.cumulative_rewards, device=device),
            "trajectory_lengths": torch.tensor(self.trajectory_lengths, device=device),
        }

    def clear(self) -> None:
        """Clear all stored trajectories (keeps in-progress accumulators)."""
        self.start_states.clear()
        self.end_states.clear()
        self.cumulative_rewards.clear()
        self.trajectory_lengths.clear()


class VisitationCritic:
    """Visitation Critic using Conditional Flow Matching.

    Learns the conditional distribution of start states given trajectory quality labels,
    then generates reset states from the "not good" distribution to create a training
    curriculum.
    """

    def __init__(self, cfg: dict, state_dim: int, obs_groups: dict, device: str = "cpu") -> None:
        self.cfg = cfg
        self.state_dim = state_dim
        self.obs_groups = obs_groups
        self.device = device
        self._trained = False

        # Training schedule
        self.train_every_n_iters: int = cfg["train_every_n_iters"]
        self.num_warmup_iterations: int = cfg.get("num_warmup_iterations", 0)
        self.num_train_steps: int = cfg["num_train_steps"]
        self.learning_rate: float = cfg.get("learning_rate", 1e-3)
        self.warmup_steps: int = cfg.get("warmup_steps", 500)
        self.batch_size: int = cfg.get("batch_size", 1000)

        # Labeling config
        self.label_mode: str = cfg["label_mode"]
        self.l2_radius: float = cfg.get("l2_radius", 4.0)
        self.num_reward_bins: int = cfg.get("num_reward_bins", 4)
        self.reward_bin_boundaries: list[float] = cfg.get(
            "reward_bin_boundaries", [0.25, 0.5, 0.75]
        )

        # Generation config
        self.guidance_scale: float = cfg.get("guidance_scale", 5.0)
        self.num_euler_steps: int = cfg.get("num_euler_steps", 100)
        self.cfg_dropout_prob: float = cfg.get("cfg_dropout_prob", 0.25)
        self.min_scatter_states: int = cfg.get("min_scatter_states", 500)
        self.generated_states_per_class: int = cfg.get("generated_states_per_class", 500)

        # Reset behavior
        self.reset_condition_label: int = cfg.get("reset_condition_label", 0)
        self.reset_condition_value: float = cfg.get("reset_condition_value", 0.1)

        # Deterministic trajectory collection settings
        self.num_collect_trajectories: int = cfg.get("num_collect_trajectories", 10000)
        self.disable_push_during_collection: bool = cfg.get(
            "disable_push_during_collection", False
        )
        self.collection_push_term_name: str = cfg.get(
            "collection_push_term_name", "push_robot"
        )

        # Build model
        self.conditioning_type: str = cfg["conditioning_type"]
        self.model = self._build_model()
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        # Probability path and ODE solver
        self.path = GaussianConditionalProbabilityPath()
        self.solver = EulerODESolver()

        # Trajectory buffer
        self.buffer = TrajectoryBuffer(
            max_trajectories=cfg.get("max_trajectories", 10000),
            state_dim=state_dim,
            device="cpu",
        )

        # Precompute null condition for discrete mode
        if self.conditioning_type == "discrete":
            self._null_label = cfg.get("null_label", cfg.get("num_classes", 2))
        self._wandb_media: dict[str, Any] = {}

    def _build_model(self) -> nn.Module:
        """Construct the vector field network from config."""
        hidden_dims = self.cfg.get("hidden_dims", [512, 512, 512])
        activation = self.cfg.get("activation", "swish")

        if self.conditioning_type == "discrete":
            return MLPConditionalVectorField(
                state_dim=self.state_dim,
                hidden_dims=hidden_dims,
                num_classes=self.cfg.get("num_classes", 2),
                class_dim=self.cfg.get("class_dim", 8),
                activation=activation,
            )
        elif self.conditioning_type == "continuous":
            return MLPContinuousConditionalVectorField(
                state_dim=self.state_dim,
                hidden_dims=hidden_dims,
                cond_dim=self.cfg.get("cond_dim", 1),
                null_value=self.cfg.get("null_value", -1.0),
                activation=activation,
            )
        else:
            raise ValueError(f"Unknown conditioning_type: {self.conditioning_type}")

    def _reinitialize_model(self) -> None:
        """Reinitialize model weights and optimizer from scratch."""
        self.model = self._build_model()
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

    # --- Data Collection ---

    def store_trajectory_data(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """Store one timestep of trajectory data from all envs.

        Args:
            obs: Observation TensorDict from the environment.
            rewards: Rewards tensor, shape (num_envs,).
            dones: Done flags tensor, shape (num_envs,).
        """
        flat_obs = torch.cat([obs[k] for k in self.obs_groups["relative_state"]], dim=-1)
        self.buffer.add_step(flat_obs, rewards, dones)

    def _compute_end_state_norms(self, end_states: torch.Tensor) -> torch.Tensor:
        """Compute end-state L2 norm over all relative-state dimensions."""
        return torch.linalg.norm(end_states, dim=1)

    # --- Labeling ---

    def _label_trajectories(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Label trajectories based on the configured label_mode.

        Returns:
            start_states: (N, state_dim)
            labels: (N,) integer labels for discrete, (N, cond_dim) float for continuous.
        """
        start_states = data["start_states"]
        end_states = data["end_states"]
        cum_rewards = data["cumulative_rewards"]

        if self.label_mode == "l2_ball":
            norms = self._compute_end_state_norms(end_states)
            labels = (norms < self.l2_radius).long()  # 1=good, 0=bad
            return start_states, labels

        elif self.label_mode == "reward_bins":
            r_min, r_max = cum_rewards.min(), cum_rewards.max()
            if r_max - r_min > 1e-8:
                r_norm = (cum_rewards - r_min) / (r_max - r_min)
            else:
                r_norm = torch.zeros_like(cum_rewards)
            boundaries = torch.tensor(self.reward_bin_boundaries, device=cum_rewards.device)
            labels = torch.bucketize(r_norm, boundaries, right=True).long()
            return start_states, labels

        elif self.label_mode == "continuous_reward":
            r_min, r_max = cum_rewards.min(), cum_rewards.max()
            if r_max - r_min > 1e-8:
                r_norm = (cum_rewards - r_min) / (r_max - r_min)
            else:
                r_norm = torch.zeros_like(cum_rewards)
            return start_states, r_norm.unsqueeze(-1)

        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

    # --- Deterministic trajectory collection ---

    def should_collect(self, iteration: int) -> bool:
        """Check whether to pause PPO and collect a deterministic trajectory buffer.

        Matches the CFM training cadence: collect at the first post-warmup iteration
        that aligns with ``train_every_n_iters`` and then every period thereafter.
        """
        return (
            iteration >= self.num_warmup_iterations
            and iteration % self.train_every_n_iters == 0
        )

    @torch.inference_mode()
    def collect_trajectories(self, env: VecEnv, actor: nn.Module) -> int:
        """Collect deterministic trajectories into the buffer.

        Clears the buffer, force-resets all envs so every trajectory begins from the
        init-condition distribution, then steps env.step() with the deterministic
        policy output until the buffer holds at least ``self.num_collect_trajectories``
        completed trajectories. Honors ``self.disable_push_during_collection``.
        Policy normalizers are NOT updated.

        Returns:
            Number of trajectories collected (>= num_collect_trajectories).
        """
        actor_was_training = actor.training
        actor.eval()
        self.buffer.clear()
        if self.buffer._initialized:
            self.buffer._step_counts.zero_()
            self.buffer._cum_rewards.zero_()

        with self._maybe_disabled_push(env):
            obs, _ = env.reset()
            obs = obs.to(self.device)
            while self.buffer.num_trajectories < self.num_collect_trajectories:
                actions = actor(obs, stochastic_output=False)
                obs, rewards, dones, _ = env.step(actions.to(env.device))
                obs = obs.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                self.store_trajectory_data(obs, rewards, dones)

        if actor_was_training:
            actor.train()
        return self.buffer.num_trajectories

    @contextlib.contextmanager
    def _maybe_disabled_push(self, env: VecEnv):
        """Temporarily replace the push-robot event term's func with a no-op.

        Uses func-replacement (rather than zeroing velocity_range) so the
        ``push_velocity_curriculum`` can't reintroduce perturbations mid-collection.
        """
        if not self.disable_push_during_collection:
            yield
            return
        try:
            term_cfg = env.unwrapped.event_manager.get_term_cfg(self.collection_push_term_name)
        except Exception:
            yield
            return
        original_func = term_cfg.func
        term_cfg.func = lambda env, env_ids, **kwargs: None
        try:
            yield
        finally:
            term_cfg.func = original_func

    # --- Training ---

    def should_train(self, iteration: int) -> bool:
        """Check whether CFM training should happen at this iteration.

        Assumes the buffer has been freshly filled via ``collect_trajectories``
        just before this call.
        """
        return (
            iteration >= self.num_warmup_iterations
            and iteration % self.train_every_n_iters == 0
            and self.buffer.num_trajectories > 0
        )

    def train(self) -> dict[str, float]:
        """Train the CFM model on collected trajectory data.

        Reinitializes the model from scratch each round to avoid loss stagnation
        from stale weight basins when the trajectory distribution shifts.

        Returns:
            Dictionary of training metrics.
        """
        self._reinitialize_model()

        data = self.buffer.get_tensors(self.device)
        if not data:
            return {"visitation_critic/cfm_loss": 0.0}

        start_states, labels = self._label_trajectories(data)
        num_samples = start_states.shape[0]

        eps = 1e-3  # Avoid t=1

        self.model.train()
        total_loss = 0.0
        cfm_loss_curve: list[float] = []

        for step in range(self.num_train_steps):
            # Linear warmup
            if step < self.warmup_steps:
                lr = self.learning_rate * (step + 1) / self.warmup_steps
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

            # Sample a batch
            idx = torch.randint(0, num_samples, (self.batch_size,), device=self.device)
            z = start_states[idx]
            y = labels[idx]

            # Classifier-free guidance dropout
            if self.conditioning_type == "discrete":
                drop_mask = torch.rand(self.batch_size, device=self.device) < self.cfg_dropout_prob
                y = y.clone()
                y[drop_mask] = self._null_label
            elif self.conditioning_type == "continuous":
                drop_mask = torch.rand(self.batch_size, device=self.device) < self.cfg_dropout_prob
                null_emb = self.model.get_null_embedding(self.batch_size)
                y = y.clone()
                y[drop_mask] = null_emb[drop_mask]

            # Sample time and interpolated state
            t = torch.rand(self.batch_size, device=self.device) * (1.0 - eps)
            x = self.path.sample_conditional_path(z, t)

            # Forward pass
            u_theta = self.model(x, t, y)
            u_ref = self.path.conditional_vector_field(x, z, t)

            # MSE loss
            loss = torch.mean((u_theta - u_ref) ** 2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value = loss.item()

            total_loss += loss_value
            cfm_loss_curve.append(loss_value)

        self._trained = True
        avg_loss = total_loss / max(self.num_train_steps, 1)
        self._wandb_media = self._build_wandb_media(
            data, start_states, data["end_states"], labels, cfm_loss_curve
        )

        # Clear the buffer so next training round uses fresh on-policy data
        self.buffer.clear()

        return {
            "visitation_critic/cfm_loss": avg_loss}

    @property
    def wandb_media(self) -> dict[str, Any]:
        """Return media payload for W&B logging from the latest VC update."""
        return self._wandb_media

    # --- Generation ---

    @property
    def is_trained(self) -> bool:
        return self._trained

    @torch.no_grad()
    def generate_reset_states(self, num_states: int) -> torch.Tensor:
        """Generate reset states conditioned on 'bad' trajectories.

        Samples noise from N(0, I) and integrates the learned vector field from t=0 to t=1
        using classifier-free guidance conditioned on the 'bad' label/low reward.

        Args:
            num_states: Number of states to generate.

        Returns:
            Generated states, shape (num_states, state_dim).
        """
        self.model.eval()

        x0 = torch.randn(num_states, self.state_dim, device=self.device)

        if self.conditioning_type == "discrete":
            cond = torch.full((num_states,), self.reset_condition_label, dtype=torch.long, device=self.device)
            null_cond = torch.full((num_states,), self._null_label, dtype=torch.long, device=self.device)
        elif self.conditioning_type == "continuous":
            cond = torch.full((num_states, self.model.cond_dim), self.reset_condition_value, device=self.device)
            null_cond = self.model.get_null_embedding(num_states)
        else:
            raise ValueError(f"Unknown conditioning_type: {self.conditioning_type}")

        guidance_scale = self.guidance_scale

        def guided_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return cfg_guided_velocity(
                self.model, x, t, guidance_scale, cond=cond, null_cond=null_cond
            )

        generated = self.solver.solve(x0, guided_fn, num_steps=self.num_euler_steps)

        self.model.train()
        return generated

    @torch.no_grad()
    def generate_states_for_label(self, num_states: int, label: int) -> torch.Tensor:
        """Generate states conditioned on a specific discrete label."""
        if self.conditioning_type != "discrete":
            raise ValueError("generate_states_for_label is only supported for discrete conditioning.")

        self.model.eval()
        x0 = torch.randn(num_states, self.state_dim, device=self.device)
        cond = torch.full((num_states,), label, dtype=torch.long, device=self.device)
        null_cond = torch.full((num_states,), self._null_label, dtype=torch.long, device=self.device)

        def guided_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return cfg_guided_velocity(
                self.model, x, t, self.guidance_scale, cond=cond, null_cond=null_cond
            )

        generated = self.solver.solve(x0, guided_fn, num_steps=self.num_euler_steps)
        self.model.train()
        return generated

    @staticmethod
    def _state_roll_pitch(state: torch.Tensor) -> torch.Tensor:
        """Extract true Euler roll/pitch (ZYX convention) from a relative-state tensor.

        The rel-qpos block stores orientation as an axis-angle rotation vector in
        indices [3, 4, 5] (output of ``differentiate_qpos``). Plotting these raw
        components approximates roll/pitch only for small tilts; past ~30° the
        rotvec-x/y components diverge from Euler roll/pitch. This helper converts
        rotvec -> quaternion -> Euler so plot axes match the physical quantities.

        Args:
            state: (N, state_dim) relative-state tensor.

        Returns:
            (N, 2) tensor of [roll, pitch] in radians.
        """
        rotvec = state[:, 3:6]
        angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp(min=1e-10)
        axis = rotvec / angle
        half = angle * 0.5
        w = torch.cos(half).squeeze(-1)
        xyz = axis * torch.sin(half)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        return torch.stack([roll, pitch], dim=-1)

    def _build_wandb_media(
        self,
        data: dict[str, torch.Tensor],
        start_states: torch.Tensor,
        end_states: torch.Tensor,
        labels: torch.Tensor,
        cfm_loss_curve: list[float],
    ) -> dict[str, Any]:
        """Build W&B media payload with VC diagnostic scatter plots."""

        rel_qpos_dim = self.cfg.get("rel_qpos_dim", self.state_dim // 2)
        roll_rate_idx = rel_qpos_dim + 3
        pitch_rate_idx = rel_qpos_dim + 4
        if pitch_rate_idx >= self.state_dim:
            return {}

        start_cpu = start_states.detach().cpu()
        end_cpu = end_states.detach().cpu()
        labels_cpu = labels.detach().cpu() if isinstance(labels, torch.Tensor) else None
        scatter_count = max(self.min_scatter_states, 500)
        media: dict[str, Any] = {}
        # Resolve class labels and names for discrete conditioning.
        class_names: list[str] = []
        class_values: list[int] = []
        if self.conditioning_type == "discrete" and labels_cpu is not None:
            num_classes = int(self.cfg.get("num_classes", 2))
            class_names_cfg = self.cfg.get("class_names", None)
            if isinstance(class_names_cfg, (list, tuple)) and len(class_names_cfg) >= num_classes:
                class_names = [str(name) for name in class_names_cfg[:num_classes]]
            elif self.label_mode == "l2_ball" and num_classes >= 2:
                class_names = ["Bad", "Good"] + [f"Class_{i}" for i in range(2, num_classes)]
            else:
                class_names = [f"Class_{i}" for i in range(num_classes)]
            class_values = list(range(num_classes))
        else:
            class_names = ["all"]
            class_values = [0]

        # Build per-class sampled dataset tensors for start and end states.
        # Track each class's share of the full dataset for legend percentages.
        total_count = start_cpu.shape[0]
        sampled_by_class: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        class_pct: dict[int, float] = {}
        for class_id in class_values:
            if self.conditioning_type == "discrete" and labels_cpu is not None:
                class_mask = labels_cpu == class_id
                start_cls = start_cpu[class_mask]
                end_cls = end_cpu[class_mask]
                class_pct[class_id] = 100.0 * class_mask.float().mean().item() if total_count > 0 else 0.0
            else:
                start_cls = start_cpu
                end_cls = end_cpu
                class_pct[class_id] = 100.0
            if start_cls.shape[0] == 0:
                continue
            n = max(scatter_count, 500)
            idx = torch.randint(0, start_cls.shape[0], (n,))
            sampled_by_class[class_id] = (start_cls[idx], end_cls[idx])

        # Dataset plot: 2x2 grid (row1 start, row2 end; col1 roll, col2 pitch), class-colored.
        fig_dataset, axes = plt.subplots(2, 2, figsize=(12, 9))
        colors = plt.cm.tab10
        for i, class_id in enumerate(class_values):
            if class_id not in sampled_by_class:
                continue
            start_cls, end_cls = sampled_by_class[class_id]
            start_rp = self._state_roll_pitch(start_cls)
            end_rp = self._state_roll_pitch(end_cls)
            name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            pct = class_pct.get(class_id, 0.0)
            count = int(round(pct / 100.0 * total_count))
            class_label = f"{name} ({class_id}): {pct:.1f}% of {total_count} ({count})"
            c = colors(i % 10)
            axes[0, 0].scatter(
                start_rp[:, 0].numpy(),
                start_cls[:, roll_rate_idx].numpy(),
                alpha=0.28,
                s=6,
                color=c,
                label=class_label,
            )
            axes[0, 1].scatter(
                start_rp[:, 1].numpy(),
                start_cls[:, pitch_rate_idx].numpy(),
                alpha=0.28,
                s=6,
                color=c,
                label=class_label,
            )
            axes[1, 0].scatter(
                end_rp[:, 0].numpy(),
                end_cls[:, roll_rate_idx].numpy(),
                alpha=0.28,
                s=6,
                color=c,
                label=class_label,
            )
            axes[1, 1].scatter(
                end_rp[:, 1].numpy(),
                end_cls[:, pitch_rate_idx].numpy(),
                alpha=0.28,
                s=6,
                color=c,
                label=class_label,
            )
        axes[0, 0].set_title("Roll phase, start state")
        axes[0, 1].set_title("Pitch phase, start state")
        axes[1, 0].set_title("Roll phase, end state")
        axes[1, 1].set_title("Pitch phase, end state")
        axes[0, 0].set_xlabel(r"$\phi$ (rad)")
        axes[0, 0].set_ylabel(r"$\dot{\phi}$ (rad/s)")
        axes[0, 1].set_xlabel(r"$\theta$ (rad)")
        axes[0, 1].set_ylabel(r"$\dot{\theta}$ (rad/s)")
        axes[1, 0].set_xlabel(r"$\phi$ (rad)")
        axes[1, 0].set_ylabel(r"$\dot{\phi}$ (rad/s)")
        axes[1, 1].set_xlabel(r"$\theta$ (rad)")
        axes[1, 1].set_ylabel(r"$\dot{\theta}$ (rad/s)")
        for ax in axes.flat:
            ax.grid(alpha=0.2)
            ax.legend(loc="best")
        fig_dataset.suptitle(
            f"Dataset Trajectory States (N={total_count} total, ~{scatter_count}/class sampled)"
        )
        media["visitation_critic/dataset_phase_space"] = fig_dataset

        # End-state norm histogram with radius threshold line (l2_ball labeling only).
        if self.label_mode == "l2_ball" and total_count > 0:
            end_norms = self._compute_end_state_norms(end_cpu).numpy()
            good_pct = 100.0 * (end_norms < self.l2_radius).mean()
            fig_hist, ax_hist = plt.subplots(1, 1, figsize=(10, 5))
            ax_hist.hist(
                end_norms,
                bins=80,
                color="gray",
                alpha=0.8,
                edgecolor="black",
                linewidth=0.3,
            )
            ax_hist.axvline(
                self.l2_radius,
                color="tab:red",
                linestyle="--",
                lw=2.0,
                label=f"l2_radius={self.l2_radius:.2f}  (good={good_pct:.1f}%)",
            )
            ax_hist.set_xlabel(r"$\|$X_T$\|$  (relative_state L2 norm)")
            ax_hist.set_ylabel("count")
            ax_hist.set_title(
                f"End-state norm histogram  (N={total_count})"
            )
            ax_hist.grid(alpha=0.2)
            ax_hist.legend(loc="best")
            fig_hist.tight_layout()
            media["visitation_critic/end_state_norm_histogram"] = fig_hist

        # Dataset overlay plot: 1x2 (roll and pitch), start vs end distributions.
        start_overlay = start_cpu
        end_overlay = end_cpu
        if start_overlay.shape[0] > 0:
            overlay_count = max(self.min_scatter_states, 500)
            s_idx = torch.randint(0, start_overlay.shape[0], (overlay_count,))
            e_idx = torch.randint(0, end_overlay.shape[0], (overlay_count,))
            start_overlay = start_overlay[s_idx]
            end_overlay = end_overlay[e_idx]
            start_rp_overlay = self._state_roll_pitch(start_overlay)
            end_rp_overlay = self._state_roll_pitch(end_overlay)
            fig_overlay, ax_overlay = plt.subplots(1, 2, figsize=(12, 5))
            # Roll phase space
            ax_overlay[0].scatter(
                start_rp_overlay[:, 0].numpy(),
                start_overlay[:, roll_rate_idx].numpy(),
                alpha=0.25,
                s=7,
                color="royalblue",
                label="start state",
            )
            ax_overlay[0].scatter(
                end_rp_overlay[:, 0].numpy(),
                end_overlay[:, roll_rate_idx].numpy(),
                alpha=0.25,
                s=7,
                color="crimson",
                label="end state",
            )
            ax_overlay[0].set_title("Roll phase space: start vs end")
            ax_overlay[0].set_xlabel(r"$\phi$ (rad)")
            ax_overlay[0].set_ylabel(r"$\dot{\phi}$ (rad/s)")
            ax_overlay[0].grid(alpha=0.2)
            ax_overlay[0].legend(loc="best")
            # Pitch phase space
            ax_overlay[1].scatter(
                start_rp_overlay[:, 1].numpy(),
                start_overlay[:, pitch_rate_idx].numpy(),
                alpha=0.25,
                s=7,
                color="royalblue",
                label="start state",
            )
            ax_overlay[1].scatter(
                end_rp_overlay[:, 1].numpy(),
                end_overlay[:, pitch_rate_idx].numpy(),
                alpha=0.25,
                s=7,
                color="crimson",
                label="end state",
            )
            ax_overlay[1].set_title("Pitch phase space: start vs end")
            ax_overlay[1].set_xlabel(r"$\theta$ (rad)")
            ax_overlay[1].set_ylabel(r"$\dot{\theta}$ (rad/s)")
            ax_overlay[1].grid(alpha=0.2)
            ax_overlay[1].legend(loc="best")
            fig_overlay.suptitle("Trajectory Dataset Start/End Overlay")
            media["visitation_critic/dataset_start_end_overlay"] = fig_overlay

        # Full CFM inner-loop loss curve for this policy iteration.
        if cfm_loss_curve:
            fig_curve, ax_curve = plt.subplots(figsize=(7, 4))
            ax_curve.plot(cfm_loss_curve, color="royalblue", linewidth=1.5)
            ax_curve.set_title("CFM Inner-Loop Loss Curve")
            ax_curve.set_xlabel("CFM optimization step")
            ax_curve.set_ylabel("MSE loss")
            ax_curve.grid(alpha=0.25)
            media["visitation_critic/cfm_loss_curve"] = fig_curve

        # Discrete class-conditioned generated state scatter: 1x2 grid, start states only.
        if self.conditioning_type == "discrete":
            num_classes = int(self.cfg.get("num_classes", 2))
            fig_cls, ax_cls = plt.subplots(1, 2, figsize=(12, 5))
            for cls in range(num_classes):
                samples = self.generate_states_for_label(
                    max(self.generated_states_per_class, 500), cls
                ).detach().cpu()
                samples_rp = self._state_roll_pitch(samples)
                class_label = f"{class_names[cls]} ({cls})" if cls < len(class_names) else f"class_{cls}"
                color = colors(cls % 10)
                ax_cls[0].scatter(
                    samples_rp[:, 0].numpy(),
                    samples[:, roll_rate_idx].numpy(),
                    alpha=0.25,
                    s=6,
                    color=color,
                    label=class_label,
                )
                ax_cls[1].scatter(
                    samples_rp[:, 1].numpy(),
                    samples[:, pitch_rate_idx].numpy(),
                    alpha=0.25,
                    s=6,
                    color=color,
                    label=class_label,
                )
            ax_cls[0].set_title("Roll phase, generated start state")
            ax_cls[0].set_xlabel(r"$\phi$ (rad)")
            ax_cls[0].set_ylabel(r"$\dot{\phi}$ (rad/s)")
            ax_cls[1].set_title("Pitch phase, generated start state")
            ax_cls[1].set_xlabel(r"$\theta$ (rad)")
            ax_cls[1].set_ylabel(r"$\dot{\theta}$ (rad/s)")
            for ax in ax_cls:
                ax.grid(alpha=0.2)
                ax.legend(loc="best")
            fig_cls.suptitle("Generated Start States by Class")
            media["visitation_critic/generated_states_by_class"] = fig_cls

        return media

    # --- Persistence ---

    def save(self) -> dict:
        """Return state dict for checkpointing."""
        return {
            "vc_model_state": self.model.state_dict(),
            "vc_optimizer_state": self.optimizer.state_dict(),
            "vc_trained": self._trained,
        }

    def load(self, state_dict: dict) -> None:
        """Restore from checkpoint."""
        if "vc_model_state" in state_dict:
            self.model.load_state_dict(state_dict["vc_model_state"])
        if "vc_optimizer_state" in state_dict:
            self.optimizer.load_state_dict(state_dict["vc_optimizer_state"])
        self._trained = state_dict.get("vc_trained", False)


def resolve_visitation_critic_config(
    vc_cfg: dict, obs: TensorDict, obs_groups: dict[str, list[str]], env: VecEnv
) -> dict:
    """Resolve the visitation critic configuration.

    Computes ``state_dim`` from the resolved ``obs_groups["relative_state"]`` groups,
    validates that each group contains 1D observations, and stores the resolved
    ``obs_groups`` dict into ``vc_cfg`` for use inside :class:`VisitationCritic`.

    Args:
        vc_cfg: Visitation critic configuration dictionary.
        obs: Observation dictionary from the environment.
        obs_groups: Resolved observation groups dictionary (must contain ``"relative_state"``).
        env: Environment object (unused currently, kept for API symmetry with resolve_rnd_config).

    Returns:
        The updated visitation critic configuration dictionary.

    Raises:
        ValueError: If any observation group in ``relative_state`` is not 1D.
        ValueError: If ``relative_state`` resolved via the ``"policy"`` fallback, which is
            not allowed — VC must operate on true simulator relative state.
    """
    if vc_cfg is None:
        return vc_cfg

    # Guard against accidental policy-obs fallback.
    if obs_groups["relative_state"] == ["policy"]:
        raise ValueError(
            "The visitation critic 'relative_state' observation group resolved to ['policy'], "
            "which is not allowed. The environment must expose a dedicated 'relative_state' "
            "observation group containing true simulator relative state "
            "([differentiate_qpos(qpos, qpos_ref), qvel - qvel_ref])."
        )

    relative_state_groups = obs_groups["relative_state"]

    # Compute state_dim from the resolved groups.
    state_dim = 0
    for obs_group in relative_state_groups:
        if len(obs[obs_group].shape) != 2:
            raise ValueError(
                f"The visitation critic only supports 1D observations, "
                f"got shape {obs[obs_group].shape} for '{obs_group}'."
            )
        state_dim += obs[obs_group].shape[-1]

    if len(relative_state_groups) == 1:
        if state_dim % 2 != 0:
            raise ValueError(
                "The visitation critic expected an even relative_state dimension when a single "
                f"concatenated group is used, but got state_dim={state_dim}."
            )
        rel_qpos_dim = state_dim // 2
    else:
        rel_qpos_dim = obs[relative_state_groups[0]].shape[-1]

    vc_cfg["state_dim"] = state_dim
    vc_cfg["rel_qpos_dim"] = rel_qpos_dim
    vc_cfg["obs_groups"] = obs_groups

    return vc_cfg
