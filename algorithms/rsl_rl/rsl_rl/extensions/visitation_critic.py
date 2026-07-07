# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visitation Critic for PPO (rsl_rl port of SB3 VCPPO).

Models the density of locally high-value (state, action) regions with a flow-matching
model. At rollout time, a candidate action sampled from the model is alpha-blended
with the policy action; PPO's stored log_prob is recomputed on the blend so the
surrogate stays consistent. The PPO objective is otherwise unchanged.

Reference implementation:
    visitation_critic/algorithms/gymnasium_baselines/vcppo/vcppo.py
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.env import VecEnv


K_NN_FILTER = 16
_CFG_NULL_STATE_VAL = -999.0


def _build_mlp(in_dim: int, out_dim: int, hidden_sizes: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = int(in_dim)
    for h in hidden_sizes:
        layers += [nn.Linear(last, int(h)), nn.SiLU()]
        last = int(h)
    layers += [nn.Linear(last, int(out_dim))]
    return nn.Sequential(*layers)


# =============================================================================
# Flow networks
# =============================================================================


class FlowMLP(nn.Module):
    """Joint flow vector field over concat(state, action) of width state_dim+act_dim."""

    def __init__(self, dim: int, hidden_sizes: Sequence[int] = (128, 128, 128)) -> None:
        super().__init__()
        self.dim = int(dim)
        self.net = _build_mlp(self.dim + 1, self.dim, hidden_sizes)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, tau], dim=-1))


class CFGActionFlowMLP(nn.Module):
    """Conditional vector field for p(a | s). Input: cat(a, tau, s_cond)."""

    def __init__(self, state_dim: int, act_dim: int, hidden_sizes: Sequence[int] = (128, 128, 128)) -> None:
        super().__init__()
        self.net = _build_mlp(int(act_dim) + 1 + int(state_dim), int(act_dim), hidden_sizes)

    def forward(self, a: torch.Tensor, tau: torch.Tensor, s_cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([a, tau, s_cond], dim=-1))


# =============================================================================
# kNN top-MCQ admission filter
# =============================================================================


def _local_topmcq_mask(
    query_states: torch.Tensor,
    query_mcq: torch.Tensor,
    ref_states: torch.Tensor,
    ref_mcq: torch.Tensor,
    k_nn: int,
    keep_frac: float,
) -> torch.Tensor:
    """Per-row mask: True iff query_mcq is in the top keep_frac percentile of its
    k nearest neighbors in ref (by state-slice L2 distance).

    Mirrors SAFlow._local_topmcq_mask in the SB3 implementation but kept torch-only
    so we don't drag in faiss as a runtime dependency.
    """
    if query_states.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=query_states.device)
    if ref_states.numel() == 0:
        return torch.ones(query_states.shape[0], dtype=torch.bool, device=query_states.device)
    k = max(1, min(int(k_nn), ref_states.shape[0]))
    dists = torch.cdist(query_states, ref_states)  # (Q, R)
    _, nn_idx = dists.topk(k, largest=False, dim=1)
    nbr_mcq = ref_mcq[nn_idx]  # (Q, k)
    # 1 - keep_frac quantile of neighbors: anything at or above this is in the top-keep.
    thresh = torch.quantile(nbr_mcq, 1.0 - keep_frac, dim=1)
    return query_mcq >= thresh


# =============================================================================
# State-action flow model with visitation buffer
# =============================================================================


class SAFlow:
    """Flow-matching model over rows = concat(obs, action)."""

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        device: torch.device,
        lr: float = 1e-3,
        hidden_sizes: Sequence[int] = (128, 128, 128),
        max_buffer: int = 100_000,
        k_nn: int = K_NN_FILTER,
        mcq_percentile: float = 25.0,
        seed: int = 0,
    ) -> None:
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.dim = self.state_dim + self.act_dim
        self.device = torch.device(device)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.model: nn.Module = FlowMLP(dim=self.dim, hidden_sizes=self.hidden_sizes).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=float(lr))
        self.max_buffer = int(max_buffer)
        self.k_nn = int(k_nn)
        self.mcq_keep = float(mcq_percentile) / 100.0
        # Buffer lives on CPU as a single growing tensor; cheaper than a python list
        # of arrays when we have to topk across the whole thing every train call.
        self._buf_sa: torch.Tensor | None = None
        self._buf_mcq: torch.Tensor | None = None
        self.last_loss: float | None = None
        self.last_kept = 0
        self.last_admitted = 0
        self.last_candidate_count = 0
        self._torch_gen = torch.Generator(device=self.device).manual_seed(int(seed))
        # Separate CPU generator for buffer eviction (the buffer lives on CPU).
        self._cpu_gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7777)

    @property
    def buffer_size(self) -> int:
        return 0 if self._buf_sa is None else int(self._buf_sa.shape[0])

    def add(self, sa_batch: torch.Tensor, mcq_batch: torch.Tensor) -> None:
        """Append rows unconditionally. When the buffer overflows, evict uniformly
        at random down to ``max_buffer`` rows. The local top-MCQ filter is applied
        only at train time inside ``train_steps`` — pruning at admission time
        collapses diversity and the flow loses coverage.
        """
        sa_batch = sa_batch.detach().to(torch.float32).cpu().reshape(-1, self.dim)
        mcq_batch = mcq_batch.detach().to(torch.float32).cpu().reshape(-1)
        self.last_candidate_count = int(sa_batch.shape[0])
        self.last_admitted = int(sa_batch.shape[0])
        if sa_batch.shape[0] == 0:
            return
        if self._buf_sa is None:
            self._buf_sa = sa_batch.clone()
            self._buf_mcq = mcq_batch.clone()
        else:
            self._buf_sa = torch.cat([self._buf_sa, sa_batch], dim=0)
            self._buf_mcq = torch.cat([self._buf_mcq, mcq_batch], dim=0)
        n = self._buf_sa.shape[0]
        if n <= self.max_buffer:
            return
        # Uniform-random eviction: pick max_buffer indices to keep without replacement.
        # Preserves older high-quality off-policy rows in expectation; unlike FIFO,
        # which biases the buffer toward the most recent rollouts.
        keep = torch.randperm(n, generator=self._cpu_gen)[: self.max_buffer]
        keep, _ = torch.sort(keep)
        self._buf_sa = self._buf_sa[keep]
        self._buf_mcq = self._buf_mcq[keep]

    def _top_mcq_subset(self) -> torch.Tensor | None:
        """Return the (M', dim) slice of the buffer in the top-MCQ kNN region, or None
        if there isn't enough data yet to train on.
        """
        if self._buf_sa is None or self._buf_sa.shape[0] < 16:
            return None
        states = self._buf_sa[:, : self.state_dim]
        mask = _local_topmcq_mask(states, self._buf_mcq, states, self._buf_mcq, self.k_nn, self.mcq_keep)
        sa_prime = self._buf_sa[mask]
        return sa_prime if sa_prime.shape[0] > 0 else None

    def train_steps(self, n_steps: int = 80, batch_size: int = 256) -> float | None:
        sa_prime = self._top_mcq_subset()
        if sa_prime is None:
            self.last_loss = None
            self.last_kept = self.buffer_size
            return None
        sa_prime = sa_prime.to(self.device)
        m_prime = sa_prime.shape[0]
        losses: list[float] = []
        self.model.train()
        for _ in range(int(n_steps)):
            idx = torch.randint(0, m_prime, (int(batch_size),), device=self.device, generator=self._torch_gen)
            x1 = sa_prime[idx]
            x0 = torch.randn(x1.shape, dtype=x1.dtype, device=self.device, generator=self._torch_gen)
            tau = torch.rand(int(batch_size), 1, device=self.device, generator=self._torch_gen)
            x_tau = (1.0 - tau) * x0 + tau * x1
            target = x1 - x0
            pred = self.model(x_tau, tau)
            loss = ((pred - target) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(float(loss.item()))
        self.last_loss = float(np.mean(losses))
        self.last_kept = self.buffer_size
        return self.last_loss

    def _condition_tensors(
        self, s_obs: torch.Tensor, a_cond: torch.Tensor, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        num_samples = max(1, int(num_samples))
        s_obs = s_obs.to(device=self.device, dtype=torch.float32).reshape(-1, self.state_dim)
        a_cond = a_cond.to(device=self.device, dtype=torch.float32).reshape(-1, self.act_dim)
        x_clean = torch.cat([s_obs, a_cond], dim=-1)
        x_clean_rep = x_clean.repeat_interleave(num_samples, dim=0)
        return x_clean, x_clean_rep, num_samples

    def _euler_integrate(
        self,
        x: torch.Tensor,
        tau_start: float | torch.Tensor,
        tau_end: float | torch.Tensor,
        n_steps: int,
        state_clamp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_steps = max(1, int(n_steps))
        n = x.shape[0]
        if not torch.is_tensor(tau_start):
            tau_start = torch.full((n, 1), float(tau_start), device=self.device)
        if not torch.is_tensor(tau_end):
            tau_end = torch.full((n, 1), float(tau_end), device=self.device)
        dt = (tau_end - tau_start) / n_steps
        for k in range(n_steps):
            tau = tau_start + (float(k) / n_steps) * (tau_end - tau_start)
            if state_clamp is None:
                x = x + dt * self.model(x, tau)
            else:
                x[:, : self.state_dim] = state_clamp
                vel = self.model(x, tau)
                x[:, self.state_dim :] = x[:, self.state_dim :] + dt * vel[:, self.state_dim :]
                x[:, : self.state_dim] = state_clamp
        return x

    @torch.no_grad()
    def sample_sdedit(
        self, s_obs: torch.Tensor, a_cond: torch.Tensor, tau: float = 0.5, num_samples: int = 300, n_steps: int = 50
    ) -> torch.Tensor:
        if not 0.0 <= float(tau) <= 1.0:
            raise ValueError(f"vc_tau must be in [0, 1], got {tau}")
        self.model.eval()
        x_clean, x_clean_rep, num_samples = self._condition_tensors(s_obs, a_cond, num_samples)
        eps = torch.randn(x_clean_rep.shape, dtype=x_clean_rep.dtype, device=self.device, generator=self._torch_gen)
        x = (1.0 - tau) * eps + tau * x_clean_rep
        x = self._euler_integrate(x, tau, 1.0, n_steps)
        return x.reshape(x_clean.shape[0], num_samples, self.dim)

    @torch.no_grad()
    def sample_gode(
        self,
        s_obs: torch.Tensor,
        a_cond: torch.Tensor,
        sigma: float = 0.2,
        tau: float = 0.5,
        num_samples: int = 300,
        n_steps: int = 50,
    ) -> torch.Tensor:
        if not 0.0 <= float(tau) <= 1.0:
            raise ValueError(f"vc_tau must be in [0, 1], got {tau}")
        self.model.eval()
        x_clean, x_clean_rep, num_samples = self._condition_tensors(s_obs, a_cond, num_samples)
        eps = torch.randn(x_clean_rep.shape, dtype=x_clean_rep.dtype, device=self.device, generator=self._torch_gen)
        x = x_clean_rep + sigma * eps
        x = self._euler_integrate(x, tau, 1.0, n_steps)
        return x.reshape(x_clean.shape[0], num_samples, self.dim)

    @torch.no_grad()
    def sample_inpainting(
        self, s_obs: torch.Tensor, a_cond: torch.Tensor, num_samples: int = 300, n_steps: int = 50
    ) -> torch.Tensor:
        self.model.eval()
        x_clean, x_clean_rep, num_samples = self._condition_tensors(s_obs, a_cond, num_samples)
        x = torch.randn(x_clean_rep.shape, dtype=x_clean_rep.dtype, device=self.device, generator=self._torch_gen)
        x = self._euler_integrate(x, 0.0, 1.0, n_steps, state_clamp=x_clean_rep[:, : self.state_dim])
        return x.reshape(x_clean.shape[0], num_samples, self.dim)

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "buf_sa": self._buf_sa,
            "buf_mcq": self._buf_mcq,
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        self._buf_sa = state.get("buf_sa")
        self._buf_mcq = state.get("buf_mcq")


class AFlow(SAFlow):
    """CFG conditional flow model for p(a | s). Inherits SAFlow's buffer."""

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        device: torch.device,
        cfg_dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(state_dim=state_dim, act_dim=act_dim, device=device, **kwargs)
        lr = kwargs.get("lr", 1e-3)
        self.model = CFGActionFlowMLP(
            state_dim=self.state_dim, act_dim=self.act_dim, hidden_sizes=self.hidden_sizes
        ).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=float(lr))
        self.cfg_dropout = float(cfg_dropout)

    def train_steps(self, n_steps: int = 80, batch_size: int = 256) -> float | None:
        # CFG-dropout training; SB3 trains on the whole buffer (no MCQ filter here).
        if self._buf_sa is None or self._buf_sa.shape[0] < 16:
            self.last_loss = None
            self.last_kept = self.buffer_size
            return None
        sa = self._buf_sa.to(self.device)
        m = sa.shape[0]
        null_state = torch.full((self.state_dim,), _CFG_NULL_STATE_VAL, dtype=torch.float32, device=self.device)
        losses: list[float] = []
        self.model.train()
        for _ in range(int(n_steps)):
            idx = torch.randint(0, m, (int(batch_size),), device=self.device, generator=self._torch_gen)
            s1 = sa[idx, : self.state_dim]
            a1 = sa[idx, self.state_dim :]
            dropout_mask = torch.rand(int(batch_size), device=self.device, generator=self._torch_gen) < self.cfg_dropout
            s_cond = s1.clone()
            s_cond[dropout_mask] = null_state
            a0 = torch.randn(a1.shape, dtype=a1.dtype, device=self.device, generator=self._torch_gen)
            tau = torch.rand(int(batch_size), 1, device=self.device, generator=self._torch_gen)
            a_tau = (1.0 - tau) * a0 + tau * a1
            target = a1 - a0
            pred = self.model(a_tau, tau, s_cond)
            loss = ((pred - target) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(float(loss.item()))
        self.last_loss = float(np.mean(losses))
        self.last_kept = self.buffer_size
        return self.last_loss

    @torch.no_grad()
    def sample_cfg(
        self, s_obs: torch.Tensor, num_samples: int = 300, guidance_scale: float = 1.0, n_steps: int = 50
    ) -> torch.Tensor:
        """Returns (n_envs, num_samples, act_dim) — actions only."""
        self.model.eval()
        num_samples = max(1, int(num_samples))
        s_obs = s_obs.to(device=self.device, dtype=torch.float32).reshape(-1, self.state_dim)
        n_envs = s_obs.shape[0]
        s_rep = s_obs.repeat_interleave(num_samples, dim=0)
        null_s = torch.full_like(s_rep, _CFG_NULL_STATE_VAL)
        a = torch.randn(n_envs * num_samples, self.act_dim, device=self.device, generator=self._torch_gen)
        dt = 1.0 / int(n_steps)
        for k in range(int(n_steps)):
            tau = torch.full((n_envs * num_samples, 1), k * dt, device=self.device)
            u_cond = self.model(a, tau, s_rep)
            u_uncond = self.model(a, tau, null_s)
            u = (1.0 - guidance_scale) * u_uncond + guidance_scale * u_cond
            a = a + dt * u
        return a.reshape(n_envs, num_samples, self.act_dim)


# =============================================================================
# Optional side Q-critic (SAC twin-Q)
# =============================================================================


class TwinQNet(nn.Module):
    """Two independent Q heads on (state, action). Min over heads is the conservative estimate."""

    def __init__(self, state_dim: int, act_dim: int, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        hidden = tuple(int(h) for h in hidden)
        self.heads = nn.ModuleList()
        for _ in range(2):
            layers: list[nn.Module] = []
            in_dim = state_dim + act_dim
            for h in hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            self.heads.append(nn.Sequential(*layers))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> list[torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return [head(x) for head in self.heads]


class QCritic:
    """Side SAC-style critic with a torch-tensor ring replay. Not used unless q_mode != "off"."""

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        device: torch.device,
        net_arch: Sequence[int] = (256, 256),
        lr: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        seed: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.q_net = TwinQNet(self.state_dim, self.act_dim, hidden=net_arch).to(self.device)
        self.q_net_target = TwinQNet(self.state_dim, self.act_dim, hidden=net_arch).to(self.device)
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        for p in self.q_net_target.parameters():
            p.requires_grad_(False)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=float(lr))
        # Replay buffer on CPU (typical SB3 replay)
        self._capacity = int(buffer_size)
        self._obs = torch.zeros(self._capacity, self.state_dim, dtype=torch.float32)
        self._next_obs = torch.zeros(self._capacity, self.state_dim, dtype=torch.float32)
        self._actions = torch.zeros(self._capacity, self.act_dim, dtype=torch.float32)
        self._rewards = torch.zeros(self._capacity, 1, dtype=torch.float32)
        self._dones = torch.zeros(self._capacity, 1, dtype=torch.float32)
        self._pos = 0
        self._size = 0
        self._gen = torch.Generator().manual_seed(int(seed) + 13)
        self.last_loss: float | None = None
        self.last_target_q_mean: float | None = None

    def size(self) -> int:
        return int(self._size)

    def add_transition(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        obs = obs.detach().to(torch.float32).cpu().reshape(-1, self.state_dim)
        next_obs = next_obs.detach().to(torch.float32).cpu().reshape(-1, self.state_dim)
        action = action.detach().to(torch.float32).cpu().reshape(-1, self.act_dim)
        reward = reward.detach().to(torch.float32).cpu().reshape(-1, 1)
        done = done.detach().to(torch.float32).cpu().reshape(-1, 1)
        n = obs.shape[0]
        # Wrap-around write
        end = self._pos + n
        if end <= self._capacity:
            sl = slice(self._pos, end)
            self._obs[sl] = obs
            self._next_obs[sl] = next_obs
            self._actions[sl] = action
            self._rewards[sl] = reward
            self._dones[sl] = done
        else:
            first = self._capacity - self._pos
            self._obs[self._pos :] = obs[:first]
            self._next_obs[self._pos :] = next_obs[:first]
            self._actions[self._pos :] = action[:first]
            self._rewards[self._pos :] = reward[:first]
            self._dones[self._pos :] = done[:first]
            rem = n - first
            self._obs[:rem] = obs[first:]
            self._next_obs[:rem] = next_obs[first:]
            self._actions[:rem] = action[first:]
            self._rewards[:rem] = reward[first:]
            self._dones[:rem] = done[first:]
        self._pos = (self._pos + n) % self._capacity
        self._size = min(self._size + n, self._capacity)

    def train_steps(self, actor_mean_fn, n_steps: int = 1) -> float | None:
        """`actor_mean_fn` maps a CPU obs tensor to a CPU action tensor (the policy's
        deterministic mean), evaluated under torch.no_grad. Mirrors SB3 VCQCritic.train_step
        but de-couples the actor type (rsl_rl actors take TensorDicts, not flat tensors).
        """
        if self._size < self.batch_size:
            self.last_loss = None
            return None
        losses: list[float] = []
        target_q_means: list[float] = []
        for _ in range(int(n_steps)):
            idx = torch.randint(0, self._size, (self.batch_size,), generator=self._gen)
            obs_b = self._obs[idx].to(self.device)
            next_obs_b = self._next_obs[idx].to(self.device)
            actions_b = self._actions[idx].to(self.device)
            rewards_b = self._rewards[idx].to(self.device)
            dones_b = self._dones[idx].to(self.device)
            with torch.no_grad():
                next_actions = actor_mean_fn(next_obs_b)
                next_q_heads = self.q_net_target(next_obs_b, next_actions)
                next_q_stack = torch.cat(next_q_heads, dim=1)
                next_q_min, _ = torch.min(next_q_stack, dim=1, keepdim=True)
                target_q = rewards_b + (1.0 - dones_b) * self.gamma * next_q_min
                target_q_means.append(float(target_q.mean().item()))
            current_q_heads = self.q_net(obs_b, actions_b)
            loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q_heads)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                for p, p_targ in zip(self.q_net.parameters(), self.q_net_target.parameters()):
                    p_targ.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            losses.append(float(loss.item()))
        self.last_loss = float(np.mean(losses))
        self.last_target_q_mean = float(np.mean(target_q_means)) if target_q_means else None
        return self.last_loss

    @torch.no_grad()
    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=torch.float32)
        action = action.to(device=self.device, dtype=torch.float32)
        q_heads = self.q_net(obs, action)
        stacked = torch.cat(q_heads, dim=-1)
        q_min, _ = torch.min(stacked, dim=-1)
        return q_min

    def state_dict(self) -> dict:
        return {
            "q_net": self.q_net.state_dict(),
            "q_net_target": self.q_net_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs": self._obs,
            "next_obs": self._next_obs,
            "actions": self._actions,
            "rewards": self._rewards,
            "dones": self._dones,
            "pos": self._pos,
            "size": self._size,
        }

    def load_state_dict(self, state: dict) -> None:
        self.q_net.load_state_dict(state["q_net"])
        self.q_net_target.load_state_dict(state["q_net_target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._obs = state["obs"]
        self._next_obs = state["next_obs"]
        self._actions = state["actions"]
        self._rewards = state["rewards"]
        self._dones = state["dones"]
        self._pos = int(state["pos"])
        self._size = int(state["size"])


# =============================================================================
# Alpha schedule
# =============================================================================


def resolve_alpha_schedule(alpha):
    """Return a callable iter -> alpha. Accepts float, int, or dict spec.

    Dict form: {"alpha": float, "mode": "constant"|"linear_decay", "stop_iter": int}.
    """
    if isinstance(alpha, (int, float)):
        a0 = float(alpha)
        return lambda it: a0
    if isinstance(alpha, dict):
        a0 = float(alpha["alpha"])
        mode = alpha.get("mode", "constant")
        if mode == "constant":
            return lambda it: a0
        if mode == "linear_decay":
            stop = int(alpha["stop_iter"])

            def _sched(it: int) -> float:
                if stop <= 0 or it >= stop:
                    return 0.0
                return a0 * max(0.0, 1.0 - it / stop)

            return _sched
        raise ValueError(f"unknown alpha mode: {mode!r}")
    raise TypeError(f"alpha must be float or dict, got {type(alpha).__name__}")


# =============================================================================
# Top-level orchestrator
# =============================================================================


class VisitationCritic:
    """Owns the flow model, optional Q-critic, per-env episode buffers, and the alpha schedule.

    Iteration-driven (rsl_rl works at iteration granularity; SB3 works at env-step granularity).
    The envelope and schedule below use iteration counts.
    """

    def __init__(
        self,
        cfg: dict,
        state_dim: int,
        act_dim: int,
        obs_groups: dict[str, list[str]],
        num_envs: int,
        device: torch.device | str,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.num_envs = int(num_envs)
        # Reuse the actor's obs groups (concatenated in order).
        self.actor_groups: list[str] = list(obs_groups["actor"])

        self.sample_method = str(cfg.get("sample_method", "sdedit"))
        if self.sample_method not in {"sdedit", "gode", "inpainting", "cfg"}:
            raise ValueError(f"unknown sample_method: {self.sample_method!r}")
        self.tau = float(cfg.get("tau", 0.9))
        self.sigma = float(cfg.get("sigma", 0.2))
        self.guidance_scale = float(cfg.get("guidance_scale", 1.0))
        self.cfg_dropout = float(cfg.get("cfg_dropout", 0.1))
        self.num_samples = int(cfg.get("num_samples", 50))
        self.policy_trust_std = float(cfg.get("policy_trust_std", 3.0))
        self.warmup_iters = int(cfg.get("warmup_iters", 250))
        self.decay_start_iter: int | None = (
            None if cfg.get("decay_start_iter") is None else int(cfg["decay_start_iter"])
        )
        self.stop_iter: int | None = None if cfg.get("stop_iter") is None else int(cfg["stop_iter"])
        self.model_train_steps = int(cfg.get("model_train_steps", 80))
        self.model_train_every = max(1, int(cfg.get("model_train_every", 1)))
        self.model_batch_size = int(cfg.get("model_batch_size", 256))
        self.model_lambda_steps = int(cfg.get("model_lambda_steps", 50))
        self.gamma_mcq = float(cfg.get("gamma_mcq", 0.99))
        self._alpha_schedule = resolve_alpha_schedule(cfg.get("alpha", 0.5))

        flow_kwargs = dict(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            device=self.device,
            lr=float(cfg.get("model_lr", 1e-3)),
            hidden_sizes=tuple(int(h) for h in cfg.get("model_net", (128, 128, 128))),
            max_buffer=int(cfg.get("buffer_size", 100_000)),
            k_nn=int(cfg.get("q_filter_k", K_NN_FILTER)),
            mcq_percentile=float(cfg.get("q_top_fraction", 0.25)) * 100.0,
            seed=int(cfg.get("seed", 0)),
        )
        if self.sample_method == "cfg":
            self.flow: SAFlow = AFlow(cfg_dropout=self.cfg_dropout, **flow_kwargs)
        else:
            self.flow = SAFlow(**flow_kwargs)

        self.q_mode = str(cfg.get("q_mode", "off"))
        if self.q_mode not in {"off", "selection", "analysis"}:
            raise ValueError(f"unknown q_mode: {self.q_mode!r}")
        self.q_train_steps = int(cfg.get("q_train_steps", 300))
        self.q_critic: QCritic | None = None
        if self.q_mode != "off":
            self.q_critic = QCritic(
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                device=self.device,
                net_arch=tuple(int(h) for h in cfg.get("q_net", (256, 256))),
                lr=float(cfg.get("q_lr", 3e-4)),
                buffer_size=int(cfg.get("q_replay_size", 1_000_000)),
                batch_size=int(cfg.get("q_batch_size", 256)),
                tau=float(cfg.get("q_tau", 0.005)),
                gamma=float(cfg.get("gamma", 0.99)),
                seed=int(cfg.get("seed", 0)),
            )

        # Per-env pending-episode buffers (CPU-side python lists of tensors).
        self._pending_obs: list[list[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._pending_actions: list[list[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._pending_rewards: list[list[float]] = [[] for _ in range(self.num_envs)]
        # EMA of true single-step state distances, for the sdedit/gode manifold gate.
        self._step_dist_ema: float | None = None
        self._step_dist_ema_alpha: float = 0.01
        # Latest-step metrics for logging
        self.last_alpha = 0.0
        self.last_blend_zscore = 0.0
        self.last_blend_tail_frac = 0.0
        # Latest action selection diagnostics
        self.last_vc_active_frac = 0.0

    # ------------------------------------------------------------------ helpers
    def _extract_state(self, obs_td: TensorDict) -> torch.Tensor:
        """Concatenate the actor's obs groups into a flat (batch, state_dim) tensor."""
        parts = [obs_td[g] for g in self.actor_groups]
        return torch.cat(parts, dim=-1)

    def alpha(self, iteration: int) -> float:
        """Effective blend coefficient at ``iteration`` — ``base_alpha * ramp * fade``.

        Default schedule (matches SB3 ``_vc_alpha_envelope``): ramp linearly 0 → 1
        over ``warmup_iters``, then **hold at base_alpha forever** — no decay.

        Decay is opt-in: set ``stop_iter`` to enable a trapezoidal envelope that
        linearly fades alpha → 0 between ``decay_start_iter`` (defaults to
        ``stop_iter // 2``) and ``stop_iter``.

        The runner's gate is ``alpha > 0``, so the VC blend branch *is* active
        throughout warmup (as soon as the flow has trained at least once); the
        a_vc contribution is just weighted less in the early iterations.
        """
        base = float(self._alpha_schedule(iteration))
        if self.warmup_iters > 0 and iteration < self.warmup_iters:
            ramp = float(iteration) / float(self.warmup_iters)
        else:
            ramp = 1.0
        # Decay envelope is opt-in: only when stop_iter is explicitly set.
        if self.stop_iter is not None and self.stop_iter > 0:
            decay_start = (
                self.decay_start_iter if self.decay_start_iter is not None else self.stop_iter // 2
            )
            if iteration >= self.stop_iter:
                fade = 0.0
            elif iteration > decay_start and self.stop_iter > decay_start:
                fade = max(0.0, 1.0 - float(iteration - decay_start) / float(self.stop_iter - decay_start))
            else:
                fade = 1.0
        else:
            fade = 1.0
        return float(base * ramp * fade)

    def should_train(self, iteration: int) -> bool:
        # No point training before we have a buffer worth using.
        return iteration % self.model_train_every == 0

    def is_ready(self) -> bool:
        return self.flow.last_loss is not None

    # ------------------------------------------------------------- selection
    @torch.no_grad()
    def vc_action(
        self,
        obs_td: TensorDict,
        a_ppo: torch.Tensor,
        a_mean: torch.Tensor,
        a_std: torch.Tensor,
    ) -> torch.Tensor:
        """Per-env VC action. Falls back to a_ppo for envs with no valid candidate
        (and globally when the flow hasn't trained yet).
        """
        if not self.is_ready():
            self.last_vc_active_frac = 0.0
            return a_ppo
        state = self._extract_state(obs_td)
        n_envs = state.shape[0]
        D, A = self.state_dim, self.act_dim

        if self.sample_method in ("sdedit", "gode"):
            if self.sample_method == "sdedit":
                cands = self.flow.sample_sdedit(
                    state, a_ppo, tau=self.tau, num_samples=self.num_samples, n_steps=self.model_lambda_steps
                )
            else:
                cands = self.flow.sample_gode(
                    state,
                    a_ppo,
                    sigma=self.sigma,
                    tau=self.tau,
                    num_samples=self.num_samples,
                    n_steps=self.model_lambda_steps,
                )
            n_cand = cands.shape[1]
            cand_states = cands[:, :, :D]
            cand_actions = cands[:, :, D : D + A]
            # Manifold gate: candidate state stays within typical single-step distance.
            dists = (cand_states - state[:, None, :]).norm(dim=-1)
            ema = self._step_dist_ema
            in_manifold = dists < float(ema) if (ema is not None and ema > 0.0) else torch.ones_like(dists, dtype=torch.bool)
            # Trust band: action within policy mean ± trust * sigma.
            trust = float(self.policy_trust_std)
            mean_b = a_mean.reshape(n_envs, 1, A)
            std_b = a_std.reshape(n_envs, 1, A).clamp_min(1e-8)
            in_trust = (
                (cand_actions >= mean_b - trust * std_b) & (cand_actions <= mean_b + trust * std_b)
            ).all(dim=-1)
            valid = in_manifold & in_trust
            has_valid = valid.any(dim=1)
            q_ready = (
                self.q_mode == "selection"
                and self.q_critic is not None
                and self.q_critic.size() >= self.q_critic.batch_size
            )
            if q_ready:
                obs_flat = state.to(self.device).reshape(n_envs, -1)
                obs_rep = obs_flat.unsqueeze(1).expand(n_envs, n_cand, -1).reshape(-1, obs_flat.shape[-1])
                cand_flat = cand_actions.reshape(-1, A).to(self.device)
                q_scores = self.q_critic.q_min(obs_rep, cand_flat).reshape(n_envs, n_cand)
                neg_inf = torch.tensor(float("-inf"), device=q_scores.device, dtype=q_scores.dtype)
                selected_idx = torch.where(valid, q_scores, neg_inf).argmax(dim=1)
            else:
                # Pick the candidate whose state is closest to the current state.
                masked_dists = torch.where(valid, dists, torch.full_like(dists, float("inf")))
                selected_idx = masked_dists.argmin(dim=1)
            a_vc = cand_actions[torch.arange(n_envs, device=state.device), selected_idx, :]
            out = torch.where(has_valid[:, None], a_vc, a_ppo)
            self.last_vc_active_frac = float(has_valid.float().mean().item())
            return out

        if self.sample_method == "inpainting":
            cands = self.flow.sample_inpainting(
                state, a_ppo, num_samples=self.num_samples, n_steps=self.model_lambda_steps
            )
            cand_actions = cands[:, :, D : D + A]
        else:  # cfg
            cand_actions = self.flow.sample_cfg(
                state,
                num_samples=self.num_samples,
                guidance_scale=self.guidance_scale,
                n_steps=self.model_lambda_steps,
            )

        n_cand = cand_actions.shape[1]
        # Trust band gate: action must lie within policy mean ± trust * sigma on every dim.
        trust = float(self.policy_trust_std)
        mean_b = a_mean.reshape(n_envs, 1, A)
        std_b = a_std.reshape(n_envs, 1, A).clamp_min(1e-8)
        valid = (
            (cand_actions >= mean_b - trust * std_b) & (cand_actions <= mean_b + trust * std_b)
        ).all(dim=-1)
        has_valid = valid.any(dim=1)
        q_ready = (
            self.q_mode == "selection"
            and self.q_critic is not None
            and self.q_critic.size() >= self.q_critic.batch_size
        )
        if q_ready:
            obs_flat = state.to(self.device).reshape(n_envs, -1)
            obs_rep = obs_flat.unsqueeze(1).expand(n_envs, n_cand, -1).reshape(-1, obs_flat.shape[-1])
            cand_flat = cand_actions.reshape(-1, A).to(self.device)
            q_scores = self.q_critic.q_min(obs_rep, cand_flat).reshape(n_envs, n_cand)
            neg_inf = torch.tensor(float("-inf"), device=q_scores.device, dtype=q_scores.dtype)
            selected_idx = torch.where(valid, q_scores, neg_inf).argmax(dim=1)
        else:
            # Uniform random pick across the trust-band-valid candidates for each env.
            # Envs with no valid candidate are masked out below; the index value stored
            # here for those envs is arbitrary.
            selected_idx = torch.zeros(n_envs, dtype=torch.long, device=cand_actions.device)
            valid_env_ids = torch.nonzero(has_valid, as_tuple=False).flatten()
            for env_i in valid_env_ids.tolist():
                choices = torch.nonzero(valid[env_i], as_tuple=False).flatten()
                pick = torch.randint(choices.shape[0], (1,), device=choices.device)
                selected_idx[env_i] = choices[int(pick.item())]
        a_vc = cand_actions[torch.arange(n_envs, device=cand_actions.device), selected_idx, :]
        # Fallback: envs with zero trust-band-valid candidates use a_ppo instead.
        # Mirrors the new SB3 _select_flow_action behavior.
        out = torch.where(has_valid[:, None], a_vc, a_ppo)
        self.last_vc_active_frac = float(has_valid.float().mean().item())
        return out

    def record_blend_stats(self, a_blend: torch.Tensor, a_mean: torch.Tensor, a_std: torch.Tensor) -> None:
        """Per-(env, dim) z-score of the blended action under the policy distribution."""
        zscore = (a_blend - a_mean).abs() / a_std.clamp_min(1e-8)
        self.last_blend_zscore = float(zscore.mean().item())
        self.last_blend_tail_frac = float((zscore > 3.0).float().mean().item())

    # ------------------------------------------------------------ data ingest
    @torch.no_grad()
    def record_step(
        self,
        obs_td: TensorDict,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs_td: TensorDict,
        done: torch.Tensor,
    ) -> None:
        """Push one rollout step into the per-env pending buffer. On done, complete the episode."""
        state = self._extract_state(obs_td).detach()
        next_state = self._extract_state(next_obs_td).detach()
        action_cpu = action.detach().to(torch.float32).cpu()
        state_cpu = state.detach().to(torch.float32).cpu()
        reward_cpu = reward.detach().to(torch.float32).reshape(-1).cpu()
        done_cpu = done.detach().to(torch.float32).reshape(-1).cpu()

        # Update the manifold-gate EMA from non-done steps (true env transitions only).
        keep = (done.detach().reshape(-1) <= 0).cpu()
        if keep.any():
            step_dists = (next_state.cpu() - state_cpu).norm(dim=-1)
            mean_step = float(step_dists[keep].mean().item())
            if self._step_dist_ema is None:
                self._step_dist_ema = mean_step
            else:
                a = self._step_dist_ema_alpha
                self._step_dist_ema = (1.0 - a) * self._step_dist_ema + a * mean_step

        for env_i in range(self.num_envs):
            self._pending_obs[env_i].append(state_cpu[env_i])
            self._pending_actions[env_i].append(action_cpu[env_i])
            self._pending_rewards[env_i].append(float(reward_cpu[env_i]))

        # Q-critic ingest (raw rewards, pre-bootstrap).
        if self.q_critic is not None:
            self.q_critic.add_transition(state, action, reward, next_state, done)

        # Flush completed episodes into the visitation buffer.
        done_idx = (done_cpu > 0).nonzero(as_tuple=True)[0].tolist()
        if done_idx:
            sa_rows: list[torch.Tensor] = []
            mcq_vals: list[float] = []
            for env_i in done_idx:
                obs_seq = self._pending_obs[env_i]
                act_seq = self._pending_actions[env_i]
                r_seq = self._pending_rewards[env_i]
                if len(r_seq) > 0:
                    # Reverse MC return-to-go.
                    g = 0.0
                    running: list[float] = [0.0] * len(r_seq)
                    for t in range(len(r_seq) - 1, -1, -1):
                        g = float(r_seq[t]) + self.gamma_mcq * g
                        running[t] = g
                    sa = torch.stack(
                        [torch.cat([obs_seq[t], act_seq[t]], dim=-1) for t in range(len(r_seq))],
                        dim=0,
                    )
                    sa_rows.append(sa)
                    mcq_vals.extend(running)
                self._pending_obs[env_i] = []
                self._pending_actions[env_i] = []
                self._pending_rewards[env_i] = []
            if sa_rows:
                self.flow.add(torch.cat(sa_rows, dim=0), torch.tensor(mcq_vals, dtype=torch.float32))

    # ------------------------------------------------------------------ train
    def train(self, actor_mean_fn=None) -> dict[str, float]:
        """Train the flow (and optionally the Q-critic). actor_mean_fn is called with
        a (B, state_dim) tensor and must return (B, act_dim); only required when q_mode != "off".
        """
        loss_dict: dict[str, float] = {}
        flow_loss = self.flow.train_steps(n_steps=self.model_train_steps, batch_size=self.model_batch_size)
        if flow_loss is not None:
            loss_dict["vc/flow_loss"] = flow_loss
        loss_dict["vc/buffer_size"] = float(self.flow.buffer_size)
        loss_dict["vc/last_alpha"] = self.last_alpha
        loss_dict["vc/blend_zscore_mean"] = self.last_blend_zscore
        loss_dict["vc/blend_zscore_tail_frac"] = self.last_blend_tail_frac
        loss_dict["vc/vc_active_frac"] = self.last_vc_active_frac
        if self.q_critic is not None and actor_mean_fn is not None:
            q_loss = self.q_critic.train_steps(actor_mean_fn, n_steps=self.q_train_steps)
            if q_loss is not None:
                loss_dict["vc/q_loss"] = q_loss
        if self._step_dist_ema is not None:
            loss_dict["vc/step_dist_ema"] = float(self._step_dist_ema)
        return loss_dict

    # ----------------------------------------------------------------- persist
    def save(self) -> dict:
        out = {"vc_flow": self.flow.state_dict()}
        if self.q_critic is not None:
            out["vc_q_critic"] = self.q_critic.state_dict()
        out["vc_step_dist_ema"] = self._step_dist_ema
        return out

    def load(self, state: dict) -> None:
        if "vc_flow" in state:
            self.flow.load_state_dict(state["vc_flow"])
        if self.q_critic is not None and "vc_q_critic" in state:
            self.q_critic.load_state_dict(state["vc_q_critic"])
        self._step_dist_ema = state.get("vc_step_dist_ema")


def resolve_visitation_critic_config(
    alg_cfg: dict,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    env: VecEnv,
) -> dict:
    """Fill in state_dim / act_dim / obs_groups / num_envs on the VC config.

    Called from PPO.construct_algorithm when VC is enabled. The VC reuses the actor
    obs group as its state input; we compute state_dim by summing the actor group widths.
    """
    cfg = alg_cfg.get("visitation_critic_cfg") or {}
    state_dim = 0
    for g in obs_groups["actor"]:
        shape = obs[g].shape
        if len(shape) != 2:
            raise ValueError(
                f"VisitationCritic only supports 1D actor obs groups, got shape {shape} for '{g}'."
            )
        state_dim += int(shape[-1])
    cfg["state_dim"] = state_dim
    cfg["act_dim"] = int(env.num_actions)
    cfg["obs_groups"] = obs_groups
    cfg["num_envs"] = int(env.num_envs)
    cfg.setdefault("gamma", float(alg_cfg.get("gamma", 0.99)))
    alg_cfg["visitation_critic_cfg"] = cfg
    return alg_cfg
