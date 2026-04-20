# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from tensordict import TensorDict


class VecEnv(ABC):
    """Abstract class for a vectorized environment.

    The vectorized environment is a collection of environments that are synchronized. This means that the same type of
    action is applied to all environments and the same type of observation is returned from all environments.
    """

    num_envs: int
    """Number of environments."""

    num_actions: int
    """Number of actions."""

    max_episode_length: int | torch.Tensor
    """Maximum episode length.

    The maximum episode length can be a scalar or a tensor. If it is a scalar, it is the same for all environments.
    If it is a tensor, it is the maximum episode length for each environment. This is useful for dynamic episode
    lengths.
    """

    episode_length_buf: torch.Tensor
    """Buffer for current episode lengths."""

    device: torch.device | str
    """Device to use."""

    cfg: dict | object
    """Configuration object."""

    @abstractmethod
    def get_observations(self) -> TensorDict:
        """Return the current observations.

        Returns:
            The observations from the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Apply input action to the environment.

        Args:
            actions: Input actions to apply. Shape: (num_envs, num_actions)

        Returns:
            observations: Observations from the environment.
            rewards: Rewards from the environment. Shape: (num_envs,)
            dones: Done flags from the environment. Shape: (num_envs,)
            extras: Extra information from the environment.

        Observations:
            The observations TensorDict usually contains multiple observation groups. The `obs_groups`
            dictionary of the runner configuration specifies which observation groups are used for which
            purpose, i.e., it maps from required observation sets (e.g. actor) to lists of observation groups.
            The observation sets (keys of the `obs_groups` dictionary) currently used by rsl_rl are:

            - "actor": Specified observation groups are used as input to the actor model.
            - "critic": Specified observation groups are used as input to the critic model.
            - "student": Specified observation groups are used as input to the student model.
            - "teacher": Specified observation groups are used as input to the teacher model.
            - "rnd_state": Specified observation groups are used as input to the RND extension.
            - "relative_state": Specified observation groups are used as input to the visitation critic extension.

            Incomplete or incorrect configurations are handled in the `resolve_obs_groups()` function in
            `rsl_rl/utils/utils.py`, which provides detailed information on the expected configuration.

        Extras:
            The extras dictionary includes metrics such as the episode reward, episode length, etc. The following
            dictionary keys are used by rsl_rl:

            - "time_outs" (torch.Tensor): Timeouts for the environments. These correspond to terminations that
               happen due to time limits and not due to the environment reaching a terminal state. This is useful
               for environments that have a fixed episode length.

            - "log" (dict[str, float | torch.Tensor]): Additional information for logging and debugging purposes.
               The key should be a string and start with "/" for namespacing. The value can be a scalar or a
               tensor. If it is a tensor, the mean of the tensor is used for logging.
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_reset_states(self, env_ids: torch.Tensor, states: torch.Tensor) -> torch.Tensor | None:
        """Set custom initial states for specified environments on their next reset.

        This is an optional method used by the visitation critic to override the default
        reset behavior with CFM-generated states. Environments that do not support custom
        resets can leave this as a no-op.

        The ``states`` tensor is in **tangent-space** relative coordinates with dimension
        ``2 * nv`` (e.g., 58D for a 30-DOF humanoid). The first ``nv`` entries are the output
        of ``differentiate_qpos(qpos, qpos_ref)`` and the remaining ``nv`` entries are
        ``qvel - qvel_ref``. The implementation is responsible for converting back to absolute
        coordinates before writing into the simulator, e.g.::

            from rsl_rl.utils.qpos import integrate_qpos

            nv = model.nv  # e.g., 29 for humanoid
            rel_qpos, rel_qvel = states[:, :nv], states[:, nv:]
            abs_qpos = integrate_qpos(self.qpos_ref, rel_qpos)  # handles quaternion
            abs_qvel = rel_qvel + self.qvel_ref
            # write abs_qpos, abs_qvel into the simulator for env_ids

        Args:
            env_ids: Indices of environments to set reset states for. Shape: (num_resets,)
            states: Tangent-space relative state tensors, shape (num_resets, 2 * nv).
                Layout: [differentiate_qpos_result(nv), qvel_rel(nv)].

        Returns:
            If custom reset is implemented: absolute ``qpos`` tensor written (or intended),
            shape ``(num_resets, 7 + n_joints)``: env-local root position, root quaternion
            ``wxyz``, joint positions — same layout as ``collect_dataset.py`` ``state_qpos``.
            Otherwise ``None``.
        """
        raise NotImplementedError
