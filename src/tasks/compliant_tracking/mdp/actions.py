"""Action terms for the compliant tracking task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from .commands import CompliantMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ForcefieldWrenchAction(ActionTerm):
  """Cache the released body-local wrench once per policy step.

  IsaacLab interval events run outside the decimation loop. Its .005s
  interval therefore evaluates at50Hz in this task, then retains a local
  wrench over four physics steps. MuJoCo accepts world-frame wrenches, so
  each substep only rotates the cached local vector with the moving body.
  """

  cfg: ForcefieldWrenchActionCfg

  def __init__(self, cfg: ForcefieldWrenchActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._raw_action = torch.zeros(env.num_envs, 0, device=env.device)
    self._forces = torch.zeros(
      env.num_envs, self._entity.num_bodies, 3, device=env.device
    )
    self._torques = torch.zeros_like(self._forces)
    self._was_enabled = False
    self._local_forces = torch.zeros_like(self._forces)
    self._local_torques = torch.zeros_like(self._forces)

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_action

  def process_actions(self, actions: torch.Tensor) -> None:
    del actions

  def cache_wrench(self) -> None:
    """Called after command updates by a policy-step event, as in IsaacLab."""
    command = cast(CompliantMotionCommand, self._env.command_manager.get_term(self.cfg.command_name))
    self._local_forces.zero_()
    self._local_torques.zero_()
    if not command.dataset_forces_enabled:
      return
    active = command.active_force_mask & (command.force_body_indexes >= 0)
    if bool(active.any()):
      body_idx = command.force_body_indexes[active]
      quat = self._entity.data.body_link_quat_w[active, body_idx]
      self._local_forces[active, body_idx] = quat_apply_inverse(quat, command.forcefield_forces_w[active])
      self._local_torques[active, body_idx] = quat_apply_inverse(quat, command.forcefield_torques_w[active])

  def apply_actions(self) -> None:
    command = cast(
      CompliantMotionCommand,
      self._env.command_manager.get_term(self.cfg.command_name),
    )
    if not command.dataset_forces_enabled:
      if self._was_enabled:
        # xfrc_applied persists: relinquishing ownership must clear our load.
        self._forces.zero_()
        self._torques.zero_()
        self._entity.write_external_wrench_to_sim(self._forces, self._torques)
      self._was_enabled = False
      return
    self._was_enabled = True

    # Only transform the cached wrench here: never recompute the spring.
    quat = self._entity.data.body_link_quat_w
    self._forces.copy_(quat_apply(quat, self._local_forces))
    self._torques.copy_(quat_apply(quat, self._local_torques))
    self._entity.write_external_wrench_to_sim(self._forces, self._torques)


@dataclass(kw_only=True)
class ForcefieldWrenchActionCfg(ActionTermCfg):
  """Configuration for the cached body-local forcefield wrench."""

  command_name: str = "motion"

  def build(self, env: ManagerBasedRlEnv) -> ForcefieldWrenchAction:
    return ForcefieldWrenchAction(self, env)
