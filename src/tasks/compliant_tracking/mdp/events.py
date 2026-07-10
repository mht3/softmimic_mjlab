from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import CompliantMotionCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def apply_forcefield_wrench(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Apply the reactive forcefield wrench to the per-env force body.

  mjlab analog of SoftMimic's ``apply_forcefield_force_torque``: the command
  term computes the world-frame forcefield force/torque acting on the force
  link and this event writes it as an external wrench each step. Wrenches
  persist in ``xfrc_applied`` until the next write, so all bodies are
  rewritten every call (inactive envs get zeros). Use with ``mode="step"``.

  In interactive play the command exposes ``dataset_forces_enabled`` (a GUI
  toggle, off by default); when off this is a no-op so the GUI Force/Push
  panels own ``xfrc_applied``. Step events run after the command manager, so
  when on, the dataset wrench takes precedence over those panels' per-step
  clear — reproducing exactly what the policy saw during training.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))

  if not command.dataset_forces_enabled:
    return

  forces = torch.zeros(env.num_envs, asset.num_bodies, 3, device=env.device)
  torques = torch.zeros_like(forces)

  active = command.active_force_mask & (command.force_body_indexes >= 0)
  if bool(active.any()):
    body_idx = command.force_body_indexes[active]
    forces[active, body_idx] = command.forcefield_forces_w[active]
    torques[active, body_idx] = command.forcefield_torques_w[active]

  asset.write_external_wrench_to_sim(forces, torques)
