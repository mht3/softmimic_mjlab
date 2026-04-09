from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


class PushVelocityStage(TypedDict):
  step: int
  velocity_range: dict[str, tuple[float, float]]


def push_velocity_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_stages: list[PushVelocityStage],
) -> dict[str, torch.Tensor]:
  """Increase push_robot velocity ranges at configured training step thresholds."""
  del env_ids  # Unused.
  event_term_cfg = env.event_manager.get_term_cfg("push_robot")
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      event_term_cfg.params["velocity_range"].update(stage["velocity_range"])
  vel_range = event_term_cfg.params["velocity_range"]
  out: dict[str, torch.Tensor] = {}
  for axis, bounds in vel_range.items():
    hi = float(bounds[1])
    if axis in ("x", "y", "z"):
      prefix = f"push_vel_{axis}"
    else:
      prefix = f"push_{axis}"
    out[f"{prefix}_max"] = torch.tensor(hi)
  return out


class ResetNoiseStage(TypedDict):
  step: int
  joint_range: float
  lin_vel_range: float
  z_vel_range: float
  ang_vel_range: float
  pos_z_range: float
  rp_range: float


def reset_noise_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  noise_stages: list[ResetNoiseStage],
) -> dict[str, torch.Tensor]:
  """Increase reset init noise ranges at configured training step thresholds.

  Updates reset_base velocity_range and pose_range["z"], reset_robot_joints
  position_range, and reset_rp_noise rp_range in sync with push curriculum.
  """
  del env_ids
  active = noise_stages[0]
  for stage in noise_stages:
    if env.common_step_counter > stage["step"]:
      active = stage
  lv = active["lin_vel_range"]
  zv = active["z_vel_range"]
  av = active["ang_vel_range"]
  pz = active["pos_z_range"]
  jr = active["joint_range"]
  rp = active["rp_range"]

  base_cfg = env.event_manager.get_term_cfg("reset_base")
  base_cfg.params["velocity_range"] = {
    "x": (-lv, lv), "y": (-lv, lv), "z": (-zv, zv),
    "roll": (-av, av), "pitch": (-av, av), "yaw": (-av, av),
  }
  base_cfg.params["pose_range"]["z"] = (-pz, pz)
  env.event_manager.get_term_cfg("reset_robot_joints").params["position_range"] = (-jr, jr)
  env.event_manager.get_term_cfg("reset_rp_noise").params["rp_range"] = rp
  return {
    "init_lin_vel_max": torch.tensor(lv),
    "init_rp_max": torch.tensor(rp),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])
