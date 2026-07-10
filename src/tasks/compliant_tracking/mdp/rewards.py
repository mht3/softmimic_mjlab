from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_error_magnitude

from .commands import CompliantMotionCommand, MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


##
# SoftMimic soft (compliance) reward terms.
##


def force_command_tracking(
  env: ManagerBasedRlEnv, command_name: str, sigma: float = 20.0
) -> torch.Tensor:
  """Reward matching the applied forcefield force to the dataset target."""
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  error = torch.norm(command.target_forces_w - command.forcefield_forces_w, dim=-1)
  return torch.exp(-(error**2) / sigma**2)


def torque_command_tracking(
  env: ManagerBasedRlEnv, command_name: str, sigma: float = 2.0
) -> torch.Tensor:
  """Reward matching the applied forcefield torque to the dataset target."""
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  error = torch.norm(command.target_torques_w - command.forcefield_torques_w, dim=-1)
  return torch.exp(-(error**2) / sigma**2)


def force_link_position_tracking_exp(
  env: ManagerBasedRlEnv, command_name: str, sigma: float = 0.1
) -> torch.Tensor:
  """Reward tracking the adapted position of the force-target body.

  Uses the same anchored relative frame as the body tracking rewards. Returns
  1 for environments without an active force event.
  """
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  body_idx = command.force_body_indexes.clamp(min=0)
  env_ids = torch.arange(env.num_envs, device=env.device)
  robot_pos = command.robot.data.body_link_pos_w[env_ids, body_idx]
  error = torch.norm(command.force_body_pos_relative_w - robot_pos, dim=-1)
  reward = torch.exp(-(error**2) / sigma**2)
  reward[~command.active_force_mask] = 1.0
  return reward


def force_link_orientation_tracking_exp(
  env: ManagerBasedRlEnv, command_name: str, sigma: float = 0.1
) -> torch.Tensor:
  """Reward tracking the adapted orientation of the force-target body."""
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  body_idx = command.force_body_indexes.clamp(min=0)
  env_ids = torch.arange(env.num_envs, device=env.device)
  robot_quat = command.robot.data.body_link_quat_w[env_ids, body_idx]
  error = quat_error_magnitude(command.force_body_quat_relative_w, robot_quat)
  reward = torch.exp(-(error**2) / sigma**2)
  reward[~command.active_force_mask] = 1.0
  return reward


##
# SoftMimic base tracking / stability terms (paper Table VI).
##


def motion_base_gravity_error_exp(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Reward aligning the base's projected gravity with the reference's.

  SoftMimic's ``target_orientation_exp`` (gravity_exp): yaw-invariant roll/
  pitch tracking of the pelvis, kernel ``exp(-sin(angle))``.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
  ref_gravity_b = quat_apply_inverse(command.body_quat_w[:, 0], gravity)
  robot_gravity_b = quat_apply_inverse(command.robot_body_quat_w[:, 0], gravity)
  dot = torch.sum(ref_gravity_b * robot_gravity_b, dim=-1).clamp(-1.0, 1.0)
  return torch.exp(-torch.sin(torch.acos(dot).abs()))


def motion_base_lin_vel_local_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float = 0.5
) -> torch.Tensor:
  """Reward base linear velocity tracking in each body's own frame.

  SoftMimic's ``track_lin_vel_local``: the reference velocity expressed in the
  reference base frame is compared with the robot velocity in the robot base
  frame, making the term yaw-drift invariant. Body 0 is the pelvis.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  ref_vel_b = quat_apply_inverse(
    command.body_quat_w[:, 0], command.body_lin_vel_w[:, 0]
  )
  robot_vel_b = quat_apply_inverse(
    command.robot_body_quat_w[:, 0], command.robot_body_lin_vel_w[:, 0]
  )
  error = torch.sum(torch.square(ref_vel_b - robot_vel_b), dim=-1)
  return torch.exp(-error / std**2)


def motion_base_ang_vel_local_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float = 2.0
) -> torch.Tensor:
  """Reward base angular velocity tracking in each body's own frame."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  ref_vel_b = quat_apply_inverse(
    command.body_quat_w[:, 0], command.body_ang_vel_w[:, 0]
  )
  robot_vel_b = quat_apply_inverse(
    command.robot_body_quat_w[:, 0], command.robot_body_ang_vel_w[:, 0]
  )
  error = torch.sum(torch.square(ref_vel_b - robot_vel_b), dim=-1)
  return torch.exp(-error / std**2)


def feet_slide_proportional(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
) -> torch.Tensor:
  """Penalize foot sliding proportionally to the contact force.

  SoftMimic's stance-foot stability term: sum over feet of the planar foot
  velocity magnitude weighted by the contact force magnitude.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  contact_force = torch.norm(sensor.data.force, dim=-1)  # [B, num_feet]
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  foot_vel_xy = command.robot.data.body_link_lin_vel_w[
    :, command.foot_body_indexes, :2
  ]
  return torch.sum(foot_vel_xy.norm(dim=-1) * contact_force, dim=1)


def foot_contact_schedule_mismatch(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize feet whose contact state deviates from the reference schedule.

  SoftMimic's ``joint_deviation_from_command_contacts_prob``: squared error
  between the actual (boolean) foot contact and the reference contact
  probability from the motion data, summed over both feet.
  """
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  contacts = (sensor.data.found.view(env.num_envs, -1) > 0).float()  # [B, num_feet]
  desired = command.desired_foot_contacts  # [B, num_feet]
  return torch.sum(torch.square(contacts - desired), dim=1)
