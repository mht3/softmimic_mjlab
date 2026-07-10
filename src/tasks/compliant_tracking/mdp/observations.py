from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .commands import CompliantMotionCommand, MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


##
# Velocity-conditioning observations (actor + critic). These expose the
# reference root velocity in the reference heading frame so the policy learns
# to associate a commanded velocity with the gait; the GUI joystick overrides
# them at play time to steer the robot.
##


def reference_root_lin_vel_b(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Reference root linear velocity (heading frame). xy is the steering command."""
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  return command.reference_root_lin_vel_b[:, :2]


def reference_root_ang_vel_b(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Reference root yaw rate (heading frame). The steering yaw command."""
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  return command.reference_root_ang_vel_b[:, 2:3]


##
# SoftMimic compliance observations (critic-only; the actor keeps the
# deployable tracking observation layout).
##


def forcefield_force_applied(
  env: ManagerBasedRlEnv, command_name: str, scale: float = 1.0 / 50.0
) -> torch.Tensor:
  """Reactive forcefield force on the force body, world frame."""
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  return command.forcefield_forces_w * scale


def forcefield_torque_applied(
  env: ManagerBasedRlEnv, command_name: str, scale: float = 1.0 / 5.0
) -> torch.Tensor:
  """Reactive forcefield torque on the force body, world frame."""
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  return command.forcefield_torques_w * scale


def forcefield_force_desired(
  env: ManagerBasedRlEnv, command_name: str, scale: float = 1.0 / 50.0
) -> torch.Tensor:
  """Feedforward force target from the dataset, world frame."""
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  return command.target_forces_w * scale


def forcefield_torque_desired(
  env: ManagerBasedRlEnv, command_name: str, scale: float = 1.0 / 5.0
) -> torch.Tensor:
  """Feedforward torque target from the dataset, world frame."""
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  return command.target_torques_w * scale


def desired_stiffness_log(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  return torch.log(command.desired_stiffness + 1e-6).view(env.num_envs, 1)


def desired_rotational_stiffness_log(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  return torch.log(command.desired_rotational_stiffness + 1e-6).view(env.num_envs, 1)


def forcefield_stiffness_log(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  return torch.log(command.forcefield_stiffness + 1e-6).view(env.num_envs, 1)


def forcefield_rotational_stiffness_log(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  command = cast(CompliantMotionCommand, env.command_manager.get_term(command_name))
  return torch.log(command.forcefield_rotational_stiffness + 1e-6).view(
    env.num_envs, 1
  )
