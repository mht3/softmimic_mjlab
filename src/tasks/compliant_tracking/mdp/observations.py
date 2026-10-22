from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_mul,
  quat_apply,
  quat_inv,
  yaw_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from .commands import CompliantMotionCommand, MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str, nominal: bool = False) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    (command.motion.nominal_body_pos_w[command.time_steps, command.motion_anchor_body_index]
     + env.scene.env_origins) if nominal else command.anchor_pos_w,
    command.motion.nominal_body_quat_w[command.time_steps, command.motion_anchor_body_index]
    if nominal else command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str, nominal: bool = False) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    (command.motion.nominal_body_pos_w[command.time_steps, command.motion_anchor_body_index]
     + env.scene.env_origins) if nominal else command.anchor_pos_w,
    command.motion.nominal_body_quat_w[command.time_steps, command.motion_anchor_body_index]
    if nominal else command.anchor_quat_w,
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


def reference_frame_index(command, step=0):
  return command.future_time_steps((step,))[:, 0]


def aligned_reference_body_targets(env, command, step=0):
  """Align to the LIVE pelvis XY/yaw, preserving reference absolute height.

  Recompute at evaluation time: reward/termination run after physics but before
  command.compute, so cached targets would use the preceding robot pose.
  """
  idx = reference_frame_index(command, step)
  ref_pos = command.motion.body_pos_w[idx] + env.scene.env_origins[:, None, :]
  ref_quat = command.motion.body_quat_w[idx]
  robot_data = command.robot.data
  delta_yaw = quat_mul(yaw_quat(robot_data.root_link_quat_w),
                       quat_inv(yaw_quat(ref_quat[:, 0])))
  root_ref = ref_pos[:, 0].clone(); root_ref[:, 2] = 0.
  root_robot = robot_data.root_link_pos_w.clone(); root_robot[:, 2] = 0.
  delta = delta_yaw[:, None].expand(-1, ref_pos.shape[1], -1)
  target_pos = root_robot[:, None] + quat_apply(delta, ref_pos - root_ref[:, None])
  return target_pos, quat_mul(delta, ref_quat)


def force_link_reference_targets(env, command, step=1):
  """Future adapted link pose transformed by the fixed force-onset frame."""
  idx = reference_frame_index(command, step)
  body = command.motion.force_body_index[idx]
  position = command.motion.force_body_pos_w[idx] + env.scene.env_origins
  position = command._ff_anchor_pos + quat_apply(
    command._ff_anchor_rot, position - command._ff_anchor_ref_pos)
  quaternion = quat_mul(command._ff_anchor_rot, command.motion.force_body_quat_w[idx])
  return body, position, quaternion


def desired_wrench_next(command, rotational=False):
  """Released desired wrench uses horizon1; reactive spring metadata uses0."""
  idx = reference_frame_index(command, 1)
  vector = command.motion.torque_vector if rotational else command.motion.force_vector
  return quat_apply(command._ff_anchor_rot, vector[idx])


# Released SoftMimic reference channels. Nominal and adapted references remain
# separate even for a static standing clip; their values are not interchangeable.
def _reference(env, command_name, field, future_steps=(0,), nominal=True):
  command = env.command_manager.get_term(command_name)
  idx = command.future_time_steps(tuple(future_steps))
  prefix = "nominal_" if nominal else ""
  motion = command.motion
  if field == "joint_pos":
    value = getattr(motion, prefix + "joint_pos")[idx]
  elif field == "foot_contacts":
    value = getattr(motion, prefix + "foot_contacts")[idx]
  else:
    # body arrays are already in the configured keypoint order; pelvis is0.
    if field == "height":
      value = getattr(motion, prefix + "body_pos_w")[idx, 0, 2:3]
    else:
      quat = getattr(motion, prefix + "body_quat_w")[idx, 0]
      if field == "gravity":
        vector = torch.zeros_like(quat[..., :3]); vector[..., 2] = -1.
      else:
        key = "body_lin_vel_w" if field == "xy_vel" else "body_ang_vel_w"
        vector = getattr(motion, prefix + key)[idx, 0]
      value = quat_apply_inverse(quat, vector)
      if field == "xy_vel":
        value = value[..., :2]
      elif field == "yaw_vel":
        value = value[..., 2:3]
  return value.reshape(env.num_envs, -1)


def reference_dof_pos(env, command_name, future_steps=(0,), nominal=True):
  return _reference(env, command_name, "joint_pos", future_steps, nominal)


def reference_xy_vel(env, command_name, future_steps=(0,), nominal=True):
  return _reference(env, command_name, "xy_vel", future_steps, nominal)


def reference_ang_vel(env, command_name, future_steps=(0,), nominal=True):
  return _reference(env, command_name, "yaw_vel", future_steps, nominal)


def reference_root_height(env, command_name, future_steps=(0,), nominal=True):
  return _reference(env, command_name, "height", future_steps, nominal)


def reference_gravity_vector(env, command_name, future_steps=(0,), nominal=True):
  return _reference(env, command_name, "gravity", future_steps, nominal)


def reference_foot_contacts(env, command_name, future_steps=(0,), nominal=True):
  return _reference(env, command_name, "foot_contacts", future_steps, nominal)


def _hand_wrench(env, command_name, attribute, scale):
  """Two body-local hand slots, preserving force location information."""
  command = env.command_manager.get_term(command_name)
  body_idx = command.force_body_indexes
  rows = torch.arange(env.num_envs, device=env.device)
  quat = command.robot.data.body_link_quat_w[rows, body_idx.clamp(min=0)]
  if attribute == "target_forces_w":
    vector = desired_wrench_next(command)
  elif attribute == "target_torques_w":
    vector = desired_wrench_next(command, rotational=True)
  else:
    vector = getattr(command, attribute)
  local = quat_apply_inverse(quat, vector) * scale
  result = local.new_zeros((env.num_envs, 2, 3))
  for slot, name in enumerate(("left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand")):
    idx = command.robot.body_names.index(name)
    result[:, slot] = torch.where((body_idx == idx)[:, None], local, 0.)
  return result.flatten(1)


def forcefield_force_applied(env, command_name, scale=1. / 50.):
  return _hand_wrench(env, command_name, "forcefield_forces_w", scale)


def forcefield_torque_applied(env, command_name, scale=1. / 5.):
  return _hand_wrench(env, command_name, "forcefield_torques_w", scale)


def forcefield_force_desired(env, command_name, scale=1. / 50.):
  return _hand_wrench(env, command_name, "target_forces_w", scale)


def forcefield_torque_desired(env, command_name, scale=1. / 5.):
  return _hand_wrench(env, command_name, "target_torques_w", scale)


def desired_stiffness_log(env, command_name):
  c = env.command_manager.get_term(command_name)
  return torch.log(c.desired_stiffness + 1e-6).reshape(env.num_envs, 1)


def desired_rotational_stiffness_log(env, command_name):
  c = env.command_manager.get_term(command_name)
  return torch.log(c.desired_rotational_stiffness + 1e-6).reshape(env.num_envs, 1)


def forcefield_stiffness_log(env, command_name):
  c = env.command_manager.get_term(command_name)
  return torch.log(c.forcefield_stiffness + 1e-6).reshape(env.num_envs, 1)


def forcefield_rotational_stiffness_log(env, command_name):
  c = env.command_manager.get_term(command_name)
  return torch.log(c.forcefield_rotational_stiffness + 1e-6).reshape(env.num_envs, 1)


def root_height(env):
  return env.scene["robot"].data.root_link_pos_w[:, 2:3]


def contact_force_magnitudes(env, sensor_name):
  """Maximum force norm over the last three physics substeps per foot."""
  data = env.scene[sensor_name].data
  if data.force_history is not None:
    return data.force_history.norm(dim=-1).amax(dim=2)
  if data.force is None:
    raise ValueError("Foot contact sensor must expose force")
  return data.force.norm(dim=-1)


def foot_contact(env, sensor_name):
  return (contact_force_magnitudes(env, sensor_name) > 1.).float()


def foot_contact_forces(env, sensor_name, scale=1. / 600.):
  return contact_force_magnitudes(env, sensor_name) * scale


def foot_velocities(env, command_name, scale=.05):
  command = env.command_manager.get_term(command_name)
  vel = command.robot.data.body_link_lin_vel_w[:, command.foot_body_indexes]
  return vel.reshape(env.num_envs, -1) * scale


def keypoint_error_local_shifted(env, command_name):
  """Current-minus-adapted keypoint errors in the current pelvis yaw frame."""
  command = env.command_manager.get_term(command_name)
  target, _ = aligned_reference_body_targets(env, command, step=0)
  error = command.robot_body_pos_w - target
  quat = yaw_quat(command.robot.data.root_link_quat_w)[:, None].expand(-1, error.shape[1], -1)
  return quat_apply_inverse(quat, error).reshape(env.num_envs, -1)


def all_adapted_keypoints_pos_error_local(env, command_name):
  # Upstream's historical name actually denotes only the active force link.
  command = env.command_manager.get_term(command_name)
  rows = torch.arange(env.num_envs, device=env.device)
  body, target, _ = force_link_reference_targets(env, command, step=1)
  error = target - command.robot.data.body_link_pos_w[rows, body.clamp(min=0)]
  error = quat_apply_inverse(command._ff_anchor_rot, error)
  return torch.where((command.active_force_mask & (body >= 0))[:, None], error, 0.)


def all_adapted_keypoints_quat_error_local(env, command_name):
  command = env.command_manager.get_term(command_name)
  rows = torch.arange(env.num_envs, device=env.device)
  body, _, target = force_link_reference_targets(env, command, step=1)
  quat = command.robot.data.body_link_quat_w[rows, body.clamp(min=0)]
  error = quat_mul(target, quat_inv(quat))
  error = torch.where(error[:, :1] < 0., -error, error)
  # Release uses 2*quaternion-vector, not exact axis-angle. Guard invalid IDs.
  return torch.where((body >= 0)[:, None], 2. * error[:, 1:], 0.)


# Compatibility aliases for analysis scripts that previously used adapted-only
# future-reference helpers. Training uses explicit nominal=True/False above.
def motion_future_joint_pos(env, command_name, future_steps):
  return reference_dof_pos(env, command_name, future_steps, nominal=False)


def motion_future_root_height(env, command_name, future_steps):
  return reference_root_height(env, command_name, future_steps, nominal=False)


def motion_future_gravity_vector(env, command_name, future_steps):
  return reference_gravity_vector(env, command_name, future_steps, nominal=False)


def motion_future_foot_contacts(env, command_name, future_steps):
  return reference_foot_contacts(env, command_name, future_steps, nominal=False)


def motion_future_xy_vel(env, command_name, future_steps):
  return reference_xy_vel(env, command_name, future_steps, nominal=False)


def motion_future_yaw_vel(env, command_name, future_steps):
  return reference_ang_vel(env, command_name, future_steps, nominal=False)
