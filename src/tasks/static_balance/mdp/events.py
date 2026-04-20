"""Custom event functions for static balance task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.events import push_by_setting_velocity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def push_by_setting_velocity_with_cutoff(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  time_cutoff_s: float = 17.0,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Apply a velocity push, but skip envs whose episode time has passed `time_cutoff_s`.

  This leaves the final `max_episode_length_s - time_cutoff_s` seconds push-free so
  the agent can recover before episode end.
  """
  cutoff_step = int(time_cutoff_s / env.step_dt)
  mask = env.episode_length_buf[env_ids] < cutoff_step
  if not bool(mask.any()):
    return
  push_by_setting_velocity(env, env_ids[mask], velocity_range, asset_cfg)


def reset_rp_noise(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  rp_range: float,
) -> None:
  """Apply random roll/pitch rotation to robot base after reset.

  Rotates the root orientation by a sampled (roll, pitch) delta while
  preserving position and world-frame velocities.

  Reads pose/velocity directly from qpos/qvel rather than from
  robot.data.root_link_quat_w (which reflects xquat, a derived quantity
  that is only updated by mj_forward — not yet called between reset events).

  Args:
      rp_range: Symmetric range (rad) for roll and pitch offsets.
                Pass 0.0 to skip (no-op).
  """
  if rp_range == 0.0:
    return
  robot = env.scene["robot"]
  n = len(env_ids)

  roll = torch.zeros(n, device=env.device).uniform_(-rp_range, rp_range)
  pitch = torch.zeros(n, device=env.device).uniform_(-rp_range, rp_range)
  delta_quat = quat_from_euler_xyz(roll, pitch, torch.zeros(n, device=env.device))

  # read_pose / read_vel directly from qpos/qvel:
  #   qpos[free_joint] = [x, y, z, qw, qx, qy, qz]  (world frame)
  #   qvel[free_joint] = [vx, vy, vz, wx, wy, wz]  where lin_vel is world frame
  #                      and ang_vel is body frame (MuJoCo convention)
  q_adr = robot.data.indexing.free_joint_q_adr  # shape (7,)
  v_adr = robot.data.indexing.free_joint_v_adr  # shape (6,)

  ids2d = env_ids[:, None]  # (n, 1) for batched advanced indexing
  pose = robot.data.data.qpos[ids2d, q_adr]   # (n, 7)
  vel = robot.data.data.qvel[ids2d, v_adr]    # (n, 6)

  pos = pose[:, :3]
  current_quat = pose[:, 3:7]
  new_quat = quat_mul(delta_quat, current_quat)

  lin_vel_w = vel[:, :3]                            # world frame — pass through
  ang_vel_b = vel[:, 3:6]                           # body frame — rotate to world
  ang_vel_w = quat_apply(current_quat, ang_vel_b)

  root_state = torch.cat([pos, new_quat, lin_vel_w, ang_vel_w], dim=-1)
  robot.write_root_state_to_sim(root_state, env_ids=env_ids)
