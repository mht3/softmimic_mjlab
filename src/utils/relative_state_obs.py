from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_inv, quat_mul, yaw_quat

from rsl_rl.utils.qpos import differentiate_qpos


def relative_state_from_sim(env) -> torch.Tensor:
    """Return simulator-grounded relative state [rel_qpos, rel_qvel].

    This function builds absolute state from robot simulator tensors and then maps to
    tangent-space relative coordinates expected by the visitation critic.
    """
    robot: Entity = env.scene["robot"]
    env_origins = env.scene.env_origins

    # Absolute state from simulator-facing robot data.
    pos_w = robot.data.root_link_pos_w
    quat_w = robot.data.root_link_quat_w
    lin_vel_b = robot.data.root_link_lin_vel_b
    ang_vel_b = robot.data.root_link_ang_vel_b
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel

    state_qpos = torch.cat([pos_w - env_origins, quat_w, joint_pos], dim=-1)
    state_qvel = torch.cat([lin_vel_b, ang_vel_b, joint_vel], dim=-1)

    # Tracking task: use moving command target as reference if available.
    ref_qpos = None
    ref_qvel = None
    if getattr(env, "command_manager", None) is not None and "motion" in getattr(env.cfg, "commands", {}):
        motion_cmd = env.command_manager.get_term("motion")
        if hasattr(motion_cmd, "anchor_pos_w") and hasattr(motion_cmd, "anchor_quat_w") and hasattr(motion_cmd, "joint_pos"):
            ref_qpos = torch.cat(
                [motion_cmd.anchor_pos_w - env_origins, motion_cmd.anchor_quat_w, motion_cmd.joint_pos], dim=-1
            )
            anchor_lin_vel_w = getattr(motion_cmd, "anchor_lin_vel_w", None)
            anchor_ang_vel_w = getattr(motion_cmd, "anchor_ang_vel_w", None)
            cmd_joint_vel = getattr(motion_cmd, "joint_vel", None)
            if anchor_lin_vel_w is not None and anchor_ang_vel_w is not None and cmd_joint_vel is not None:
                ref_qvel = torch.cat([anchor_lin_vel_w, anchor_ang_vel_w, cmd_joint_vel], dim=-1)

    # Velocity task: use commanded twist terms as qvel reference.
    # Supports [vx, vy, yaw_rate] and [vx, vy, vz, wx, wy, wz].
    if ref_qvel is None and getattr(env, "command_manager", None) is not None and "twist" in getattr(env.cfg, "commands", {}):
        twist_cmd = env.command_manager.get_term("twist")
        cmd_vel_b = getattr(twist_cmd, "command", None)
        if cmd_vel_b is not None:
            ref_lin_vel_b = torch.zeros_like(lin_vel_b)
            ref_ang_vel_b = torch.zeros_like(ang_vel_b)
            cmd_dim = cmd_vel_b.shape[-1]

            if cmd_dim >= 1:
                ref_lin_vel_b[:, 0] = cmd_vel_b[:, 0]
            if cmd_dim >= 2:
                ref_lin_vel_b[:, 1] = cmd_vel_b[:, 1]
            if cmd_dim == 3:
                ref_ang_vel_b[:, 2] = cmd_vel_b[:, 2]
            elif cmd_dim >= 6:
                ref_lin_vel_b[:, 2] = cmd_vel_b[:, 2]
                ref_ang_vel_b[:, 0] = cmd_vel_b[:, 3]
                ref_ang_vel_b[:, 1] = cmd_vel_b[:, 4]
                ref_ang_vel_b[:, 2] = cmd_vel_b[:, 5]
            ref_qvel = torch.cat([ref_lin_vel_b, ref_ang_vel_b, robot.data.default_joint_vel], dim=-1)

    # Non-tracking fallback:
    # - keep per-episode randomized x/y/yaw from the first state in the episode
    # - keep equilibrium z/roll/pitch and default joints
    if ref_qpos is None:
        default_root_state = robot.data.default_root_state
        num_envs = state_qpos.shape[0]

        if not hasattr(env, "_vc_anchor_xy"):
            env._vc_anchor_xy = state_qpos[:, :2].clone()
            env._vc_anchor_yaw_quat = yaw_quat(state_qpos[:, 3:7]).clone()

        episode_length_buf = getattr(env, "episode_length_buf", None)
        if episode_length_buf is not None:
            reset_env_ids = torch.where(episode_length_buf == 0)[0]
            if len(reset_env_ids) > 0:
                env._vc_anchor_xy[reset_env_ids] = state_qpos[reset_env_ids, :2]
                env._vc_anchor_yaw_quat[reset_env_ids] = yaw_quat(state_qpos[reset_env_ids, 3:7])
        elif env._vc_anchor_xy.shape[0] != num_envs:
            env._vc_anchor_xy = state_qpos[:, :2].clone()
            env._vc_anchor_yaw_quat = yaw_quat(state_qpos[:, 3:7]).clone()

        # `default_root_state` is already in environment-local frame.
        ref_root_pos = default_root_state[:, :3].clone()
        ref_root_pos[:, :2] = env._vc_anchor_xy
        # Keep equilibrium roll/pitch from default root while swapping only yaw.
        default_root_quat = default_root_state[:, 3:7]
        default_root_yaw = yaw_quat(default_root_quat)
        default_root_roll_pitch = quat_mul(quat_inv(default_root_yaw), default_root_quat)
        ref_root_quat = quat_mul(env._vc_anchor_yaw_quat, default_root_roll_pitch)
        ref_qpos = torch.cat([ref_root_pos, ref_root_quat, robot.data.default_joint_pos], dim=-1)
    if ref_qvel is None:
        default_root_state = robot.data.default_root_state
        default_root_lin_vel_w = default_root_state[:, 7:10]
        default_root_ang_vel_w = default_root_state[:, 10:13]
        ref_qvel = torch.cat([default_root_lin_vel_w, default_root_ang_vel_w, robot.data.default_joint_vel], dim=-1)

    rel_qpos = differentiate_qpos(state_qpos, ref_qpos)
    rel_qvel = state_qvel - ref_qvel
    return torch.cat([rel_qpos, rel_qvel], dim=-1)
