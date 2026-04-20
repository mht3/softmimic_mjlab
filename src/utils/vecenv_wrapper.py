"""VecEnv wrapper with visitation-critic reset-state support."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat

from rsl_rl.utils.qpos import integrate_qpos


class RslRlVecEnvSpecialResetWrapper(RslRlVecEnvWrapper):
    """Extends RslRlVecEnvWrapper with set_reset_states for the visitation critic."""

    def set_reset_states(self, env_ids: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Override reset states for specified envs with VC-generated relative states.

        Converts tangent-space relative states (2*nv) to absolute coordinates and
        writes them into the simulator, overriding the default reset that already
        ran for these envs during env.step().

        The reference frame keeps each env's current x, y, yaw (from the post-reset
        position) and uses equilibrium values for z, roll, pitch, and all joints.

        Args:
            env_ids: Indices of environments to override. Shape: (num_resets,)
            states: Relative states from the VC. Shape: (num_resets, 2 * nv).
                Layout: [rel_qpos(nv), rel_qvel(nv)].

        Returns:
            Absolute ``qpos`` tensor that was integrated and written: shape
            ``(num_resets, 7 + n_joints)`` — env-local root pos, ``wxyz`` quat, joint pos.
        """
        env: ManagerBasedRlEnv = self.unwrapped
        robot: Entity = env.scene["robot"]

        nv = states.shape[-1] // 2
        rel_qpos = states[:, :nv]
        rel_qvel = states[:, nv:]

        # Build the reference qpos for each env: keep current x, y, yaw from
        # the post-reset state; use equilibrium for everything else.
        default_root_state = robot.data.default_root_state[env_ids].clone()
        # Current post-reset position (in world frame, subtract env_origins to
        # get env-local coordinates matching the qpos convention).
        cur_pos_w = robot.data.root_link_pos_w[env_ids]
        cur_quat_w = robot.data.root_link_quat_w[env_ids]
        cur_pos_local = cur_pos_w - env.scene.env_origins[env_ids]

        # Reference position: current x/y, default z.
        ref_root_pos = default_root_state[:, :3].clone()
        ref_root_pos[:, :2] = cur_pos_local[:, :2]

        # Reference orientation: current yaw, default roll/pitch.
        cur_yaw_quat = yaw_quat(cur_quat_w)
        default_quat = default_root_state[:, 3:7]
        default_yaw = yaw_quat(default_quat)
        default_roll_pitch = quat_mul(quat_inv(default_yaw), default_quat)
        ref_root_quat = quat_mul(cur_yaw_quat, default_roll_pitch)

        # Reference joints: default positions.
        ref_joint_pos = robot.data.default_joint_pos[env_ids]

        # Full reference qpos: [pos(3), quat(4), joints(nq-7)].
        ref_qpos = torch.cat([ref_root_pos, ref_root_quat, ref_joint_pos], dim=-1)

        # Reference qvel: default (typically zeros).
        default_root_lin_vel = default_root_state[:, 7:10]
        default_root_ang_vel = default_root_state[:, 10:13]
        default_joint_vel = robot.data.default_joint_vel[env_ids]
        ref_qvel = torch.cat([default_root_lin_vel, default_root_ang_vel, default_joint_vel], dim=-1)

        # Convert relative -> absolute.
        abs_qpos = integrate_qpos(ref_qpos, rel_qpos)
        abs_qvel = rel_qvel + ref_qvel

        # Write absolute state into the simulator.
        root_pos_w = abs_qpos[:, :3] + env.scene.env_origins[env_ids]
        root_quat = abs_qpos[:, 3:7]

        # The relative state observation records body-frame velocities
        # (root_link_lin_vel_b, root_link_ang_vel_b). Convert to world frame
        # for write_root_state_to_sim, matching collect_dataset.py.
        root_lin_vel_b = abs_qvel[:, :3]
        root_ang_vel_b = abs_qvel[:, 3:6]
        root_lin_vel_w = quat_apply(root_quat, root_lin_vel_b)
        root_ang_vel_w = quat_apply(root_quat, root_ang_vel_b)

        root_state = torch.cat([root_pos_w, root_quat, root_lin_vel_w, root_ang_vel_w], dim=-1)
        robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        # Joint state.
        joint_pos = abs_qpos[:, 7:]
        joint_vel = abs_qvel[:, 6:]
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Refresh entity data so get_observations() sees the new state.
        robot.clear_state(env_ids=env_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()
        robot.update(env.step_dt)

        return abs_qpos
