from typing import Dict, Optional, Tuple

import mujoco
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import mink

from config import SimulationConfig
from constants import KEYPOINT_BODY_NAMES, MAX_RELEASE_ANGULAR_VEL, MAX_RELEASE_LINEAR_VEL
from ik_solver import G1_Mink_IK_Solver
from physics import calculate_cop_aware_com_target


def perform_mink_ik_step(
    ik_solver: G1_Mink_IK_Solver,
    qpos_ref: np.ndarray,
    current_task_body_name: str,
    force_ext: np.ndarray,
    torque_ext: np.ndarray,
    com_force_for_ik: np.ndarray,
    stiffness: float,
    rotational_stiffness: float,
    dt: float,
    ik_target_overrides: Optional[Dict[str, Tuple[np.ndarray, Rotation]]] = None,
) -> Tuple[np.ndarray, Rotation, np.ndarray]:
    """Run a single constrained IK solve."""
    ref_config = mink.Configuration(ik_solver.model)
    ref_config.update(q=qpos_ref)
    mujoco.mj_forward(ref_config.model, ref_config.data)

    active_tasks = []
    ik_solver.posture_task.set_target(qpos_ref)
    active_tasks.append(ik_solver.posture_task)
    if ik_solver.waist_task:
        ik_solver.waist_task.set_target(qpos_ref)
        active_tasks.append(ik_solver.waist_task)
    if ik_solver.knee_task:
        ik_solver.knee_task.set_target(qpos_ref)
        active_tasks.append(ik_solver.knee_task)
    ik_solver.pelvis_pitch_task.set_target(
        ref_config.get_transform_frame_to_world("pelvis", "body")
    )
    active_tasks.append(ik_solver.pelvis_pitch_task)
    for name, task in ik_solver.foot_tasks.items():
        task.set_target(ref_config.get_transform_frame_to_world(name, "body"))
        active_tasks.append(task)

    if ik_solver.torso_orientation_task is not None:
        ik_solver.torso_orientation_task.set_target(
            ref_config.get_transform_frame_to_world("torso_link", "body")
        )
        active_tasks.append(ik_solver.torso_orientation_task)

    com_target = calculate_cop_aware_com_target(
        ref_config.data,
        ik_solver.total_mass,
        com_force_for_ik,
        torque_ext,
        current_task_body_name,
    )

    ik_solver.com_task.set_target(com_target)
    active_tasks.append(ik_solver.com_task)

    is_force_active = np.linalg.norm(force_ext) > 1e-2 or np.linalg.norm(torque_ext) > 1e-2
    if ik_target_overrides is None:
        ik_target_overrides = {}
    for name, task in ik_solver.keypoint_tasks.items():
        if name in ik_target_overrides:
            pos, rot = ik_target_overrides[name]
            quat_xyzw = rot.as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            target_pose = mink.SE3(np.concatenate([quat_wxyz, pos]))
            release_task = ik_solver.force_tasks[name]
            release_task.set_target(target_pose)
            active_tasks.append(release_task)
        elif not (is_force_active and name == current_task_body_name):
            task.set_target(ref_config.get_transform_frame_to_world(name, "body"))
            active_tasks.append(task)

    task_ref_pos = ref_config.data.body(current_task_body_name).xpos.copy()
    task_ref_rot = Rotation.from_matrix(
        ref_config.data.body(current_task_body_name).xmat.reshape(3, 3)
    )
    task_target_pos, task_target_rot = task_ref_pos, task_ref_rot
    if is_force_active:
        task_target_pos = task_ref_pos + force_ext / max(stiffness, 1e-6)
        if np.linalg.norm(torque_ext) > 1e-4 and rotational_stiffness > 1e-4:
            task_target_rot = Rotation.from_rotvec(torque_ext / rotational_stiffness) * task_ref_rot
        quat_xyzw = task_target_rot.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        target_pose = mink.SE3(np.concatenate([quat_wxyz, task_target_pos]))
        active_force_task = ik_solver.force_tasks[current_task_body_name]
        active_force_task.set_target(target_pose)
        active_tasks.append(active_force_task)
    elif current_task_body_name in ik_target_overrides:
        task_target_pos, task_target_rot = ik_target_overrides[current_task_body_name]

    vel = mink.solve_ik(
        ik_solver.configuration,
        active_tasks,
        dt,
        solver="daqp",
        limits=ik_solver.limits,
        damping=1e-5,
    )
    ik_solver.configuration.integrate_inplace(vel, dt)
    mujoco.mj_forward(ik_solver.model, ik_solver.data)
    return task_target_pos, task_target_rot, vel


def perform_single_ik_update(
    ik_solver: G1_Mink_IK_Solver,
    ref_data: mj.MjData,
    event: Optional[Dict],
    release_info: Optional[Dict],
    sim_time: float,
    dt: float,
    config: SimulationConfig,
    qpos_ref: np.ndarray,
) -> Tuple:
    force_ext, torque_ext = np.zeros(3), np.zeros(3)
    com_force_for_ik = np.zeros(3)
    task_body_name = (
        event["link_name"]
        if event
        else (release_info["link_name"] if release_info else "torso_link")
    )
    stiffness, rot_stiffness = 140.0, 1.0

    ik_target_overrides: Dict[str, Tuple[np.ndarray, Rotation]] = {}
    if release_info and not release_info.get("finished", False):
        ref_data.qpos[:] = qpos_ref
        mujoco.mj_forward(ref_data.model, ref_data)
        target_pos = ref_data.body(release_info["link_name"]).xpos
        target_rot = Rotation.from_matrix(
            ref_data.body(release_info["link_name"]).xmat.reshape(3, 3)
        )
        total_dist = np.linalg.norm(target_pos - release_info["start_pos"])
        total_angle = (release_info["start_rot"].inv() * target_rot).magnitude()
        time_needed_pos = (
            total_dist / MAX_RELEASE_LINEAR_VEL if MAX_RELEASE_LINEAR_VEL > 1e-6 else 0.0
        )
        time_needed_rot = (
            total_angle / MAX_RELEASE_ANGULAR_VEL if MAX_RELEASE_ANGULAR_VEL > 1e-6 else 0.0
        )
        total_time_needed = max(time_needed_pos, time_needed_rot, 1e-5)
        time_since_release = sim_time - release_info["start_time"]

        if time_since_release >= total_time_needed:
            release_info["finished"] = True
            com_force_for_ik = np.zeros(3)
        else:
            interp_factor = 1.0 - (time_since_release / total_time_needed)
            slerp = Slerp([0, 1], Rotation.concatenate([target_rot, release_info["start_rot"]]))
            interp_rot = slerp(interp_factor)
            interp_pos = target_pos + interp_factor * (release_info["start_pos"] - target_pos)
            ik_target_overrides[release_info["link_name"]] = (interp_pos, interp_rot)
            com_force_for_ik = release_info["start_force"] * interp_factor

    if event and sim_time >= event["start_time"]:
        stiffness = event["stiffness"]
        rot_stiffness = event.get("rotational_stiffness", 1.0)

        main_magnitude_factor = 0.0
        if sim_time < event["hold_start_time"]:
            main_magnitude_factor = (
                (sim_time - event["start_time"]) / event["ramp_duration"]
                if event["ramp_duration"] > 1e-5
                else 1.0
            )
        elif sim_time < event["hold_end_time"]:
            main_magnitude_factor = 1.0
        elif sim_time <= event["end_time"]:
            main_magnitude_factor = (
                1.0
                - (sim_time - event["hold_end_time"]) / event["ramp_duration"]
                if event["ramp_duration"] > 1e-5
                else 0.0
            )
        main_magnitude_factor = np.clip(main_magnitude_factor, 0.0, 1.0)

        ref_data.qpos[:] = qpos_ref
        mujoco.mj_forward(ref_data.model, ref_data)
        p_ref = ref_data.body(task_body_name).xpos.copy()
        r_ref = Rotation.from_matrix(ref_data.body(task_body_name).xmat.reshape(3, 3))

        if config.force_mode == "triangle":
            force_ext = event["amplitude"] * main_magnitude_factor * event["force_axis"]
            torque_ext = event["torque_amplitude"] * main_magnitude_factor * event["torque_axis"]

        elif config.force_mode == "forcefield":
            force_ext = event["amplitude"] * main_magnitude_factor * event["force_axis"]
            torque_ext = event["torque_amplitude"] * main_magnitude_factor * event["torque_axis"]

            k_robot_lin, k_ff_lin = stiffness, event["forcefield_stiffness"]
            k_robot_rot = rot_stiffness
            k_ff_rot = event["rotational_forcefield_stiffness"]

            p_ff = p_ref + force_ext / max(k_ff_lin, 1e-6) + force_ext / max(k_robot_lin, 1e-6)
            rot_ff = (
                Rotation.from_rotvec(torque_ext / max(k_ff_rot, 1e-6))
                * Rotation.from_rotvec(torque_ext / max(k_robot_rot, 1e-6))
                * r_ref
            )

            event["forcefield_setpoint_pos"] = p_ff
            event["forcefield_setpoint_rot"] = rot_ff

        elif config.force_mode == "collision-emulator":
            k_robot_lin, k_ff_lin = stiffness, event["forcefield_stiffness"]
            k_eff_lin = (
                (k_robot_lin * k_ff_lin) / (k_robot_lin + k_ff_lin)
                if (k_robot_lin + k_ff_lin) > 1e-6
                else 0.0
            )

            k_robot_rot, k_ff_rot = rot_stiffness, event["rotational_forcefield_stiffness"]
            k_eff_rot = (
                (k_robot_rot * k_ff_rot) / (k_robot_rot + k_ff_rot)
                if (k_robot_rot + k_ff_rot) > 1e-6
                else 0.0
            )

            delta_p = event["forcefield_setpoint_pos"] - p_ref
            delta_rot_vec = (event["forcefield_setpoint_rot"] * r_ref.inv()).as_rotvec()

            force_ext = k_eff_lin * delta_p
            torque_ext = k_eff_rot * delta_rot_vec

        elif config.force_mode == "collision-emulator-1d":
            if not event.get("terminated", False):
                k_ff_eff = event["forcefield_stiffness"]
                time_since_spawn = sim_time - event["initial_spawn_time"]
                current_plane_origin = (
                    event["collision_plane_origin"]
                    + event["plane_velocity_vec"] * time_since_spawn
                )
                penetration = -np.dot(
                    p_ref - current_plane_origin, event["collision_plane_normal"]
                )
                if penetration > 0:
                    denominator = stiffness + k_ff_eff
                    force_ext = (
                        (stiffness * k_ff_eff / denominator)
                        * penetration
                        * event["collision_plane_normal"]
                        if denominator > 1e-6
                        else np.zeros(3)
                    )
                else:
                    event["terminated"] = True

            if "torque_start_time" in event and sim_time >= event["torque_start_time"]:
                torque_magnitude_factor = 0.0
                if sim_time < event["torque_hold_start_time"]:
                    torque_magnitude_factor = (
                        (sim_time - event["torque_start_time"]) / event["torque_ramp_duration"]
                        if event["torque_ramp_duration"] > 1e-5
                        else 1.0
                    )
                elif sim_time < event["torque_hold_end_time"]:
                    torque_magnitude_factor = 1.0
                elif sim_time <= event["torque_end_time"]:
                    torque_magnitude_factor = (
                        1.0
                        - (sim_time - event["torque_hold_end_time"]) / event["torque_ramp_duration"]
                        if event["torque_ramp_duration"] > 1e-5
                        else 0.0
                    )
                torque_ext = (
                    event["torque_amplitude"]
                    * np.clip(torque_magnitude_factor, 0.0, 1.0)
                    * event["torque_axis"]
                )

        com_force_for_ik = force_ext

    elif not release_info or release_info.get("finished", False):
        ik_solver.configuration.update(q=qpos_ref)
        mujoco.mj_forward(ik_solver.model, ik_solver.data)

    task_target_pos, task_target_rot, _ = perform_mink_ik_step(
        ik_solver,
        qpos_ref,
        task_body_name,
        force_ext,
        torque_ext,
        com_force_for_ik,
        stiffness,
        rot_stiffness,
        dt,
        ik_target_overrides,
    )

    ref_data.qpos[:] = qpos_ref
    mujoco.mj_forward(ref_data.model, ref_data)
    task_ref_pos_for_viz = ref_data.body(task_body_name).xpos.copy()
    keypoint_poses = {
        name: (
            ref_data.body(name).xpos.copy(),
            Rotation.from_matrix(ref_data.body(name).xmat.reshape(3, 3)),
        )
        for name in KEYPOINT_BODY_NAMES
    }
    return (
        task_body_name,
        force_ext,
        torque_ext,
        task_ref_pos_for_viz,
        task_target_pos,
        task_target_rot,
        keypoint_poses,
    )
