from typing import Dict, Tuple

import mujoco as mj
import numpy as np

from .constants import (
    FOOT_NAMES,
    IK_CHECK_MAX_COM_TRACKING_ERROR,
    IK_CHECK_MAX_DISPLACEMENT_MAGNITUDE,
    IK_CHECK_MAX_FOOT_DISP,
    IK_CHECK_MAX_FORCE_MAGNITUDE,
    IK_CHECK_MAX_IK_TRACKING_ERROR,
    IK_CHECK_MAX_ROTATIONAL_DISPLACEMENT_MAGNITUDE,
    IK_CHECK_MAX_TORQUE_MAGNITUDE,
)


def calculate_cop_aware_com_target(
    ref_data: mj.MjData,
    total_mass: float,
    com_force_for_ik: np.ndarray,
    com_torque_for_ik: np.ndarray,
    task_body_name: str,
) -> np.ndarray:
    left_foot_pos = ref_data.body("left_ankle_roll_link").xpos
    right_foot_pos = ref_data.body("right_ankle_roll_link").xpos
    target_cop = (left_foot_pos + right_foot_pos) / 2.0

    force_application_pos = ref_data.body(task_body_name).xpos
    lever_arm = force_application_pos - target_cop

    moment_from_force = np.cross(lever_arm, com_force_for_ik)
    total_moment_ext = moment_from_force + com_torque_for_ik

    mg = total_mass * 9.81
    ref_com = ref_data.subtree_com[0]
    if mg < 1e-3:
        return ref_com

    return ref_com + np.array(
        [-total_moment_ext[1] / mg, total_moment_ext[0] / mg, 0.0]
    )


def is_ik_solution_feasible(
    ik_data: mj.MjData,
    ref_data: mj.MjData,
    link_name: str,
    force_ext: np.ndarray,
    torque_ext: np.ndarray,
    stiffness: float,
    rotational_stiffness: float,
    total_mass: float,
) -> Tuple[bool, Dict[str, float]]:
    """Check if the IK solution stays within physical and tracking limits."""
    violations: Dict[str, float] = {}
    is_feasible = True

    force_mag = np.linalg.norm(force_ext)
    violations["force_magnitude"] = force_mag
    if force_mag > IK_CHECK_MAX_FORCE_MAGNITUDE:
        is_feasible = False

    displacement_mag = force_mag / stiffness if stiffness > 1e-6 else 0.0
    violations["displacement_magnitude"] = displacement_mag
    if displacement_mag > IK_CHECK_MAX_DISPLACEMENT_MAGNITUDE:
        is_feasible = False

    torque_mag = np.linalg.norm(torque_ext)
    violations["torque_magnitude"] = torque_mag
    if torque_mag > IK_CHECK_MAX_TORQUE_MAGNITUDE:
        is_feasible = False

    rot_displacement_mag = (
        torque_mag / rotational_stiffness if rotational_stiffness > 1e-6 else 0.0
    )
    violations["rotational_displacement_magnitude"] = rot_displacement_mag
    if rot_displacement_mag > IK_CHECK_MAX_ROTATIONAL_DISPLACEMENT_MAGNITUDE:
        is_feasible = False

    p_ref_link = ref_data.body(link_name).xpos
    p_target_link = p_ref_link + force_ext / max(stiffness, 1e-6)
    p_ik_link = ik_data.body(link_name).xpos
    link_tracking_error = np.linalg.norm(p_ik_link - p_target_link)
    violations["link_tracking_error"] = link_tracking_error
    if link_tracking_error > IK_CHECK_MAX_IK_TRACKING_ERROR:
        is_feasible = False

    max_foot_disp = 0.0
    for foot_name in FOOT_NAMES:
        p_ref_foot = ref_data.body(foot_name).xpos
        p_ik_foot = ik_data.body(foot_name).xpos
        foot_disp = np.linalg.norm(p_ik_foot - p_ref_foot)
        max_foot_disp = max(max_foot_disp, foot_disp)
    violations["max_foot_disp"] = max_foot_disp
    if max_foot_disp > IK_CHECK_MAX_FOOT_DISP:
        is_feasible = False

    com_target = calculate_cop_aware_com_target(
        ref_data,
        total_mass,
        force_ext,
        torque_ext,
        link_name,
    )
    com_ik = ik_data.subtree_com[0]
    com_tracking_error_xy = np.linalg.norm(com_ik[:2] - com_target[:2])
    violations["com_tracking_error"] = com_tracking_error_xy
    if com_tracking_error_xy > IK_CHECK_MAX_COM_TRACKING_ERROR:
        is_feasible = False

    return is_feasible, violations
