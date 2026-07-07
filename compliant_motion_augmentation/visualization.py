from typing import Dict, List, Optional, Tuple

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation


def add_marker(
    scene: mj.MjvScene,
    pos: np.ndarray,
    size: List[float],
    rgba: List[float],
    type: int = mj.mjtGeom.mjGEOM_SPHERE,
    mat: np.ndarray = np.eye(3).flatten(),
):
    if scene.ngeom < scene.maxgeom:
        mj.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=type,
            size=np.asarray(size, dtype=np.float32),
            pos=pos,
            mat=mat,
            rgba=np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1


def add_arrow(
    scene: mj.MjvScene,
    from_: np.ndarray,
    to: np.ndarray,
    radius: float = 0.015,
    rgba: List[float] = [1.0, 0.7, 0.1, 1.0],
):
    if scene.ngeom < scene.maxgeom:
        mj.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.zeros(9),
            rgba=np.asarray(rgba).astype(np.float32),
        )
        mj.mjv_connector(
            scene.geoms[scene.ngeom], type=mj.mjtGeom.mjGEOM_ARROW, width=radius, from_=from_, to=to
        )
        scene.ngeom += 1


def add_visual_overlays(
    scene: mj.MjvScene,
    current_data: mj.MjData,
    task_body_name: str,
    force_ext: np.ndarray,
    torque_ext: np.ndarray,
    task_ref_pos: np.ndarray,
    task_target_pos: np.ndarray,
    task_target_rot: Rotation,
    keypoint_target_6d_poses: Dict[str, Tuple[np.ndarray, Rotation]],
    event: Optional[Dict],
    sim_time: float,
    rewind_indicator_until: float,
):
    """Renders all visual guides for the IK simulation."""
    if np.linalg.norm(force_ext) > 1e-2:
        arrow_start = current_data.body(task_body_name).xpos
        arrow_end = arrow_start + force_ext * 0.05
        add_arrow(scene, arrow_start, arrow_end, rgba=[1, 0.7, 0.1, 1])
        add_marker(scene, task_target_pos, [0.03, 0, 0], [0, 1, 0, 0.5])

    if event:
        if "forcefield_setpoint_pos" in event:
            add_marker(
                scene,
                event["forcefield_setpoint_pos"],
                [0.04, 0, 0],
                [1, 0.4, 0.8, 0.7],
            )
        if "collision_plane_normal" in event and not event.get("terminated", False):
            time_since_spawn = sim_time - event["initial_spawn_time"]
            plane_pos = event["collision_plane_origin"] + event.get(
                "plane_velocity_vec", np.zeros(3)
            ) * max(0, time_since_spawn)
            plane_normal = event["collision_plane_normal"]
            z_axis = np.array([0, 0, 1])
            rot_axis = np.cross(z_axis, plane_normal)
            angle = np.arccos(np.clip(np.dot(z_axis, plane_normal), -1.0, 1.0))
            if np.linalg.norm(rot_axis) > 1e-6:
                orientation = Rotation.from_rotvec(angle * rot_axis / np.linalg.norm(rot_axis))
            else:
                orientation = Rotation.identity()
            add_marker(
                scene,
                plane_pos,
                [0.2, 0.2, 0.005],
                [0.1, 0.8, 0.9, 0.3],
                type=mj.mjtGeom.mjGEOM_BOX,
                mat=orientation.as_matrix().flatten(),
            )

    if np.linalg.norm(torque_ext) > 1e-2:
        axis_len, axis_radius = 0.15, 0.007
        target_axes = task_target_rot.as_matrix()
        add_arrow(
            scene,
            task_target_pos,
            task_target_pos + axis_len * target_axes[:, 0],
            radius=axis_radius,
            rgba=[1, 0, 0, 0.7],
        )
        add_arrow(
            scene,
            task_target_pos,
            task_target_pos + axis_len * target_axes[:, 1],
            radius=axis_radius,
            rgba=[0, 1, 0, 0.7],
        )
        add_arrow(
            scene,
            task_target_pos,
            task_target_pos + axis_len * target_axes[:, 2],
            radius=axis_radius,
            rgba=[0, 0, 1, 0.7],
        )

    add_marker(scene, task_ref_pos, [0.03, 0, 0], [1, 0, 0, 0.5])
    add_marker(scene, current_data.subtree_com[0], [0.035, 0, 0], [0.9, 0.2, 0.8, 0.8])
    for name, (target_pos, target_rot) in keypoint_target_6d_poses.items():
        add_marker(scene, target_pos, [0.02, 0, 0], [0.2, 0.5, 1, 0.5])

    if sim_time < rewind_indicator_until:
        alpha = 0.6 * (rewind_indicator_until - sim_time)
        add_marker(scene, current_data.subtree_com[0], [0.4, 0, 0], [1, 0.1, 0.1, alpha])
