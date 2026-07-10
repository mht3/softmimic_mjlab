#!/usr/bin/env python3

import pathlib
import numpy as np
import torch
import mujoco

from softmimic_deploy.src.motion_lib.motion_lib_from_multi_csv import JointConfig


def get_isaac_to_mujoco_joint_map(robot_type: str, num_joints_in_data: int) -> list[int]:
    """Map Isaac joint ordering to MuJoCo ordering for the G1 robot."""
    if robot_type != "g1":
        raise ValueError(f"Unsupported robot type for joint mapping: {robot_type}. Expected 'g1'.")
    if num_joints_in_data == 29:
        return [
            0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28
        ]
    if num_joints_in_data == 23:
        # The 23-DOF CSV column order (the 29-DOF isaac order with waist
        # roll/pitch and both wrist pitch/yaw dropped, see
        # scripts/csv_29dof_to_23dof.py) is already identical to the qpos joint
        # order of g1_23dof.xml, so no reordering is needed.
        return list(range(23))
    raise ValueError(f"Expected 23 or 29 joints for G1, but got {num_joints_in_data}")


class KeypointExtractor:
    """Extract 3D keypoints from robot poses using standard MuJoCo forward kinematics."""

    def __init__(self, robot_type: str = "g1", num_joints: int = 29):
        if robot_type != "g1":
            raise ValueError(f"Unsupported robot type: {robot_type}. KeypointExtractor only supports 'g1'.")
        if num_joints not in (23, 29):
            raise ValueError(f"KeypointExtractor supports 23 or 29 joints for g1, got {num_joints}.")

        self.robot_type = "g1"
        self.num_joints = num_joints
        repo_base = pathlib.Path(__file__).absolute().parents[3]

        if num_joints == 29:
            self.joint_config = JointConfig(
                num_joints=29,
                left_leg_indices={"hip_yaw": 2, "hip_roll": 1, "hip_pitch": 0, "knee": 3, "ankle_pitch": 4, "ankle_roll": 5},
                right_leg_indices={"hip_yaw": 8, "hip_roll": 7, "hip_pitch": 6, "knee": 9, "ankle_pitch": 10, "ankle_roll": 11},
                left_arm_indices={
                    "shoulder_pitch": 15,
                    "shoulder_roll": 16,
                    "shoulder_yaw": 17,
                    "elbow": 18,
                    "wrist_pitch": 19,
                    "wrist_roll": 20,
                    "wrist_yaw": 21,
                },
                right_arm_indices={
                    "shoulder_pitch": 22,
                    "shoulder_roll": 23,
                    "shoulder_yaw": 24,
                    "elbow": 25,
                    "wrist_pitch": 26,
                    "wrist_roll": 27,
                    "wrist_yaw": 28,
                },
                thigh_length=0.3,
                calf_length=0.3,
            )
            self.mjcf_path = str(repo_base / "softmimic_deploy/src/assets/g1/g1_29dof_w_ghost.xml")
        else:  # 23-DOF: waist roll/pitch and wrist pitch/yaw dropped.
            self.joint_config = JointConfig(
                num_joints=23,
                left_leg_indices={"hip_yaw": 2, "hip_roll": 1, "hip_pitch": 0, "knee": 3, "ankle_pitch": 4, "ankle_roll": 5},
                right_leg_indices={"hip_yaw": 8, "hip_roll": 7, "hip_pitch": 6, "knee": 9, "ankle_pitch": 10, "ankle_roll": 11},
                left_arm_indices={
                    "shoulder_pitch": 13,
                    "shoulder_roll": 14,
                    "shoulder_yaw": 15,
                    "elbow": 16,
                    "wrist_roll": 17,
                },
                right_arm_indices={
                    "shoulder_pitch": 18,
                    "shoulder_roll": 19,
                    "shoulder_yaw": 20,
                    "elbow": 21,
                    "wrist_roll": 22,
                },
                thigh_length=0.3,
                calf_length=0.3,
            )
            self.mjcf_path = str(repo_base / "softmimic_deploy/src/assets/g1/g1_23dof.xml")

        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.keypoint_body_ids = self._get_keypoint_bodies()

    def _get_keypoint_bodies(self) -> dict[str, int]:
        """Collect body ids for keypoints of interest."""
        keypoints: dict[str, int] = {}

        target_keypoints = [
            "pelvis",
            "torso",
            "left_hip",
            "left_knee",
            "left_ankle",
            "left_foot",
            "right_hip",
            "right_knee",
            "right_ankle",
            "right_foot",
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "left_hand",
            "right_shoulder",
            "right_elbow",
            "right_wrist",
            "right_hand",
            "head",
        ]

        name_patterns = {
            "pelvis": ["pelvis", "base"],
            "torso": ["torso", "chest", "trunk"],
            "left_hip": ["left_hip", "left_thigh"],
            "right_hip": ["right_hip", "right_thigh"],
            "left_knee": ["left_knee", "left_calf"],
            "right_knee": ["right_knee", "right_calf"],
            "left_ankle": ["left_ankle", "left_foot", "left_ankle_pitch"],
            "right_ankle": ["right_ankle", "right_foot", "right_ankle_pitch"],
            "left_foot": ["left_foot", "left_sole"],
            "right_foot": ["right_foot", "right_sole"],
            "left_shoulder": ["left_shoulder", "left_arm"],
            "right_shoulder": ["right_shoulder", "right_arm"],
            "left_elbow": ["left_elbow", "left_forearm"],
            "right_elbow": ["right_elbow", "right_forearm"],
            "left_wrist": ["left_wrist", "left_hand", "left_palm"],
            "right_wrist": ["right_wrist", "right_hand", "right_palm"],
            "left_hand": ["left_hand", "left_palm", "left_hand_base"],
            "right_hand": ["right_hand", "right_palm", "right_hand_base"],
            "head": ["head", "neck", "skull"],
        }

        body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)
        ]
        for i, body_name in enumerate(body_names):
            if body_name is None:
                continue
            for keypoint, patterns in name_patterns.items():
                if any(pattern in body_name and "ghost" not in body_name for pattern in patterns):
                    keypoints[keypoint] = i
                    break

        return keypoints

    def _set_pose(self, root_pos: np.ndarray, root_rot: np.ndarray, joint_pos: np.ndarray) -> None:
        """Write pose data into MuJoCo state."""
        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_rot[[3, 0, 1, 2]]
        joint_index = 7
        num_joints_to_set = min(len(joint_pos), self.joint_config.num_joints)
        self.data.qpos[joint_index : joint_index + num_joints_to_set] = joint_pos[:num_joints_to_set]
        mujoco.mj_forward(self.model, self.data)

    def compute_keypoints_batch(
        self,
        root_positions: torch.Tensor,
        root_rotations: torch.Tensor,
        joint_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Compute keypoint positions/orientations for a batch of poses."""
        batch_size = root_positions.shape[0]
        device = root_positions.device

        root_pos_np = root_positions.cpu().numpy()
        root_rot_np = root_rotations.cpu().numpy()
        joint_pos_np = joint_positions.cpu().numpy()

        isaac_to_mujoco_indices = get_isaac_to_mujoco_joint_map(self.robot_type, joint_pos_np.shape[1])
        joint_pos_mujoco_np = joint_pos_np[:, isaac_to_mujoco_indices]

        keypoint_names = list(self.keypoint_body_ids.keys())
        num_keypoints = len(keypoint_names)
        keypoints_batch_np = np.zeros((batch_size, num_keypoints, 3), dtype=np.float32)
        keypoint_rotations_batch_np = np.zeros((batch_size, num_keypoints, 4), dtype=np.float32)

        for i in range(batch_size):
            self._set_pose(root_pos_np[i], root_rot_np[i], joint_pos_mujoco_np[i])
            for j, name in enumerate(keypoint_names):
                body_id = self.keypoint_body_ids[name]
                keypoints_batch_np[i, j, :] = self.data.xpos[body_id]
                keypoint_rotations_batch_np[i, j, :] = self.data.xquat[body_id]

        keypoints_batch = torch.from_numpy(keypoints_batch_np).to(device)
        keypoint_rotations_batch = torch.from_numpy(keypoint_rotations_batch_np).to(device)
        return keypoints_batch, keypoint_rotations_batch, keypoint_names
