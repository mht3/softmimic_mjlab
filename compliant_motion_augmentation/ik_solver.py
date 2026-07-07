from typing import Dict, List, Optional, Tuple
import os

import mujoco
import mujoco as mj
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import mink
from softmimic_deploy.src.motion_lib.motion_lib_from_multi_csv import (
    JointConfig,
    ProceduralMotionLibFromDemo,
)

from .constants import FORCEABLE_LINKS, FOOT_NAMES, KEYPOINT_BODY_NAMES
from .tasks import KneeBendingTask


class G1_Mink_IK_Solver:
    """A wrapper for the mink IK library tailored for the G1 robot and this script's needs."""

    def __init__(
        self,
        model_path: str,
        motion_path: Optional[str] = None,
        repeat_frame_time: Optional[float] = None,
        com_cost: float = 0.5,
        com_cost_z_factor: float = 1.0,
        upper_joint_cost: float = 0.0,
        torso_orientation_cost: float = 0.0,
        waist_cost: float = 0.01,
        knee_cost: float = 0.01,
    ):
        if mink is None:
            raise ImportError("The 'mink' library is required for G1_Mink_IK_Solver.")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.configuration = mink.Configuration(self.model)
        self.data = self.configuration.data
        self.total_mass = sum(self.model.body_mass)

        actuated_joint_ids = [
            self.model.actuator_trnid[i, 0]
            for i in range(self.model.nu)
            if self.model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
        ]
        self.actuated_qpos_indices = [
            self.model.jnt_qposadr[jid] for jid in np.unique(actuated_joint_ids)
        ]
        self.num_dofs = len(self.actuated_qpos_indices)

        self.posture_task = mink.PostureTask(self.model, cost=1e-4)
        com_cost_vector = np.array([com_cost, com_cost, com_cost * com_cost_z_factor])
        self.com_task = mink.ComTask(cost=com_cost_vector)
        self.com_task.set_cost(com_cost_vector)

        self.waist_task = None
        if waist_cost > 1e-5:
            waist_joint_names = ["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"]
            waist_cost_vector = np.zeros(self.model.nv)
            found_joints = []
            for name in waist_joint_names:
                try:
                    joint_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                    dof_adr = self.model.jnt_dofadr[joint_id]
                    waist_cost_vector[dof_adr] = waist_cost
                    found_joints.append(name)
                except KeyError:
                    print(
                        f"Warning: Waist joint '{name}' not found in model. Skipping for waist task."
                    )
            if found_joints:
                self.waist_task = mink.PostureTask(self.model, cost=waist_cost_vector)

        self.upper_task = None
        if upper_joint_cost > 1e-5:
            upper_joint_names = [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "left_elbow_joint",
                "right_elbow_joint",
                "left_wrist_pitch_joint",
                "left_wrist_roll_joint",
                "left_wrist_yaw_joint",
                "right_wrist_pitch_joint",
                "right_wrist_roll_joint",
                "right_wrist_yaw_joint",
            ]
            upper_cost_vector = np.zeros(self.model.nv)
            found_joints = []
            for name in upper_joint_names:
                try:
                    joint_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                    dof_adr = self.model.jnt_dofadr[joint_id]
                    upper_cost_vector[dof_adr] = upper_joint_cost
                    found_joints.append(name)
                except KeyError:
                    print(
                        f"Warning: Upper joint '{name}' not found in model. Skipping for upper task."
                    )
            if found_joints:
                self.upper_task = mink.PostureTask(self.model, cost=upper_cost_vector)

        self.torso_orientation_task = None
        if torso_orientation_cost > 1e-5:
            self.torso_orientation_task = mink.FrameTask(
                "torso_link",
                "body",
                position_cost=0.0,
                orientation_cost=np.array([torso_orientation_cost] * 3),
            )
        self.pelvis_pitch_task = mink.FrameTask(
            "pelvis", "body", position_cost=0.0, orientation_cost=np.array([0.03, 0.03, 0.03])
        )
        self.knee_task = None
        if knee_cost > 1e-5:
            self.knee_task = KneeBendingTask(
                self.model, cost=knee_cost, joint_names=["left_knee_joint", "right_knee_joint"]
            )

        self.keypoint_tasks = {
            name: mink.FrameTask(name, "body", position_cost=1e-2, orientation_cost=1e-3)
            for name in KEYPOINT_BODY_NAMES
        }
        self.foot_tasks = {
            name: mink.FrameTask(name, "body", position_cost=2.5, orientation_cost=0.5)
            for name in FOOT_NAMES
        }
        self.force_tasks = {
            name: mink.FrameTask(name, "body", position_cost=5.0, orientation_cost=1.0)
            for name in FORCEABLE_LINKS
        }

        joint_names = [
            self.model.joint(i).name
            for i in range(self.model.njnt)
            if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
        ]
        velocity_limits = {name: np.pi * 2 for name in joint_names}
        self.velocity_limit = mink.VelocityLimit(self.model, velocity_limits)
        self.limits = [mink.ConfigurationLimit(self.model), self.velocity_limit]
        self.repeat_frame_time = repeat_frame_time
        self.static_qpos_ref: Optional[np.ndarray] = None
        self.static_foot_contacts_ref: Optional[np.ndarray] = None
        self.motion_lib = None
        if motion_path:
            self._load_motion(motion_path)
        self._initialize_reference_pose()

    def _load_motion(self, motion_path: str):
        if ProceduralMotionLibFromDemo is None or not os.path.exists(motion_path):
            return
        print(f"Loading reference motion from: {motion_path}")
        joint_config = JointConfig(
            num_joints=29,
            left_leg_indices={
                "hip_yaw": 2,
                "hip_roll": 1,
                "hip_pitch": 0,
                "knee": 3,
                "ankle_pitch": 4,
                "ankle_roll": 5,
            },
            right_leg_indices={
                "hip_yaw": 8,
                "hip_roll": 7,
                "hip_pitch": 6,
                "knee": 9,
                "ankle_pitch": 10,
                "ankle_roll": 11,
            },
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
        self.motion_lib = ProceduralMotionLibFromDemo(
            input_path=motion_path,
            device="cpu",
            motion_dt=0.02,
            n_future_steps=0,
            joint_config=joint_config,
            feet_contacts=True,
            reindex_mapping=None,
        )
        print("Motion library loaded successfully.")

    def _initialize_reference_pose(self):
        qpos_ref, _, _ = self.get_reference_motion(0.0)
        self.configuration.update(q=qpos_ref)
        mujoco.mj_forward(self.model, self.data)

    def get_reference_motion(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.repeat_frame_time is not None:
            if self.static_qpos_ref is None:
                q_ref, _, c_ref = self._get_raw_motion_from_lib(self.repeat_frame_time)
                self.static_qpos_ref = q_ref.copy()
                self.static_foot_contacts_ref = c_ref.copy()
            return (
                self.static_qpos_ref,
                np.zeros(self.model.nv),
                self.static_foot_contacts_ref,
            )
        if not self.motion_lib:
            q_ref = np.zeros(self.model.nq)
            q_ref[2] = 0.77
            q_ref[3] = 1.0
            return q_ref, np.zeros(self.model.nv), np.array([1.0, 1.0])
        return self._get_raw_motion_from_lib(t)

    def _get_raw_motion_from_lib(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        motion_data = self.motion_lib.get_motion_state(
            torch.tensor([0]), torch.tensor([t], dtype=torch.float32)
        )
        root_pos = motion_data["root_pos"].squeeze(0).cpu().numpy()[0]
        root_rot = motion_data["root_rot"].squeeze(0).cpu().numpy()[0]
        dof_pos = motion_data["dof_pos"].squeeze(0).cpu().numpy()[0]
        foot_contacts = motion_data["foot_contacts"].squeeze(0).cpu().numpy()[0]
        qpos_ref_new = np.zeros(self.model.nq)
        qpos_ref_new[0:3] = root_pos
        qpos_ref_new[3:7] = root_rot[[3, 0, 1, 2]]
        min_dofs = min(len(dof_pos), self.num_dofs)
        for i in range(min_dofs):
            self_idx = self.actuated_qpos_indices[i]
            qpos_ref_new[self_idx] = dof_pos[i]
        return qpos_ref_new, np.zeros(self.model.nv), foot_contacts
