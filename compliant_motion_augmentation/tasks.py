from typing import List, Optional

import mujoco
import mujoco as mj
import numpy as np

import mink


class KneeBendingTask(mink.tasks.task.Task):
    """A custom mink task to penalize knee hyperextension."""

    def __init__(self, model: mj.MjModel, cost: float, joint_names: List[str]):
        super().__init__(cost=cost)
        self.model = model
        self.cost = np.array([cost] * len(joint_names), dtype=np.float32)
        self.joint_names = joint_names
        self.target_q: Optional[np.ndarray] = None
        self.dof_indices: List[int] = []
        for name in self.joint_names:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.dof_indices.append(self.model.jnt_dofadr[joint_id])
            except KeyError:
                print(
                    f"Warning: Knee joint '{name}' not found. It will be ignored by the KneeBendingTask."
                )
        self.num_knees = len(self.dof_indices)

    def set_target(self, target_q: np.ndarray):
        self.target_q = target_q

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        if self.target_q is None:
            return np.zeros(self.num_knees)
        error = np.zeros(self.num_knees)
        current_q = configuration.q
        for i, dof_idx in enumerate(self.dof_indices):
            joint_id = self.model.dof_jntid[dof_idx]
            qpos_adr = self.model.jnt_qposadr[joint_id]
            e = self.target_q[qpos_adr] - current_q[qpos_adr]
            error[i] = max(0.0, e)
        return error

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        jacobian = np.zeros((self.num_knees, self.model.nv))
        current_q = configuration.q
        for i, dof_idx in enumerate(self.dof_indices):
            joint_id = self.model.dof_jntid[dof_idx]
            qpos_adr = self.model.jnt_qposadr[joint_id]
            if current_q[qpos_adr] < self.target_q[qpos_adr]:
                jacobian[i, dof_idx] = -1.0
        return jacobian
