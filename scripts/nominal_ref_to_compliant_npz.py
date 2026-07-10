"""Convert a raw 23-DOF reference CSV to a nominal compliant-tracking NPZ.

Unlike ``scripts/compliant_csv_to_npz.py`` (which needs the 83-column augmented
CSV with force/forcefield blocks), this consumes the *raw* reference CSV in
``src/assets/compliant_motions_ref23/`` (32 columns: 7 base + 23 joints + 2 foot
contacts) and produces an NPZ suitable for deployment as an un-augmented
reference. It replays the reference through the same MuJoCo FK pipeline as
``scripts/csv_to_npz.py`` to fill the BeyondMimic body arrays, then appends the
compliance channels the ``CompliantMotionLoader`` requires, all set to their
inert (zero-wrench / no-forcefield) values.

Usage:
  python scripts/nominal_ref_to_compliant_npz.py \
    --input-file src/assets/compliant_motions_ref23/stand.csv \
    --output-file deploy/robots/g1_23dof/config/policy/compliant_mimic/stand/params/stand.npz \
    --input-fps 30 --output-fps 50 --num-frames 50
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from src.tasks.tracking.config.g1_23dof.env_cfgs import (
  unitree_g1_23dof_flat_tracking_env_cfg,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csv_to_npz import MotionLoader  # noqa: E402

# Raw reference CSV layout (23-DOF): 7 base + 23 joints + 2 foot contacts.
NUM_JOINTS = 23
REF_COLS = 7 + NUM_JOINTS  # 30 (base + joints; MotionLoader reads [:, :30])
FOOT_CONTACT_COLS = slice(30, 32)  # [left, right]
TOTAL_COLS = 32

# Same order csv_to_npz.py / compliant_csv_to_npz.py use for the 23-DOF model.
JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
]


class _RefMotionLoader(MotionLoader):
  """MotionLoader over an in-memory raw reference block (base + joints)."""

  def __init__(self, motion: np.ndarray, input_fps: float, output_fps: float, device):
    self._motion_array = motion
    super().__init__(
      motion_file="<in-memory>",
      input_fps=input_fps,
      output_fps=output_fps,
      device=device,
    )

  def _load_motion(self):
    motion = torch.from_numpy(self._motion_array).to(torch.float32).to(self.device)
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz
    self.motion_dof_poss_input = motion[:, 7:REF_COLS]
    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt


def main(
  input_file: str = "src/assets/compliant_motions_ref23/stand.csv",
  output_file: str = (
    "deploy/robots/g1_23dof/config/policy/compliant_mimic/stand/params/stand.npz"
  ),
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  num_frames: int | None = 50,
  device: str = "cuda:0",
):
  """Convert a raw 23-DOF reference CSV to a nominal (inert) compliant NPZ.

  Args:
    input_file: Raw reference CSV (32 cols: 7 base + 23 joints + 2 foot contacts).
    output_file: Output NPZ path.
    input_fps: Frame rate of the CSV.
    output_fps: Desired output frame rate.
    num_frames: If set, use only the first ``num_frames`` input rows (the pose is
      static, so a short clip loops seamlessly). None uses the whole CSV.
    device: Device to use.
  """
  data = np.loadtxt(input_file, delimiter=",")
  assert data.shape[1] == TOTAL_COLS, (
    f"{input_file}: expected {TOTAL_COLS} cols (raw 23-DOF reference), "
    f"got {data.shape[1]}"
  )
  if num_frames is not None:
    data = data[:num_frames]

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  scene = Scene(unitree_g1_23dof_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]

  motion = _RefMotionLoader(
    data[:, :REF_COLS], input_fps=input_fps, output_fps=output_fps, device=sim.device
  )

  # Nearest input frame per output frame, for the foot-contact zero-order hold.
  times = np.arange(0, motion.duration, motion.output_dt)[: motion.output_frames]
  phase = times / motion.duration if motion.duration > 0 else np.zeros_like(times)
  frame_idx = np.round(phase * (motion.input_frames - 1)).astype(np.int64)

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }

  scene.reset()
  for _ in range(motion.output_frames):
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      _,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
    log["body_lin_vel_w"].append(
      robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
    )
    log["body_ang_vel_w"].append(
      robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
    )

  for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w"):
    log[k] = np.stack(log[k], axis=0)

  n = motion.output_frames

  # Inert compliance channels: zero wrench, no forcefield. Force maps to the
  # root body (index 0); at zero stiffness it is never applied. force_body_pos/
  # quat therefore just track the root's FK pose. ff_setpoint_rot is identity
  # (wxyz), ff_normal a unit +z so no NaNs downstream even though inactive.
  log["force_body_index"] = np.zeros(n, dtype=np.int64)
  log["force_vector"] = np.zeros((n, 3), dtype=np.float32)
  log["torque_vector"] = np.zeros((n, 3), dtype=np.float32)
  log["stiffness"] = np.zeros(n, dtype=np.float32)
  log["rotational_stiffness"] = np.zeros(n, dtype=np.float32)
  log["ff_stiffness"] = np.zeros(n, dtype=np.float32)
  log["ff_rotational_stiffness"] = np.zeros(n, dtype=np.float32)
  log["ff_origin"] = np.zeros((n, 3), dtype=np.float32)
  ff_setpoint_rot = np.zeros((n, 4), dtype=np.float32)
  ff_setpoint_rot[:, 0] = 1.0  # identity wxyz
  log["ff_setpoint_rot"] = ff_setpoint_rot
  ff_normal = np.zeros((n, 3), dtype=np.float32)
  ff_normal[:, 2] = 1.0
  log["ff_normal"] = ff_normal
  log["force_body_pos_w"] = log["body_pos_w"][:, 0].copy()
  log["force_body_quat_w"] = log["body_quat_w"][:, 0].copy()
  log["foot_contacts"] = data[:, FOOT_CONTACT_COLS][frame_idx].astype(np.float32)

  out_path = Path(output_file)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out_path, **log)  # type: ignore[arg-type]
  print(f"Wrote nominal compliant NPZ ({n} frames) -> {out_path}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
