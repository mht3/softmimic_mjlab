"""Convert augmented compliant-motion CSVs (23-DOF) to training NPZ files.

Each augmented CSV (see ``scripts/csv_29dof_to_23dof.py``) holds:
  cols  0:32  reference motion (7 base + 23 joints + 2 foot contacts)
  cols 32:62  adapted motion   (7 base + 23 joints)
  cols 62:71  force block      (link id, force xyz, torque xyz, stiffness, rot stiffness)
  cols 71:83  forcefield block (2 stiffness, origin xyz, setpoint quat xyzw, normal xyz)

The adapted motion is replayed through the MuJoCo model (same FK pipeline as
``scripts/csv_to_npz.py``) to produce BeyondMimic-style body arrays, and the
compliance channels are resampled to the output fps and appended. Quaternions
are converted from the CSV's xyzw to wxyz.

Usage (directory of CSVs, mirrors layout). The CSVs and NPZs can share a
directory (``scripts/generate_compliant_23dof.py`` writes them side by side):
  python scripts/compliant_csv_to_npz.py \
    --input-dir src/assets/compliant_motions/g1_23dof/stand \
    --output-dir src/assets/compliant_motions/g1_23dof/stand \
    --input-fps 30 --output-fps 50
"""

from pathlib import Path
from typing import Any
import sys

# Also support direct invocation from any working directory without relying
# on an editable installation whose checkout may have moved.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "algorithms" / "rsl_rl")]

import mujoco
import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from src.tasks.tracking.config.g1_23dof.env_cfgs import (
  unitree_g1_23dof_flat_tracking_env_cfg,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csv_to_npz import MotionLoader  # noqa: E402
from compliant_motion_augmentation.collision_checks import self_collision_pairs

# Column layout of the 23-DOF augmented CSV.
NUM_JOINTS = 23
REF_FOOT_CONTACT_COLS = slice(30, 32)  # [left, right] contact probability
REF_COLS = 7 + NUM_JOINTS + 2  # 32
ADAPTED_COLS = slice(REF_COLS, REF_COLS + 7 + NUM_JOINTS)  # 32:62
FORCE_LINK_COL = 62
FORCE_VEC_COLS = slice(63, 66)
TORQUE_VEC_COLS = slice(66, 69)
STIFFNESS_COL = 69
ROT_STIFFNESS_COL = 70
FF_STIFFNESS_COL = 71
FF_ROT_STIFFNESS_COL = 72
FF_ORIGIN_COLS = slice(73, 76)
FF_SETPOINT_ROT_COLS = slice(76, 80)  # xyzw
FF_NORMAL_COLS = slice(80, 83)
TOTAL_COLS = 83

# CSV force-link ids are MuJoCo body ids of the model the augmentation ran on
# (worldbody = 0), so they must be decoded with that same model — see
# build_force_link_map(). The default is the 23-DOF generation model used by
# scripts/generate_compliant_23dof.py; pass --gen-model-xml with the 29-DOF XML
# when converting legacy 29-DOF-generated CSVs.
DEFAULT_GEN_MODEL_XML = (
  Path(__file__).resolve().parent.parent
  / "softmimic_deploy"
  / "src"
  / "assets"
  / "g1"
  / "g1_23dof.xml"
)

# Bodies that only exist on the 29-DOF model, mapped to the nearest 23-DOF
# equivalent (same table as compliant_motion_augmentation.constants).
LINK_NAME_29_TO_23 = {
  "left_wrist_yaw_link": "left_wrist_roll_rubber_hand",
  "right_wrist_yaw_link": "right_wrist_roll_rubber_hand",
}


def build_force_link_map(gen_model_xml, robot_body_names) -> dict[int, int]:
  """Map generation-model body ids (CSV link-id column) to robot body indices.

  The CSVs store ``mj_name2id(gen_model, ...)`` ids, so the only correct
  decode is ``mj_id2name`` on that same model followed by a name lookup on the
  training robot. 29-DOF-only bodies fall back via LINK_NAME_29_TO_23.
  """
  gen_model = mujoco.MjModel.from_xml_path(str(gen_model_xml))
  link_map: dict[int, int] = {}
  for body_id in range(gen_model.nbody):
    name = mujoco.mj_id2name(gen_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    name = LINK_NAME_29_TO_23.get(name, name)
    if name in robot_body_names:
      link_map[body_id] = robot_body_names.index(name)
  return link_map

# CSV joint column order (LAFAN1 retargeting convention, 23-DOF).
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


class _AdaptedMotionLoader(MotionLoader):
  """MotionLoader over the in-memory adapted block of an augmented CSV."""

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
    self.motion_dof_poss_input = motion[:, 7:]
    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt


def _nearest_input_frames(
  num_output_frames: int, num_input_frames: int, duration: float, output_dt: float
) -> np.ndarray:
  """Nearest input frame per output frame (zero-order hold for force events)."""
  times = np.arange(0, duration, output_dt)[:num_output_frames]
  phase = times / duration
  return np.round(phase * (num_input_frames - 1)).astype(np.int64)


def convert_file(
  csv_path: Path,
  output_path: Path,
  sim: Simulation,
  scene: Scene,
  robot_joint_indexes,
  force_link_map: dict[int, int],
  input_fps: float,
  output_fps: float,
  collision_model=None,
) -> None:
  data = np.loadtxt(csv_path, delimiter=",")
  assert data.shape[1] == TOTAL_COLS, (
    f"{csv_path}: expected {TOTAL_COLS} cols (23-DOF augmented), got {data.shape[1]}"
  )

  motion = _AdaptedMotionLoader(
    data[:, ADAPTED_COLS], input_fps=input_fps, output_fps=output_fps, device=sim.device
  )
  robot: Entity = scene["robot"]
  # Check resampled targets too: interpolation can cross the body even when
  # the original CSV endpoints are clear. Use the generator's collision mesh.
  collision_model = collision_model or mujoco.MjModel.from_xml_path(str(DEFAULT_GEN_MODEL_XML))
  collision_data = mujoco.MjData(collision_model)
  def check_pose(pos, quat, joints, frame, label):
    collision_data.qpos[:3] = pos.cpu().numpy().reshape(-1)
    collision_data.qpos[3:7] = quat.cpu().numpy().reshape(-1)
    collision_data.qpos[7:] = joints.cpu().numpy().reshape(-1)
    mujoco.mj_forward(collision_model, collision_data)
    pairs = self_collision_pairs(collision_model, collision_data)
    if pairs:
      raise ValueError(f"{csv_path}: {label} self-collision at output frame {frame}: {pairs}")

  frame_idx = _nearest_input_frames(
    motion.output_frames, motion.input_frames, motion.duration, motion.output_dt
  )

  # Map MuJoCo 29-DOF link ids to 23-DOF robot body indices (-1 = no force).
  raw_link_ids = data[:, FORCE_LINK_COL].astype(np.int64)[frame_idx]
  unknown = set(raw_link_ids.tolist()) - set(force_link_map) - {-1, 0}
  if unknown:
    raise ValueError(f"{csv_path}: unmapped force body IDs {sorted(unknown)}; check --gen-model-xml")
  force_body_index = np.array(
    [force_link_map.get(int(i), -1) for i in raw_link_ids], dtype=np.int64
  )

  setpoint_rot = data[:, FF_SETPOINT_ROT_COLS][frame_idx][:, [3, 0, 1, 2]]  # -> wxyz

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
    "force_body_pos_w": [],
    "force_body_quat_w": [],
  }

  scene.reset()
  for i in range(motion.output_frames):
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
    check_pose(motion_base_pos, motion_base_rot, motion_dof_pos, i, "adapted")

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
    body_idx = max(int(force_body_index[i]), 0)
    log["force_body_pos_w"].append(
      robot.data.body_link_pos_w[0, body_idx].cpu().numpy().copy()
    )
    log["force_body_quat_w"].append(
      robot.data.body_link_quat_w[0, body_idx].cpu().numpy().copy()
    )

  # Preserve the original reference for the actor. Adapted references remain
  # privileged reward/critic targets; exposing them to the actor leaks force.
  nominal = _AdaptedMotionLoader(
    data[:, :30], input_fps=input_fps, output_fps=output_fps, device=sim.device
  )
  for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
              "body_lin_vel_w", "body_ang_vel_w"):
    log["nominal_" + key] = []
  for nominal_frame in range(nominal.output_frames):
    (pos, quat, lin, ang, q, dq), _ = nominal.get_next_state()
    check_pose(pos, quat, q, nominal_frame, "nominal")
    root_states = robot.data.default_root_state.clone()
    root_states[:, :3] = pos + scene.env_origins
    root_states[:, 3:7] = quat
    root_states[:, 7:10] = lin
    root_states[:, 10:] = ang
    robot.write_root_state_to_sim(root_states)
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = q
    joint_vel[:, robot_joint_indexes] = dq
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    for key, value in (("joint_pos", robot.data.joint_pos),
                       ("joint_vel", robot.data.joint_vel),
                       ("body_pos_w", robot.data.body_link_pos_w),
                       ("body_quat_w", robot.data.body_link_quat_w),
                       ("body_lin_vel_w", robot.data.body_link_lin_vel_w),
                       ("body_ang_vel_w", robot.data.body_link_ang_vel_w)):
      log["nominal_" + key].append(value[0].cpu().numpy().copy())

  for k in list(log.keys()):
    if k != "fps":
      log[k] = np.stack(log[k], axis=0)

  log["force_body_index"] = force_body_index
  log["force_vector"] = data[:, FORCE_VEC_COLS][frame_idx]
  log["torque_vector"] = data[:, TORQUE_VEC_COLS][frame_idx]
  log["stiffness"] = data[:, STIFFNESS_COL][frame_idx]
  log["rotational_stiffness"] = data[:, ROT_STIFFNESS_COL][frame_idx]
  log["ff_stiffness"] = data[:, FF_STIFFNESS_COL][frame_idx]
  log["ff_rotational_stiffness"] = data[:, FF_ROT_STIFFNESS_COL][frame_idx]
  log["ff_origin"] = data[:, FF_ORIGIN_COLS][frame_idx]
  log["ff_setpoint_rot"] = setpoint_rot
  log["ff_normal"] = data[:, FF_NORMAL_COLS][frame_idx]
  log["foot_contacts"] = data[:, REF_FOOT_CONTACT_COLS][frame_idx]
  log["nominal_foot_contacts"] = log["foot_contacts"].copy()

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_path, **log)  # type: ignore[arg-type]


def main(
  input_dir: str | None = None,
  input_file: str | None = None,
  output_dir: str = "src/assets/compliant_motions/g1_23dof",
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  gen_model_xml: str = str(DEFAULT_GEN_MODEL_XML),
):
  """Convert 23-DOF augmented compliant-motion CSVs to NPZ training files.

  Args:
    input_dir: Directory of augmented 23-DOF CSVs (converted recursively,
      mirroring the directory layout, e.g. forcefield/collision-emulator/
      zero-wrench subdirectories).
    input_file: Single CSV to convert instead of a directory.
    output_dir: Output directory for the NPZ files.
    input_fps: Frame rate of the CSVs (augmentation runs at 30 fps).
    output_fps: Desired output frame rate (training runs at 50 fps).
    device: Device to use.
    gen_model_xml: MuJoCo XML the augmentation solved IK on; the CSV link-id
      column stores body ids of this model. Pass the 29-DOF XML for CSVs
      produced by the legacy 29-DOF pipeline.
  """
  if (input_dir is None) == (input_file is None):
    raise ValueError("Pass exactly one of --input-dir or --input-file.")

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  scene = Scene(unitree_g1_23dof_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]
  force_link_map = build_force_link_map(gen_model_xml, robot.body_names)

  out_root = Path(output_dir)
  if input_file is not None:
    files = [(Path(input_file), out_root / (Path(input_file).stem + ".npz"))]
  else:
    in_root = Path(input_dir)
    csvs = sorted(in_root.rglob("*.csv"))
    if not csvs:
      raise FileNotFoundError(f"No CSV files found under {in_root}")
    files = [
      (csv, (out_root / csv.relative_to(in_root)).with_suffix(".npz")) for csv in csvs
    ]

  for csv_path, output_path in tqdm(files, desc="Converting", unit="file"):
    convert_file(
      csv_path,
      output_path,
      sim,
      scene,
      robot_joint_indexes,
      force_link_map,
      input_fps,
      output_fps,
    )
  print(f"Converted {len(files)} files -> {out_root}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
