"""Force-link id decoding for the augmented-CSV -> NPZ converter.

The augmentation CSVs store MuJoCo body ids of the model the IK ran on. The
converter used to decode them with a hardcoded 29-DOF table, so ids written by
the 23-DOF pipeline (19 = left hand, 24 = right hand) were silently mapped to
shoulder links — training then applied every external wrench to the wrong
body. These tests pin the model-based decoding for both generations of CSVs.
"""

import sys
from pathlib import Path

import mujoco
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MODEL_23 = REPO_ROOT / "softmimic_deploy" / "src" / "assets" / "g1" / "g1_23dof.xml"
MODEL_29 = REPO_ROOT / "softmimic_deploy" / "src" / "assets" / "g1" / "g1_29dof.xml"


@pytest.fixture(scope="module")
def build_force_link_map():
  # compliant_csv_to_npz imports mjlab at module scope; the mapping helper
  # itself only needs mujoco, but skip cleanly where mjlab is unavailable.
  mod = pytest.importorskip("compliant_csv_to_npz")
  return mod.build_force_link_map


@pytest.fixture(scope="module")
def robot_body_names():
  """Training-robot body list: model bodies without the world body."""
  model = mujoco.MjModel.from_xml_path(str(MODEL_23))
  return [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    for i in range(1, model.nbody)
  ]


def test_23dof_ids_decode_to_hands(build_force_link_map, robot_body_names):
  link_map = build_force_link_map(MODEL_23, robot_body_names)
  # 23-DOF generation model: 19/24 are the wrist-roll hands, 14 the torso.
  assert link_map[19] == robot_body_names.index("left_wrist_roll_rubber_hand")
  assert link_map[24] == robot_body_names.index("right_wrist_roll_rubber_hand")
  assert link_map[14] == robot_body_names.index("torso_link")
  # The old 29-DOF table decoded 24 as the right shoulder — pin the regression.
  assert link_map[24] != robot_body_names.index("right_shoulder_pitch_link")


def test_29dof_ids_decode_via_name_fallback(build_force_link_map, robot_body_names):
  link_map = build_force_link_map(MODEL_29, robot_body_names)
  model29 = mujoco.MjModel.from_xml_path(str(MODEL_29))

  def id29(name: str) -> int:
    return mujoco.mj_name2id(model29, mujoco.mjtObj.mjOBJ_BODY, name)

  # 29-DOF wrist-yaw end effectors fall back to the 23-DOF rubber hands.
  assert link_map[id29("left_wrist_yaw_link")] == robot_body_names.index(
    "left_wrist_roll_rubber_hand"
  )
  assert link_map[id29("right_wrist_yaw_link")] == robot_body_names.index(
    "right_wrist_roll_rubber_hand"
  )
  assert link_map[id29("torso_link")] == robot_body_names.index("torso_link")


def test_unknown_ids_map_to_no_force(build_force_link_map, robot_body_names):
  link_map = build_force_link_map(MODEL_23, robot_body_names)
  assert link_map.get(-1, -1) == -1  # "no force" id passes through
  assert 0 not in link_map  # worldbody never maps to a robot body
