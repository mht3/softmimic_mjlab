import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from csv_29dof_to_23dof import (  # noqa: E402
  AUGMENTED_29,
  BASE_COLS,
  CONTACT_29,
  KEEP_JOINT_IDX,
  PLAIN_29,
  convert,
  convert_array,
)

DROPPED_JOINT_IDX = [13, 14, 20, 21, 27, 28]

# 29-DOF joint order of the LAFAN1 retargeting convention. The kept indices
# must drop exactly the waist roll/pitch and wrist pitch/yaw joints.
JOINT_NAMES_29 = [
  "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
  "left_ankle_pitch", "left_ankle_roll",
  "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
  "right_ankle_pitch", "right_ankle_roll",
  "waist_yaw", "waist_roll", "waist_pitch",
  "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
  "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
  "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
  "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]


def _tagged_motion_block(num_frames: int, joint_offset: float) -> np.ndarray:
  """Base cols hold negative tags; joint j holds joint_offset + j."""
  base = -np.arange(1, BASE_COLS + 1, dtype=float)[None, :].repeat(num_frames, 0)
  joints = joint_offset + np.arange(29, dtype=float)[None, :].repeat(num_frames, 0)
  return np.hstack([base, joints])


def test_keep_idx_drops_expected_joints():
  assert len(KEEP_JOINT_IDX) == 23
  assert sorted(KEEP_JOINT_IDX + DROPPED_JOINT_IDX) == list(range(29))
  dropped_names = [JOINT_NAMES_29[i] for i in DROPPED_JOINT_IDX]
  assert dropped_names == [
    "waist_roll", "waist_pitch",
    "left_wrist_pitch", "left_wrist_yaw",
    "right_wrist_pitch", "right_wrist_yaw",
  ]


def test_convert_plain_36col():
  data = _tagged_motion_block(4, joint_offset=100.0)
  out = convert_array(data)
  assert out.shape == (4, BASE_COLS + 23)
  np.testing.assert_array_equal(out[:, :BASE_COLS], data[:, :BASE_COLS])
  np.testing.assert_array_equal(
    out[:, BASE_COLS:], 100.0 + np.array(KEEP_JOINT_IDX, dtype=float)[None, :].repeat(4, 0)
  )


def test_convert_contacts_38col():
  contacts = np.full((4, 2), 0.5)
  data = np.hstack([_tagged_motion_block(4, 100.0), contacts])
  out = convert_array(data)
  assert out.shape == (4, BASE_COLS + 23 + 2)
  np.testing.assert_array_equal(out[:, :BASE_COLS], data[:, :BASE_COLS])
  np.testing.assert_array_equal(
    out[:, BASE_COLS : BASE_COLS + 23],
    100.0 + np.array(KEEP_JOINT_IDX, dtype=float)[None, :].repeat(4, 0),
  )
  np.testing.assert_array_equal(out[:, -2:], contacts)


def test_convert_augmented_95col():
  contacts = np.full((4, 2), 0.5)
  ref = np.hstack([_tagged_motion_block(4, 100.0), contacts])
  adapted = _tagged_motion_block(4, 200.0)
  force = 300.0 + np.arange(9, dtype=float)[None, :].repeat(4, 0)
  forcefield = 400.0 + np.arange(12, dtype=float)[None, :].repeat(4, 0)
  data = np.hstack([ref, adapted, force, forcefield])
  assert data.shape[1] == AUGMENTED_29

  out = convert_array(data)
  assert out.shape == (4, 83)

  kept = np.array(KEEP_JOINT_IDX, dtype=float)[None, :].repeat(4, 0)
  # Reference block: base + 23 kept joints + contacts.
  np.testing.assert_array_equal(out[:, :BASE_COLS], ref[:, :BASE_COLS])
  np.testing.assert_array_equal(out[:, BASE_COLS : BASE_COLS + 23], 100.0 + kept)
  np.testing.assert_array_equal(out[:, 30:32], contacts)
  # Adapted block: base + 23 kept joints.
  np.testing.assert_array_equal(out[:, 32 : 32 + BASE_COLS], adapted[:, :BASE_COLS])
  np.testing.assert_array_equal(out[:, 32 + BASE_COLS : 62], 200.0 + kept)
  # Force and forcefield blocks pass through unchanged.
  np.testing.assert_array_equal(out[:, 62:71], force)
  np.testing.assert_array_equal(out[:, 71:83], forcefield)


def test_convert_rejects_unknown_format():
  with pytest.raises(ValueError, match="Unsupported column count"):
    convert_array(np.zeros((2, 40)))


@pytest.mark.parametrize(
  "csv_rel",
  [
    "compliant_motion_augmentation/release_examples/forcefield/stand/stand_augmented_mink_001.csv",
    "compliant_motion_augmentation/release_examples/zero-wrench/stand/stand_augmented_mink_001.csv",
  ],
)
def test_convert_release_example_roundtrip(tmp_path, csv_rel):
  """End-to-end file conversion on a real augmented CSV."""
  src = REPO_ROOT / csv_rel
  if not src.exists():
    pytest.skip(f"release example not found: {src}")
  dst = tmp_path / "out.csv"
  convert(src, dst)
  original = np.loadtxt(src, delimiter=",")
  converted = np.loadtxt(dst, delimiter=",")
  assert original.shape[1] == AUGMENTED_29
  assert converted.shape == (original.shape[0], 83)
  # Kept joint columns of both motion blocks match the source values.
  np.testing.assert_allclose(
    converted[:, BASE_COLS : BASE_COLS + 23],
    original[:, BASE_COLS : BASE_COLS + 29][:, KEEP_JOINT_IDX],
  )
  np.testing.assert_allclose(
    converted[:, 32 + BASE_COLS : 62],
    original[:, CONTACT_29 + BASE_COLS : CONTACT_29 + PLAIN_29][:, KEEP_JOINT_IDX],
  )
  # Force + forcefield blocks are untouched.
  np.testing.assert_allclose(converted[:, 62:], original[:, CONTACT_29 + PLAIN_29 :])
