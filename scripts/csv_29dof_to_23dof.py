"""Convert a 29-DOF G1 motion CSV to a 23-DOF CSV by dropping 6 joints.

Dropped joints (indices 13, 14, 20, 21, 27, 28 in the 29-joint block):
  waist_roll_joint, waist_pitch_joint,
  left_wrist_pitch_joint, left_wrist_yaw_joint,
  right_wrist_pitch_joint, right_wrist_yaw_joint
"""

import argparse
from pathlib import Path

import numpy as np


# Indices into the 29-joint block to keep (0-indexed).
# Matches KEEP_29_TO_23_IDX from the legacy Isaac Lab converter.
KEEP_JOINT_IDX = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
  15, 16, 17, 18, 19,
  22, 23, 24, 25, 26,
]

BASE_COLS = 7  # 3 pos + 4 quat


def convert(input_file: Path, output_file: Path) -> None:
  data = np.loadtxt(input_file, delimiter=",")
  assert data.shape[1] == BASE_COLS + 29, (
    f"Expected 36 columns (7 base + 29 joints), got {data.shape[1]}"
  )

  base = data[:, :BASE_COLS]
  joints_29 = data[:, BASE_COLS:]
  joints_23 = joints_29[:, KEEP_JOINT_IDX]

  out = np.hstack([base, joints_23])
  assert out.shape[1] == BASE_COLS + 23

  output_file.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(output_file, out, delimiter=",")
  print(f"Wrote {out.shape[0]} frames x {out.shape[1]} cols -> {output_file}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input-file",
    type=Path,
    required=True,
    help="Path to 29-DOF CSV.",
  )
  parser.add_argument(
    "--output-file",
    type=Path,
    required=True,
    help="Path to write 23-DOF CSV.",
  )
  args = parser.parse_args()
  convert(args.input_file, args.output_file)
