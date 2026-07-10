"""Convert a 29-DOF G1 motion CSV to a 23-DOF CSV by dropping 6 joints.

Dropped joints (indices 13, 14, 20, 21, 27, 28 in the 29-joint block):
  waist_roll_joint, waist_pitch_joint,
  left_wrist_pitch_joint, left_wrist_yaw_joint,
  right_wrist_pitch_joint, right_wrist_yaw_joint

Supported input formats (auto-detected by column count):
  36 cols: 7 base + 29 joints (LAFAN1 retarget)              -> 30 cols
  38 cols: 7 base + 29 joints + 2 foot contacts (softmimic)  -> 32 cols
  95 cols: augmented compliant motion
           38 (reference) + 36 (adapted) + 9 (force) + 12 (forcefield)
           -> 32 + 30 + 9 + 12 = 83 cols
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
CONTACT_COLS = 2  # left/right foot contact flags
FORCE_COLS = 9  # 1 link id + 3 force + 3 torque + 1 stiff + 1 rot stiff
FORCEFIELD_COLS = 12  # 2 stiffness + 3 origin + 4 setpoint quat + 3 normal

# Column counts of the supported 29-DOF input formats.
PLAIN_29 = BASE_COLS + 29  # 36
CONTACT_29 = BASE_COLS + 29 + CONTACT_COLS  # 38
AUGMENTED_29 = CONTACT_29 + PLAIN_29 + FORCE_COLS + FORCEFIELD_COLS  # 95


def _filter_block(block: np.ndarray, num_joints: int) -> np.ndarray:
  """Keep base cols + 23 of 29 joints (+ any trailing cols) of a motion block."""
  assert num_joints == 29
  base = block[:, :BASE_COLS]
  joints = block[:, BASE_COLS : BASE_COLS + num_joints]
  trailing = block[:, BASE_COLS + num_joints :]
  return np.hstack([base, joints[:, KEEP_JOINT_IDX], trailing])


def convert_array(data: np.ndarray) -> np.ndarray:
  """Convert a 29-DOF motion array to 23-DOF, auto-detecting the format."""
  num_cols = data.shape[1]
  if num_cols == PLAIN_29:
    out = _filter_block(data, 29)
    assert out.shape[1] == BASE_COLS + 23
  elif num_cols == CONTACT_29:
    out = _filter_block(data, 29)
    assert out.shape[1] == BASE_COLS + 23 + CONTACT_COLS
  elif num_cols == AUGMENTED_29:
    ref = _filter_block(data[:, :CONTACT_29], 29)
    adapted = _filter_block(data[:, CONTACT_29 : CONTACT_29 + PLAIN_29], 29)
    rest = data[:, CONTACT_29 + PLAIN_29 :]
    out = np.hstack([ref, adapted, rest])
    assert out.shape[1] == 32 + 30 + FORCE_COLS + FORCEFIELD_COLS
  else:
    raise ValueError(
      f"Unsupported column count {num_cols}; expected one of "
      f"{PLAIN_29} (plain), {CONTACT_29} (with contacts), {AUGMENTED_29} (augmented)"
    )
  return out


def convert(input_file: Path, output_file: Path) -> None:
  data = np.loadtxt(input_file, delimiter=",", ndmin=2)
  out = convert_array(data)
  output_file.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(output_file, out, delimiter=",")
  print(f"Wrote {out.shape[0]} frames x {out.shape[1]} cols -> {output_file}")


def convert_dir(input_dir: Path, output_dir: Path) -> None:
  """Convert every CSV under input_dir, mirroring the directory layout."""
  csv_files = sorted(input_dir.rglob("*.csv"))
  if not csv_files:
    raise FileNotFoundError(f"No CSV files found under {input_dir}")
  for csv_file in csv_files:
    convert(csv_file, output_dir / csv_file.relative_to(input_dir))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input-file",
    type=Path,
    help="Path to 29-DOF CSV.",
  )
  parser.add_argument(
    "--output-file",
    type=Path,
    help="Path to write 23-DOF CSV.",
  )
  parser.add_argument(
    "--input-dir",
    type=Path,
    help="Convert all CSVs under this directory (recursively).",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    help="Output directory mirroring the input layout (with --input-dir).",
  )
  args = parser.parse_args()
  if args.input_dir is not None:
    if args.output_dir is None:
      parser.error("--output-dir is required with --input-dir")
    convert_dir(args.input_dir, args.output_dir)
  else:
    if args.input_file is None or args.output_file is None:
      parser.error("--input-file and --output-file are required")
    convert(args.input_file, args.output_file)
