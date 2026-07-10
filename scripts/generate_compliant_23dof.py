"""End-to-end 23-DOF compliant-motion augmentation pipeline.

Runs the reordered pipeline that solves inverse kinematics directly on the
23-DOF G1 (the robot we deploy) rather than on the 29-DOF model:

    29-DOF reference CSV
      -> filter to 23-DOF reference CSV      (scripts/csv_29dof_to_23dof.py)
      -> mink augmentation on g1_23dof.xml    (compliant_motion_augmentation)
      -> convert to training NPZ              (scripts/compliant_csv_to_npz.py)

The augmented CSVs and their NPZs are written side by side under

    src/assets/compliant_motions/g1_23dof/<motion>/<mode>/
        <motion>_augmented_mink_NNN.csv
        <motion>_augmented_mink_NNN.npz

Example:
    python scripts/generate_compliant_23dof.py \
        --motions stand walk \
        --modes forcefield collision-emulator zero-wrench \
        --num-files 40 40 5
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUG_DIR = REPO_ROOT / "compliant_motion_augmentation"
OUT_ROOT = REPO_ROOT / "src" / "assets" / "compliant_motions" / "g1_23dof"
REF23_DIR = REPO_ROOT / "src" / "assets" / "compliant_motions_ref23"
DEFAULT_MOTIONS_DIR = REPO_ROOT / "datasets" / "motions_csv"

PYTHON = sys.executable


def _run(cmd, cwd=None, env=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def filter_reference(motion: str, motions_dir: Path) -> Path:
    """Filter a 29-DOF reference CSV to a 23-DOF one (kept in its own folder)."""
    src = motions_dir / f"{motion}.csv"
    if not src.is_file():
        raise FileNotFoundError(f"Reference motion not found: {src}")
    dst = REF23_DIR / f"{motion}.csv"
    _run([
        PYTHON, REPO_ROOT / "scripts" / "csv_29dof_to_23dof.py",
        "--input-file", src, "--output-file", dst,
    ])
    return dst


def augment(motion: str, mode: str, ref23: Path, num_files: int, seed: int) -> Path:
    """Run the mink augmentation on the 23-DOF model into the output folder."""
    out_dir = OUT_ROOT / motion / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    # The augmentation package resolves sibling imports relative to its own dir.
    _run([
        PYTHON, "mink_generator_ff.py", "generate-data",
        "--motion_path", ref23,
        "--force_mode", mode,
        "--num_files", num_files,
        "--seed", seed,
        "--output_dir", out_dir,
    ], cwd=AUG_DIR)
    return out_dir


def to_npz(out_dir: Path, input_fps: float, output_fps: float, device: str) -> None:
    """Convert every augmented CSV in out_dir to an NPZ beside it."""
    _run([
        PYTHON, REPO_ROOT / "scripts" / "compliant_csv_to_npz.py",
        "--input-dir", out_dir,
        "--output-dir", out_dir,  # CSV and NPZ live together
        "--input-fps", input_fps,
        "--output-fps", output_fps,
        "--device", device,
    ], cwd=REPO_ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motions", nargs="+", default=["stand", "walk"])
    p.add_argument("--modes", nargs="+", default=["forcefield", "collision-emulator", "zero-wrench"])
    p.add_argument("--num-files", nargs="+", type=int, default=None,
                   help="Files per mode (one int, or one per --modes). Default 40/40/5 for the standard modes, else 10.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--motions-dir", type=Path, default=DEFAULT_MOTIONS_DIR)
    p.add_argument("--input-fps", type=float, default=30.0)
    p.add_argument("--output-fps", type=float, default=50.0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip-npz", action="store_true", help="Only generate CSVs, skip NPZ conversion.")
    args = p.parse_args()

    # Resolve per-mode file counts.
    if args.num_files is None:
        default_counts = {"forcefield": 40, "collision-emulator": 40, "zero-wrench": 5}
        counts = [default_counts.get(m, 10) for m in args.modes]
    elif len(args.num_files) == 1:
        counts = args.num_files * len(args.modes)
    else:
        if len(args.num_files) != len(args.modes):
            p.error("--num-files must be a single value or one per --modes")
        counts = args.num_files

    print(f"Output root: {OUT_ROOT}")
    for motion in args.motions:
        ref23 = filter_reference(motion, args.motions_dir)
        for mode, n in zip(args.modes, counts):
            print(f"\n===== {motion} / {mode} ({n} files) =====")
            out_dir = augment(motion, mode, ref23, n, args.seed)
            if not args.skip_npz:
                to_npz(out_dir, args.input_fps, args.output_fps, args.device)

    print(f"\nDone. Datasets under {OUT_ROOT}")


if __name__ == "__main__":
    main()
