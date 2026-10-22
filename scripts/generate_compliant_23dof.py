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
from datetime import datetime
import json
import hashlib
import shutil
import tempfile
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
    # Child scripts must resolve this checkout even when an editable install
    # still points at an old workspace or the caller did not set PYTHONPATH.
    child_env = dict(os.environ if env is None else env)
    import_roots = [str(REPO_ROOT), str(REPO_ROOT / "algorithms" / "rsl_rl")]
    if child_env.get("PYTHONPATH"):
        import_roots.append(child_env["PYTHONPATH"])
    child_env["PYTHONPATH"] = os.pathsep.join(import_roots)
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=child_env, check=True)


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


def augment(motion: str, mode: str, ref23: Path, num_files: int, seed: int, output_root: Path = OUT_ROOT, torso_orientation_cost: float = 0.) -> Path:
    """Run the mink augmentation on the 23-DOF model into the output folder."""
    out_dir = output_root.resolve() / motion / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    # The augmentation package resolves sibling imports relative to its own dir.
    _run([
        PYTHON, "mink_generator_ff.py", "generate-data",
        "--motion_path", ref23,
        "--force_mode", mode,
        "--num_files", num_files,
        "--seed", seed,
        "--output_dir", out_dir,
        "--torso_orientation_cost", torso_orientation_cost,
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
    p.add_argument("--modes", nargs="+", choices=["triangle","forcefield","collision-emulator","collision-emulator-1d","zero-wrench"], default=["forcefield", "collision-emulator", "zero-wrench"])
    p.add_argument("--num-files", nargs="+", type=int, default=None,
                   help="Files per mode (one int, or one per --modes). Default 40/40/5 for the standard modes, else 10.")
    p.add_argument("--replace", action="store_true", help="Stage and audit a complete replacement, then back up and replace each requested motion directory")
    p.add_argument("--resume-stage", type=Path, help="Reuse completed CSV modes in a failed .generate-* stage; requires --replace and one motion")
    p.add_argument("--output-root", type=Path, default=OUT_ROOT)
    p.add_argument("--torso-orientation-cost", type=float, default=0.)
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

    if any(n<1 for n in counts):p.error("file counts must be positive")
    if args.replace and args.skip_npz:p.error("--replace requires NPZ conversion")
    if args.resume_stage and (not args.replace or len(args.motions)!=1):
        p.error("--resume-stage requires --replace and exactly one motion")
    output_root=args.output_root.resolve()
    if args.resume_stage:
        resume=args.resume_stage.resolve()
        if not resume.is_dir() or resume.parent!=output_root or not resume.name.startswith('.generate-'):
            p.error("--resume-stage must be an existing .generate-* directory immediately under --output-root")
    output_root.mkdir(parents=True,exist_ok=True)
    print(f"Output root: {output_root}")
    for motion in args.motions:
        if Path(motion).name!=motion or motion in ('.','..'):p.error("motion must be a filename stem")
        # The existing training dataset remains available until every selected
        # mode has generated, converted and passed the structural audit.
        stage=resume if args.resume_stage else (Path(tempfile.mkdtemp(prefix='.generate-',dir=output_root)) if args.replace else output_root)
        request={'motion':motion,'modes':args.modes,'num_files':counts,'seed':args.seed,
                 'input_fps':args.input_fps,'output_fps':args.output_fps,
                 'torso_orientation_cost':args.torso_orientation_cost,
                 'source_sha256':{str(path.relative_to(REPO_ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(AUG_DIR.glob('*.py'))}}
        request_path=stage/'generation_request.json'
        legacy_resume=bool(args.resume_stage and not request_path.exists())
        if args.resume_stage and request_path.exists() and json.loads(request_path.read_text())!=request:
            raise ValueError("Resume parameters differ from the stage's saved generation_request.json")
        if args.replace:request_path.write_text(json.dumps(request,indent=2)+'\n')
        if legacy_resume:print('Resuming legacy stage: original generation settings have no saved manifest; supplied settings are recorded, not independently verified.')
        if args.resume_stage:
            unexpected=[p.name for p in (stage/motion).iterdir() if p.is_dir() and p.name not in args.modes]
            if unexpected:raise ValueError(f"Resume stage contains unrequested modes: {unexpected}")
        ref23 = filter_reference(motion, args.motions_dir)
        for mode, n in zip(args.modes, counts):
            print(f"\n===== {motion} / {mode} ({n} files) =====")
            out_dir=stage/motion/mode
            existing=sorted(out_dir.glob('*.csv')) if args.resume_stage else []
            if existing:
                expected={f'{motion}_augmented_mink_{i:03d}.csv' for i in range(1,n+1)}
                if {p.name for p in existing}!=expected:
                    raise ValueError(f"Incomplete or unexpected CSV set in {out_dir}; refusing to mix old and regenerated clips")
                print(f"Reusing {len(existing)} completed CSVs in {out_dir}; rebuilding NPZs")
            else:
                out_dir = augment(motion, mode, ref23, n, args.seed, stage, args.torso_orientation_cost)
            if not args.skip_npz:
                to_npz(out_dir, args.input_fps, args.output_fps, args.device)
        if args.replace:
            generated=stage/motion
            audit_path=generated/'audit.json'
            _run([PYTHON,REPO_ROOT/'scripts/audit_compliant_dataset.py',generated,
                  '--stride','1','--input-fps',args.input_fps,'--output',audit_path],cwd=REPO_ROOT)
            audit=json.loads(audit_path.read_text())
            if len(audit)!=sum(counts) or any('error' in row or not row.get('npz_finite',False)
                    or row.get('npz_joint_count')!=23 or not row.get('npz_has_nominal',False)
                    or row.get('npz_errors',[])
                    or row.get('max_joint_limit_violation_rad',0)>1e-3
                    or row.get('max_joint_speed_rad_s',0)>2 * 3.141592653589793 + 1e-5
                    or row.get('sampled_max_foot_displacement_m',0)>.05 + 1e-6
                    or row.get('sampled_max_F_over_K_error_m',0)>.05 + 1e-6 for row in audit):
                raise RuntimeError(f"Generated dataset failed structural or kinematic checks; old data retained. Inspect {audit_path}")
            destination=output_root/motion
            backup=output_root/'.backups'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')/motion
            if destination.exists():
                backup.parent.mkdir(parents=True,exist_ok=True)
                destination.rename(backup)
                print(f"Previous dataset backed up to {backup}")
            try:generated.rename(destination)
            except Exception:
                if backup.exists():backup.rename(destination)
                raise
            request_path.unlink(missing_ok=True)
            stage.rmdir()
            report_name=f'{motion}_dataset_audit.json' if output_root==OUT_ROOT.resolve() else f'{output_root.name}_{motion}_dataset_audit.json'
            report=REPO_ROOT/'logs'/report_name
            report.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(destination/'audit.json',report)
            manifest={'motion':motion,'modes':args.modes,'num_files':counts,'seed':args.seed,
                      'resumed_stage':str(args.resume_stage) if args.resume_stage else None,
                      'legacy_resume_unverified_generation_settings':legacy_resume,
                      'input_fps':args.input_fps,'output_fps':args.output_fps,
                      'torso_orientation_cost':args.torso_orientation_cost,
                      'model':'softmimic_deploy/src/assets/g1/g1_23dof.xml',
                      'reference_sha256':hashlib.sha256(ref23.read_bytes()).hexdigest(),
                      'source_sha256':{str(path.relative_to(REPO_ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(AUG_DIR.glob('*.py'))},
                      'generator_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'converter_script_sha256':hashlib.sha256((REPO_ROOT/'scripts/compliant_csv_to_npz.py').read_bytes()).hexdigest()}
            (destination/'generation.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(f"Replaced {destination}; audit: {report}")

    print(f"\nDone. Datasets under {output_root}")


if __name__ == "__main__":
    main()
