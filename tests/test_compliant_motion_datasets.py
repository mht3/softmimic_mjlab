"""Content checks for the generated compliant-motion datasets.

These guard against the failure mode where the augmentation pipeline silently
emits a frozen stand pose for every task (see
``test_ik_solver_motion_loading``). A "walk" dataset must actually translate the
root; a "stand" dataset must stay in place. The checks run against whatever
stage of the pipeline is present:

* the augmented 29-DOF CSVs under ``compliant_motion_augmentation/release_examples``
* the 23-DOF augmented CSVs under ``src/assets/compliant_motions/g1_23dof``
* the training NPZs (beside the CSVs) under ``src/assets/compliant_motions/g1_23dof``

Each stage is skipped (not failed) when its files are absent, so the suite is
useful both before and after a regeneration.
"""

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

RELEASE_ROOT = REPO_ROOT / "compliant_motion_augmentation" / "release_examples"
# 23-DOF augmented CSVs and their NPZs live side by side under g1_23dof/.
DATA23_ROOT = REPO_ROOT / "src" / "assets" / "compliant_motions" / "g1_23dof"
CSV23_ROOT = DATA23_ROOT
NPZ_ROOT = DATA23_ROOT

# A walking clip should move the root at least this far (m) over the clip; a
# standing clip should stay well under the stand tolerance. The walk reference
# travels several metres, so this margin is comfortable either way.
WALK_MIN_DISPLACEMENT = 1.0
STAND_MAX_DISPLACEMENT = 0.2

# 29-DOF augmented CSV layout (see scripts/csv_29dof_to_23dof.py).
_AUG29_REF_BASE = slice(0, 3)  # reference root xyz
# 23-DOF augmented CSV layout: reference block is 7 base + 23 joints + 2 contact.
_AUG23_REF_BASE = slice(0, 3)


def _root_xy_displacement_csv(csv_path: Path, base_cols: slice) -> float:
  data = np.loadtxt(csv_path, delimiter=",", ndmin=2)
  root_xy = data[:, base_cols][:, :2]
  return float(np.linalg.norm(root_xy[-1] - root_xy[0]))


def _first_file(root: Path, *parts: str, ext: str = "csv") -> Path | None:
  # CSVs and NPZs now share a directory, so callers pick the type explicitly.
  d = root.joinpath(*parts)
  if not d.is_dir():
    return None
  files = sorted(d.glob(f"*.{ext}"))
  return files[0] if files else None


# --- Augmented 29-DOF CSVs (release_examples) -------------------------------


@pytest.mark.parametrize("mode", ["zero-wrench", "forcefield", "collision-emulator"])
def test_release_walk_translates(mode):
  f = _first_file(RELEASE_ROOT, mode, "walk")
  if f is None:
    pytest.skip(f"no release_examples/{mode}/walk CSVs")
  disp = _root_xy_displacement_csv(f, _AUG29_REF_BASE)
  assert disp >= WALK_MIN_DISPLACEMENT, (
    f"walk reference barely moved ({disp:.3f} m) in {f} — likely the frozen "
    "stand-pose fallback"
  )


@pytest.mark.parametrize("mode", ["zero-wrench", "forcefield", "collision-emulator"])
def test_release_stand_is_static(mode):
  f = _first_file(RELEASE_ROOT, mode, "stand")
  if f is None:
    pytest.skip(f"no release_examples/{mode}/stand CSVs")
  disp = _root_xy_displacement_csv(f, _AUG29_REF_BASE)
  assert disp <= STAND_MAX_DISPLACEMENT, (
    f"stand reference drifted {disp:.3f} m in {f}"
  )


def test_release_walk_and_stand_differ():
  walk = _first_file(RELEASE_ROOT, "zero-wrench", "walk")
  stand = _first_file(RELEASE_ROOT, "zero-wrench", "stand")
  if walk is None or stand is None:
    pytest.skip("release_examples zero-wrench walk/stand missing")
  w = np.loadtxt(walk, delimiter=",", ndmin=2)
  s = np.loadtxt(stand, delimiter=",", ndmin=2)
  assert not (w.shape == s.shape and np.allclose(w, s)), (
    "walk and stand release CSVs are identical — the walk motion never loaded"
  )


# --- 23-DOF CSVs ------------------------------------------------------------


@pytest.mark.parametrize("mode", ["zero-wrench", "forcefield", "collision-emulator"])
def test_csv23_walk_translates(mode):
  f = _first_file(CSV23_ROOT, "walk", mode, ext="csv")
  if f is None:
    pytest.skip(f"no g1_23dof/walk/{mode} CSVs")
  disp = _root_xy_displacement_csv(f, _AUG23_REF_BASE)
  assert disp >= WALK_MIN_DISPLACEMENT, f"23-DOF walk barely moved ({disp:.3f} m)"


# --- Training NPZs ----------------------------------------------------------


def _root_xy_displacement_npz(npz_path: Path) -> float:
  d = np.load(npz_path)
  root_xy = d["body_pos_w"][:, 0, :2]
  return float(np.linalg.norm(root_xy[-1] - root_xy[0]))


def test_npz_walk_translates():
  f = _first_file(NPZ_ROOT, "walk", "zero-wrench", ext="npz")
  if f is None:
    pytest.skip("no g1_23dof/walk/zero-wrench NPZs")
  disp = _root_xy_displacement_npz(f)
  assert disp >= WALK_MIN_DISPLACEMENT, (
    f"walk NPZ root barely moved ({disp:.3f} m) — frozen stand fallback"
  )


def test_npz_stand_is_static():
  f = _first_file(NPZ_ROOT, "stand", "zero-wrench", ext="npz")
  if f is None:
    pytest.skip("no g1_23dof/stand/zero-wrench NPZs")
  disp = _root_xy_displacement_npz(f)
  assert disp <= STAND_MAX_DISPLACEMENT, f"stand NPZ drifted {disp:.3f} m"


def test_npz_walk_joints_move():
  """A real walk clip cycles the leg joints; the frozen fallback did not."""
  f = _first_file(NPZ_ROOT, "walk", "zero-wrench", ext="npz")
  if f is None:
    pytest.skip("no g1_23dof/walk/zero-wrench NPZs")
  d = np.load(f)
  assert d["joint_pos"].std(axis=0).max() > 0.05, (
    "walk NPZ joints are frozen — not a real motion"
  )
