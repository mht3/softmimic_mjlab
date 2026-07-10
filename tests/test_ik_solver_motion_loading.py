"""Regression tests for reference-motion loading in the augmentation pipeline.

A silent fallback here previously produced a whole "walk" dataset that was
actually a frozen stand pose: ``G1_Mink_IK_Solver._load_motion`` returned early
when the motion CSV path did not exist, and ``get_reference_motion`` then served
a hardcoded standing pose. These tests pin the contract that a missing motion
path fails loudly instead.
"""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module", autouse=True)
def ensure_softmimic_symlink():
  # Mirror test_mink_generator: the augmentation package imports
  # ``softmimic_deploy.src...`` which is only importable via this symlink.
  symlink_dir = REPO_ROOT / "softmimic_deploy" / "softmimic_deploy"
  target_dir = REPO_ROOT / "softmimic_deploy"
  created = False
  if not symlink_dir.exists():
    symlink_dir.symlink_to(target_dir, target_is_directory=True)
    created = True
  try:
    yield
  finally:
    if created and symlink_dir.is_symlink():
      symlink_dir.unlink()


@pytest.fixture(scope="module")
def ik_solver_cls():
  mink = pytest.importorskip("mink", reason="augmentation pipeline needs mink")
  del mink
  module = importlib.import_module("compliant_motion_augmentation.ik_solver")
  return module.G1_Mink_IK_Solver


@pytest.fixture(scope="module")
def model_path():
  path = (
    REPO_ROOT / "softmimic_deploy" / "src" / "assets" / "g1" / "g1_29dof.xml"
  )
  if not path.exists():
    pytest.skip(f"G1 model not found at {path}")
  return str(path)


@pytest.fixture(scope="module")
def model_path_23dof():
  path = REPO_ROOT / "softmimic_deploy" / "src" / "assets" / "g1" / "g1_23dof.xml"
  if not path.exists():
    pytest.skip(f"G1 23-DOF model not found at {path}")
  return str(path)


def _resolve_stand_csv() -> Path | None:
  """Mirror generate_all.sh / test_mink_generator's motion-dir discovery."""
  import os

  override = os.environ.get("SOFTMIMIC_MOTIONS_DIR")
  for c in [
    Path(override) if override else None,
    REPO_ROOT / "datasets" / "motions_csv",
    REPO_ROOT.parent / "softmimic" / "datasets" / "motions_csv",
  ]:
    if c is not None and (c / "stand.csv").is_file():
      return c / "stand.csv"
  return None


def test_missing_motion_path_raises(ik_solver_cls, model_path, tmp_path):
  """A non-existent motion CSV must raise, not silently stand still."""
  missing = tmp_path / "does_not_exist.csv"
  with pytest.raises(FileNotFoundError):
    ik_solver_cls(model_path, motion_path=str(missing))


def test_no_motion_path_uses_static_pose(ik_solver_cls, model_path):
  """Passing no motion path is an explicit choice and stays a static pose.

  (This documents the intentional ``motion_path=None`` path so a future
  refactor doesn't conflate "no motion requested" with "motion not found".)
  """
  solver = ik_solver_cls(model_path, motion_path=None)
  assert solver.motion_lib is None


def test_23dof_loads_filtered_reference(ik_solver_cls, model_path_23dof, tmp_path):
  """The 23-DOF solver loads a filtered 23-DOF reference and exposes 23 DOFs.

  Exercises the reordered pipeline: filter a 29-DOF reference CSV to 23-DOF,
  then solve IK on the 23-DOF model. Also pins the forceable-link retarget
  (wrist-yaw links, absent on 23-DOF, become the wrist-roll hands).
  """
  import sys

  stand_csv = _resolve_stand_csv()
  if stand_csv is None:
    pytest.skip("reference motion CSVs not found; set SOFTMIMIC_MOTIONS_DIR")

  # Filter the 29-DOF reference to 23-DOF.
  sys.path.insert(0, str(REPO_ROOT / "scripts"))
  from csv_29dof_to_23dof import convert  # noqa: E402

  ref23 = tmp_path / "stand_23dof.csv"
  convert(stand_csv, ref23)

  solver = ik_solver_cls(str(model_path_23dof), motion_path=str(ref23))
  assert solver.num_dofs == 23
  assert solver.motion_lib is not None
  assert solver.motion_lib.joint_config.num_joints == 23

  # The wrist-yaw links do not exist on the 23-DOF model; the forceable links
  # must have been retargeted to bodies that do exist.
  import mujoco

  from compliant_motion_augmentation.constants import FORCEABLE_LINKS

  for link in FORCEABLE_LINKS:
    body_id = mujoco.mj_name2id(solver.model, mujoco.mjtObj.mjOBJ_BODY, link)
    assert body_id != -1, f"forceable link '{link}' missing from 23-DOF model"

  # The reference pose must be finite and correctly sized.
  qpos_ref, _, _ = solver.get_reference_motion(0.0)
  assert qpos_ref.shape[0] == solver.model.nq
  assert np.all(np.isfinite(qpos_ref))
