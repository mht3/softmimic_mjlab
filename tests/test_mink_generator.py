import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODULE_ENV_VAR = "MINK_GENERATOR_MODULE"
DEFAULT_MODULE = "compliant_motion_augmentation.mink_generator_ff"

@pytest.fixture(scope="module", autouse=True)
def ensure_softmimic_symlink():
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
def generator_module():
    module_name = os.environ.get(MODULE_ENV_VAR, DEFAULT_MODULE)
    return importlib.import_module(module_name)


def _resolve_motions_dir(repo_root: Path) -> Path | None:
    """Locate the reference motion CSVs (mirrors generate_all.sh).

    The CSVs live in the sibling ``softmimic`` repo, not under this repo, so
    they are not always present. ``SOFTMIMIC_MOTIONS_DIR`` overrides the guess.
    """
    override = os.environ.get("SOFTMIMIC_MOTIONS_DIR")
    candidates = [
        Path(override) if override else None,
        repo_root / "datasets" / "motions_csv",
        repo_root.parent / "softmimic" / "datasets" / "motions_csv",
    ]
    for c in candidates:
        if c is not None and (c / "stand.csv").is_file():
            return c
    return None


@pytest.fixture(scope="module")
def paths(generator_module):
    module_dir = Path(generator_module.__file__).resolve().parent
    repo_root = module_dir.parent
    motions_dir = _resolve_motions_dir(repo_root)
    if motions_dir is None:
        pytest.skip(
            "reference motion CSVs not found; set SOFTMIMIC_MOTIONS_DIR to the "
            "softmimic datasets/motions_csv directory"
        )
    model_path = (
        repo_root
        / "softmimic_deploy"
        / "src"
        / "assets"
        / "g1"
        / "g1_29dof.xml"
    )
    release_root = module_dir / "release_examples"
    return {
        "module_dir": module_dir,
        "motion_path": motions_dir / "stand.csv",
        "model_path": model_path,
        "release_root": release_root,
    }


@pytest.mark.parametrize("mode", ["zero-wrench", "forcefield", "collision-emulator"])
def test_generated_csv_matches_release_example(generator_module, paths, tmp_path, mode):
    SimulationConfig = generator_module.SimulationConfig
    run_simulation_or_generation = generator_module.run_simulation_or_generation

    config = SimulationConfig(
        mode="generate-data",
        model_path=str(paths["model_path"]),
        motion_path=str(paths["motion_path"]),
        seed=42,
        com_cost=0.1,
        com_cost_z_factor=1e-5,
        force_mode=mode,
        upper_joint_cost=0.0,
        torso_orientation_cost=0.0,
        repeat_frame_time=None,
        record_video=False,
        output_filename="ik_simulation.mp4",
        num_files=1,
        output_dir=str(tmp_path),
    )

    run_simulation_or_generation(config, file_index=0)

    generated_path = tmp_path / "stand_augmented_mink_001.csv"
    expected_path = paths["release_root"] / mode / "stand" / "stand_augmented_mink_001.csv"

    assert generated_path.exists(), f"Generated file missing: {generated_path}"
    assert expected_path.exists(), f"Reference file missing: {expected_path}"

    generated = np.loadtxt(generated_path, delimiter=",")
    expected = np.loadtxt(expected_path, delimiter=",")

    assert generated.shape == expected.shape
    np.testing.assert_allclose(generated, expected, rtol=1e-6, atol=1e-7)
