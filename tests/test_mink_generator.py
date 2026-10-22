import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import mujoco


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODULE_ENV_VAR = "MINK_GENERATOR_MODULE"
DEFAULT_MODULE = "compliant_motion_augmentation.mink_generator_ff"

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

    num_joints = 29  # this test generates against the 29-DOF model
    ref_cols = 7 + num_joints + 2
    if mode != "forcefield":
        # These modes are unforced for this static reference. Preserve exact
        # release parity rather than replacing the original golden fixtures.
        assert generated.shape == expected.shape
        np.testing.assert_allclose(generated, expected, rtol=1e-6, atol=1e-7)
    else:
        # The old forcefield golden contains duplicated rewind frames and the
        # old release limiter. Check the corrected timeline against the unchanged
        # nominal clip, then constrain the adapted result independently via FK.
        nominal_path = paths["release_root"] / "zero-wrench/stand/stand_augmented_mink_001.csv"
        nominal = np.loadtxt(nominal_path, delimiter=",")
        assert generated.shape == nominal.shape
        np.testing.assert_allclose(generated[:, :ref_cols], nominal[:, :ref_cols],
                                   rtol=1e-6, atol=1e-7)

    assert np.isfinite(generated).all()
    force_col = ref_cols + 7 + num_joints
    for quat_col in (3, ref_cols + 3, force_col + 14):
        np.testing.assert_allclose(np.linalg.norm(generated[:, quat_col:quat_col+4], axis=1),
                                   1., rtol=0., atol=1e-6)

    # Decay and IK share one complete-frame velocity budget. A kinematic rate
    # bound does not by itself establish dynamic feasibility.
    from compliant_motion_augmentation.constants import MAX_IK_JOINT_VEL
    adapted_joints = generated[:, ref_cols + 7 : ref_cols + 7 + num_joints]
    fd_vel = np.abs(np.diff(adapted_joints, axis=0)) * 30.0  # generation fps
    vel_limit = MAX_IK_JOINT_VEL
    assert fd_vel.max() <= vel_limit + 1e-6, (
        f"adapted reference contains joint jumps of {fd_vel.max():.1f} rad/s "
        f"(limit {vel_limit:.1f}); release/rewind discontinuity regression"
    )

    if mode == "forcefield":
        model = mujoco.MjModel.from_xml_path(str(paths["model_path"]))
        reference_data, adapted_data = mujoco.MjData(model), mujoco.MjData(model)
        feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
        joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        assert np.all(adapted_joints >= model.jnt_range[joint_ids, 0] - 1e-6)
        assert np.all(adapted_joints <= model.jnt_range[joint_ids, 1] + 1e-6)
        foot_errors, force_target_errors = [], []
        from compliant_motion_augmentation.collision_checks import self_collision_pairs
        for row in generated:
            for data, offset in ((reference_data, 0), (adapted_data, ref_cols)):
                data.qpos[:3] = row[offset:offset+3]
                data.qpos[3:7] = row[[offset+6, offset+3, offset+4, offset+5]]
                data.qpos[7:] = row[offset+7:offset+7+num_joints]
                mujoco.mj_forward(model, data)
            assert not self_collision_pairs(model, adapted_data)
            foot_errors.append(np.linalg.norm(adapted_data.xpos[feet] - reference_data.xpos[feet], axis=1).max())
            force = row[force_col+1:force_col+4]
            body = int(row[force_col])
            if np.linalg.norm(force) > 1.:
                assert 0 < body < model.nbody
                stiffness = row[force_col+7]
                assert stiffness > 0
                force_target_errors.append(np.linalg.norm(
                    adapted_data.xpos[body] - reference_data.xpos[body] - force/stiffness))
        assert force_target_errors, "forcefield fixture must exercise nonzero forcing"
        # Collision avoidance changes the accepted trajectory: the old seed's
        # incidental 3cm F/K bound is not the generator's feasibility contract.
        # Keep the production 5cm bound unchanged and explicitly check collision
        # rejection above, instead of matching an unconstrained trajectory.
        from compliant_motion_augmentation.constants import IK_CHECK_MAX_IK_TRACKING_ERROR
        assert max(foot_errors) < .025
        assert max(force_target_errors) <= IK_CHECK_MAX_IK_TRACKING_ERROR + 1e-6
