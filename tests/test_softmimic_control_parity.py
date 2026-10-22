"""CPU-only contract checks against the retained joints of local SoftMimic."""
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "algorithms/rsl_rl")]

from mjlab.entity import Entity
from src.assets.robots.unitree_g1.g1_23dof_constants import get_g1_23dof_robot_cfg
from src.tasks.compliant_tracking.config.g1_23dof.rl_cfg import (
    unitree_g1_23dof_compliant_tracking_ppo_runner_cfg,
)


def test_softmimic_retained_joint_control_contract():
    cfg = get_g1_23dof_robot_cfg("softmimic")
    model = Entity(cfg).spec.compile()
    joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    assert len(joint_ids) == 23
    names = [model.joint(int(i)).name for i in joint_ids]
    expected_names = [
        f"{side}_{joint}_joint" for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ] + ["waist_yaw_joint"] + [
        f"{side}_{joint}_joint" for side in ("left", "right")
        for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
    ]
    assert names == expected_names
    expected_q = [-.20, 0., 0., .42, -.23, 0.] * 2 + [0.] + [
        .35, .16, 0., .87, 0., .35, -.16, 0., .87, 0.
    ]
    np.testing.assert_allclose(model.key_qpos[0, model.jnt_qposadr[joint_ids]], expected_q)
    assert model.key_qpos[0, 2] == .74
    expected_armature = [.01] * 13 + ([.01] * 4 + [.001]) * 2
    np.testing.assert_allclose(model.dof_armature[model.jnt_dofadr[joint_ids]], expected_armature)
    np.testing.assert_allclose(model.dof_frictionloss[model.jnt_dofadr[joint_ids]], 0.)
    np.testing.assert_allclose(model.dof_damping[model.jnt_dofadr[joint_ids]], 0.)
    expected_effort = [88., 88., 88., 139., 50., 50.] * 2 + [88.] + [25.] * 10
    expected_kp = [200., 150., 150., 200., 20., 20.] * 2 + [200.] + [40., 40., 40., 40., 5.] * 2
    expected_kd = [5., 5., 5., 5., 2., 2.] * 2 + [5.] + [5., 5., 5., 5., 1.] * 2
    for j, effort, kp, kd in zip(joint_ids, expected_effort, expected_kp, expected_kd):
        actuator = np.flatnonzero(model.actuator_trnid[:, 0] == j)
        assert len(actuator) == 1
        i = actuator[0]
        np.testing.assert_allclose(model.actuator_forcerange[i], [-effort, effort])
        assert model.actuator_gainprm[i, 0] == kp
        np.testing.assert_allclose(model.actuator_biasprm[i, 1:3], [-kp, -kd])
    np.testing.assert_allclose(cfg.hardware_stiffness, expected_kp)
    np.testing.assert_allclose(cfg.hardware_damping, expected_kd)
    assert all(c.contype == 0 and c.conaffinity == 1 for c in cfg.collisions)


def test_natural_frequency_profile_keeps_existing_pose_and_self_collisions():
    cfg = get_g1_23dof_robot_cfg("natural_freq")
    model = Entity(cfg).spec.compile()
    assert model.key_qpos[0, 2] == .8
    for name, expected in (("left_hip_pitch_joint", -.1), ("left_knee_joint", .3),
                           ("left_ankle_pitch_joint", -.2), ("left_shoulder_roll_joint", .18)):
        assert model.key_qpos[0, model.joint(name).qposadr[0]] == expected
    assert any(model.geom_contype != 0)
    assert model.dof_armature[model.joint("left_wrist_roll_joint").dofadr[0]] != .001


def test_softmimic_ppo_contract_and_requested_logging():
    cfg = unitree_g1_23dof_compliant_tracking_ppo_runner_cfg()
    assert not cfg.actor.obs_normalization and not cfg.critic.obs_normalization
    assert cfg.actor.hidden_dims == (512, 512, 256, 128)
    assert cfg.critic.hidden_dims == (512, 512, 512, 512)
    assert cfg.actor.activation == cfg.critic.activation == "elu"
    assert cfg.actor.distribution_cfg["init_std"] == 1.
    assert cfg.actor.distribution_cfg["std_type"] == "scalar"
    for field, expected in {
        "value_loss_coef": 1., "use_clipped_value_loss": True, "clip_param": .2,
        "entropy_coef": .002, "num_learning_epochs": 5, "num_mini_batches": 4,
        "learning_rate": .001, "schedule": "adaptive", "gamma": .99,
        "lam": .95, "desired_kl": .01, "max_grad_norm": 1.,
    }.items():
        assert getattr(cfg.algorithm, field) == expected
    assert cfg.algorithm.optimizer == "adam"
    assert not cfg.algorithm.normalize_advantage_per_mini_batch
    assert cfg.num_steps_per_env == 24 and cfg.save_interval == 500
    assert cfg.max_iterations == 100000
    assert cfg.logger == "wandb" and cfg.wandb_project == "g1_compliance"
