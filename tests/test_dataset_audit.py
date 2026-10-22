"""Structural acceptance must reject stale, truncated and incomplete NPZs."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from scripts.audit_compliant_dataset import audit, audit_npz
from scripts.check_compliant_dataset_ready import check


MODEL = Path(__file__).resolve().parents[1] / 'softmimic_deploy/src/assets/g1/g1_23dof.xml'


def valid_arrays(frames=5, bodies=24):
    arrays = {'fps': np.array([50.])}
    for key in ('joint_pos', 'joint_vel', 'nominal_joint_pos', 'nominal_joint_vel'):
        arrays[key] = np.zeros((frames, 23))
    for key in ('body_pos_w', 'body_lin_vel_w', 'body_ang_vel_w', 'nominal_body_pos_w',
                'nominal_body_lin_vel_w', 'nominal_body_ang_vel_w'):
        arrays[key] = np.zeros((frames, bodies, 3))
    for key in ('body_quat_w', 'nominal_body_quat_w'):
        arrays[key] = np.zeros((frames, bodies, 4)); arrays[key][..., 0] = 1
    for key in ('force_body_pos_w', 'force_vector', 'torque_vector', 'ff_origin', 'ff_normal'):
        arrays[key] = np.zeros((frames, 3))
    for key in ('force_body_quat_w', 'ff_setpoint_rot'):
        arrays[key] = np.zeros((frames, 4)); arrays[key][:, 0] = 1
    for key in ('stiffness', 'rotational_stiffness', 'ff_stiffness', 'ff_rotational_stiffness'):
        arrays[key] = np.zeros(frames)
    arrays['force_body_index'] = np.full(frames, -1)
    arrays['foot_contacts'] = np.ones((frames, 2))
    arrays['nominal_foot_contacts'] = arrays['foot_contacts'].copy()
    return arrays


def test_complete_npz_timeline_and_contract(tmp_path):
    path = tmp_path / 'clip.npz'
    np.savez(path, **valid_arrays())
    result, errors = audit_npz(path, 4, 30., 24)
    assert errors == []
    assert result['npz_structural_valid'] and result['npz_has_nominal']
    assert result['npz_expected_frames'] == result['npz_frames'] == 5
    # The exact same file cannot be approved against a different CSV timebase.
    result, errors = audit_npz(path, 4, 15., 24)
    assert not result['npz_structural_valid']
    assert result['npz_expected_frames'] == 10


@pytest.mark.parametrize('corruption,expected', [
    ('missing_nominal', 'missing NPZ array nominal_body_quat_w'),
    ('truncated_nominal', 'nominal_joint_vel shape'),
    ('missing_nominal_velocity', 'missing NPZ array nominal_body_ang_vel_w'),
    ('missing_nominal_contacts', 'missing NPZ array nominal_foot_contacts'),
    ('wrong_body_count', 'body_pos_w shape'),
    ('wrong_joint_count', 'joint_pos shape'),
    ('invalid_quaternion', 'non-unit quaternions'),
    ('nan', 'nonfinite'),
    ('invalid_fps', 'fps must contain'),
    ('invalid_body_index', 'invalid body indices'),
])
def test_rejects_structural_corruptions(tmp_path, corruption, expected):
    arrays = valid_arrays()
    if corruption == 'missing_nominal': arrays.pop('nominal_body_quat_w')
    elif corruption == 'missing_nominal_velocity': arrays.pop('nominal_body_ang_vel_w')
    elif corruption == 'missing_nominal_contacts': arrays.pop('nominal_foot_contacts')
    elif corruption == 'truncated_nominal': arrays['nominal_joint_vel'] = np.zeros((4, 23))
    elif corruption == 'wrong_body_count': arrays['body_pos_w'] = np.zeros((5, 23, 3))
    elif corruption == 'wrong_joint_count': arrays['joint_pos'] = np.zeros((5, 29))
    elif corruption == 'invalid_quaternion': arrays['nominal_body_quat_w'][0, 0] = 0
    elif corruption == 'nan': arrays['force_vector'][0, 0] = np.nan
    elif corruption == 'invalid_fps': arrays['fps'][0] = 0
    elif corruption == 'invalid_body_index': arrays['force_body_index'][0] = 24
    path = tmp_path / 'clip.npz'; np.savez(path, **arrays)
    result, errors = audit_npz(path, 4, 30., 24)
    assert not result['npz_structural_valid']
    assert any(expected in error for error in errors)


def test_audit_provenance_hidden_scope_and_missing_npz(tmp_path):
    # An explicitly selected hidden stage is valid; hidden descendants are not.
    root = tmp_path / '.generate-selected'; root.mkdir()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    q = model.qpos0
    pose = np.r_[q[:3], q[[4, 5, 6, 3]], q[7:]]
    row = np.r_[pose, [1., 1.], pose, -1., np.zeros(20)]
    row[79] = 1.  # CSV forcefield quaternion is xyzw.
    csv = root / 'clip.csv'; np.savetxt(csv, np.repeat(row[None], 4, axis=0), delimiter=',')
    hidden = root / '.backups'; hidden.mkdir()
    (hidden / 'old.csv').write_text('invalid hidden data')
    (root / '.partial.csv').write_text('invalid hidden data')
    result = audit(root, MODEL, stride=1)
    assert len(result) == 1
    assert result[0]['dataset_root'] == str(root.resolve())
    assert result[0]['csv_sha256'] == hashlib.sha256(csv.read_bytes()).hexdigest()
    assert result[0]['error'] == 'missing paired NPZ'
    np.savez(csv.with_suffix('.npz'), **valid_arrays())
    result = audit(root, MODEL, stride=1)
    assert 'error' not in result[0]
    assert result[0]['npz_structural_valid']


def test_zero_force_body_intersection_is_rejected(tmp_path):
    # The actual pose from the user's video: legal joint angles, no force,
    # but a right upper arm and elbow passing through the torso.
    row = np.loadtxt(Path(__file__).parent / 'fixtures/stand_clip018_self_collision.csv', delimiter=',')
    assert np.linalg.norm(row[63:66]) == 0
    csv = tmp_path / 'clip.csv'
    np.savetxt(csv, np.repeat(row[None], 4, axis=0), delimiter=',')
    np.savez(csv.with_suffix('.npz'), **valid_arrays())
    result = audit(tmp_path, MODEL, stride=1)
    entry = result[0]
    assert entry['npz_structural_valid']
    assert entry['max_joint_limit_violation_rad'] == 0
    assert entry['sampled_nominal_self_collision_frames'] == 0
    assert entry['sampled_self_collision_frames'] == 4
    assert entry['max_self_penetration_m'] > .09
    assert not entry['self_collision_free']
    (tmp_path / 'audit.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='failed collision-aware'):
        check(tmp_path)


def test_training_gate_requires_current_collision_audit(tmp_path):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    q = model.qpos0
    pose = np.r_[q[:3], q[[4, 5, 6, 3]], q[7:]]
    row = np.r_[pose, [1., 1.], pose, -1., np.zeros(20)]
    row[79] = 1.
    csv = tmp_path / 'clip.csv'
    np.savetxt(csv, np.repeat(row[None], 4, axis=0), delimiter=',')
    np.savez(csv.with_suffix('.npz'), **valid_arrays())
    report = audit(tmp_path, MODEL, stride=1)
    audit_path = tmp_path / 'audit.json'
    audit_path.write_text(json.dumps(report))
    assert check(tmp_path) == 1
    assert check(csv.with_suffix('.npz')) == 1
    old = dict(report[0]); old.pop('self_collision_checked')
    audit_path.write_text(json.dumps([old]))
    with pytest.raises(ValueError, match='collision-aware'):
        check(tmp_path)
    audit_path.write_text(json.dumps(report))
    csv.write_text(csv.read_text() + '\n')
    with pytest.raises(ValueError, match='Stale audit hash'):
        check(tmp_path)


def test_collision_free_but_stuck_standing_hand_is_rejected(tmp_path):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    q = model.qpos0
    pose = np.r_[q[:3], q[[4, 5, 6, 3]], q[7:]]
    adapted = pose.copy()
    joint = model.joint('right_shoulder_pitch_joint').id
    adapted[model.jnt_qposadr[joint]] = -.5
    row = np.r_[pose, [1., 1.], adapted, -1., np.zeros(20)]
    row[79] = 1.
    csv = tmp_path / 'stuck.csv'
    np.savetxt(csv, np.repeat(row[None], 121, axis=0), delimiter=',')
    np.savez(csv.with_suffix('.npz'), **valid_arrays(frames=200))
    entry = audit(tmp_path, MODEL, stride=1)[0]
    assert entry['self_collision_free']
    assert entry['npz_structural_valid']
    assert entry['max_settled_hand_error_m'] > .05
    assert 'standing release failed to settle' in entry['error']
