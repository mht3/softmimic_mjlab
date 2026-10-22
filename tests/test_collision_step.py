"""Finite integration can collide even when differential IK limits pass."""
import mujoco
import numpy as np
import pytest
import json
from pathlib import Path

from compliant_motion_augmentation.collision_checks import collision_free_step, self_collision_pairs


def model_and_data():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="torso"><geom type="sphere" size=".1"/></body>
      <body name="hand"><joint name="j" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>''')
    return model, mujoco.MjData(model)


@pytest.mark.parametrize('candidate', [0., .4])
def test_rejects_collision_at_endpoint_or_between_clear_endpoints(candidate):
    model, data = model_and_data()
    previous = np.array([-.4])
    q, fraction = collision_free_step(model, data, previous, np.array([candidate]))
    assert 0 < fraction < 1
    assert q[0] <= -.2
    for phase in np.linspace(0, 1, 101):
        data.qpos[:] = previous + (q-previous)*phase
        mujoco.mj_forward(model, data)
        assert not self_collision_pairs(model, data, tolerance=0.)


def test_clear_step_is_preserved():
    model, data = model_and_data()
    q, fraction = collision_free_step(model, data, np.array([-.4]), np.array([-.3]))
    assert fraction == 1
    np.testing.assert_allclose(q, [-.3])


def test_invalid_start_is_not_silently_accepted():
    model, data = model_and_data()
    with pytest.raises(ValueError, match='already self-colliding'):
        collision_free_step(model, data, np.array([0.]), np.array([-.4]))


def test_actual_seed43_release_collision_is_prevented():
    root = Path(__file__).resolve().parents[1]
    fixture = json.loads((Path(__file__).parent / 'fixtures/release_collision_step.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(root / 'softmimic_deploy/src/assets/g1/g1_23dof.xml'))
    data = mujoco.MjData(model)
    previous, candidate = np.array(fixture['previous']), np.array(fixture['candidate'])
    data.qpos[:] = candidate
    mujoco.mj_forward(model, data)
    assert max(self_collision_pairs(model, data).values()) > .004
    corrected, fraction = collision_free_step(model, data, previous, candidate)
    assert fraction == .5
    tangent = np.empty(model.nv)
    mujoco.mj_differentiatePos(model, tangent, 1., previous, corrected)
    for phase in np.linspace(0, 1, 101):
        data.qpos[:] = previous
        mujoco.mj_integratePos(model, data.qpos, tangent, float(phase))
        mujoco.mj_forward(model, data)
        assert not self_collision_pairs(model, data, tolerance=0.)
    assert np.max(np.abs(corrected[7:] - previous[7:])) / fixture['dt'] <= 2*np.pi+1e-6
