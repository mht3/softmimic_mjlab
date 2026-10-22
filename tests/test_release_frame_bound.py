"""Regression for a real seed49 release frame with a floating-base/leg mismatch."""
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from compliant_motion_augmentation.ik_update import bound_frame_joint_step
from compliant_motion_augmentation.constants import MAX_IK_JOINT_VEL


def test_release_step_preserves_coordinated_root_leg_motion():
    root = Path(__file__).resolve().parents[1]
    fixture = json.loads((Path(__file__).parent / 'fixtures/release_frame_1734.json').read_text())
    previous = np.array(fixture['previous']); candidate = np.array(fixture['candidate'])
    dt = fixture['dt']
    model = mujoco.MjModel.from_xml_path(str(root / 'softmimic_deploy/src/assets/g1/g1_23dof.xml'))
    data = mujoco.MjData(model)
    feet = [model.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
    def foot_positions(q):
        data.qpos[:] = q; mujoco.mj_forward(model, data)
        return data.xpos[feet].copy()
    old_clipped = candidate.copy()
    old_clipped[7:] = previous[7:] + np.clip(candidate[7:] - previous[7:],
                                          -MAX_IK_JOINT_VEL*dt, MAX_IK_JOINT_VEL*dt)
    # The former limiter produced a >6cm foot jump despite an IK candidate
    # that coordinated the base and legs. Test the actual captured transition.
    previous_feet = foot_positions(previous)
    old_jump = np.linalg.norm(foot_positions(old_clipped) - previous_feet, axis=1).max()
    assert old_jump > .06
    configuration = SimpleNamespace(q=candidate.copy())
    configuration.update = lambda q: setattr(configuration, 'q', q.copy())
    solver = SimpleNamespace(configuration=configuration, model=model, data=data,
                             actuated_qpos_indices=list(range(7, 30)))
    bound_frame_joint_step(solver, previous, dt)
    corrected = configuration.q
    new_jump = np.linalg.norm(foot_positions(corrected) - previous_feet, axis=1).max()
    assert new_jump < .01
    assert np.max(np.abs(corrected[7:] - previous[7:])) <= MAX_IK_JOINT_VEL*dt + 1e-12
    assert np.isclose(np.linalg.norm(corrected[3:7]), 1.)
    assert not np.allclose(corrected[:7], candidate[:7])


def test_unlimited_step_preserves_ik_solution():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    configuration = SimpleNamespace(q=np.array([.001]))
    configuration.update = lambda q: setattr(configuration, 'q', q.copy())
    solver = SimpleNamespace(configuration=configuration, model=model, data=data,
                             actuated_qpos_indices=[0])
    bound_frame_joint_step(solver, np.zeros(1), 1/30)
    np.testing.assert_array_equal(configuration.q, [.001])
