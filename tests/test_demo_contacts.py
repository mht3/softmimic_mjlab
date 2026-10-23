"""Compiled MuJoCo contact parameters and passive sliding regression (CPU only)."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'demos/sonic_web'))
from contact_model import configure_contacts


def model_for(profile):
    source = REPO / 'src/assets/robots/unitree_g1/xmls/g1_23dof.xml'
    root = ET.parse(source).getroot()
    root.find('compiler').set('meshdir', str(source.parent / 'assets'))
    ET.SubElement(root, 'option', timestep='.005', iterations='10',
                  ls_iterations='20', integrator='implicitfast')
    ET.SubElement(root.find('worldbody'), 'geom', name='floor', type='plane',
                  size='0 0 .05', friction='1 .005 .0001')
    actuators = ET.SubElement(root, 'actuator')
    for joint in root.findall('.//worldbody//joint'):
        joint.set('armature', '.001' if 'wrist' in joint.get('name') else '.01')
        # Same fall-cutoff mode as the browser: kp=0, kd=3, no commanded torque.
        ET.SubElement(actuators, 'general', joint=joint.get('name'),
                      gainprm='0', biastype='affine', biasprm='0 0 -3')
    configure_contacts(root, profile)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def test_body_friction_stops_passive_slide_without_changing_feet():
    models = {profile: model_for(profile) for profile in ('training', 'frictional-body')}
    old, new = models.values()
    for i in range(old.ngeom):
        name = old.geom(i).name
        if '_foot' in name:
            assert old.geom_condim[i] == new.geom_condim[i] == 3
            np.testing.assert_array_equal(old.geom_friction[i], new.geom_friction[i])
        assert old.geom_contype[i] == new.geom_contype[i]
        assert old.geom_conaffinity[i] == new.geom_conaffinity[i]
    # Settle once, then use the identical fallen pose and 1 m/s horizontal
    # velocity for both profiles. No controller, drag, or velocity clamps.
    settled = mujoco.MjData(old)
    settled.qpos[:3] = [0, 0, .75]
    settled.qpos[3:7] = [2**-.5, 2**-.5, 0, 0]
    for _ in range(800):
        mujoco.mj_step(old, settled)
    results = {}
    for profile, model in models.items():
        data = mujoco.MjData(model)
        data.qpos[:] = settled.qpos
        data.qvel[0] = 1.
        mujoco.mj_forward(model, data)
        start = data.subtree_com[1, :2].copy()
        body_contacts = 0
        for _ in range(800):
            mujoco.mj_step(model, data)
            for contact in data.contact:
                names = [model.geom(int(i)).name for i in contact.geom]
                if 'floor' in names and not any('_foot' in name for name in names):
                    body_contacts += 1
                    assert contact.dim == (1 if profile == 'training' else 3)
                    if profile == 'frictional-body':
                        assert contact.friction[0] == .5
        assert body_contacts > 0
        results[profile] = dict(distance=float(np.linalg.norm(data.subtree_com[1, :2]-start)),
                                final_speed=float(np.linalg.norm(data.qvel[:2])))
    print(results)
    assert results['training']['distance'] > 2.
    assert results['frictional-body']['distance'] < .5
    assert results['frictional-body']['final_speed'] < .02
