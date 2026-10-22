"""Build the 23DOF assets; optionally install a corrected exported standing policy."""
from pathlib import Path
import argparse, hashlib, json, shutil, xml.etree.ElementTree as ET
import numpy as np
import yaml
from reference import read_reference,validate_reference

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True,help='The nominal CSV used for training, not an augmented compliant target')
    p.add_argument('--reference-fps',type=float,default=30.)
    p.add_argument('--deploy', type=Path, help='Corrected run params/deploy.yaml')
    p.add_argument('--policy', type=Path, help='The same run exported policy.onnx')
    p.add_argument('--stiffness', type=float, default=200.)
    p.add_argument('--rotational-stiffness', type=float, default=1.)
    args = p.parse_args()
    if not (40 <= args.stiffness <= 1000 and .1 <= args.rotational_stiffness <= 10):
        p.error('Stiffness commands must stay within the training ranges')
    if bool(args.deploy) != bool(args.policy):
        p.error('--deploy and --policy must be supplied together')
    motion=read_reference(args.reference,args.reference_fps)
    if args.deploy:validate_reference(motion,yaml.safe_load(args.deploy.read_text()))
    assets = HERE / 'assets'
    robot = assets / 'robot'
    (robot / 'meshes').mkdir(parents=True, exist_ok=True)
    source = REPO / 'src/assets/robots/unitree_g1/xmls/g1_23dof.xml'
    tree = ET.parse(source); root = tree.getroot()
    root.find('compiler').set('meshdir', 'meshes')
    ET.SubElement(root, 'option', timestep='0.005', iterations='10', ls_iterations='20', integrator='implicitfast')
    ET.SubElement(root.find('worldbody'), 'geom', name='floor', type='plane', size='0 0 0.05', contype='1', conaffinity='1', friction='1 0.005 0.0001', rgba='0.3 0.3 0.3 1')
    joints = root.findall('.//worldbody//joint')
    assert len(joints) == 23
    if motion['joint_names'] is not None and motion['joint_names'] != [j.get('name') for j in joints]:
        raise ValueError('Reference named joint order differs from the robot')
    effort = []
    actuators = ET.SubElement(root, 'actuator')
    for j in joints:
        name = j.attrib['name']
        limit = 139 if 'knee' in name else 50 if 'ankle' in name else 88 if ('hip' in name or 'waist' in name) else 25
        j.set('armature', '0.001' if 'wrist' in name else '0.01')
        ET.SubElement(actuators, 'motor', name=name, joint=name, ctrllimited='true', ctrlrange=f'-{limit} {limit}')
        effort.append(limit)
    # Same collision masks as get_g1_23dof_robot_cfg(gains='softmimic').
    for g in root.findall('.//worldbody//geom'):
        name = g.get('name', '')
        if name.endswith('_collision'):
            foot = '_foot' in name
            g.attrib.update(contype='0', conaffinity='1', condim='3' if foot else '1')
            if foot: g.set('friction','0.6 0.005 0.0001')
    for mesh in root.findall('./asset/mesh'):
        f = mesh.attrib['file']; shutil.copy2(source.parent/'assets'/f, robot/'meshes'/f)
    ET.indent(tree); tree.write(robot/'scene.xml')
    (assets/'motions').mkdir(exist_ok=True)
    (assets/'motions/stand.json').write_text(json.dumps(motion))
    (assets/'motions/index.json').write_text(json.dumps([dict(id='stand',display=motion['name'],frames=motion['frames'],fps=motion['fps'])]))
    (assets/'robot/contract.json').write_text(json.dumps(dict(joint_names=[j.get('name') for j in joints],effort_limits=effort,step_dt=.02,substeps=4)))
    if args.policy:
        import onnx
        c = yaml.safe_load(args.deploy.read_text())
        assert c.get('reference_mode') == 'constant_nominal' and c.get('observation_dim') == 327
        assert len(c['default_joint_pos']) == 23 and c['step_dt'] == .02
        assert c['joint_ids_map'] == [*range(13),15,16,17,18,19,22,23,24,25,26]
        model = onnx.load(args.policy); onnx.checker.check_model(model)
        dim = lambda value: [x.dim_value for x in value.type.tensor_type.shape.dim]
        assert len(model.graph.input)==1 and dim(model.graph.input[0])[-1]==327
        assert len(model.graph.output)==1 and dim(model.graph.output[0])[-1]==23
        validate_reference(motion,c)
        c['effort_limits']=effort
        c['stiffness_command']=args.stiffness; c['rotational_stiffness_command']=args.rotational_stiffness
        c['source_policy_sha256']=hashlib.sha256(args.policy.read_bytes()).hexdigest()
        c['source_policy']=str(args.policy)
        metadata={prop.key:prop.value for prop in model.metadata_props}
        if 'checkpoint_iteration' in metadata:
            c['checkpoint_iteration']=int(metadata['checkpoint_iteration'])
            c['checkpoint_sha256']=metadata.get('checkpoint_sha256')
            c['training_status']=metadata.get('training_status')
        c['reference_sha256']=motion['source_sha256']
        (assets/'policy').mkdir(exist_ok=True)
        from onnx.external_data_helper import convert_model_from_external_data
        convert_model_from_external_data(model)
        onnx.save_model(model, assets/'policy/policy.onnx', save_as_external_data=False)
        (assets/'policy/config.json').write_text(json.dumps(c,indent=2))
    print('Built 23DOF reference assets.' + (' Installed ONNX policy; behavioral validation still required.' if args.policy else ' No policy installed.'))

if __name__ == '__main__': main()
