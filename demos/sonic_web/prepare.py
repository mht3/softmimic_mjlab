"""Build the 23DOF assets; optionally install a corrected exported standing policy."""
from pathlib import Path
import argparse, hashlib, json, shutil, xml.etree.ElementTree as ET
import numpy as np
import yaml
from reference import read_reference,validate_reference
from contact_model import add_contact_arguments,configure_contacts

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets-dir',type=Path,default=HERE/'assets',help='Generated assets directory; use a separate directory for each preview')
    p.add_argument('--reference',type=Path,required=True,help='The nominal CSV used for training, not an augmented compliant target')
    p.add_argument('--reference-fps',type=float,default=30.)
    p.add_argument('--deploy', type=Path, help='Corrected run params/deploy.yaml')
    p.add_argument('--policy', type=Path, help='The same run exported policy.onnx')
    p.add_argument('--stiffness', type=float, default=200.)
    p.add_argument('--rotational-stiffness', type=float, default=1.)
    add_contact_arguments(p)
    args = p.parse_args()
    if not (40 <= args.stiffness <= 1000 and .1 <= args.rotational_stiffness <= 10):
        p.error('Stiffness commands must stay within the training ranges')
    if bool(args.deploy) != bool(args.policy):
        p.error('--deploy and --policy must be supplied together')
    motion=read_reference(args.reference,args.reference_fps)
    if args.deploy:validate_reference(motion,yaml.safe_load(args.deploy.read_text()))
    # Reject a bad controller before changing any served asset.
    if args.policy:
        import onnx
        c=yaml.safe_load(args.deploy.read_text())
        expected={'base_ang_vel':3,'projected_gravity':3,'joint_pos_rel':23,'joint_vel_rel':23,'last_action':23,
                  'reference_dof_pos':23,'reference_xy_vel':2,'reference_yaw_vel':1,'reference_root_height':1,
                  'reference_gravity_vector':3,'reference_foot_contact':2,'desired_stiffness_log':1,'desired_rotational_stiffness_log':1}
        if c.get('observation_dim')!=327 or c.get('step_dt')!=.02 or list(c['observations'])!=list(expected):raise ValueError('Unsupported policy observation contract')
        for name,width in expected.items():
            term=c['observations'][name]
            if term['history_length']!=3 or len(term['scale'])!=width or not np.isfinite(term['scale']).all():raise ValueError(f'Invalid observation term {name}')
        if c['joint_ids_map']!=[*range(13),15,16,17,18,19,22,23,24,25,26]:raise ValueError('Incompatible hardware joint order')
        for name in ('default_joint_pos','stiffness','damping'):
            if len(c[name])!=23 or not np.isfinite(c[name]).all():raise ValueError(f'Invalid {name}')
        action=c['actions']['JointPositionAction']
        if action['clip'] is not None or action['joint_ids'] is not None:raise ValueError('Unsupported action mapping')
        for name in ('scale','offset'):
            if len(action[name])!=23 or not np.isfinite(action[name]).all():raise ValueError('Invalid action scale/offset')
        model=onnx.load(args.policy);onnx.checker.check_model(model)
        dimensions=lambda x:[d.dim_value for d in x.type.tensor_type.shape.dim]
        if len(model.graph.input)!=1 or len(model.graph.output)!=1 or dimensions(model.graph.input[0])!=[1,327] or dimensions(model.graph.output[0])!=[1,23]:raise ValueError('Incompatible ONNX dimensions')
    assets = args.assets_dir.resolve()
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
    deploy_cfg = yaml.safe_load(args.deploy.read_text()) if args.deploy else None
    effort = []
    actuators = ET.SubElement(root, 'actuator')
    for j in joints:
        name = j.attrib['name']
        limit = 139 if 'knee' in name else 50 if 'ankle' in name else 88 if ('hip' in name or 'waist' in name) else 25
        j.set('armature', '0.001' if 'wrist' in name else '0.01')
        if deploy_cfg:
            i = len(effort)
            kp, kd = deploy_cfg['stiffness'][i], deploy_cfg['damping'][i]
            ET.SubElement(actuators, 'general', name=name, joint=name,
                          gainprm=f'{kp}', biasprm=f'0 {-kp} {-kd}', biastype='affine',
                          forcelimited='true', forcerange=f'-{limit} {limit}')
        else:
            ET.SubElement(actuators, 'motor', name=name, joint=name, ctrllimited='true', ctrlrange=f'-{limit} {limit}')
        effort.append(limit)
    contact_model = configure_contacts(root,args.contact_profile,args.body_friction)
    for mesh in root.findall('./asset/mesh'):
        f = mesh.attrib['file']; shutil.copy2(source.parent/'assets'/f, robot/'meshes'/f)
    ET.indent(tree); tree.write(robot/'scene.xml')
    (assets/'motions').mkdir(exist_ok=True)
    (assets/'motions/stand.json').write_text(json.dumps(motion))
    (assets/'motions/index.json').write_text(json.dumps([dict(id='stand',display=motion['name'],frames=motion['frames'],fps=motion['fps'])]))
    (assets/'robot/contract.json').write_text(json.dumps(dict(joint_names=[j.get('name') for j in joints],effort_limits=effort,step_dt=.02,substeps=4,actuation='implicit_pd' if deploy_cfg else 'torque',contacts=contact_model)))
    if args.policy:
        import onnx
        c = yaml.safe_load(args.deploy.read_text())
        assert c.get('reference_mode') in ('constant_nominal','nominal_motion') and c.get('observation_dim') == 327
        assert len(c['default_joint_pos']) == 23 and c['step_dt'] == .02
        assert c['joint_ids_map'] == [*range(13),15,16,17,18,19,22,23,24,25,26]
        model = onnx.load(args.policy); onnx.checker.check_model(model)
        dim = lambda value: [x.dim_value for x in value.type.tensor_type.shape.dim]
        assert len(model.graph.input)==1 and dim(model.graph.input[0])[-1]==327
        assert len(model.graph.output)==1 and dim(model.graph.output[0])[-1]==23
        validate_reference(motion,c)
        c['effort_limits']=effort
        c['actuation']='implicit_pd'
        controller=args.deploy.parent.parent/'controller_config.yaml'
        native=yaml.safe_load(controller.read_text())['FSM']['Mimic_Compliant_Stand'] if controller.exists() else {}
        c['fall_detection']={'limit_angle':float(native.get('limit_angle',1.25)), 'passive_damping':3.}
        if not 0<c['fall_detection']['limit_angle']<np.pi:raise ValueError('Invalid fall detection angle')
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
