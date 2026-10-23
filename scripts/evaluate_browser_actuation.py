"""CPU comparison of explicit versus training-matched implicit PD, fixed ONNX.

No training, web-asset installation or hardware calls. This is a nominal MuJoCo
rollout, not a browser-WASM or hardware performance certificate.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from onnx.reference import ReferenceEvaluator
import yaml


def run(bundle, seconds, mode):
    repo = Path(__file__).resolve().parents[1]
    c = yaml.safe_load((bundle/'params/deploy.yaml').read_text())
    q0 = np.loadtxt(bundle/'params/reference.csv', delimiter=',', comments='#')[0]
    source = repo/'src/assets/robots/unitree_g1/xmls/g1_23dof.xml'
    root = ET.parse(source).getroot()
    root.find('compiler').set('meshdir', str(source.parent/'assets'))
    ET.SubElement(root,'option',timestep='.005',iterations='10',ls_iterations='20',integrator='implicitfast')
    ET.SubElement(root.find('worldbody'),'geom',name='floor',type='plane',size='0 0 .05',contype='1',conaffinity='1',friction='1 .005 .0001')
    actuators=ET.SubElement(root,'actuator')
    effort=np.array([88,88,88,139,50,50]*2+[88]+[25]*10)
    for i,j in enumerate(root.findall('.//worldbody//joint')):
        name=j.get('name');j.set('armature','.001' if 'wrist' in name else '.01')
        if mode=='implicit':
            ET.SubElement(actuators,'general',name=name,joint=name,biastype='affine',gainprm=str(c['stiffness'][i]),biasprm=f"0 {-c['stiffness'][i]} {-c['damping'][i]}",forcelimited='true',forcerange=f'{-effort[i]} {effort[i]}')
        else:
            ET.SubElement(actuators,'motor',name=name,joint=name,ctrllimited='true',ctrlrange=f'{-effort[i]} {effort[i]}')
    for g in root.findall('.//worldbody//geom'):
        name=g.get('name','')
        if name.endswith('_collision'):
            g.set('contype','0');g.set('conaffinity','1');g.set('condim','3' if '_foot' in name else '1')
            if '_foot' in name:g.set('friction','.6 .005 .0001')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model)
    data.qpos[:3]=q0[:3];data.qpos[3:7]=q0[[6,3,4,5]];data.qpos[7:]=q0[7:30]
    mujoco.mj_forward(model,data)
    session=ReferenceEvaluator(str(bundle/'exported/policy.onnx'))
    history={};last=np.zeros(23);speeds=[];deltas=[];heights=[];peaks=[]
    for step in range(round(seconds/.02)):
        w,x,y,z=data.qpos[3:7]
        live=dict(base_ang_vel=data.qvel[3:6],projected_gravity=np.array([2*(w*y-x*z),-2*(y*z+w*x),2*(x*x+y*y)-1]),joint_pos_rel=data.qpos[7:]-c['default_joint_pos'],joint_vel_rel=data.qvel[6:],last_action=last,desired_stiffness_log=[np.log(200.+1e-6)],desired_rotational_stiffness_log=[np.log(1.+1e-6)])
        for name,term in c['observations'].items():
            raw=np.array(live[name] if name in live else term['params']['value'])
            if term['clip']:raw=np.clip(raw,*term['clip'])
            value=raw*term['scale']
            history[name]=history[name][1:]+[value] if name in history else [value.copy() for _ in range(term['history_length'])]
        obs=np.concatenate([np.concatenate(v) for v in history.values()]).astype(np.float32)[None]
        action=session.run(None,{session.input_names[0]:obs})[0][0]
        if not np.isfinite(action).all():raise RuntimeError('Non-finite action')
        peaks.append(float(np.abs(action).max()))
        if c.get('policy_action_clip'):action=np.clip(action,-c['policy_action_clip'],c['policy_action_clip'])
        deltas.append((action[[17,22]]-last[[17,22]])/.02);last=action.copy()
        target=action*np.array(c['actions']['JointPositionAction']['scale'])+c['actions']['JointPositionAction']['offset']
        for _ in range(4):
            data.ctrl[:]=target if mode=='implicit' else np.clip(np.array(c['stiffness'])*(target-data.qpos[7:])-np.array(c['damping'])*data.qvel[6:],-effort,effort)
            mujoco.mj_step(model,data)
            speeds.append(data.qvel[[23,28]].copy())
        heights.append(float(data.qpos[2]))
        if not np.isfinite(data.qpos).all() or data.qpos[2]<.3:break
    return dict(mode=mode,seconds=len(heights)*.02,minimum_root_height_m=min(heights),wrist_speed_rms_rad_s=float(np.sqrt(np.mean(np.array(speeds)**2))),wrist_action_rate_rms=float(np.sqrt(np.mean(np.array(deltas)**2))),max_raw_action=max(peaks))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--seconds',type=float,default=20);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=dict(bundle=str(a.bundle.resolve()),results=[run(a.bundle,a.seconds,mode) for mode in ('explicit','implicit')],hardware_validated=False)
    a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
