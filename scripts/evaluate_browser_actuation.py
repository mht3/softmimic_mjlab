"""CPU comparison of explicit versus training-matched implicit PD, fixed ONNX.

No training, web-asset installation or hardware calls. This is a nominal MuJoCo
rollout, not a browser-WASM or hardware performance certificate.
"""
import argparse
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from onnx.reference import ReferenceEvaluator
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'demos/sonic_web'))
from contact_model import add_contact_arguments, configure_contacts

LOADS = {
    'none': {},
    'left_pull': {'left_palm': [0., 20., 0.]},
    'right_pull': {'right_palm': [0., -20., 0.]},
    'downward': {'left_palm': [0., 0., -30.], 'right_palm': [0., 0., -30.]},
}


def load_envelope(t):
    """One-second cosine ramps, two-second hold, complete release at 9 s."""
    if t < 5 or t >= 9:
        return 0.
    if t < 6:
        return .5 - .5*np.cos(np.pi*(t-5))
    if t < 8:
        return 1.
    return .5 + .5*np.cos(np.pi*(t-8))


def run(bundle, seconds, mode, sensor_noise=False, delay_steps=0, seed=42, contact_profile='training', body_friction=.5, load='none', noise_scales=None, randomization=None):
    # Diagnostics only: consume every noise draw even for disabled channels so
    # paired ablations retain identical noise in the remaining sensor channels.
    noise_bounds = {'base_ang_vel': .2, 'projected_gravity': .01,
                    'joint_pos_rel': .01, 'joint_vel_rel': 1.5}
    if noise_scales is not None and (set(noise_scales) - set(noise_bounds) or
            any(not np.isfinite(v) or v < 0 for v in noise_scales.values())):
        raise ValueError('Noise scales must be finite nonnegative values for known sensor channels')
    if load not in LOADS or (load != 'none' and seconds < 15):
        raise ValueError('Unknown load or insufficient recovery duration (15 s minimum)')
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
    contacts=configure_contacts(root,contact_profile,body_friction)
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model)
    randomized_values=None
    if randomization is not None:
        if mode != 'implicit': raise ValueError('Randomization requires implicit PD')
        from domain_randomization import apply_draw
        randomized_values=apply_draw(model,**randomization)
        if randomization.get('training_contacts',True):
            contacts.update(profile='training',body_sliding_friction=0.)
        if randomized_values['foot_friction'] is not None:
            contacts['foot_sliding_friction']=randomized_values['foot_friction']
        mujoco.mj_setConst(model,data)
        mujoco.mj_resetData(model,data)
    data.qpos[:3]=q0[:3];data.qpos[3:7]=q0[[6,3,4,5]];data.qpos[7:]=q0[7:30]
    mujoco.mj_forward(model,data)
    session=ReferenceEvaluator(str(bundle/'exported/policy.onnx'))
    rng=np.random.default_rng(seed);sensor_queue=[]
    sites={name:model.site(name).id for name in ('left_palm','right_palm')}
    wrist_joints=[model.joint(name).id for name in ('left_wrist_roll_joint','right_wrist_roll_joint')]
    wrist_dofs=model.jnt_dofadr[wrist_joints]
    wrist_actions=[model.actuator(name).id for name in ('left_wrist_roll_joint','right_wrist_roll_joint')]
    history={};last=np.zeros(23);speeds=[];deltas=[];heights=[];peaks=[];tilts=[];poses=[];fallen=False
    peak_force=0.;peak_torque=0.
    for step in range(round(seconds/.02)):
        w,x,y,z=data.qpos[3:7]
        live=dict(base_ang_vel=data.qvel[3:6],projected_gravity=np.array([2*(w*y-x*z),-2*(y*z+w*x),2*(x*x+y*y)-1]),joint_pos_rel=data.qpos[7:]-c['default_joint_pos'],joint_vel_rel=data.qvel[6:],last_action=last,desired_stiffness_log=[np.log(200.+1e-6)],desired_rotational_stiffness_log=[np.log(1.+1e-6)])
        measured={name:np.array(live[name],copy=True) for name in ('base_ang_vel','projected_gravity','joint_pos_rel','joint_vel_rel')}
        if sensor_noise:
            for name,bound in noise_bounds.items():
                scale = 1. if noise_scales is None else noise_scales.get(name, 1.)
                measured[name]+=rng.uniform(-bound,bound,size=measured[name].shape)*scale
        sensor_queue.append(measured)
        while len(sensor_queue)>delay_steps+1:sensor_queue.pop(0)
        live.update(sensor_queue[0])
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
        deltas.append((action[wrist_actions]-last[wrist_actions])/.02);last=action.copy()
        target=action*np.array(c['actions']['JointPositionAction']['scale'])+c['actions']['JointPositionAction']['offset']
        for _ in range(4):
            data.xfrc_applied[:]=0
            for name,force in LOADS[load].items():
                site=sites[name];body=model.site_bodyid[site]
                f=np.array(force)*load_envelope(float(data.time))
                torque=np.cross(data.site_xpos[site]-data.xipos[body],f)
                data.xfrc_applied[body,:3]+=f;data.xfrc_applied[body,3:]+=torque
                peak_force=max(peak_force,float(np.linalg.norm(f)))
                peak_torque=max(peak_torque,float(np.linalg.norm(torque)))
            data.ctrl[:]=target if mode=='implicit' else np.clip(np.array(c['stiffness'])*(target-data.qpos[7:])-np.array(c['damping'])*data.qvel[6:],-effort,effort)
            mujoco.mj_step(model,data)
            speeds.append(data.qvel[wrist_dofs].copy())
            q=data.qpos[3:7];tilt=float(np.arccos(np.clip(1-2*np.sum((q[1:3]/np.linalg.norm(q))**2),-1,1)))
            tilts.append(tilt)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or tilt>1.25 or data.qpos[2]<.3:
                fallen=True;break
        heights.append(float(data.qpos[2]))
        poses.append(np.concatenate([data.qpos[:3],*[data.site_xpos[s].copy() for s in sites.values()]]))
        if fallen:break
    recovery=None
    if not fallen and seconds>=15:
        pose=np.array(poses);before=pose[150:250].mean(axis=0);after=pose[-250:].mean(axis=0)
        recovery=dict(root_height_error_m=float(abs(after[2]-before[2])),root_xy_drift_m=float(np.linalg.norm(after[:2]-before[:2])),
                      maximum_palm_error_m=float(max(np.linalg.norm(after[3:6]-before[3:6]),np.linalg.norm(after[6:9]-before[6:9]))),
                      mean_final_tilt_rad=float(np.mean(tilts[-1000:])))
    return dict(randomization=randomization,randomized_values=randomized_values,contacts=contacts,mode=mode,sensor_noise=sensor_noise,noise_scales=noise_scales,extra_sensor_delay_ms=delay_steps*20,seed=seed,seconds=float(data.time),
                load=load,load_forces_world_N=LOADS[load],load_window_s=[5,9],peak_per_hand_force_N=peak_force,peak_lever_arm_torque_Nm=peak_torque,
                fallen=fallen,recovery=recovery,maximum_tilt_rad=max(tilts),minimum_root_height_m=min(heights),wrist_speed_rms_rad_s=float(np.sqrt(np.mean(np.array(speeds)**2))),wrist_action_rate_rms=float(np.sqrt(np.mean(np.array(deltas)**2))),max_raw_action=max(peaks))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=20)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('both','implicit','explicit'),default='both')
    p.add_argument('--sensor-noise',action='store_true')
    p.add_argument('--delay-steps',type=int,choices=(0,1,2),default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--load',choices=tuple(LOADS),default='none')
    add_contact_arguments(p)
    a=p.parse_args()
    if not 0 < a.seconds <= 300: p.error('--seconds must be in (0,300]')
    modes=('explicit','implicit') if a.mode=='both' else (a.mode,)
    result=dict(bundle=str(a.bundle.resolve()),results=[run(a.bundle,a.seconds,mode,a.sensor_noise,a.delay_steps,a.seed,a.contact_profile,a.body_friction,a.load) for mode in modes],hardware_validated=False)
    a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
