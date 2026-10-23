"""Nominal-reference interchange shared by C++ and browser policy playback.

Inputs are nominal_* arrays from the training converter, never adapted IK poses.
Velocities and foot contacts are preserved rather than re-estimated in a viewer.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

WIDTHS = dict(reference_dof_pos=23,reference_xy_vel=2,reference_yaw_vel=1,
              reference_root_height=1,reference_gravity_vector=3,reference_foot_contact=2)


def channels(arrays):
    q=arrays['body_quat_w'][:,0];w=q[:,:1];v=q[:,1:]
    if not np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-5):raise ValueError('Nominal quaternion is not normalized')
    def inverse(x):
        return x-2*w*np.cross(v,x)+2*np.cross(v,np.cross(v,x))
    gravity=np.zeros_like(v);gravity[:,2]=-1
    return dict(reference_dof_pos=arrays['joint_pos'],
                reference_xy_vel=inverse(arrays['body_lin_vel_w'][:,0])[:,:2],
                reference_yaw_vel=inverse(arrays['body_ang_vel_w'][:,0])[:,2:3],
                reference_root_height=arrays['body_pos_w'][:,0,2:3],
                reference_gravity_vector=inverse(gravity),
                reference_foot_contact=arrays['foot_contacts'])


def prepare_nominal(source,output,start=0.,duration=30.):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError(output)
    if not np.isfinite([start,duration]).all() or start<0 or not 0<duration<=30:raise ValueError('Use a nonnegative start and duration at most 30 seconds')
    names=('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w','foot_contacts')
    with np.load(source,allow_pickle=False) as data:
        if float(data['fps'].reshape(-1)[0])!=50:raise ValueError('Nominal deployment requires converter output at 50 Hz')
        first=round(start*50);last=first+round(duration*50)+1
        arrays={k:np.array(data['nominal_'+k][first:last],dtype=np.float32) for k in names}
    n=len(arrays['joint_pos'])
    if n<2 or any(len(v)!=n or not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Invalid nominal arrays/frame range')
    if arrays['joint_pos'].shape!=(n,23) or arrays['joint_vel'].shape!=(n,23) or arrays['foot_contacts'].shape!=(n,2):raise ValueError('Expected 23DOF nominal channels')
    for k,width in [('body_pos_w',3),('body_quat_w',4),('body_lin_vel_w',3),('body_ang_vel_w',3)]:
        if arrays[k].ndim!=3 or arrays[k].shape[1]<1 or arrays[k].shape[2]!=width:raise ValueError(f'Invalid {k}')
        arrays[k]=arrays[k][:,:1].copy()  # Pelvis is body zero in converted NPZ.
    arrays.update(channels(arrays));arrays.update(fps=np.array([50],np.float32),nominal_reference_version=np.array([1],np.float32))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as file:np.savez(file,**arrays)
    read_nominal_motion(output)
    output.with_suffix('.json').write_text(json.dumps(dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),start=start,frames=n,nominal_only=True),indent=2)+'\n')
    return output


def read_nominal_motion(path):
    path=Path(path)
    with np.load(path,allow_pickle=False) as a:
        if a['nominal_reference_version'].tolist()!=[1.] or a['fps'].tolist()!=[50.]:raise ValueError('Unsupported nominal motion schema')
        n=len(a['joint_pos'])
        if not 2<=n<=1501:raise ValueError('Nominal viewer reference must fit 30 seconds')
        expected=channels(a)
        for name,width in WIDTHS.items():
            if a[name].shape!=(n,width) or not np.isfinite(a[name]).all() or not np.allclose(a[name],expected[name],atol=1e-5):raise ValueError(f'Invalid nominal channel {name}')
        return dict(fps=50,frames=n,name=path.stem,nominal_reference_version=1,
                    root_pos=a['body_pos_w'][:,0].tolist(),root_quat=a['body_quat_w'][:,0].tolist(),joint_pos=a['joint_pos'].tolist(),foot_contacts=a['foot_contacts'].tolist(),
                    channels={k:a[k].tolist() for k in WIDTHS},source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_file=path.name,joint_names=None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-npz',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=float,default=0);p.add_argument('--duration',type=float,default=30);a=p.parse_args()
    print(prepare_nominal(a.source_npz,a.output,a.start,a.duration))
if __name__=='__main__':main()
