"""Reference file contract shared by browser preparation and deployment packaging."""
from pathlib import Path
import json,hashlib
import numpy as np

def read_reference(path,fps=30):
    path=Path(path)
    if path.suffix == '.npz':
        from nominal_motion import read_nominal_motion
        return read_nominal_motion(path)
    with path.open() as stream:first=stream.readline()
    metadata=json.loads(first[2:]) if first.startswith('# {') else None
    if metadata:
        if metadata.get('schema')!='unitree_g1_23dof_reference_v1' or metadata.get('quaternion_order')!='xyzw':raise ValueError('Unsupported streamer CSV schema')
        fps=float(metadata['fps'])
    if metadata:
        expected=json.loads((Path(__file__).resolve().parents[1]/'ardy_web/robot_contract.json').read_text())['joint_order']
        if metadata.get('joint_names')!=expected:raise ValueError('Reference named joint order differs from the robot')
    raw=np.loadtxt(path,delimiter=',',ndmin=2)
    if raw.shape[1] not in (30,32) or not 2<=len(raw)<=3601 or not np.isfinite(raw).all() or not np.isfinite(fps) or not 1<=fps<=120:raise ValueError('Expected a finite 23DOF XYZW reference CSV at 1–120 Hz')
    if not np.allclose(np.linalg.norm(raw[:,3:7],axis=1),1,atol=1e-5):raise ValueError('Invalid reference quaternion')
    if (len(raw)-1)/fps>30.01:raise ValueError('Viewer references must fit the 30-second timeline')
    return dict(fps=fps,frames=len(raw),name=path.stem,root_pos=raw[:,:3].tolist(),root_quat=raw[:,[6,3,4,5]].tolist(),joint_pos=raw[:,7:30].tolist(),foot_contacts=raw[:,30:32].tolist() if raw.shape[1]==32 else None,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_file=path.name,joint_names=metadata['joint_names'] if metadata else None)

def validate_reference(reference,config):
    if config.get('reference_mode') == 'nominal_motion':
        if reference.get('nominal_reference_version') != 1 or reference['fps'] != 50:
            raise ValueError('Dynamic deployment requires the prepared nominal NPZ')
        for name, term in config['observations'].items():
            if name.startswith('reference_'):
                data=np.asarray(reference['channels'].get(name))
                if data.shape != (reference['frames'],len(term['scale'])) or not np.isfinite(data).all():
                    raise ValueError(f'Invalid nominal channel {name}')
        return
    if config.get('reference_mode')!='constant_nominal':raise ValueError('This deployment adapter requires a constant_nominal export')
    q=np.array(reference['root_quat']);j=np.array(reference['joint_pos']);xyz=np.array(reference['root_pos'])
    if not np.allclose(j,j[:1],atol=1e-5,rtol=0) or not np.allclose(xyz,xyz[:1],atol=1e-5,rtol=0) or not np.allclose(q,q[:1],atol=1e-5,rtol=0):raise ValueError('A moving reference cannot be paired with a constant-standing policy')
    w,x,y,z=q[0];gravity=[2*(w*y-x*z),-2*(y*z+w*x),2*(x*x+y*y)-1]
    expected={'reference_dof_pos':j[0],'reference_root_height':[xyz[0,2]],'reference_gravity_vector':gravity,'reference_xy_vel':[0,0],'reference_yaw_vel':[0],'reference_foot_contact':[1,1]}
    if reference['foot_contacts'] is not None and not np.allclose(reference['foot_contacts'],[1,1]):raise ValueError('Standing reference feet must remain planted')
    for name,value in expected.items():
        if not np.allclose(config['observations'][name]['params']['value'],value,atol=1e-5,rtol=0):raise ValueError(f'Reference differs from the trained policy: {name}')


def validate_nominal_csv_pair(csv, nominal):
    """Require the launch CSV and exact nominal-channel NPZ to describe one clip."""
    times=np.arange(nominal['frames'])/nominal['fps'];source_times=np.arange(csv['frames'])/csv['fps']
    if times[-1]>source_times[-1]+1e-5:raise ValueError('Nominal NPZ exceeds supplied CSV duration')
    for key in ('joint_pos','root_pos'):
        values=np.asarray(csv[key]);expected=np.stack([np.interp(times,source_times,values[:,j]) for j in range(values.shape[1])],axis=1)
        if not np.allclose(expected,nominal[key],atol=1e-4,rtol=0):raise ValueError(f'CSV and nominal NPZ differ: {key}')
    q=np.asarray(csv['root_quat']);x=times*csv['fps'];i=np.floor(x).astype(int);j=np.minimum(i+1,len(q)-1);f=(x-i)[:,None]
    a=q[i];b=q[j];dot=(a*b).sum(axis=1,keepdims=True);b=np.where(dot<0,-b,b);dot=np.abs(dot)
    angle=np.arccos(np.clip(dot,-1,1));den=np.sin(angle);small=dot>.9995
    # SLERP with a normalized linear fallback near zero angle.
    denominator=np.where(small,1.,den)
    expected=np.where(small,a*(1-f)+b*f,(a*np.sin((1-f)*angle)+b*np.sin(f*angle))/denominator)
    expected/=np.linalg.norm(expected,axis=1,keepdims=True)
    if not np.allclose(np.abs((expected*np.asarray(nominal['root_quat'])).sum(axis=1)),1,atol=1e-5):raise ValueError('CSV and nominal NPZ differ: root orientation')
    if csv.get('foot_contacts') is not None:
        contacts=np.asarray(csv['foot_contacts'])[np.minimum(np.rint(x).astype(int),len(q)-1)]
        if not np.allclose(contacts,nominal['foot_contacts'],atol=1e-5):raise ValueError('CSV and nominal NPZ differ: foot contacts')


def write_nominal_csv(path, nominal):
    """Bundle a matching CSV alongside the exact nominal velocity/contact NPZ."""
    names=json.loads((Path(__file__).resolve().parents[1]/'ardy_web/robot_contract.json').read_text())['joint_order']
    metadata=dict(schema='unitree_g1_23dof_reference_v1',fps=nominal['fps'],
                  quaternion_order='xyzw',joint_names=names)
    q=np.asarray(nominal['root_quat'])
    values=np.column_stack((nominal['root_pos'],q[:,[1,2,3,0]],nominal['joint_pos'],nominal['foot_contacts']))
    with Path(path).open('x') as stream:
        np.savetxt(stream,values,delimiter=',',fmt='%.9g',header=json.dumps(metadata))
    validate_nominal_csv_pair(read_reference(path),nominal)
