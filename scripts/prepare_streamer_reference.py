"""Read the streamer's named XYZW CSV and prepare a 30 Hz, 32-column IK reference."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))

def read_csv(path):
    path=Path(path)
    with path.open() as stream:first=stream.readline().strip()
    if not first.startswith('# {'):raise ValueError('Use Save CSV in Unitree_G1_23dof Motion Streamer; metadata is required.')
    meta=json.loads(first[2:])
    if meta.get('schema')!='unitree_g1_23dof_reference_v1' or meta.get('quaternion_order')!='xyzw':raise ValueError('Unsupported CSV schema/quaternion order')
    fps=float(meta['fps']);data=np.loadtxt(path,delimiter=',',ndmin=2)
    if not np.isfinite(fps) or not 1<=fps<=120 or data.shape[1]!=30 or not 2<=len(data)<=3601 or not np.isfinite(data).all():raise ValueError('Invalid reference array or frame rate')
    if (len(data)-1)/fps>30.001:raise ValueError('Reference exceeds 30 seconds')
    if not np.allclose(np.linalg.norm(data[:,3:7],axis=1),1,atol=1e-5):raise ValueError('Root quaternions must be normalized')
    return data,meta

def prepare(source,output,standing=False,duration=30):
    import mujoco
    from compliant_motion_augmentation.collision_checks import self_collision_pairs
    raw,meta=read_csv(source)
    model=mujoco.MjModel.from_xml_path(str(REPO/'softmimic_deploy/src/assets/g1/g1_23dof.xml'));d=mujoco.MjData(model)
    names=[model.joint(i).name for i in range(1,model.njnt)]
    if meta['joint_names']!=names:raise ValueError('CSV named joint order does not match G1 23DOF')
    static=np.max(np.abs(raw-raw[:1]))<1e-6
    if standing and not static:raise ValueError('Standing deployment requires a fixed pose. Use + Standing still reference in the streamer, not an ARDY standing animation.')
    end=duration if standing else (len(raw)-1)/meta['fps'];times=np.arange(int(round(end*30))+1)/30
    if not 0<end<=30:raise ValueError('Duration must be in (0,30]')
    if static:out=np.repeat(raw[:1],len(times),axis=0)
    else:
        src=np.arange(len(raw))/meta['fps'];times=np.minimum(times,src[-1]);out=np.column_stack([np.interp(times,src,raw[:,i]) for i in range(30)])
        out[:,3:7]=Slerp(src,Rotation.from_quat(raw[:,3:7]))(times).as_quat()
    # Check every saved input frame as well as the resampled IK input.
    for sequence in (raw,out):
        for index,row in enumerate(sequence):
            if np.any(row[7:]<model.jnt_range[1:,0]-1e-5) or np.any(row[7:]>model.jnt_range[1:,1]+1e-5):raise ValueError(f'Joint limit exceeded in source frame {index}')
            d.qpos[:]=np.r_[row[:3],row[6],row[3:6],row[7:]];mujoco.mj_forward(model,d)
            bad=self_collision_pairs(model,d,tolerance=0.)
            if bad:raise ValueError(f'Source self-collision at frame {index}: {bad}. Edit this motion before augmentation.')
    feet=[]
    for row in out:
        d.qpos[:]=np.r_[row[:3],row[6],row[3:6],row[7:]];mujoco.mj_forward(model,d)
        feet.append([d.body(n).xpos.copy() for n in ('left_ankle_roll_link','right_ankle_roll_link')])
    feet=np.array(feet)
    if standing:contacts=np.ones((len(out),2))
    else:
        speed=np.linalg.norm(np.gradient(feet,1/30,axis=0),axis=-1)
        contacts=((feet[:,:,2]<=feet[:,:,2].min(axis=0)+.03)&(speed<.3)).astype(float)
    output=Path(output)
    if output.exists():raise FileExistsError(f'Reference already exists: {output}; choose a new name')
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(output,np.c_[out,contacts],delimiter=',')
    report=dict(source=str(Path(source).resolve()),source_metadata=meta,fps=30,frames=len(out),static=bool(static),contacts='both planted' if standing else 'FK height/speed heuristic; review before moving-motion training',source_self_collision_free=True,continuous_collision_certificate=False)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--standing',action='store_true');p.add_argument('--duration',type=float,default=30);a=p.parse_args()
    print(prepare(a.input,a.output,a.standing,a.duration))
if __name__=='__main__':main()
