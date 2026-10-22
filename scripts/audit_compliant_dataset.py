"""Read-only CSV/NPZ coverage and kinematic audit; not a dynamics certificate."""
import argparse,json,hashlib,math,zipfile,sys
from pathlib import Path
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compliant_motion_augmentation.collision_checks import self_collision_pairs, SELF_COLLISION_TOLERANCE_M


NOMINAL_KEYS = ('nominal_joint_pos', 'nominal_joint_vel',
                'nominal_body_pos_w', 'nominal_body_quat_w',
                'nominal_body_lin_vel_w', 'nominal_body_ang_vel_w',
                'nominal_foot_contacts')


def audit_npz(path, csv_frames, input_fps, num_bodies):
    """Check the converter's array contract without importing simulation/GPU code."""
    entry = {'npz_present': path.is_file(), 'npz_has_nominal': False,
             'npz_finite': False, 'npz_structural_valid': False}
    errors = []
    if not path.is_file():
        return entry, ['missing paired NPZ']
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        return entry, [f'cannot read NPZ: {exc}']
    entry['npz_has_nominal'] = all(key in arrays for key in NOMINAL_KEYS)
    entry['npz_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        entry['npz_finite'] = all(np.isfinite(value).all() for value in arrays.values())
    except TypeError:
        errors.append('NPZ arrays must be numeric')
    if not entry['npz_finite']:
        errors.append('NPZ contains nonfinite or nonnumeric values')
    fps_array = arrays.get('fps', np.empty(0))
    fps_valid = (fps_array.shape == (1,) and fps_array.dtype.kind in 'fiu'
                 and np.isfinite(fps_array[0]) and fps_array[0] > 0)
    if not fps_valid:
        errors.append('fps must contain one finite positive value')
        expected_frames = None
    else:
        output_fps = float(fps_array[0])
        entry['npz_fps'] = output_fps
        # Match MotionLoader's half-open [0, duration) output timeline.
        expected_frames = math.ceil(((csv_frames - 1) / input_fps) / (1 / output_fps))
        entry['npz_expected_frames'] = expected_frames
    joint_pos = arrays.get('joint_pos', np.empty(0))
    frames = joint_pos.shape[0] if joint_pos.ndim >= 1 else 0
    entry['npz_frames'] = frames
    entry['npz_joint_count'] = joint_pos.shape[1] if joint_pos.ndim == 2 else None
    if expected_frames is not None and frames != expected_frames:
        errors.append(f'NPZ frames {frames} != expected {expected_frames} from CSV timing')
    shapes = {
        'joint_pos': (frames, 23), 'joint_vel': (frames, 23),
        'body_pos_w': (frames, num_bodies, 3),
        'body_quat_w': (frames, num_bodies, 4),
        'body_lin_vel_w': (frames, num_bodies, 3),
        'body_ang_vel_w': (frames, num_bodies, 3),
        'nominal_joint_pos': (frames, 23), 'nominal_joint_vel': (frames, 23),
        'nominal_body_pos_w': (frames, num_bodies, 3),
        'nominal_body_quat_w': (frames, num_bodies, 4),
        'nominal_body_lin_vel_w': (frames, num_bodies, 3),
        'nominal_body_ang_vel_w': (frames, num_bodies, 3),
        'nominal_foot_contacts': (frames, 2),
        'force_body_pos_w': (frames, 3), 'force_body_quat_w': (frames, 4),
        'force_body_index': (frames,), 'force_vector': (frames, 3),
        'torque_vector': (frames, 3), 'stiffness': (frames,),
        'rotational_stiffness': (frames,), 'ff_stiffness': (frames,),
        'ff_rotational_stiffness': (frames,), 'ff_origin': (frames, 3),
        'ff_setpoint_rot': (frames, 4), 'ff_normal': (frames, 3),
        'foot_contacts': (frames, 2),
    }
    for key, expected in shapes.items():
        if key not in arrays:
            errors.append(f'missing NPZ array {key}')
        elif arrays[key].shape != expected:
            errors.append(f'{key} shape {arrays[key].shape} != {expected}')
    for key, value in arrays.items():
        if key != 'fps' and (value.ndim == 0 or value.shape[0] != frames):
            errors.append(f'{key} has inconsistent time dimension')
    for key in ('body_quat_w', 'nominal_body_quat_w',
                'force_body_quat_w', 'ff_setpoint_rot'):
        value = arrays.get(key)
        if value is not None and value.shape == shapes[key] and value.dtype.kind in 'fiu':
            if not np.allclose(np.linalg.norm(value, axis=-1), 1., atol=1e-4, rtol=0.):
                errors.append(f'{key} contains non-unit quaternions')
    ids = arrays.get('force_body_index')
    if ids is not None and ids.dtype.kind in 'fiu':
        if not np.all(np.isfinite(ids) & (ids == np.floor(ids)) & (ids >= -1) & (ids < num_bodies)):
            errors.append('force_body_index contains invalid body indices')
    entry['npz_structural_valid'] = not errors
    return entry, errors


def audit(root,model_path,stride=30,input_fps=30.):
    root=Path(root).resolve()
    if stride < 1 or not np.isfinite(input_fps) or input_fps <= 0:
        raise ValueError('stride and input_fps must be positive')
    model=mujoco.MjModel.from_xml_path(str(model_path));ref=mujoco.MjData(model);adapt=mujoco.MjData(model)
    joint_ids=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)
    assert len(joint_ids)==23,'audit expects direct 23-DOF generator model'
    limits=model.jnt_range[joint_ids];feet=[model.body(s+'_ankle_roll_link').id for s in ('left','right')]
    hands=[model.body(s+'_wrist_roll_rubber_hand').id for s in ('left','right')]
    results=[]
    for path in sorted(root.rglob('*.csv')):
        if any(part.startswith('.') for part in path.relative_to(root).parts):continue
        provenance={'dataset_root':str(root),'file':str(path.relative_to(root)),
                    'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'input_fps':input_fps,'audit_stride':stride}
        try:a=np.loadtxt(path,delimiter=',',ndmin=2)
        except (ValueError,OSError) as exc:
            results.append({**provenance,'error':f'cannot read CSV: {exc}'});continue
        if a.shape[1]!=83 or len(a)<2 or not np.isfinite(a).all():
            results.append({**provenance,'error':'expected at least two rows with 83 finite columns'});continue
        if any(not np.allclose(np.linalg.norm(a[:,s],axis=1),1.,atol=1e-4,rtol=0.)
               for s in (slice(3,7),slice(35,39),slice(76,80))):
            results.append({**provenance,'error':'CSV contains non-unit quaternions'});continue
        force=np.linalg.norm(a[:,63:66],axis=1);torque=np.linalg.norm(a[:,66:69],axis=1)
        joints=a[:,39:62];violation=np.maximum(limits[:,0]-joints,joints-limits[:,1])
        speed=np.abs(np.diff(joints,axis=0)*input_fps)
        footerr=[];handerr=[];depth=[]
        # A collision-free but stuck arm is not a valid settled standing target.
        static_reference=bool(np.allclose(a[:,:30],a[0,:30],atol=1e-5,rtol=0.))
        unforced=(force<.01)&(torque<.01)&(a[:,71]<.01)&(a[:,72]<.01)
        quiet_age=np.zeros(len(a),dtype=int)
        for i in range(len(a)):
            if unforced[i]:quiet_age[i]=(quiet_age[i-1] if i else 0)+1
        settled_errors=[]
        collision_frames=0;collision_pairs={};worst_collision_frame=None;max_penetration=0.;nominal_collision_frames=0
        for sample_index,row in enumerate(a[::stride]):
            for data,offset in ((ref,0),(adapt,32)):
                data.qpos[:3]=row[offset:offset+3]
                data.qpos[3:7]=row[[offset+6,offset+3,offset+4,offset+5]]
                data.qpos[7:]=row[offset+7:offset+30];mujoco.mj_forward(model,data)
            pairs=self_collision_pairs(model,adapt)
            nominal_collision_frames+=bool(self_collision_pairs(model,ref))
            if pairs:
                collision_frames+=1
                penetration=max(pairs.values())
                if penetration>max_penetration:
                    max_penetration=penetration;worst_collision_frame=sample_index*stride
                for pair,value in pairs.items():collision_pairs[pair]=max(collision_pairs.get(pair,0.),value)
            footerr.append(float(np.linalg.norm(adapt.xpos[feet]-ref.xpos[feet],axis=1).max()))
            if static_reference and quiet_age[sample_index*stride]>=math.ceil(2.*input_fps):
                settled_errors.append(float(np.linalg.norm(adapt.xpos[hands]-ref.xpos[hands],axis=1).max()))
            body=int(row[62])
            if 0<body<model.nbody and np.linalg.norm(row[63:66])>1:
                handerr.append(float(np.linalg.norm(adapt.xpos[body]-ref.xpos[body]-row[63:66]/max(row[69],1e-6))))
                if row[65]<-10:depth.append(float(ref.qpos[2]-adapt.qpos[2]))
        entry={**provenance,'frames':len(a),'active_force_fraction':float(np.mean(force>1)),
            'active_torque_fraction':float(np.mean(torque>.01)),'spring_fraction':float(np.mean(np.maximum(a[:,71],a[:,72])>.1)),
            'max_force_N':float(force.max()),'downward_frames_over10N':int(np.sum(a[:,65]<-10)),
            'force_body_ids':np.unique(a[force>1,62]).astype(int).tolist(),
            'max_joint_limit_violation_rad':float(max(violation.max(),0)),
            'max_joint_speed_rad_s':float(speed.max()) if len(speed) else 0.,
            'speed_over_2pi_fraction':float(np.mean(speed>2*np.pi+1e-4)) if len(speed) else 0.,
            'sampled_max_foot_displacement_m':max(footerr,default=0),
            'sampled_max_F_over_K_error_m':max(handerr,default=0),
            'sampled_downward_pelvis_drop_median_m':float(np.median(depth)) if depth else None}
        entry.update(standing_recovery_checked=static_reference,
                     settled_hand_samples=len(settled_errors),
                     max_settled_hand_error_m=max(settled_errors,default=0.))
        npz_result,errors=audit_npz(path.with_suffix('.npz'),len(a),input_fps,model.nbody-1)
        entry.update(npz_result)
        entry.update(self_collision_checked=True, self_collision_free=collision_frames==0 and nominal_collision_frames==0,
                     self_collision_tolerance_m=SELF_COLLISION_TOLERANCE_M,
                     sampled_self_collision_frames=collision_frames,
                     sampled_nominal_self_collision_frames=nominal_collision_frames,
                     max_self_penetration_m=max_penetration,
                     worst_self_collision_frame=worst_collision_frame,
                     self_collision_pairs_max_penetration_m=collision_pairs)
        if collision_frames:
            errors.append(f'self-collision in {collision_frames} sampled frames; maximum penetration {max_penetration:.6f} m')
        if nominal_collision_frames:
            errors.append(f'nominal self-collision in {nominal_collision_frames} sampled frames')
        if settled_errors and max(settled_errors)>.05+1e-6:
            errors.append(f'standing release failed to settle: hand error {max(settled_errors):.6f} m after 2 s without force')
        if errors:
            entry['structural_errors']=errors
            entry['error']='; '.join(errors)
        results.append(entry)
    return results


def main():
    repo=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--model',type=Path,default=repo/'softmimic_deploy/src/assets/g1/g1_23dof.xml')
    p.add_argument('--stride',type=int,default=30);p.add_argument('--output',type=Path,default=Path('dataset_audit.json'))
    p.add_argument('--input-fps',type=float,default=30.)
    args=p.parse_args()
    if args.stride<1:p.error('stride must be positive')
    if not np.isfinite(args.input_fps) or args.input_fps<=0:p.error('input-fps must be finite and positive')
    result=audit(args.directory,args.model,args.stride,args.input_fps)
    if not result:raise ValueError('No CSVs found')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'files':len(result),'errors':sum('error' in r for r in result),
        'unforced_files':sum(r.get('active_force_fraction',0)==0 and r.get('spring_fraction',0)==0 for r in result),
        'joint_limit_violations':sum(r.get('max_joint_limit_violation_rad',0)>1e-3 for r in result),
        'output':str(args.output)},indent=2))

if __name__=='__main__':main()
