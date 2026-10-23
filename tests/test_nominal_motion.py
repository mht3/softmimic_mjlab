from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'demos/sonic_web'))
from nominal_motion import prepare_nominal,read_nominal_motion
from reference import validate_reference, validate_nominal_csv_pair, write_nominal_csv, read_reference


def test_moving_channels_preserve_nominal_body_frame_velocities(tmp_path):
    n=10;q=np.tile([2**-.5,0,0,2**-.5],(n,1,1));joint=np.zeros((n,23));joint[:,2]=np.arange(n)*.02
    source=tmp_path/'source.npz'
    np.savez(source,fps=[50],nominal_joint_pos=joint,nominal_joint_vel=np.zeros((n,23)),nominal_body_pos_w=np.tile([0,0,.74],(n,1,1)),nominal_body_quat_w=q,nominal_body_lin_vel_w=np.tile([1,2,0],(n,1,1)),nominal_body_ang_vel_w=np.tile([0,0,.5],(n,1,1)),nominal_foot_contacts=np.tile([1,0],(n,1)),joint_pos=np.ones((n,23))*99)
    path=prepare_nominal(source,tmp_path/'nominal.npz');m=read_nominal_motion(path)
    write_nominal_csv(tmp_path/'reference.csv',m)
    validate_nominal_csv_pair(read_reference(tmp_path/'reference.csv'),m)
    np.testing.assert_allclose(m['channels']['reference_xy_vel'],np.tile([2,-1],(n,1)),atol=1e-6)
    np.testing.assert_allclose(m['joint_pos'],joint,atol=1e-6)
    c={'reference_mode':'nominal_motion','observations':{k:{'scale':[1]*len(v[0])} for k,v in m['channels'].items()}}
    validate_reference(m,c)
    assert m['joint_pos'][0]!=m['joint_pos'][-1]
    m['channels']['reference_dof_pos'][0]=[]
    with pytest.raises(ValueError):validate_reference(m,c)


def test_adapted_arrays_cannot_be_used_as_nominal(tmp_path):
    source=tmp_path/'bad.npz';np.savez(source,fps=[50],joint_pos=np.zeros((2,23)))
    with pytest.raises(KeyError):prepare_nominal(source,tmp_path/'nominal.npz')


def test_launch_csv_must_match_dynamic_reference():
    import copy
    # A 100 Hz editor CSV and its 50 Hz converted reference agree after sampling.
    t=np.arange(21)/100
    csv=dict(frames=len(t),fps=100,root_pos=np.stack([t,0*t,.74+0*t],axis=1),
             root_quat=np.tile([1.,0,0,0],(len(t),1)),
             joint_pos=np.tile(t[:,None],(1,23)),foot_contacts=np.ones((len(t),2)))
    nominal={k:np.asarray(v)[::2].copy() for k,v in csv.items() if k not in ('frames','fps')}
    nominal.update(frames=11,fps=50)
    validate_nominal_csv_pair(csv,nominal)
    # Equivalent quaternion signs must not reject an otherwise identical clip.
    nominal['root_quat']*=-1
    validate_nominal_csv_pair(csv,nominal)
    for key in ('root_pos','joint_pos','foot_contacts'):
        bad=copy.deepcopy(nominal);bad[key][5,0]+=.1
        with pytest.raises(ValueError,match='differ'):validate_nominal_csv_pair(csv,bad)
    short=dict(csv,frames=10,**{k:np.asarray(csv[k])[:10] for k in nominal if k not in ('frames','fps')})
    with pytest.raises(ValueError,match='duration'):validate_nominal_csv_pair(short,nominal)
