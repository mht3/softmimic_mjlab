import importlib.util
from pathlib import Path
import sys

scripts=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('audit_handoff',scripts/'audit_then_train_stand.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def rows():
    return [dict(file=f'{mode}/{i}.csv',npz_structural_valid=True,npz_has_nominal=True,
                 max_joint_limit_violation_rad=0.,max_joint_speed_rad_s=6.,
                 sampled_max_foot_displacement_m=.03,sampled_max_F_over_K_error_m=.02,
                 audit_stride=1,npz_joint_count=23,npz_fps=50.,self_collision_checked=True,self_collision_free=True,
                 standing_recovery_checked=True,max_settled_hand_error_m=0.)
            for mode,count in m.MODES.items() for i in range(count)]


def test_accepts_only_complete_validated_dataset():
    assert m.validate_report(rows())==[]
    assert m.validate_report(rows()[:-1])


def test_rejects_missing_validation_and_real_release_artifact():
    data=rows();data[0]['sampled_max_foot_displacement_m']=.0718
    assert any('foot_displacement' in s for s in m.validate_report(data))
    data=rows();data[0].pop('npz_structural_valid')
    assert m.validate_report(data)
    data=rows();data[0]['max_joint_speed_rad_s']=float('nan')
    assert m.validate_report(data)
