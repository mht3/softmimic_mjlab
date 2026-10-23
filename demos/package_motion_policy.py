"""Package a nominal-motion compliance export. Does not convert standing weights."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import onnx
import yaml
sys.path.insert(0,str(Path(__file__).parent/'sonic_web'))
from reference import read_reference,validate_reference,write_nominal_csv


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    c=yaml.safe_load((a.run/'params/deploy.yaml').read_text())
    if c.get('reference_mode')!='nominal_motion':raise ValueError('Requires a motion-conditioned export; do not relabel a standing checkpoint')
    if c.get('observation_dim')!=327 or c['step_dt']!=.02 or c['joint_ids_map']!=[*range(13),15,16,17,18,19,22,23,24,25,26]:raise ValueError('Incompatible policy contract')
    motion=read_reference(a.reference);validate_reference(motion,c)
    source=a.run/'policy.onnx';before=hashlib.sha256(source.read_bytes()).hexdigest();model=onnx.load(source);onnx.checker.check_model(model)
    dims=lambda x:[d.dim_value for d in x.type.tensor_type.shape.dim]
    if len(model.graph.input)!=1 or len(model.graph.output)!=1 or dims(model.graph.input[0])!=[1,327] or dims(model.graph.output[0])!=[1,23]:raise ValueError('Incompatible ONNX')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=before:raise RuntimeError('Export changed during packaging')
    (a.output/'params').mkdir(parents=True);(a.output/'exported').mkdir()
    onnx.external_data_helper.convert_model_from_external_data(model);onnx.save(model,a.output/'exported/policy.onnx')
    shutil.copy2(a.run/'params/deploy.yaml',a.output/'params/deploy.yaml');shutil.copy2(a.reference,a.output/'params/nominal.npz')
    (a.output/'params/reference.json').write_text(json.dumps(motion))
    write_nominal_csv(a.output/'params/reference.csv',motion)
    meta=dict(source_run=str(a.run.resolve()),reference_sha256=motion['source_sha256'],source_onnx_sha256=before,stiffness=200.,rotational_stiffness=1.,behavior_validated=False,hardware_started=False)
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    patch={'FSM':{'Mimic_Compliant_Motion':dict(policy_dir=str(a.output.resolve()),motion_file=str((a.output/'params/nominal.npz').resolve()),time_start=0.,time_end=(motion['frames']-1)/50,loop=False,end_state='FixStand',desired_stiffness=200.,desired_rotational_stiffness=1.)}}
    (a.output/'fsm_patch.yaml').write_text(yaml.safe_dump(patch));print(a.output)
if __name__=='__main__':main()
