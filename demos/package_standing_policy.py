"""Create a matched standing-policy bundle for C++ deployment and the web viewer.

This packages interfaces only; it neither starts hardware nor certifies behavior.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"sonic_web"))
from reference import read_reference,validate_reference
import numpy as np,onnx,yaml
from onnx.external_data_helper import convert_model_from_external_data
REPO=Path(__file__).resolve().parents[1]
EXPECTED=[*range(13),15,16,17,18,19,22,23,24,25,26]
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--reference',type=Path,required=True)
 p.add_argument('--dataset',type=Path,required=True)
 p.add_argument('--reference-fps',type=float,default=30)
 p.add_argument('--stiffness',type=float,default=200);p.add_argument('--rotational-stiffness',type=float,default=1)
 a=p.parse_args();run=a.run.resolve();out=a.output.resolve()
 if out.exists():p.error('Output must be a new directory; existing bundles are immutable.')
 c=yaml.safe_load((run/'params/deploy.yaml').read_text())
 assert c['joint_ids_map']==EXPECTED and c['observation_dim']==327 and c['reference_mode']=='constant_nominal'
 assert c['step_dt']==.02
 assert 40<=a.stiffness<=1000 and .1<=a.rotational_stiffness<=10
 policy=run/'policy.onnx';before=hashlib.sha256(policy.read_bytes()).hexdigest()
 model=onnx.load(policy);onnx.checker.check_model(model)
 assert hashlib.sha256(policy.read_bytes()).hexdigest()==before,'Export changed while reading; retry after save finishes'
 assert len(model.graph.input)==len(model.graph.output)==1
 dims=lambda x:[d.dim_value for d in x.type.tensor_type.shape.dim]
 assert dims(model.graph.input[0])==[1,327] and dims(model.graph.output[0])==[1,23]
 convert_model_from_external_data(model)
 reference_motion=read_reference(a.reference,a.reference_fps);validate_reference(reference_motion,c)
 dataset=a.dataset.resolve();sys.path.insert(0,str(REPO/'scripts'))
 from check_compliant_dataset_ready import check
 assert check(dataset)==85,'Standing bundle requires all 85 audited files'
 audit=json.loads((dataset/'audit.json').read_text())
 assert len(audit)==85 and all(r['self_collision_free'] and r['npz_structural_valid'] for r in audit)
 row=next(r for r in audit if r['file'].startswith('zero-wrench/'))
 ref=(dataset/row['file']).with_suffix('.npz');assert hashlib.sha256(ref.read_bytes()).hexdigest()==row['npz_sha256']
 with np.load(ref) as data:
  reference={key:data['nominal_'+key][:2].astype(np.float32) for key in ['joint_pos','joint_vel','body_pos_w','body_quat_w']}
 np.testing.assert_allclose(reference['joint_pos'][0],c['observations']['reference_dof_pos']['params']['value'],atol=1e-6)
 (out/'exported').mkdir(parents=True);(out/'params').mkdir()
 onnx.save_model(model,out/'exported/policy.onnx',save_as_external_data=False)
 shutil.copy2(run/'params/deploy.yaml',out/'params/deploy.yaml')
 shutil.copy2(a.reference,out/'params/reference.csv')
 (out/'params/reference.json').write_text(json.dumps(reference_motion))
 np.savez(out/'params/stand.npz',fps=np.array([50],dtype=np.float32),**reference)
 patch={'FSM':{'Mimic_Compliant_Stand':{'policy_dir':str(out),'motion_file':str(out/'params/stand.npz'),'desired_stiffness':a.stiffness,'desired_rotational_stiffness':a.rotational_stiffness,'time_start':0.,'time_end':.04,'loop':True}}}
 (out/'fsm_patch.yaml').write_text(yaml.safe_dump(patch,sort_keys=False))
 # Emit an isolated standing-only configuration; never replace active hardware settings.
 original=yaml.safe_load((REPO/'deploy/robots/g1_23dof/config/config.yaml').read_text())['FSM']
 passive=original['Passive'];passive['transitions']={'FixStand':'LT + up.on_pressed'}
 fix=original['FixStand'];fix['transitions']={'Passive':'LT + B.on_pressed','Mimic_Compliant_Stand':'RB + A.on_pressed'}
 pose=[0.]*29
 for i,hardware in enumerate(EXPECTED):pose[hardware]=reference_motion['joint_pos'][0][i]
 fix['qs']=[[],pose]
 state=dict(patch['FSM']['Mimic_Compliant_Stand'],limit_angle=1.25,transitions={'Passive':'LT + B.on_pressed'})
 standalone={'FSM':{'_':{'Passive':{'id':1},'FixStand':{'id':2},'Mimic_Compliant_Stand':{'id':5,'type':'Mimic'}},'Passive':passive,'FixStand':fix,'Mimic_Compliant_Stand':state}}
 (out/'controller_config.yaml').write_text(yaml.safe_dump(standalone,sort_keys=False))
 meta=dict(source_run=str(run),reference_sha256=reference_motion['source_sha256'],dataset=str(dataset),source_onnx_sha256=before,bundle_onnx_sha256=hashlib.sha256((out/'exported/policy.onnx').read_bytes()).hexdigest(),observation_dim=327,action_dim=23,stiffness=a.stiffness,rotational_stiffness=a.rotational_stiffness,behavior_validated=False,hardware_started=False)
 (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
 (out/'README.txt').write_text('Matched 23DOF standing export. Interface validation is not behavioral approval.\nUse params/deploy.yaml and exported/policy.onnx together.\nMerge fsm_patch.yaml into the 23DOF controller configuration only after sim validation.\nDo not use the separate legacy 29DOF softmimic_deploy Python configuration.\n')
 print(out)
if __name__=='__main__':main()
