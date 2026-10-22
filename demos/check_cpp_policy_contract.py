"""Exercise the real C++ observation/action/ONNX managers without DDS or a robot."""
from pathlib import Path
import argparse,subprocess,tempfile,re,json
import numpy as np,yaml
from onnx.reference import ReferenceEvaluator
ROOT=Path(__file__).resolve().parents[2]/'softmimic_mjlab'
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);a=p.parse_args();bundle=a.bundle.resolve()
c=yaml.safe_load((bundle/'params/deploy.yaml').read_text());meta=json.loads((bundle/'manifest.json').read_text())
rng=np.random.default_rng(17);frames=[]
for i in range(6):
 q=rng.normal(size=30);q[3:7]/=np.linalg.norm(q[3:7]);v=rng.normal(size=29);act=rng.normal(size=23)
 if i in (0,4):act*=0
 frames.append(dict(q=q.tolist(),v=v.tolist(),action=act.tolist(),reset=i in (0,4)))
source=(ROOT/'deploy/robots/g1_23dof/src/State_Mimic.cpp').read_text()
registrations='\n'.join(re.search(r'REGISTER_OBSERVATION\('+name+r'\)\s*\{[^}]+\};?\s*\}',source).group(0) for name in ['desired_stiffness_log','desired_rotational_stiffness_log'])
cpp=r'''
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
float desired_stiffness=200.f,desired_rotational_stiffness=1.f;
namespace isaaclab { namespace mdp { REGISTRATIONS } }
struct Robot:isaaclab::Articulation {
 YAML::Node frame;
 void update() override {
  if(!frame)return;
  auto q=frame["q"].as<std::vector<float>>(),v=frame["v"].as<std::vector<float>>();
  data.root_quat_w=Eigen::Quaternionf(q[3],q[4],q[5],q[6]);
  data.projected_gravity_b=data.root_quat_w.conjugate()*data.GRAVITY_VEC_W;
  for(int j=0;j<3;j++)data.root_ang_vel_b[j]=v[3+j];
  for(int j=0;j<23;j++){data.joint_pos[j]=q[7+j];data.joint_vel[j]=v[6+j];}
 }
};
int main(int argc,char**argv){
 auto cfg=YAML::LoadFile(argv[1]);auto fixture=YAML::LoadFile(argv[3]);
 desired_stiffness=fixture["stiffness"].as<float>();desired_rotational_stiffness=fixture["rotational_stiffness"].as<float>();
 auto robot=std::make_shared<Robot>();robot->frame=fixture["frames"][0];
 isaaclab::ManagerBasedRLEnv env(cfg,robot);isaaclab::OrtRunner runner(argv[2]);YAML::Node result;
 for(size_t index=0;index<fixture["frames"].size();++index){
  YAML::Node f=fixture["frames"][index];
  robot->frame=f;robot->update();
  if(f["reset"].as<bool>()){
   env.reset();
   auto targets=env.action_manager->processed_actions();auto defaults=cfg["actions"]["JointPositionAction"]["offset"].as<std::vector<float>>();
   if(targets!=defaults)throw std::runtime_error("Reset left stale motor targets");
  }else env.action_manager->process_action(f["action"].as<std::vector<float>>());
  auto obs=env.observation_manager->compute();auto action=runner.act(obs);env.action_manager->process_action(action);
  YAML::Node row;row["obs"]=obs.at("obs");row["action"]=action;row["targets"]=env.action_manager->processed_actions();result["frames"].push_back(row);
 }
 int rejected=0;
 try{runner.act({{"obs",std::vector<float>(326,0)}});}catch(const std::runtime_error&){rejected++;}
 auto invalid=std::vector<float>(327,0);invalid[0]=std::numeric_limits<float>::quiet_NaN();
 try{runner.act({{"obs",invalid}});}catch(const std::runtime_error&){rejected++;}
 try{env.action_manager->process_action(std::vector<float>(22,0));}catch(const std::runtime_error&){rejected++;}
 if(rejected!=3)throw std::runtime_error("Malformed input was accepted");
 YAML::Emitter out;out.SetFloatPrecision(9);out<<result;std::cout<<out.c_str();
}
'''.replace('REGISTRATIONS',registrations)
ort=ROOT/'deploy/thirdparty/onnxruntime-linux-x64-1.22.0'
with tempfile.TemporaryDirectory() as d:
 d=Path(d);(d/'test.cpp').write_text(cpp);(d/'fixture.yaml').write_text(yaml.safe_dump(dict(frames=frames,stiffness=meta['stiffness'],rotational_stiffness=meta['rotational_stiffness'])))
 cmd=['g++','-std=c++17','-O1',str(d/'test.cpp'),'-I'+str(ROOT/'deploy/include'),'-I'+str(ort/'include'),'-I/usr/include/eigen3','-I/usr/local/include','-lyaml-cpp','-lfmt',str(ort/'lib/libonnxruntime.so.1.22.0'),'-Wl,-rpath,'+str(ort/'lib'),'-pthread','-o',str(d/'test')]
 subprocess.run(cmd,check=True)
 got=yaml.safe_load(subprocess.check_output([str(d/'test'),str(bundle/'params/deploy.yaml'),str(bundle/'exported/policy.onnx'),str(d/'fixture.yaml')]))
 # The browser adapter produces the comparison observations and targets.
 c.update(effort_limits=[88,88,88,139,50,50]*2+[88]+[25]*10,stiffness_command=meta['stiffness'],rotational_stiffness_command=meta['rotational_stiffness'])
 (d/'fixture.json').write_text(json.dumps(dict(config=c,frames=frames)))
 module=(Path(__file__).parent/'sonic_web/js/policy.js').resolve().as_uri()
 js="import fs from 'node:fs';import {SoftMimicPolicy} from "+json.dumps(module)+";const f=JSON.parse(fs.readFileSync(process.argv[1]));const p=new SoftMimicPolicy();p.configure(f.config);console.log(JSON.stringify(f.frames.map(x=>{if(x.reset)p.reset();p.lastAction=x.action;p.pushHistory(x.q,x.v);return Array.from(p.observation)})));"
 expected=json.loads(subprocess.check_output(['node','--input-type=module','-e',js,str(d/'fixture.json')]))
 model=ReferenceEvaluator(str(bundle/'exported/policy.onnx'));max_action=0
 for i,row in enumerate(got['frames']):
  np.testing.assert_allclose(row['obs'],expected[i],atol=1e-6,rtol=1e-6)
  action=model.run(None,{model.input_names[0]:np.array([expected[i]],dtype=np.float32)})[0][0]
  np.testing.assert_allclose(row['action'],action,atol=2e-5,rtol=2e-5)
  targets=action*np.array(c['actions']['JointPositionAction']['scale'])+c['actions']['JointPositionAction']['offset']
  np.testing.assert_allclose(row['targets'],targets,atol=2e-5,rtol=2e-5)
  max_action=max(max_action,float(np.max(np.abs(action-row['action']))))
 result=dict(passed=True,frames=6,observation_dim=327,action_dim=23,max_action_error=max_action,checks=['C++ versus browser observations','C++ ONNX versus CPU reference','action scaling and offset','reset clears motor targets','wrong dimensions and NaN rejected'],hardware_started=False)
 (bundle/'cpp_contract_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
