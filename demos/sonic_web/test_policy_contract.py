"""CPU cross-language comparison against mjlab's actual quaternion/history utilities."""
import argparse,json,subprocess,tempfile
from pathlib import Path
import numpy as np
import torch,yaml
from mjlab.utils.buffers import CircularBuffer
from mjlab.utils.lab_api.math import quat_apply_inverse

p=argparse.ArgumentParser();p.add_argument('--deploy',required=True,type=Path);args=p.parse_args()
c=yaml.safe_load(args.deploy.read_text());c.update(effort_limits=[88,88,88,139,50,50]*2+[88]+[25]*10,stiffness_command=100,rotational_stiffness_command=10)
rng=np.random.default_rng(31);frames=[];expected=[];buffers={}
for t in range(6):
 q=rng.normal(size=30);q[3:7]/=np.linalg.norm(q[3:7]);v=rng.normal(size=29)
 action=rng.normal(size=23)
 raw={'base_ang_vel':v[3:6], 'projected_gravity':quat_apply_inverse(torch.tensor(q[None,3:7],dtype=torch.float32),torch.tensor([[0.,0.,-1.]])).numpy()[0], 'joint_pos_rel':q[7:]-c['default_joint_pos'],'joint_vel_rel':v[6:],'last_action':action,'desired_stiffness_log':[np.log(100+1e-6)],'desired_rotational_stiffness_log':[np.log(10+1e-6)]}
 out=[]
 for name,term in c['observations'].items():
  val=torch.tensor(raw.get(name,term['params'].get('value')),dtype=torch.float32).reshape(1,-1)
  if term['clip']:val=val.clamp(*term['clip'])
  val*=torch.tensor(term['scale'])
  if name not in buffers:buffers[name]=CircularBuffer(3,1,'cpu')
  if t==4:buffers[name].reset()
  buffers[name].append(val);out.extend(buffers[name].buffer.flatten().tolist())
 expected.append(out);frames.append(dict(q=q.tolist(),v=v.tolist(),action=action.tolist(),reset=t==4))
with tempfile.TemporaryDirectory() as d:
 f=Path(d)/'fixture.json';f.write_text(json.dumps(dict(config=c,frames=frames)))
 module=(Path(__file__).parent/'js/policy.js').resolve().as_uri()
 code='''import fs from 'node:fs';import {SoftMimicPolicy} from MODULE;
 const f=JSON.parse(fs.readFileSync(process.argv[1]));const p=new SoftMimicPolicy();p.configure(f.config);
 const outputs=f.frames.map(x=>{if(x.reset)p.reset();p.lastAction=x.action;p.pushHistory(x.q,x.v);return Array.from(p.observation)});
 let rejected=false;try{p.configure({...f.config,observation_dim:994})}catch(e){rejected=true}if(!rejected)throw Error('Accepted SONIC contract');
 console.log(JSON.stringify(outputs));'''.replace('MODULE',json.dumps(module))
 got=json.loads(subprocess.check_output(['node','--input-type=module','-e',code,str(f)]))
 np.testing.assert_allclose(got,expected,atol=1e-6,rtol=1e-6)
 print('PASS: all 327 channels across six frames, history rollover/reset, quaternion transform and incompatible-policy rejection.')
