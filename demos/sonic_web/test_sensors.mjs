import assert from 'node:assert/strict';
import {SensorModel} from './js/sensors.js';
import {SoftMimicPolicy} from './js/policy.js';
import fs from 'node:fs';
const q=Array(30).fill(0),v=Array(29).fill(0);q[3]=1;
const s=new SensorModel(),clean=s.sample(q,v,{noise:false});
q[0]=100;q[2]=-100;v[0]=50;
assert.deepEqual(s.sample(q,v,{noise:false}),clean,'Global pose/linear velocity must not enter measured actor channels');
s.reset();let first=s.sample(q,v);s.reset();assert.deepEqual(s.sample(q,v),first);
for(const [key,bound] of Object.entries({base_ang_vel:.2,projected_gravity:.01,joint_pos:.01,joint_vel:1.5}))
 first[key].forEach((x,i)=>assert.ok(Math.abs(x-clean[key][i])<=bound));
s.reset();s.sample(q,v,{noise:false,delaySteps:1});q[7]=.5;
assert.equal(s.sample(q,v,{noise:false,delaySteps:1}).joint_pos[0],0);
assert.equal(s.sample(q,v,{noise:false,delaySteps:1}).joint_pos[0],.5);
q[3]=0;assert.throws(()=>s.sample(q,v),/quaternion/);q[3]=1;v[6]=Infinity;
assert.throws(()=>s.sample(q,v),/Invalid sensor/);
const c=JSON.parse(fs.readFileSync(new URL('./assets/policy/config.json',import.meta.url)));const p=new SoftMimicPolicy();p.configure(c);
assert.throws(()=>p.pushHistory(q,v),/Invalid sensor/);
console.log('PASS: sensor sources, noise bounds, deterministic reset, delay, invalid-data rejection');
