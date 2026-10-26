/** Simulated pelvis IMU + motor encoders. No global position, force or contacts.
 * Noise bounds match the training actor; they are not measured hardware specs.
 */
import {gravityInBody} from './policy.js';
export class SensorModel {
  constructor() { this.reset(); }
  reset(seed=42) { this.queue=[]; this.seed=seed>>>0; }
  uniform(bound) {
    this.seed=(Math.imul(1664525,this.seed)+1013904223)>>>0;
    return (2*this.seed/4294967296-1)*bound;
  }
  sample(qpos,qvel,{noise=true,delaySteps=0,channels={}}={}) {
    if (![0,1,2].includes(delaySteps)) throw Error('Invalid sensor delay');
    const quat=Array.from(qpos.slice(3,7)),norm=Math.hypot(...quat);
    if(!Number.isFinite(norm)||norm<.9||norm>1.1)throw Error('Invalid pelvis IMU quaternion');
    const m={base_ang_vel:Array.from(qvel.slice(3,6)),projected_gravity:gravityInBody(quat.map(x=>x/norm)),
      joint_pos:Array.from(qpos.slice(7,30)),joint_vel:Array.from(qvel.slice(6,29))};
    for(const [name,bound] of Object.entries({base_ang_vel:.2,projected_gravity:.01,joint_pos:.01,joint_vel:1.5})) {
      if(!m[name].every(Number.isFinite))throw Error(`Invalid sensor: ${name}`);
      if(noise)m[name]=m[name].map(x=>x+this.uniform(bound)*(channels[name]===false?0:1));
    }
    this.queue.push(m);
    while(this.queue.length>delaySteps+1)this.queue.shift();
    return this.queue[0];
  }
}
