/** Seeded, reset-only draws from SoftMimic's parameter event ranges.
 * Distributions match training; RNG streams and startup/reset scheduling do not.
 */
export const massBodies=['pelvis','left_hip_yaw_link','left_hip_roll_link','left_hip_pitch_link','right_hip_yaw_link','right_hip_roll_link','right_hip_pitch_link'];
export function rng(seed){let state=seed>>>0;return (lo,hi)=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return lo+(hi-lo)*state/4294967296;};}
export function objectNames(model,kind){
 const bytes=model.names,adr=model[`name_${kind}adr`];
 return Array.from(adr,a=>{let s='';for(let i=a;i<bytes.length&&bytes[i];i++)s+=String.fromCharCode(bytes[i]);return s;});
}
const fields=['body_mass','body_inertia','body_ipos','dof_armature','dof_frictionloss','jnt_range','geom_friction','geom_condim','actuator_gainprm','actuator_biasprm'];
export class DomainRandomizer {
 constructor(model,recompute){
  this.model=model;this.recompute=recompute;
  this.base=Object.fromEntries(fields.map(k=>[k,model[k].slice()]));
  this.bodies=objectNames(model,'body');this.geoms=objectNames(model,'geom');
 }
 apply({seed=42,mass=false,actuators=false,limits=false,friction=false,trainingContacts=false}={}){
  if(!Number.isInteger(seed)||seed<0||seed>0xffffffff)throw Error('Seed must be an integer from 0 to 4294967295');
  const m=this.model,b=this.base,u=rng(seed),values={seed};
  for(const k of fields)m[k].set(b[k]);
  // Always draw every parameter, so switching a group off preserves paired draws.
  const torso=this.bodies.indexOf('torso_link');if(torso<0)throw Error('Missing torso_link');
  const payload=u(-2,2);values.payload_kg=mass?payload:0;
  const setMass=(id,newMass)=>{m.body_mass[id]=newMass;for(let j=0;j<3;j++)m.body_inertia[3*id+j]=b.body_inertia[3*id+j]*newMass/b.body_mass[id];};
  if(mass)setMass(torso,b.body_mass[torso]+payload);
  for(const name of massBodies){const id=this.bodies.indexOf(name),scale=u(.7,1.3);if(id<0)throw Error(`Missing ${name}`);if(mass)setMass(id,b.body_mass[id]*scale);}
  values.com_offset_m=[];
  for(let j=0;j<3;j++){const offset=u(-.02,.02);values.com_offset_m.push(mass?offset:0);if(mass)m.body_ipos[3*torso+j]+=offset;}
  for(let i=0;i<m.nu;i++){const kd=u(0,2);if(actuators)m.actuator_biasprm[i*10+2]-=kd;}
  for(let i=6;i<m.nv;i++){const armature=u(.01,.1);if(actuators)m.dof_armature[i]+=armature;}
  for(let i=6;i<m.nv;i++){const loss=u(0,.01);if(actuators)m.dof_frictionloss[i]+=loss;}
  for(let i=1;i<m.njnt;i++)for(let j=0;j<2;j++){const offset=u(-.02,.02);if(limits)m.jnt_range[2*i+j]+=offset;}
  const foot=u(.5,2);values.foot_friction=friction?foot:b.geom_friction[3*this.geoms.findIndex(n=>/^left_foot[1-7]_collision$/.test(n))];
  this.geoms.forEach((name,i)=>{if(!name.endsWith('_collision'))return;const isFoot=/^(left|right)_foot[1-7]_collision$/.test(name);if(isFoot&&friction)m.geom_friction[3*i]=foot;if(trainingContacts&&!isFoot)m.geom_condim[i]=1;});
  this.recompute();return values;
 }
}
