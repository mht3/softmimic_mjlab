// Non-destructive editing in source frames; playback never runs inference.
export const MAX_SECONDS=30;
export const length=e=>(e.outFrame-e.inFrame)/e.motion.fps;
export class Timeline {
  entries=[];
  get duration(){return this.entries.reduce((n,e)=>n+length(e),0);}
  validate(entries){
    if(entries.length>100)throw Error('At most 100 clips');
    for(const e of entries){
      if(!Number.isInteger(e.inFrame)||!Number.isInteger(e.outFrame)||e.inFrame<0||e.outFrame<=e.inFrame||e.outFrame>=e.motion.qpos.length)throw Error('Trim must retain at least two source frames');
    }
    if(entries.reduce((n,e)=>n+length(e),0)>MAX_SECONDS+1e-8)throw Error('The edit exceeds 30 seconds. Trim or remove a clip first.');
  }
  add(entry,index=this.entries.length){const next=[...this.entries];next.splice(index,0,entry);this.validate(next);this.entries=next;}
  trim(id,inFrame,outFrame){const next=this.entries.map(e=>e.id===id?{...e,inFrame,outFrame}:e);this.validate(next);this.entries=next;}
  remove(id){this.entries=this.entries.filter(e=>e.id!==id);}
  move(id,index){const entry=this.entries.find(e=>e.id===id);if(!entry)return;const next=this.entries.filter(e=>e.id!==id);next.splice(Math.max(0,index),0,entry);this.entries=next;}
  resolve(time){
    if(!this.entries.length)return null;
    let start=0;const t=Math.max(0,Math.min(this.duration,time));
    for(let i=0;i<this.entries.length;i++){
      const e=this.entries[i],end=start+length(e);
      if(t<end-1e-9||i===this.entries.length-1)return {entry:e,index:i,start,sourceTime:e.inFrame/e.motion.fps+Math.min(length(e),Math.max(0,t-start))};
      start=end;
    }
  }
}

export function clipWarnings(clip){
 const a=clip.audit;if(!a)return ['Local motion: collision audit and retargeting fit have not been verified.'];
 const messages=[];
 if(a.constraint_review_required){messages.push('Generation target fit needs review.');for(const r of a.constraint_errors||[])if(r.needs_review)messages.push(`${r.type} at ${r.time.toFixed(2)} s: ${(r.position_error_m*100).toFixed(1)} cm (threshold ${(r.review_threshold_m*100).toFixed(0)} cm)${r.max_joint_error_rad!==undefined?`; max joint error ${r.max_joint_error_rad.toFixed(2)} rad`:''}`);}
 if(a.geometry_passed===false)messages.push('Source geometry checks failed.');
 if(a.fidelity_review_required){
  messages.push('Retargeting fit needs review.');
  for(const [key,label,limit] of [['max_foot_tracking_error_m','Feet',8],['max_palm_tracking_error_m','Palms',12],['max_pelvis_tracking_error_m','Pelvis',8]]){
   if(Number.isFinite(a[key]))messages.push(`${label}: ${(a[key]*100).toFixed(1)} cm (review threshold ${limit} cm)${a[key]*100>limit?' — exceeded':''}`);
  }
 }
 return messages;
}

// Kinematic editor playback. Source files and their audits remain unchanged.
export const BLEND_SECONDS = 0.6;
const mix=(a,b,t)=>a+(b-a)*t;
const yaw=q=>Math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]));
export function slerp(a,b,t){
 let dot=a.reduce((v,x,i)=>v+x*b[i],0);if(dot<0){b=b.map(x=>-x);dot=-dot;}
 if(dot>.9995){const q=a.map((x,i)=>mix(x,b[i],t)),n=Math.hypot(...q);return q.map(x=>x/n);}
 const angle=Math.acos(Math.min(1,dot)),s=Math.sin(angle);
 return a.map((x,i)=>(Math.sin((1-t)*angle)*x+Math.sin(t*angle)*b[i])/s);
}
export function sampleMotion(motion,time){
 const x=Math.max(0,Math.min(motion.qpos.length-1,time*motion.fps)),i=Math.floor(x),f=x-i,a=motion.qpos[i],b=motion.qpos[Math.min(i+1,motion.qpos.length-1)];
 const q=a.map((v,k)=>mix(v,b[k],f));q.splice(3,4,...slerp(a.slice(3,7),b.slice(3,7),f));return q;
}
function transform(q,placement){
 const {angle,dx,dy}=placement,c=Math.cos(angle),s=Math.sin(angle),w=Math.cos(angle/2),z=Math.sin(angle/2),r=q.slice();
 r[0]=c*q[0]-s*q[1]+dx;r[1]=s*q[0]+c*q[1]+dy;
 r.splice(3,4,w*q[3]-z*q[6],w*q[4]-z*q[5],w*q[5]+z*q[4],w*q[6]+z*q[3]);return r;
}
function hermite(a,b,va,vb,t,duration,bounded){
 // Limit joint/height slopes to avoid interpolation overshoot beyond endpoints.
 if(bounded){const delta=(b-a)/duration;if(Math.abs(delta)<1e-10)va=vb=0;else{va=Math.max(0,Math.min(3,va/delta))*delta;vb=Math.max(0,Math.min(3,vb/delta))*delta;const magnitude=Math.hypot(va/delta,vb/delta);if(magnitude>3){va*=3/magnitude;vb*=3/magnitude;}}}
 const u=t/duration,u2=u*u,u3=u2*u;
 return (2*u3-3*u2+1)*a+(u3-2*u2+u)*duration*va+(-2*u3+3*u2)*b+(u3-u2)*duration*vb;
}
export class MotionPlayback {
 constructor(){this.entries=null;this.layout=[];}
 prepare(entries){
  if(this.entries===entries)return;this.entries=entries;this.layout=[];let start=0;
  for(const entry of entries){
   const duration=(entry.outFrame-entry.inFrame)/entry.motion.fps,inTime=entry.inFrame/entry.motion.fps;
   const item={entry,start,duration,inTime,placement:{angle:0,dx:0,dy:0},blend:0};
   if(this.layout.length){
    const previous=this.layout.at(-1),dt=Math.min(1/previous.entry.motion.fps,previous.duration/4),end=this.sampleItem(previous,previous.duration),before=this.sampleItem(previous,previous.duration-dt),velocity=end.map((x,k)=>(x-before[k])/dt),first=sampleMotion(entry.motion,inTime);
    const angle=yaw(end.slice(3,7))-yaw(first.slice(3,7)),c=Math.cos(angle),s=Math.sin(angle),blend=Math.min(BLEND_SECONDS,duration/2);
    // A walking-to-idle join gets a short deceleration distance instead of
    // stopping at the cut or returning to the source clip's world origin.
    item.placement={angle,dx:end[0]+velocity[0]*blend/2-(c*first[0]-s*first[1]),dy:end[1]+velocity[1]*blend/2-(s*first[0]+c*first[1])};
    item.blend=blend;item.from=end;item.velocity=velocity;
    item.to=transform(sampleMotion(entry.motion,inTime+blend),item.placement);
    const step=Math.min(1/entry.motion.fps,duration-blend),next=transform(sampleMotion(entry.motion,inTime+blend+step),item.placement);
    item.toVelocity=next.map((x,k)=>(x-item.to[k])/step);
   }
   this.layout.push(item);start+=duration;
  }
 }
 sampleItem(item,time){
  const t=Math.max(0,Math.min(item.duration,time));
  if(!item.blend||t>=item.blend)return transform(sampleMotion(item.entry.motion,item.inTime+t),item.placement);
  const q=item.from.map((a,k)=>hermite(a,item.to[k],item.velocity[k],item.toVelocity[k],t,item.blend,k>=2));
  const u=t/item.blend,s=u*u*(3-2*u);q.splice(3,4,...slerp(item.from.slice(3,7),item.to.slice(3,7),s));return q;
 }
 sample(entries,time){
  this.prepare(entries);if(!this.layout.length)return null;
  const last=this.layout.at(-1),t=Math.max(0,Math.min(last.start+last.duration,time));
  const item=this.layout.find(x=>t<x.start+x.duration)||last;
  return this.sampleItem(item,t-item.start);
 }
}

export function exportTimelineCsv(entries,jointNames,fps=100){
 if(!entries.length)throw Error('Add a motion to the timeline before saving.');
 if(jointNames.length!==23||new Set(jointNames).size!==23)throw Error('Expected 23 named joints.');
 const timeline=new Timeline();timeline.validate(entries);timeline.entries=entries;
 const playback=new MotionPlayback(),frames=Math.round(timeline.duration*fps)+1;
 const metadata={schema:'unitree_g1_23dof_reference_v1',fps,quaternion_order:'xyzw',joint_names:jointNames,duration:(frames-1)/fps,blend_seconds:BLEND_SECONDS,geometry_audited:false,
  clips:entries.map(e=>({source:e.base,prompt:e.clip.prompt,in_frame:e.inFrame,out_frame:e.outFrame,source_fps:e.motion.fps,warnings:clipWarnings(e.clip)}))};
 const lines=['# '+JSON.stringify(metadata),'# '+['root_x','root_y','root_z','qx','qy','qz','qw',...jointNames].join(',')];
 for(let i=0;i<frames;i++){
  const q=playback.sample(entries,Math.min(i/fps,timeline.duration));
  if(q.length!==30||!q.every(Number.isFinite))throw Error('Invalid pose in timeline.');
  lines.push([...q.slice(0,3),...q.slice(4,7),q[3],...q.slice(7)].map(x=>Number(x.toFixed(10))).join(','));
 }
 return lines.join('\n')+'\n';
}
