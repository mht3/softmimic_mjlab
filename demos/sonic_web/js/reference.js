import {gravityInBody} from './policy.js';
export function validateReference(m,c=null){
 if(!Number.isFinite(m.fps)||m.fps<1||m.fps>120||!Number.isInteger(m.frames)||m.frames<2||m.frames>3601||(m.frames-1)/m.fps>30.01)throw Error('Reference must contain 2–3601 frames, at most 30 seconds.');
 for(const [key,width] of [['root_pos',3],['root_quat',4],['joint_pos',23]])if(m[key]?.length!==m.frames||m[key].some(row=>row.length!==width||!row.every(Number.isFinite)))throw Error(`Invalid reference ${key}`);
 if(m.root_quat.some(q=>Math.abs(Math.hypot(...q)-1)>1e-5))throw Error('Reference quaternion is not normalized.');
 if(!c)return;
 if(c.reference_mode!=='constant_nominal')throw Error('Unsupported policy reference mode.');
 const close=(a,b)=>a.length===b.length&&a.every((x,i)=>Math.abs(x-b[i])<1e-5);
 for(const key of ['root_pos','root_quat','joint_pos'])if(m[key].some(row=>!close(row,m[key][0])))throw Error('This policy requires its constant standing reference.');
 const expected={reference_dof_pos:m.joint_pos[0],reference_root_height:[m.root_pos[0][2]],reference_gravity_vector:gravityInBody(m.root_quat[0]),reference_xy_vel:[0,0],reference_yaw_vel:[0],reference_foot_contact:[1,1]};
 for(const [name,value] of Object.entries(expected))if(!close(value,c.observations[name].params.value))throw Error(`Reference/policy mismatch: ${name}`);
 if(m.foot_contacts?.some(row=>!close(row,[1,1])))throw Error('Standing reference must have planted feet.');
}
export function parseReferenceCsv(text,names){
 const lines=text.trim().split(/\r?\n/),meta=lines[0].startsWith('# {')?JSON.parse(lines[0].slice(2)):null;
 if(meta&&(meta.schema!=='unitree_g1_23dof_reference_v1'||meta.quaternion_order!=='xyzw'||JSON.stringify(meta.joint_names)!==JSON.stringify(names)))throw Error('CSV schema, quaternion convention or named joint order differs.');
 const rows=lines.filter(l=>l.trim()&&!l.startsWith('#')).map(l=>l.split(',').map(v=>v.trim()===''?NaN:Number(v)));
 if(rows.some(row=>![30,32].includes(row.length)))throw Error('Expected 30 or 32 columns: XYZ, XYZW, 23 joints, optional foot contacts.');
 const m={fps:meta?meta.fps:30,frames:rows.length,root_pos:rows.map(r=>r.slice(0,3)),root_quat:rows.map(r=>[r[6],...r.slice(3,6)]),joint_pos:rows.map(r=>r.slice(7,30)),foot_contacts:rows[0]?.length===32?rows.map(r=>r.slice(30)):null};validateReference(m);return m;
}
export function referenceFrame(m,t){
 const x=Math.max(0,Math.min(m.frames-1,t*m.fps)),i=Math.floor(x),j=Math.min(m.frames-1,i+1),f=x-i;
 const lerp=key=>m[key][i].map((v,k)=>v*(1-f)+m[key][j][k]*f);
 let a=m.root_quat[i],b=m.root_quat[j],dot=a.reduce((v,x,k)=>v+x*b[k],0);if(dot<0){b=b.map(v=>-v);dot=-dot;}
 let q;if(dot>.9995){q=a.map((v,k)=>v*(1-f)+b[k]*f);const norm=Math.hypot(...q);q=q.map(v=>v/norm);}else{const angle=Math.acos(Math.min(1,dot)),s=Math.sin(angle);q=a.map((v,k)=>(v*Math.sin((1-f)*angle)+b[k]*Math.sin(f*angle))/s);}
 return {pos:lerp('root_pos'),quat:q,joints:lerp('joint_pos')};
}
