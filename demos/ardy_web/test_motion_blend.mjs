import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {MotionPlayback,sampleMotion} from './timeline-model.mjs';
import {Timeline,clipWarnings} from './timeline-model.mjs';
const motion=(offset=0,speed=0,heading=0)=>({fps:100,qpos:Array.from({length:401},(_,i)=>[offset+speed*i/100,0,.8,Math.cos(heading/2),0,0,Math.sin(heading/2),...Array(23).fill(offset/100)])});
const entry=(id,m)=>({id,motion:m,inFrame:0,outFrame:m.qpos.length-1});
const diff=(a,b)=>Math.max(...a.map((v,k)=>Math.abs(v-b[k])));
test('walking to standing aligns origins and decelerates without teleport or backwards travel',()=>{
 const a=entry('walk',motion(0,1)),b=entry('stand',motion(100,0)),entries=[a,b],p=new MotionPlayback(),saved=JSON.stringify(entries);
 assert.ok(diff(p.sample(entries,4-1e-6),p.sample(entries,4))<2e-6);
 let x=p.sample(entries,4)[0];for(let i=1;i<=100;i++){const q=p.sample(entries,4+i/100);assert.ok(q[0]>=x-1e-10);x=q[0];assert.ok(Math.abs(Math.hypot(...q.slice(3,7))-1)<1e-10);}
 assert.ok(Math.abs(x-4.3)<1e-8);assert.equal(JSON.stringify(entries),saved);
 assert.ok(diff(p.sample(entries,4.6-1e-6),p.sample(entries,4.6+1e-6))<1e-5);
});
test('heading is carried forward, short trimmed clips stay deterministic after reorder',()=>{
 const t=new Timeline();t.add(entry('a',motion(0,1,1)));t.add(entry('b',motion(20,0,-2)));t.trim('b',100,120);const p=new MotionPlayback();
 assert.ok(diff(p.sample(t.entries,4-1e-6),p.sample(t.entries,4))<2e-6);
 const end=p.sample(t.entries,t.duration);p.sample(t.entries,0);assert.deepEqual(p.sample(t.entries,t.duration),end);
 t.move('b',0);assert.deepEqual(p.sample(t.entries,0),sampleMotion(t.entries[0].motion,1));
});
test('source review warnings include all errors and their thresholds',()=>{
 assert.deepEqual(clipWarnings({audit:{geometry_passed:true,fidelity_review_required:false}}),[]);
 const warnings=clipWarnings({audit:{fidelity_review_required:true,max_foot_tracking_error_m:.092,max_palm_tracking_error_m:.157,max_pelvis_tracking_error_m:.104}}).join('\n');
 assert.match(warnings,/Feet: 9.2 cm/);assert.match(warnings,/Palms: 15.7 cm/);assert.match(warnings,/Pelvis: 10.4 cm/);assert.match(warnings,/exceeded/);
});
test('real saved walking→standing joins remain continuous at the boundary and blend end',()=>{
 const root=new URL('../../../sonic23/ardy_motions/',import.meta.url);
 const read=id=>JSON.parse(fs.readFileSync(new URL(`${id}/motion.json`,root)));
 const entries=[entry('walk',read('f0cef4ab1b1f486c99da78450f59a11d')),entry('stand',read('8fe5d11abc5a47f8b2e207235733e985'))],p=new MotionPlayback(),cut=3.96;
 const originalJump=Math.hypot(...entries[0].motion.qpos.at(-1).slice(0,2).map((x,i)=>x-entries[1].motion.qpos[0][i]));
 for(const t of [cut,cut+.6])assert.ok(diff(p.sample(entries,t-1e-6),p.sample(entries,t+1e-6))<1e-4);
 for(let i=0;i<=60;i++){const q=p.sample(entries,cut+i/100);assert.ok(q.every(Number.isFinite));assert.ok(Math.abs(Math.hypot(...q.slice(3,7))-1)<1e-8);}
 console.log(`Original saved-clip XY jump: ${originalJump.toFixed(3)} m; blended boundary continuous within 1e-4 in pose coordinates.`);
});
