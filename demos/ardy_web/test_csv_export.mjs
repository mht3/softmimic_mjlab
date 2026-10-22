import test from 'node:test';
import assert from 'node:assert/strict';
import {exportTimelineCsv} from './timeline-model.mjs';
import {parseReferenceCsv,validateReference,referenceFrame} from '../sonic_web/js/reference.js';
const names=Array.from({length:23},(_,i)=>`joint_${i}`);
const make=(x=0)=>({fps:100,qpos:Array.from({length:101},(_,i)=>[x+i/100,0,.8,Math.SQRT1_2,0,0,Math.SQRT1_2,...Array(23).fill(.1)])});
const entry=(id,motion)=>({id,clip:{prompt:id},motion,inFrame:0,outFrame:100});
test('Save CSV preserves named order, units, trim and WXYZ→XYZW→WXYZ round trip',()=>{
 const e=entry('walk',make());e.inFrame=25;e.outFrame=75;
 const csv=exportTimelineCsv([e],names),rows=csv.trim().split('\n').filter(l=>!l.startsWith('#')).map(l=>l.split(',').map(Number));
 assert.equal(rows.length,51);assert.equal(rows[0].length,30);
 assert.deepEqual(rows[0].slice(3,7),[0,0,.7071067812,.7071067812]);
 const m=parseReferenceCsv(csv,names);assert.equal(m.fps,100);assert.equal(m.root_pos[0][0],.25);assert.equal(m.root_pos.at(-1)[0],.75);
 assert.ok(Math.abs(referenceFrame(m,.25).quat[0]-Math.SQRT1_2)<1e-9);
 assert.throws(()=>parseReferenceCsv(csv,[...names].reverse()),/joint order/);
});
test('saved edited sequence contains aligned blend rather than a raw teleport',()=>{
 const csv=exportTimelineCsv([entry('walk',make()),entry('next',make(100))],names);
 const m=parseReferenceCsv(csv,names);assert.equal(m.frames,201);
 for(let i=1;i<m.frames;i++)assert.ok(Math.abs(m.root_pos[i][0]-m.root_pos[i-1][0])<.1);
 assert.ok(m.root_pos.at(-1)[0]<3);
});
test('malformed reference values and policy mismatches fail closed',()=>{
 const csv=exportTimelineCsv([entry('walk',make())],names),m=parseReferenceCsv(csv,names);
 assert.throws(()=>parseReferenceCsv(csv.replace(',0.8,',',,'),names),/Invalid reference/);
 assert.throws(()=>validateReference(m,{reference_mode:'constant_nominal'}),/constant standing/);
 assert.throws(()=>exportTimelineCsv([],names),/Add a motion/i);
});
