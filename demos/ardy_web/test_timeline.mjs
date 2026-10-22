import test from 'node:test';
import assert from 'node:assert/strict';
import {Timeline,length} from './timeline-model.mjs';
const clip=(id,seconds=4,fps=100)=>({id,inFrame:0,outFrame:seconds*fps,motion:{fps,qpos:Array.from({length:seconds*fps+1},(_,i)=>[i])}});
test('playback follows actual saved clips and switches exactly at cuts',()=>{
 const t=new Timeline();const a=clip('walk'),b=clip('stand');t.add(a);t.add(b);
 assert.equal(t.duration,8);assert.equal(t.resolve(3.99).entry,a);assert.equal(t.resolve(4).entry,b);assert.equal(t.resolve(4).sourceTime,0);assert.equal(t.resolve(8).sourceTime,4);
});
test('trims are non-destructive and seeking includes the source in point',()=>{
 const t=new Timeline(),a=clip('walk');t.add(a);t.trim('walk',100,350);assert.equal(t.duration,2.5);assert.equal(t.resolve(0).sourceTime,1);assert.equal(t.resolve(2).sourceTime,3);assert.equal(a.motion.qpos.length,401);assert.equal(a.inFrame,0);
});
test('move, delete and frame timing at mixed frame rates',()=>{
 const t=new Timeline();t.add(clip('a',4,25));t.add(clip('b',3,100));t.move('b',0);assert.equal(t.resolve(0).entry.id,'b');assert.equal(t.resolve(3).entry.id,'a');t.remove('b');assert.equal(t.duration,4);
});
test('30 seconds allowed; overflow and invalid trims leave edit unchanged',()=>{
 const t=new Timeline();t.add(clip('a',20));t.add(clip('b',10));assert.equal(t.duration,30);assert.throws(()=>t.add(clip('c',1)),/30/);assert.equal(t.entries.length,2);assert.throws(()=>t.trim('a',4,4));assert.throws(()=>t.trim('a',-1,5));assert.throws(()=>t.trim('a',0,5000));assert.equal(t.duration,30);
});
test('trim extension cannot overflow the track and boundary seek is clamped',()=>{
 const t=new Timeline(),a=clip('a',20);a.outFrame=1000;t.add(a);t.add(clip('b',20));assert.throws(()=>t.trim('a',0,1100),/30/);assert.equal(length(t.entries[0]),10);assert.equal(t.resolve(-1).sourceTime,0);assert.equal(t.resolve(40).sourceTime,20);t.remove('a');t.remove('b');assert.equal(t.resolve(0),null);
});
