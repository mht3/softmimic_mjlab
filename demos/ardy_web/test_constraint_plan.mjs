import test from 'node:test';
import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';
if(!globalThis.crypto)globalThis.crypto=webcrypto;
import {examplePlan,checkPlan} from './constraints-model.mjs';
test('circle plan has bounded unique timed waypoints and a closed path',()=>{
 const p=examplePlan('circle');assert.equal(checkPlan(p),12);assert.equal(p.constraints.length,13);
 assert.deepEqual(p.constraints[0].position,[0,0]);assert.ok(Math.hypot(...p.constraints.at(-1).position)<1e-10);
 assert.ok(p.constraints.every(c=>c.time<12));assert.equal(new Set(p.constraints.map(c=>c.time)).size,13);
});
test('skill blending is a prompt schedule with a standing finish',()=>{
 const p=examplePlan('skills');assert.equal(checkPlan(p),16);assert.equal(p.segments.length,4);assert.match(p.segments.at(-1).prompt,/standing/);
 assert.throws(()=>checkPlan({...p,constraints:[{time:29}]}),/outside/);
 assert.throws(()=>checkPlan({segments:[{prompt:'walk',seconds:31}],constraints:[]}),/30/);
});
