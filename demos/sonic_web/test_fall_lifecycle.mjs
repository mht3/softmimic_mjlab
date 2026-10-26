/** Fault injection through the real demo event loop; no browser or GPU required. */
import fs from 'node:fs';
import assert from 'node:assert/strict';
const base=new URL('./',import.meta.url);
const moduleURL=source=>'data:text/javascript;base64,'+Buffer.from(source).toString('base64');
const elements=new Map();
const element=id=>{
 if(!elements.has(id))elements.set(id,{value:id==='speed'?'1':'0',checked:id==='sensor-noise',disabled:false,textContent:'',style:{}});
 return elements.get(id);
};
let keydown;
globalThis.document={getElementById:element};globalThis.window={addEventListener(name,fn){if(name==='keydown')keydown=fn;}};
let frame;
globalThis.requestAnimationFrame=callback=>{frame=callback;};
let inferences=0,physicsSteps=0,passiveSteps=0;
const modelURL=moduleURL(`export class MuJoCoViewer {
 constructor(){globalThis.testViewer=this;this.ready=true;this.renderer=false;this.data={qpos:new Float64Array(30),qvel:new Float64Array(29),time:0,site_xpos:new Float64Array([0,1,0,0,-1,0]),xipos:new Float64Array(9),xfrc_applied:new Float64Array(18)};this.data.qpos[3]=1;this.mujoco={mj_setConst(){}};this.model={site_bodyid:[1,2],actuator_gainprm:new Float64Array(230),actuator_biasprm:new Float64Array(230)};}
 initThree(){} async loadScene(){} initDragForce(){} clearDragForce(){} setGhost(){} render(){} finalizeStep(){}
 resetToDefault(){this.data.qpos.fill(0);this.data.qpos[3]=1;this.mujoco={mj_setConst(){}};this.model={site_bodyid:[1,2],actuator_gainprm:new Float64Array(230),actuator_biasprm:new Float64Array(230)};}
 setPolicyActuation(){this.passive=false;}setPassiveActuation(){this.passive=true;}
 stepPhysicsOnce(){this.data.xfrc_applied.fill(0);this.externalForceCallback?.();globalThis.testPhysicsStep();if(globalThis.testFall!==false){this.data.qpos[3]=Math.SQRT1_2;this.data.qpos[4]=Math.SQRT1_2;}this.data.time+=.005;}
 stepPhysics(){globalThis.testPassiveStep();if(!this.passive)throw Error('Fault lost damping');}
}`);
globalThis.testPhysicsStep=()=>physicsSteps++;
globalThis.testPassiveStep=()=>passiveSteps++;
globalThis.testInference=()=>{inferences++;return new Float32Array(23);};
const randomizationURL=moduleURL(`export class DomainRandomizer {apply(){return {payload_kg:0,foot_friction:.6}}};export function objectNames(){return ['left_palm','right_palm']}`);
const policyURL=moduleURL(`export class SoftMimicPolicy {
 async load(config){this.config=config;}reset(){}pushHistory(){}async infer(){return globalThis.testInference();}
 actionToTargets(action){return action;}
}`);
const referenceURL=moduleURL(`export function validateReference(){};export function referenceFrame(){return {pos:[0,0,.8],quat:[1,0,0,0],joints:Array(23).fill(0)}}`);
globalThis.fetch=async url=>({ok:true,json:async()=>url.includes('contract')?{joint_names:[]}:url.includes('stand.json')?{frames:1501,fps:50,name:'Fixture'}:{fall_detection:{limit_angle:1.25,passive_damping:3},actuation:'implicit_pd',checkpoint_iteration:15000,stiffness_command:200,rotational_stiffness_command:1}});
let source=fs.readFileSync(new URL('js/demo.js',base),'utf8');
for(const [path,url] of Object.entries({'./viewer.js':modelURL,'./randomization.js':randomizationURL,'./policy.js':policyURL,'./reference.js':referenceURL,'./sensors.js':new URL('js/sensors.js',base).href,'./fall_guard.js':new URL('js/fall_guard.js',base).href}))source=source.replace(`from '${path}'`,`from '${url}'`);
await import(moduleURL(source));
const drain=async()=>{for(let i=0;i<5;i++)await new Promise(resolve=>setImmediate(resolve));};
// Reset before loading cannot arm an unavailable policy.
element('btn-restart').onclick();assert.notEqual(element('btn-policy').textContent,'ON');
await drain();assert.equal(element('btn-policy').disabled,false);
assert.equal(element('btn-policy').textContent,'OFF','Initial load still starts OFF');
element('btn-policy').onclick();element('btn-play').onclick();frame(40);await drain();
assert.equal(inferences,1);assert.equal(physicsSteps,1,'Trip at first fallen substep');
assert.equal(element('btn-policy').disabled,true);assert.match(element('status-text').textContent,/Fall detected/);
frame(80);await drain();assert.equal(inferences,1);assert.equal(passiveSteps,1);
// Even restoring upright orientation must not re-enable the actor.
globalThis.testViewer.data.qpos[3]=1;globalThis.testViewer.data.qpos[4]=0;
element('btn-policy').onclick();frame(120);await drain();assert.equal(inferences,1);
element('btn-restart').onclick();assert.equal(element('btn-policy').disabled,false);assert.equal(element('btn-policy').textContent,'ON');
assert.equal(element('btn-physics').textContent,'ON');assert.equal(element('btn-play').textContent,'▶ Play');
frame(160);await drain();assert.equal(inferences,1,'Reset must not start inference');
element('btn-play').onclick();frame(200);await drain();assert.equal(inferences,2);
assert.equal(element('btn-policy').disabled,true);
// Seeking restores an upright reference pose, so it must also clear the latch.
element('scrub').value='1';element('scrub').oninput();
assert.equal(element('btn-policy').disabled,false);assert.equal(element('btn-policy').textContent,'OFF');
// A rejected inference from before reset must not fault the new session.
let rejectOld;
globalThis.testInference=()=>{inferences++;return new Promise((resolve,reject)=>{rejectOld=reject;});};
element('btn-policy').onclick();element('btn-play').onclick();frame(240);await drain();
assert.equal(inferences,3);element('btn-restart').onclick();
rejectOld(new Error('Old inference failed after reset'));await drain();
assert.equal(element('btn-policy').disabled,false);assert.equal(element('btn-policy').textContent,'ON');
assert.match(element('status-text').textContent,/Reset complete/);
// A new inference error must still stop and latch the actor.
globalThis.testInference=async()=>{inferences++;throw Error('Current inference failed');};
element('btn-play').onclick();frame(280);await drain();
assert.equal(inferences,4);assert.equal(element('btn-policy').disabled,true);
assert.match(element('status-text').textContent,/Current inference failed/);
keydown({key:'r',code:'KeyR',target:{closest(){return null;}}});
assert.equal(element('btn-policy').textContent,'ON','Keyboard reset arms the policy too');
globalThis.testInference=()=>{inferences++;return new Float32Array(23);};
element('btn-play').onclick();frame(320);await drain();
assert.equal(inferences,5);assert.equal(physicsSteps,3,'Re-armed controller reaches physics again');
console.log('PASS: fall/substep cutoff, damping, reset and seek re-arm, stale inference rejection ignored, new errors latch, Reset/R enables the policy but leaves playback paused; Play resumes inference.');

// Scripted loads must ramp, apply at palms with torque, release, and clear on reset/fall.
globalThis.testFall=false;
element('btn-restart').onclick();element('btn-play').onclick();frame(360);await drain();
element('push-down').onclick();const start=globalThis.testViewer.data.time;
globalThis.testViewer.data.time=start+2;frame(400);await drain();
assert.equal(globalThis.testViewer.data.xfrc_applied[8],-30);
assert.equal(globalThis.testViewer.data.xfrc_applied[9],-30,'left palm lever-arm moment');
assert.equal(globalThis.testViewer.data.xfrc_applied[14],-30);
assert.equal(globalThis.testViewer.data.xfrc_applied[15],30,'right palm lever-arm moment');
globalThis.testViewer.data.time=start+4.1;frame(440);await drain();
assert.ok(globalThis.testViewer.data.xfrc_applied.every(v=>v===0));
assert.match(element('load-status').textContent,/released/);
element('push-left').onclick();element('btn-restart').onclick();
assert.equal(element('load-status').textContent,'No scripted load');
element('btn-play').onclick();frame(480);await drain();
assert.ok(globalThis.testViewer.data.xfrc_applied.every(v=>v===0));
element('push-right').onclick();globalThis.testFall=true;frame(520);await drain();
assert.equal(element('btn-policy').disabled,true);assert.match(element('load-status').textContent,/cleared/);
console.log('PASS: scripted palm force/moment, automatic release, reset cancellation, fall cancellation');
