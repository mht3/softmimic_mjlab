/** Fixed-reference playback plus the matched SoftMimic ONNX controller. */
import {MuJoCoViewer} from './viewer.js';
import {FallGuard} from './fall_guard.js';
import {DomainRandomizer,objectNames} from './randomization.js';
import {SensorModel} from './sensors.js';
import {SoftMimicPolicy} from './policy.js';
import {validateReference,referenceFrame} from './reference.js';
const $=id=>document.getElementById(id),viewer=new MuJoCoViewer($('canvas'));
let motion=null,policy=null,policyReady=false,policyEnabled=false,physics=false,running=false,time=0,revision=0,busy=false,clock=0,names=[];
const sensors=new SensorModel();
let fallGuard=new FallGuard(),randomizer=null,scriptedLoad=null,siteIds={},metrics=null,preparedContacts=null;
const noiseChannels=['base_ang_vel','projected_gravity','joint_pos','joint_vel'];
function sensorOptions(){return {noise:$('sensor-noise').checked,delaySteps:Number($('sensor-delay').value),channels:Object.fromEntries(noiseChannels.map(n=>[n,$('noise-'+n).checked]))};}
function applyDraw(){
 if(!randomizer)return;
 const seed=Number($('dr-seed').value);
 const values=randomizer.apply({seed,mass:$('dr-mass').checked,actuators:$('dr-actuators').checked,limits:$('dr-limits').checked,friction:$('dr-friction').checked,trainingContacts:$('dr-contacts').checked});
 // The drive cache must follow the sampled damping, including after a fall.
 viewer._actuatorParameters={gain:viewer.model.actuator_gainprm.slice(),bias:viewer.model.actuator_biasprm.slice()};
 if(preparedContacts)$('contact-help').textContent=`Current ground friction: feet ${values.foot_friction.toFixed(2)}, body ${$('dr-contacts').checked?0:preparedContacts.body_sliding_friction} (not calibrated).`;
 $('dr-values').textContent=`Seed ${seed} · torso payload ${values.payload_kg.toFixed(2)} kg · foot μ ${values.foot_friction.toFixed(2)} · body contacts ${$('dr-contacts').checked?'training frictionless':'prepared profile'}`;
 sensors.reset(seed);metrics={sum:0,n:0,peak:0};scriptedLoad=null;$('load-status').textContent='No scripted load';$('robustness-metrics').textContent='Wrist RMS: —';
}
function applyTest(){try{reset(0,true,true);}catch(e){stopPolicy(`Randomization failed: ${e.message}`);}}
function startLoad(kind){
 if(!running||!physics||!policyEnabled){status('Enable the policy and press Play before applying a load.');return;}
 if(duration()-time<4){status('Reset first: a load needs four seconds of playback remaining.');return;}
 scriptedLoad={kind,start:viewer.data.time};$('load-status').textContent=`${kind}: applying for 4 s, then release`;
}
function loadForces(){
 if(!scriptedLoad)return;
 const t=viewer.data.time-scriptedLoad.start;
 if(t>=4){scriptedLoad=null;$('load-status').textContent='Load released — observe recovery';return;}
 const scale=t<1?.5-.5*Math.cos(Math.PI*t):t<3?1:.5+.5*Math.cos(Math.PI*(t-3));
 const forces=scriptedLoad.kind==='downward'?{left_palm:[0,0,-30],right_palm:[0,0,-30]}:scriptedLoad.kind==='left pull'?{left_palm:[0,20,0]}:{right_palm:[0,-20,0]};
 for(const [name,force] of Object.entries(forces)){
  const site=siteIds[name],body=viewer.model.site_bodyid[site],d=viewer.data;
  const f=force.map(x=>x*scale),r=[0,1,2].map(i=>d.site_xpos[3*site+i]-d.xipos[3*body+i]);
  const torque=[r[1]*f[2]-r[2]*f[1],r[2]*f[0]-r[0]*f[2],r[0]*f[1]-r[1]*f[0]];
  for(let i=0;i<3;i++){d.xfrc_applied[6*body+i]+=f[i];d.xfrc_applied[6*body+3+i]+=torque[i];}
 }
}
function measure(){
 if(!metrics)return;
 for(let j=0;j<names.length;j++)if(names[j].includes('wrist')){const v=viewer.data.qvel[6+j];metrics.sum+=v*v;metrics.n++;metrics.peak=Math.max(metrics.peak,Math.abs(v));}
 if(metrics.n%40===0)$('robustness-metrics').textContent=`Wrist speed RMS ${(Math.sqrt(metrics.sum/metrics.n)).toFixed(3)} rad/s · peak ${metrics.peak.toFixed(2)} rad/s (since reset)`;
}
const duration=()=>motion?(motion.frames-1)/motion.fps:0;
const status=message=>{$('status-text').textContent=message;};
function controls(){
 $('btn-play').textContent=running?'Ⅱ Pause':'▶ Play';$('btn-play').disabled=!motion;
 $('btn-policy').disabled=!policyReady||fallGuard.tripped;$('btn-policy').textContent=policyEnabled?'ON':policyReady?'OFF':'Awaiting matched policy';
 $('btn-physics').textContent=physics?'ON':'OFF';$('time').textContent=`${time.toFixed(2)} / ${duration().toFixed(2)} s`;$('scrub').max=duration();$('scrub').value=time;
 $('playhead').style.left=`${time/30*100}%`;
}
function ghost(){if(motion){const q=referenceFrame(motion,time);viewer.setGhost(q.pos,q.quat,q.joints,[0,$('ghost-offset').checked?1.25:0,0]);}}
function reset(at=0,clearFault=fallGuard.tripped,enablePolicy=false){
 if(clearFault){fallGuard.reset();policyEnabled=false;status('Reset complete. Policy is OFF; enable it explicitly to resume.');}
 revision++;running=false;time=Math.max(0,Math.min(duration(),at));viewer.clearDragForce();policy?.reset();applyDraw();
 if(motion){const q=referenceFrame(motion,time);viewer.resetToDefault(q.joints,q.pos[2],q.pos,q.quat);ghost();}
 if(enablePolicy&&policyReady){policyEnabled=true;physics=true;status('Reset complete. Policy is ON; press Play to simulate.');}
 controls();
}
function checkPair(){policyReady=false;if(!policy||!motion)return;try{validateReference(motion,policy.config);policyReady=true;status('Reference matches policy. Enable policy control, then press Play.');}catch(e){policyEnabled=false;status(e.message);}}
function installMotion(m,name){validateReference(m);motion=m;motion.name=name;policyEnabled=false;physics=false;checkPair();$('reference-name').textContent=name;$('reference-clip').textContent=name;$('reference-clip').style.width=`${duration()/30*100}%`;reset();}
$('btn-play').onclick=()=>{if(!motion)return;if(time>=duration())reset();running=!running;controls();};
$('btn-restart').onclick=()=>reset(0,true,true);
$('scrub').oninput=()=>reset(Number($('scrub').value));
$('ruler').onclick=e=>{const r=e.currentTarget.getBoundingClientRect();reset((e.clientX-r.left)/r.width*30);};
$('btn-policy').onclick=()=>{if(!policyReady||fallGuard.tripped)return;policyEnabled=!policyEnabled;if(policyEnabled)physics=true;reset();status(policyEnabled?'Policy enabled. Press Play to simulate.':'Policy control is OFF.');controls();};
$('btn-physics').onclick=()=>{physics=!physics;if(!physics)policyEnabled=false;reset();};
$('btn-follow').onclick=()=>{viewer.followCamera=!viewer.followCamera;$('btn-follow').textContent=viewer.followCamera?'ON':'OFF';};
$('ghost-offset').onchange=ghost;
for(const id of ['sensor-noise','sensor-delay',...noiseChannels.map(n=>'noise-'+n),...['mass','actuators','limits','friction','contacts'].map(n=>'dr-'+n)])$(id).onchange=applyTest;
$('dr-apply').onclick=applyTest;
$('dr-next').onclick=()=>{$('dr-seed').value=(Number($('dr-seed').value)+1)>>>0;applyTest();};
for(const preset of ['nominal','sensors','training'])$('dr-'+preset).onclick=()=>{
 $('sensor-noise').checked=preset!=='nominal';$('sensor-delay').value=0;
 for(const n of noiseChannels)$('noise-'+n).checked=true;
 for(const n of ['mass','actuators','limits','friction','contacts'])$('dr-'+n).checked=preset==='training';
 applyTest();
};
$('push-left').onclick=()=>startLoad('left pull');$('push-right').onclick=()=>startLoad('right pull');$('push-down').onclick=()=>startLoad('downward');
for(const [id,key] of [['stiffness','stiffness_command'],['rotational-stiffness','rotational_stiffness_command']])$(id).oninput=()=>{if(policy)policy.config[key]=Number($(id).value);$(id+'-value').textContent=$(id).value;};
window.addEventListener('keydown',e=>{if(e.target.closest('input,select,textarea'))return;if(e.code==='Space'){e.preventDefault();$('btn-play').click();}if(e.key.toLowerCase()==='r')reset(0,true,true);});
function stopPolicy(reason){
 fallGuard.trip(reason);scriptedLoad=null;$('load-status').textContent='Loads cleared — policy stopped';policyEnabled=false;revision++;viewer.clearDragForce();
 viewer.setPassiveActuation(policy?.config.fall_detection?.passive_damping??3);
 status(`${fallGuard.reason}. Policy OFF; damping only. Press Reset to restore the pose and enable policy control.`);controls();
}
function detectFall(){
 if(!fallGuard.check(viewer.data.qpos.slice(3,7)))return false;
 stopPolicy(fallGuard.reason);return true;
}
async function step(){
 const rev=revision;
 if(physics){
  if(policyEnabled&&policyReady){
   if(detectFall())return;
   viewer.setPolicyActuation(true);
   policy.pushHistory(viewer.data.qpos,viewer.data.qvel,motion,time,sensors.sample(viewer.data.qpos,viewer.data.qvel,sensorOptions()));
   const action=await policy.infer();if(rev!==revision||!running||!policyEnabled)return;
   const targets=policy.actionToTargets(action);
   for(let i=0;i<4;i++){
    if(detectFall())break;
    viewer.stepPhysicsOnce(policy.config.actuation==='implicit_pd'?targets:policy.targetsToCtrl(targets,viewer.data.qpos,viewer.data.qvel));
    measure();if(detectFall())break;
   }
   viewer.finalizeStep();
  } else {
   if(fallGuard.tripped)viewer.setPassiveActuation(policy?.config.fall_detection?.passive_damping??3);
   else viewer.setPolicyActuation(false);
   viewer.stepPhysics(new Float32Array(23));
  }
 }
 if(rev!==revision)return;time=Math.min(duration(),time+.02);if(time>=duration())running=false;ghost();controls();
}
function tick(ts){requestAnimationFrame(tick);if(viewer.ready&&running&&!busy&&ts-clock>=20/Number($('speed').value)){clock=ts;busy=true;const rev=revision;step().catch(e=>{if(rev!==revision)return;running=false;stopPolicy(`Stopped: ${e.message}`);}).finally(()=>{busy=false;});}if(viewer.renderer)viewer.render();}
(async()=>{try{
 viewer.initThree();await viewer.loadScene(status);viewer.initDragForce();const contract=await fetch('assets/robot/contract.json').then(r=>r.json());names=contract.joint_names;
 if(typeof viewer.mujoco.mj_setConst!=='function')throw Error('MuJoCo build lacks mj_setConst; physical randomization unavailable.');
 randomizer=new DomainRandomizer(viewer.model,()=>viewer.mujoco.mj_setConst(viewer.model,viewer.data));
 const sites=objectNames(viewer.model,'site');for(const n of ['left_palm','right_palm']){siteIds[n]=sites.indexOf(n);if(siteIds[n]<0)throw Error(`Missing site ${n}`);}
 viewer.externalForceCallback=loadForces;
 const contacts=contract.contacts;preparedContacts=contacts;if(contacts)$('contact-help').textContent=contacts.profile==='training'?'Training contacts: feet frictional, body frictionless.':`Ground friction: feet ${contacts.foot_sliding_friction}, body ${contacts.body_sliding_friction} (assumed).`;
 const response=await fetch('assets/motions/stand.json');if(!response.ok)throw Error('Prepare the viewer with --reference YOUR_MOTION.csv first.');const m=await response.json();installMotion(m,m.name||'Reference');
 const config=await fetch('assets/policy/config.json');if(config.ok){policy=new SoftMimicPolicy();await policy.load(await config.json(),'assets/policy/policy.onnx');fallGuard=new FallGuard(policy.config.fall_detection?.limit_angle??1.25);$('fall-help').textContent=`Fall cutoff: ${(fallGuard.limitAngle*180/Math.PI).toFixed(1)}° pelvis tilt. Policy latches OFF; damping only until Reset.`;checkPair();if(policy.config.checkpoint_iteration!==undefined)$('policy-provenance').textContent=policy.config.training_status==='completed training'?`${(policy.config.checkpoint_iteration+1).toLocaleString()} training iterations · completed run`:`Iteration ${policy.config.checkpoint_iteration} · intermediate training snapshot`;for(const [id,key] of [['stiffness','stiffness_command'],['rotational-stiffness','rotational_stiffness_command']]){$(id).value=policy.config[key];$(id+'-value').textContent=$(id).value;}}
 else status('Reference ready. Install a matched trained ONNX bundle to enable policy control.');
 $('loading-overlay').hidden=true;controls();requestAnimationFrame(tick);
}catch(e){status(e.message);$('loading-status').textContent=e.message;}})();
