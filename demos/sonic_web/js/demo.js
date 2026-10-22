/** Fixed-reference playback plus the matched SoftMimic ONNX controller. */
import {MuJoCoViewer} from './viewer.js';
import {SoftMimicPolicy} from './policy.js';
import {validateReference,parseReferenceCsv,referenceFrame} from './reference.js';
const $=id=>document.getElementById(id),viewer=new MuJoCoViewer($('canvas'));
let motion=null,policy=null,policyReady=false,policyEnabled=false,physics=false,running=false,time=0,revision=0,busy=false,clock=0,names=[];
const duration=()=>motion?(motion.frames-1)/motion.fps:0;
const status=message=>{$('status-text').textContent=message;};
function controls(){
 $('btn-play').textContent=running?'Ⅱ Pause':'▶ Play';$('btn-play').disabled=!motion;
 $('btn-policy').disabled=!policyReady;$('btn-policy').textContent=policyEnabled?'ON':policyReady?'OFF':'Awaiting matched policy';
 $('btn-physics').textContent=physics?'ON':'OFF';$('time').textContent=`${time.toFixed(2)} / ${duration().toFixed(2)} s`;$('scrub').max=duration();$('scrub').value=time;
 $('playhead').style.left=`${time/30*100}%`;
}
function ghost(){if(motion){const q=referenceFrame(motion,time);viewer.setGhost(q.pos,q.quat,q.joints,[0,$('ghost-offset').checked?1.25:0,0]);}}
function reset(at=0){
 revision++;running=false;time=Math.max(0,Math.min(duration(),at));viewer.clearDragForce();policy?.reset();
 if(motion){const q=referenceFrame(motion,time);viewer.resetToDefault(q.joints,q.pos[2],q.pos,q.quat);ghost();}controls();
}
function checkPair(){policyReady=false;if(!policy||!motion)return;try{validateReference(motion,policy.config);policyReady=true;status('Reference matches policy. Enable policy control, then press Play.');}catch(e){policyEnabled=false;status(e.message);}}
function installMotion(m,name){validateReference(m);motion=m;motion.name=name;policyEnabled=false;physics=false;checkPair();$('reference-name').textContent=name;$('reference-clip').textContent=name;$('reference-clip').style.width=`${duration()/30*100}%`;reset();}
$('reference-file').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;if(file.size>12000000)throw Error('CSV is too large');installMotion(parseReferenceCsv(await file.text(),names),file.name);}catch(e){status(e.message);}};
$('btn-play').onclick=()=>{if(!motion)return;if(time>=duration())reset();running=!running;controls();};
$('btn-restart').onclick=()=>reset();
$('scrub').oninput=()=>reset(Number($('scrub').value));
$('ruler').onclick=e=>{const r=e.currentTarget.getBoundingClientRect();reset((e.clientX-r.left)/r.width*30);};
$('btn-policy').onclick=()=>{if(!policyReady)return;policyEnabled=!policyEnabled;if(policyEnabled)physics=true;reset();status(policyEnabled?'Policy enabled. Press Play to simulate.':'Policy control is OFF.');controls();};
$('btn-physics').onclick=()=>{physics=!physics;if(!physics)policyEnabled=false;reset();};
$('btn-follow').onclick=()=>{viewer.followCamera=!viewer.followCamera;$('btn-follow').textContent=viewer.followCamera?'ON':'OFF';};
$('ghost-offset').onchange=ghost;
for(const [id,key] of [['stiffness','stiffness_command'],['rotational-stiffness','rotational_stiffness_command']])$(id).oninput=()=>{if(policy)policy.config[key]=Number($(id).value);$(id+'-value').textContent=$(id).value;};
window.addEventListener('keydown',e=>{if(e.target.closest('input,select,textarea'))return;if(e.code==='Space'){e.preventDefault();$('btn-play').click();}if(e.key.toLowerCase()==='r')reset();});
async function step(){
 const rev=revision;
 if(physics){
  if(policyEnabled&&policyReady){policy.pushHistory(viewer.data.qpos,viewer.data.qvel);const action=await policy.infer();if(rev!==revision||!running||!policyEnabled)return;const targets=policy.actionToTargets(action);for(let i=0;i<4;i++)viewer.stepPhysicsOnce(policy.targetsToCtrl(targets,viewer.data.qpos,viewer.data.qvel));viewer.finalizeStep();}
  else viewer.stepPhysics(new Float32Array(23));
 }
 if(rev!==revision)return;time=Math.min(duration(),time+.02);if(time>=duration())running=false;ghost();controls();
}
function tick(ts){requestAnimationFrame(tick);if(viewer.ready&&running&&!busy&&ts-clock>=20/Number($('speed').value)){clock=ts;busy=true;step().catch(e=>{running=false;policyEnabled=false;revision++;viewer.clearDragForce();status(`Stopped: ${e.message}`);controls();}).finally(()=>{busy=false;});}if(viewer.renderer)viewer.render();}
(async()=>{try{
 viewer.initThree();await viewer.loadScene(status);viewer.initDragForce();const contract=await fetch('assets/robot/contract.json').then(r=>r.json());names=contract.joint_names;
 const response=await fetch('assets/motions/stand.json');if(!response.ok)throw Error('Prepare the viewer with --reference YOUR_MOTION.csv first.');const m=await response.json();installMotion(m,m.name||'Reference');
 const config=await fetch('assets/policy/config.json');if(config.ok){policy=new SoftMimicPolicy();await policy.load(await config.json(),'assets/policy/policy.onnx');checkPair();if(policy.config.checkpoint_iteration!==undefined)$('policy-provenance').textContent=`Iteration ${policy.config.checkpoint_iteration} · intermediate training snapshot`;for(const [id,key] of [['stiffness','stiffness_command'],['rotational-stiffness','rotational_stiffness_command']]){$(id).value=policy.config[key];$(id+'-value').textContent=$(id).value;}}
 else status('Reference ready. Install a matched trained ONNX bundle to enable policy control.');
 $('loading-overlay').hidden=true;controls();requestAnimationFrame(tick);
}catch(e){status(e.message);$('loading-status').textContent=e.message;}})();
