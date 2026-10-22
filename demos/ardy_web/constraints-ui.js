import * as THREE from 'three';
import {TRACKS,snap,examplePlan,checkPlan} from './constraints-model.mjs';
const $=id=>document.getElementById(id);
const element=(tag,text='',cls='')=>{const e=document.createElement(tag);e.textContent=text;e.className=cls;return e;};
export function setupConstraints({viewer,api,capture,seek,notice}){
 let plan={segments:[{prompt:'A person is standing.',seconds:4}],constraints:[]},selected=null,selectedPrompt=null,planning=false,drag=null,down=null;
 try{const saved=JSON.parse(localStorage.getItem('ardyConstraintPlan'));if(saved){checkPlan(saved);plan=saved;}}catch{}
 const group=new THREE.Group();viewer.scene.add(group);group.visible=false;
 const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();
 const duration=()=>plan.segments.reduce((n,s)=>n+s.seconds,0);
 function save(){localStorage.setItem('ardyConstraintPlan',JSON.stringify(plan));}
 function setMode(value){planning=value;document.body.classList.toggle('planning',value);$('generation-editor').hidden=!value;group.visible=value;$('constraints-toggle').classList.toggle('active',value);for(const id of ['save-csv','save-name','undo','clear-sequence','sequence-time'])$(id).hidden=value;document.querySelector('.editor-heading strong').textContent=value?'Generation plan':'Motion timeline';window.dispatchEvent(new Event('resize'));render();}
 function clearMarkers(){while(group.children.length){const o=group.children[0];group.remove(o);o.geometry?.dispose();o.material?.dispose();}}
 function world(c){return c.type==='root2d'?[...c.position,.025]:c.type==='fullbody'?c.pose.slice(0,3):(c.anchor||c.pose.slice(0,3)).map((x,i)=>x+c.offset[i]);}
 function markers(){clearMarkers();const roots=[];for(const c of plan.constraints){const pos=world(c),m=new THREE.Mesh(new THREE.SphereGeometry(c.type==='root2d'?.055:.035,14,10),new THREE.MeshBasicMaterial({color:c.id===selected?0xffd086:c.type==='root2d'?0x57d9ff:0xb798ff,depthTest:false}));m.position.set(...pos);m.userData.constraintId=c.id;m.renderOrder=50;group.add(m);if(c.type==='root2d')roots.push(c);}
  roots.sort((a,b)=>a.time-b.time);if(roots.length>1){const geometry=new THREE.BufferGeometry().setFromPoints(roots.map(c=>new THREE.Vector3(...world(c))));group.add(new THREE.Line(geometry,new THREE.LineBasicMaterial({color:0x57d9ff,transparent:true,opacity:.6})));}}
 function render(){
  const rows=$('constraint-tracks');rows.replaceChildren();let start=0;
  const pr=element('div','','constraint-row');pr.append(element('span','Prompts','track-label'));const lane=element('div','','constraint-lane');
  plan.segments.forEach((s,i)=>{const b=element('button',s.prompt,'prompt-goal');b.style.left=`${start/30*100}%`;b.style.width=`${s.seconds/30*100}%`;b.title=`${s.prompt} · ${s.seconds} s`;b.onclick=()=>{selectedPrompt=i;selected=null;render();};lane.append(b);start+=s.seconds;});pr.append(lane);rows.append(pr);
  for(const [kind,label] of TRACKS){const row=element('div','','constraint-row');row.append(element('span',label,'track-label'));const lane=element('div','','constraint-lane');lane.setAttribute('aria-label',`${label} keyframe track`);lane.onclick=e=>{if(e.target!==lane)return;const rect=lane.getBoundingClientRect(),time=snap((e.clientX-rect.left)/rect.width*30);$('goal-time').value=Math.min(time,Math.max(0,duration()-.04));$('goal-type').value=kind;seek(Number($('goal-time').value));notice('Choose Add keyframe to capture this pose or place a target.');};
   for(const c of plan.constraints.filter(c=>c.type===kind)){const b=element('button','◆','goal-key'+(selected===c.id?' selected':''));b.style.left=`${c.time/30*100}%`;b.title=`${label} at ${c.time.toFixed(2)} s`;b.setAttribute('aria-label',b.title);b.onclick=e=>{e.stopPropagation();selected=c.id;selectedPrompt=null;render();};b.onpointerdown=e=>{if(e.button!==0)return;e.stopPropagation();b.setPointerCapture(e.pointerId);const rect=lane.getBoundingClientRect();let moved=false;const move=ev=>{moved=true;c.time=snap(Math.max(0,Math.min(duration()-.04,(ev.clientX-rect.left)/rect.width*30)));b.style.left=`${c.time/30*100}%`;$('goal-time').value=c.time;};const up=()=>{b.removeEventListener('pointermove',move);b.removeEventListener('pointerup',up);if(moved){selected=c.id;save();render();}};b.addEventListener('pointermove',move);b.addEventListener('pointerup',up);};lane.append(b);}row.append(lane);rows.append(row);}
  const c=plan.constraints.find(c=>c.id===selected);$('goal-delete').disabled=!c;$('goal-apply').disabled=!c;
  $('goal-form').hidden=selectedPrompt!==null;$('prompt-form').hidden=selectedPrompt===null;
  if(selectedPrompt!==null){const p=plan.segments[selectedPrompt];$('plan-prompt').value=p.prompt;$('plan-seconds').value=p.seconds;}
  if(c){$('goal-type').value=c.type;$('goal-time').value=c.time;const p=c.type==='root2d'?c.position:c.offset||[0,0,0];['x','y','z'].forEach((axis,i)=>$('goal-'+axis).value=p[i]||0);}
  const kind=$('goal-type').value;for(const axis of ['x','y','z'])$('goal-'+axis).disabled=kind==='fullbody'||(kind==='root2d'&&axis==='z');
  $('goal-coordinate-label').textContent=kind==='root2d'?'World XY (m)':kind==='fullbody'?'Capture displayed 23DOF pose':'XYZ offset from captured hand/foot (m)';
  $('constraint-summary').textContent=`${duration().toFixed(2)} / 30 s · ${plan.constraints.length} goals · next generation only`;
  markers();
 }
 async function add(position=null){try{
  const kind=position?'root2d':$('goal-type').value,time=snap(Number($('goal-time').value));if(time<0||time>=duration())throw Error('Choose a time within the generation duration.');
  if(plan.constraints.filter(c=>c.type===kind).length>=20)throw Error('At most 20 goals on each track.');
  const c={id:crypto.randomUUID(),type:kind,time};if(kind==='root2d')c.position=position||[Number($('goal-x').value),Number($('goal-y').value)];else{c.pose=capture();if(kind!=='fullbody'){c.offset=['x','y','z'].map(x=>Number($('goal-'+x).value));const anchors=await api('/api/constraint-pose',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pose:c.pose})});c.anchor=anchors[kind];}}
  if(plan.constraints.some(x=>x.type===kind&&Math.abs(x.time-time)<.02))throw Error('A keyframe already exists on this track at this time.');
  plan.constraints.push(c);selected=c.id;selectedPrompt=null;save();render();
 }catch(e){notice(e.message);}}
 $('plan-generate').onclick=()=>{setMode(true);$('generate').click();};
 $('constraints-toggle').onclick=()=>setMode(!planning);$('open-constraints').onclick=()=>setMode(true);
 $('goal-type').onchange=()=>{selected=null;selectedPrompt=null;['x','y','z'].forEach(a=>$('goal-'+a).value=0);render();};
 $('goal-add').onclick=()=>add();$('goal-delete').onclick=()=>{plan.constraints=plan.constraints.filter(c=>c.id!==selected);selected=null;save();render();};
 $('goal-apply').onclick=()=>{const c=plan.constraints.find(c=>c.id===selected);if(!c)return;const time=snap(Number($('goal-time').value)),vals=['x','y','z'].map(a=>Number($('goal-'+a).value));if(!Number.isFinite(time)||time<0||time>=duration()||!vals.every(Number.isFinite)){notice('Invalid goal time or coordinates.');return;}c.time=time;if(c.type==='root2d')c.position=vals.slice(0,2);else if(c.type!=='fullbody')c.offset=vals;save();render();};
 $('goal-clear').onclick=()=>{plan.constraints=[];selected=null;save();render();};
 $('prompt-add').onclick=()=>{const s={prompt:$('prompt').value,seconds:Number($('seconds').value)};try{checkPlan({...plan,segments:[...plan.segments,s]});plan.segments.push(s);selectedPrompt=plan.segments.length-1;save();render();}catch(e){notice(e.message);}};
 $('prompt-apply').onclick=()=>{const s={prompt:$('plan-prompt').value,seconds:Number($('plan-seconds').value)};try{const next=plan.segments.map((p,i)=>i===selectedPrompt?s:p);checkPlan({...plan,segments:next});plan.segments=next;save();render();}catch(e){notice(e.message);}};
 $('prompt-delete').onclick=()=>{if(plan.segments.length===1){notice('Keep at least one prompt block.');return;}const next=plan.segments.filter((p,i)=>i!==selectedPrompt);try{checkPlan({...plan,segments:next});plan.segments=next;selectedPrompt=null;save();render();}catch(e){notice(e.message);}};
 $('plan-example').onchange=()=>{const kind=$('plan-example').value;if(!kind)return;plan=examplePlan(kind);selected=null;selectedPrompt=null;$('waypoint-mode').checked=kind==='waypoints';$('goal-time').value=2;save();render();notice('Example plan loaded. Generate to synthesize and audit a new 23DOF clip.');};
 function rayAt(e){const r=viewer.canvas.getBoundingClientRect();mouse.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(mouse,viewer.camera);}
 const canvas=viewer.canvas;
 canvas.addEventListener('pointerdown',e=>{if(!planning||e.button!==0)return;down=[e.clientX,e.clientY];rayAt(e);const hit=ray.intersectObjects(group.children).find(x=>x.object.userData.constraintId);if(hit){selected=hit.object.userData.constraintId;selectedPrompt=null;const c=plan.constraints.find(c=>c.id===selected);const normal=c.type==='root2d'?new THREE.Vector3(0,0,1):viewer.camera.getWorldDirection(new THREE.Vector3());drag={c,plane:new THREE.Plane().setFromNormalAndCoplanarPoint(normal,new THREE.Vector3(...world(c)))};viewer.controls.enabled=false;canvas.setPointerCapture(e.pointerId);e.stopImmediatePropagation();render();}},true);
 canvas.addEventListener('pointermove',e=>{if(!drag)return;rayAt(e);const hit=ray.ray.intersectPlane(drag.plane,new THREE.Vector3());if(!hit)return;const c=drag.c;if(c.type==='root2d')c.position=[hit.x,hit.y];else if(c.type!=='fullbody')c.offset=[hit.x,hit.y,hit.z].map((x,i)=>Math.max(-2,Math.min(2,x-c.anchor[i])));render();},true);
 canvas.addEventListener('pointerup',e=>{if(drag){drag=null;viewer.controls.enabled=true;save();e.stopImmediatePropagation();down=null;return;}if(!planning||!$('waypoint-mode').checked||!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>5)return;down=null;rayAt(e);const hit=ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0,0,1),0),new THREE.Vector3());if(hit)add([hit.x,hit.y]).then(()=>{$('goal-time').value=Math.min(duration()-.04,Number($('goal-time').value)+Number($('waypoint-interval').value));});},true);
 canvas.addEventListener('pointercancel',()=>{drag=null;down=null;viewer.controls.enabled=true;});
 render();return {isActive:()=>planning,request:()=>{checkPlan(plan);return {segments:plan.segments,constraints:plan.constraints};}};
}
