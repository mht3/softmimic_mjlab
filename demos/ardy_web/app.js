import {setupConstraints} from './constraints-ui.js';
import {MuJoCoViewer} from './shared/js/viewer.js';
import {MotionPlayback,sampleMotion,exportTimelineCsv} from './timeline-model.mjs';
import {timeline,setupTimeline,addClip,draggable,updatePlayhead} from './sequence.js';
const $=id=>document.getElementById(id),viewer=new MuJoCoViewer($('canvas'));
let constraintsUI=null,targetNames=null,preview=null,mode='timeline',playing=false,t=0,last=performance.now(),activeJob=sessionStorage.getItem('ardyActiveJob'),knownClips='',previewRevision=0,streamWatching=false,terminalNotice='',latestPreview=null;
const playback=new MotionPlayback(),cache=new Map();
async function api(url,options){const r=await fetch(url,options),v=await r.json();if(!r.ok)throw Error(v.error||r.statusText);return v;}
function text(tag,value,cls){const e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e;}
function showError(e){$('status').textContent=e.message;$('status').className='warning';}
function validateMotion(c){
 if(!c||c.quaternion_order!=='wxyz'||JSON.stringify(c.joint_names)!==JSON.stringify(targetNames)||!Number.isFinite(c.fps)||c.fps<1||c.fps>120||!Array.isArray(c.qpos)||c.qpos.length<2||c.qpos.length>3601||(c.qpos.length-1)/c.fps>30.001||c.qpos.some(q=>!Array.isArray(q)||q.length!==30||q.some(x=>!Number.isFinite(x))||Math.abs(Math.hypot(...q.slice(3,7))-1)>.001))throw Error('Invalid 23-DOF motion file (maximum 30 s)');
}
async function fetchMotion(base){
 if(!cache.has(base))cache.set(base,api(`${base}/motion.json`).then(c=>{validateMotion(c);return c;}).catch(e=>{cache.delete(base);throw e;}));
 return cache.get(base);
}
function endTime(){return mode==='timeline'?timeline.duration:preview?(preview.motion.qpos.length-1)/preview.motion.fps:0;}
function controls(){
 const name=mode==='timeline'?'timeline':'clip';$('play').textContent=`${playing?'Ⅱ Pause':'▶ Play'} ${name}`;$('play').disabled=endTime()<=0;
 $('timeline-mode').classList.toggle('active',mode==='timeline');$('preview-mode').classList.toggle('active',mode==='preview');$('preview-mode').disabled=!preview;
 $('scrub').max=endTime();$('scrub').value=t;$('time').textContent=`${t.toFixed(2)} / ${endTime().toFixed(2)} s`;
}
function frame(){
 if(!viewer.ready)return;
 const q=mode==='timeline'?playback.sample(timeline.entries,t):preview?sampleMotion(preview.motion,t):null;
 if(q)viewer.setFrame(q.slice(0,3),q.slice(3,7),q.slice(7));
 updatePlayhead(t,mode==='timeline');$('scrub').value=t;$('time').textContent=`${t.toFixed(2)} / ${endTime().toFixed(2)} s`;
}
function activateTimeline(seek=null){mode='timeline';streamWatching=false;playing=false;t=seek===null?Math.min(t,timeline.duration):Math.max(0,Math.min(timeline.duration,seek));controls();frame();}
async function loadClip(clip,base=`/clips/${clip.id}`,preserve=false){
 const motion=await fetchMotion(base);
 // Generation only enters preview mode after an explicit Preview action.
 if(preserve&&(!streamWatching||mode!=='preview'))return;
 preview={clip,base,motion};mode='preview';if(!preserve){t=0;playing=false;streamWatching=false;}
 t=Math.min(t,endTime());controls();frame();
}
async function importFile(file,index){
 if(file.size>12*1024*1024||!file.name.endsWith('.json'))throw Error('Drop an exported motion.json (up to 12 MB).');
 const motion=JSON.parse(await file.text());validateMotion(motion);const clip={id:crypto.randomUUID(),prompt:file.name,frames:motion.qpos.length,fps:motion.fps};
 await addClip(clip,`local:${clip.id}`,index,motion);$('edit-status').textContent='Local file added. Re-import local files after a page reload.';
}
async function library(){
 const clips=await api('/api/clips'),signature=clips.map(c=>c.id).join();if(signature===knownClips)return;knownClips=signature;$('clips').replaceChildren();
 for(const clip of clips){const row=text('div','','library-item'),b=text('div',clip.prompt,'clip-preview');b.tabIndex=0;b.setAttribute('role','button');b.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();loadClip(clip).catch(showError);}};b.append(text('small',`${clip.seconds}s · seed ${clip.seed}${(clip.audit.fidelity_review_required||clip.audit.constraint_review_required)?' · review fit':''}`));draggable(b,clip);b.onclick=()=>loadClip(clip).catch(showError);
 const add=text('button','+','clip-add');add.title='Add saved clip to timeline';add.setAttribute('aria-label',`Add ${clip.prompt} to timeline`);add.onclick=async()=>{add.disabled=true;try{await addClip(clip);}catch(e){$('edit-status').textContent=e.message;}finally{add.disabled=false;}};row.append(b,add);$('clips').append(row);}
}
$('save-csv').onclick=()=>{try{
 const csv=exportTimelineCsv(timeline.entries,targetNames),name=$('save-name').value.trim();
 if(!/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(name))throw Error('Use a motion name containing letters, numbers, underscores or hyphens.');
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv'})),a=document.createElement('a');a.href=url;a.download=`${name}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);
 $('edit-status').textContent=`Saved ${name}.csv — complete timeline, 100 Hz, including trims and blends.`;
}catch(e){$('edit-status').textContent=e.message;}};
$('standing-reference').onclick=async()=>{try{await addClip({id:'standing-reference',prompt:'Standing still reference',audit:{fidelity_review_required:false}},'/builtin/stand');$('save-name').value='stand';}catch(e){showError(e);}};
$('generate').onclick=async()=>{try{
 let constraints=[];let segments=[{prompt:$('prompt').value,seconds:Number($('seconds').value)}];if($('finish-standing').checked)segments.push({prompt:'A person is standing still.',seconds:4});
 if(constraintsUI?.isActive())({segments,constraints}=constraintsUI.request());
 if(segments.reduce((n,s)=>n+s.seconds,0)>30)throw Error('Generation plus standing finish must fit within 30 seconds.');
 const job=await api('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({segments,constraints,seed:Number($('seed').value)})});
 terminalNotice='';activeJob=job.id;sessionStorage.setItem('ardyActiveJob',activeJob);previewRevision=0;streamWatching=false;latestPreview=null;
 $('generate').disabled=true;$('cancel').hidden=false;$('status').textContent='Queued on CPU. Your timeline remains playable.';$('status').className='';$('stream-progress').textContent='Waiting for the first audited preview…';
}catch(e){showError(e);}};
$('watch-stream').onclick=async()=>{if(latestPreview){try{await loadClip(latestPreview.clip,latestPreview.base_url);streamWatching=true;playing=true;controls();}catch(e){showError(e);}}};
$('cancel').onclick=async()=>{if(activeJob)try{await api(`/api/jobs/${activeJob}/cancel`,{method:'POST'});$('status').textContent='Stopping after the current step/audit…';}catch(e){showError(e);}};
$('play').onclick=()=>{if(endTime()>0){if(!playing&&t>=endTime()-1e-6)t=0;playing=!playing;controls();frame();}};
$('reset').onclick=()=>{t=0;playing=false;controls();frame();};
$('scrub').oninput=()=>{t=Number($('scrub').value);playing=false;controls();frame();};
$('timeline-mode').onclick=()=>activateTimeline(0);
$('preview-mode').onclick=()=>{if(preview){mode='preview';playing=false;t=0;controls();frame();}};
$('follow').onclick=()=>{viewer.followCamera=!viewer.followCamera;$('follow').textContent=`Follow ${viewer.followCamera?'ON':'OFF'}`;};
async function poll(){try{
 const state=await api('/api/status');let status=terminalNotice||state.phase;const running=Object.values(state.jobs).filter(j=>!['complete','failed','cancelled'].includes(j.status));
 $('active-streams-section').hidden=!running.length||!!activeJob;
 if(!activeJob){$('active-streams').replaceChildren();for(const j of running){const b=text('button',`Follow generation · ${j.seconds} s`);b.onclick=()=>{activeJob=j.id;sessionStorage.setItem('ardyActiveJob',activeJob);previewRevision=0;};$('active-streams').append(b);}}
 const job=state.jobs[activeJob];if(activeJob&&!job){activeJob=null;sessionStorage.removeItem('ardyActiveJob');status='CPU worker restarted. Your timeline is unchanged.';}
 if(job){status=job.status;if(job.preview&&!job.cancel_requested&&job.preview.revision!==previewRevision){latestPreview=job.preview;if(streamWatching&&mode==='preview')await loadClip(job.preview.clip,job.preview.base_url,true);previewRevision=job.preview.revision;$('stream-progress').textContent=`${job.preview.seconds.toFixed(2)} / ${job.seconds.toFixed(2)} s audited · ${job.preview.final?'Saved to library':'Preview available'}`;}
 if(['complete','failed','cancelled'].includes(job.status)){activeJob=null;sessionStorage.removeItem('ardyActiveJob');streamWatching=false;status=job.status==='complete'?'Saved to library. Drag the clip into your timeline.':job.error;terminalNotice=status;}}
 $('status').textContent=status;$('generate').disabled=!state.ready||!!activeJob;$('cancel').hidden=!activeJob;$('watch-stream').hidden=!latestPreview;
 $('plan-generate').disabled=$('generate').disabled;
 $('generate').textContent=activeJob?'Generating on CPU…':state.ready?(constraintsUI?.isActive()?'Generate plan':'Generate new motion'):'Loading models…';await library();
 }catch(e){showError(e);}setTimeout(poll,1500);
}
(async()=>{try{
 viewer.initThree();viewer.camera.position.set(2.3,-2.3,1.8);await viewer.loadScene(m=>$('loading').textContent=m);if(viewer.model.nq!==30)throw Error('Viewer embodiment is not 23 DOF');const stand=await api('/assets/robot/contract.json');targetNames=stand.joint_names;if(targetNames.length!==23)throw Error('Expected 23 DOF');
 const initial=(await fetchMotion('/builtin/stand')).qpos[0];viewer.setFrame(initial.slice(0,3),initial.slice(3,7),initial.slice(7));$('loading').hidden=true;
 await setupTimeline({fetchMotion,preview:loadClip,importFile,pause:()=>{playing=false;controls();},change:()=>{if(mode==='timeline'){playing=false;t=Math.min(t,timeline.duration);controls();frame();}},activate:()=>activateTimeline(),seek:time=>activateTimeline(time)});
 constraintsUI=setupConstraints({viewer,api,capture:()=>Array.from(viewer.data.qpos),seek:time=>{playing=false;t=Math.min(endTime(),time);controls();frame();},notice:message=>{$('edit-status').textContent=message;}});
 const prompts=await api('/api/presets');for(const p of prompts){const b=text('button',p);b.onclick=()=>{$('prompt').value=p;};$('presets').append(b);}controls();poll();
}catch(e){$('loading').textContent=e.message;showError(e);}})();
function animate(now){const dt=Math.min((now-last)/1000,.1);last=now;if(playing&&endTime()>0){t+=dt*Number($('speed').value);if(t>=endTime()){t=endTime();if(!(mode==='preview'&&activeJob&&streamWatching)){playing=false;controls();}}frame();}if(viewer.renderer)viewer.render();requestAnimationFrame(animate);}requestAnimationFrame(animate);
