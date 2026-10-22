import {Timeline,MAX_SECONDS,length,clipWarnings} from './timeline-model.mjs';
const $=id=>document.getElementById(id),TYPE='application/x-ardy-motion';
export const timeline=new Timeline();
let selected=null,callbacks={},history=[],restoring=false,trimGesture=false;
function message(value){$('edit-status').textContent=value;}
function remember(){history.push(timeline.entries.map(e=>({...e})));if(history.length>40)history.shift();}
function save(){if(restoring)return;try{localStorage.setItem('ardyClipTimelineV1',JSON.stringify(timeline.entries.map(({id,clip,base,inFrame,outFrame})=>({id,clip,base,inFrame,outFrame}))));}catch{}}
function selectedEntry(){return timeline.entries.find(e=>e.id===selected);}
function changed({seek=false,render=true}={}){save();if(render)draw();callbacks.change?.(seek);}
function select(id,seek=false){selected=id;draw();callbacks.select?.(selectedEntry());if(seek){let start=0;for(const e of timeline.entries){if(e.id===id)break;start+=length(e);}callbacks.seek?.(start);}}
function mutate(fn){const before=timeline.entries;try{fn();history.push(before.map(e=>({...e})));if(history.length>40)history.shift();changed();}catch(e){message(e.message);}}
function draw(){
 document.getElementById('clip-warning-tooltip')?.remove();
 $('track-clips').replaceChildren();let start=0;
 timeline.entries.forEach((entry,index)=>{
  const card=document.createElement('div');card.className='timeline-clip'+(selected===entry.id?' selected':'');card.draggable=true;card.dataset.id=entry.id;card.tabIndex=0;card.setAttribute('role','button');card.setAttribute('aria-label',`Timeline clip ${index+1}: ${entry.clip.prompt}`);card.title=entry.clip.prompt;
  card.style.left=`${start/MAX_SECONDS*100}%`;card.style.width=`${length(entry)/MAX_SECONDS*100}%`;
  const label=document.createElement('span');label.className='clip-title';const shortName=entry.clip.prompt.replace(/^a person (is )?/i,'').replace(/\.$/,'');label.textContent=shortName.charAt(0).toUpperCase()+shortName.slice(1);
  const range=document.createElement('span');range.className='clip-range';range.textContent=`${length(entry).toFixed(2)} s · ${entry.inFrame/entry.motion.fps}s in`;card.append(label,range);
  const warnings=clipWarnings(entry.clip);
  if(warnings.length){
   card.classList.add('has-warning');const warning=document.createElement('span');warning.className='clip-warning';warning.textContent='⚠';warning.tabIndex=0;warning.setAttribute('role','img');warning.setAttribute('aria-label',warnings.join(' '));
   const hide=()=>{document.getElementById('clip-warning-tooltip')?.remove();warning.removeAttribute('aria-describedby');};
   const show=()=>{hide();const tip=document.createElement('div');tip.id='clip-warning-tooltip';tip.className='clip-warning-tooltip';tip.setAttribute('role','tooltip');tip.textContent=warnings.join('\n');document.body.append(tip);warning.setAttribute('aria-describedby',tip.id);const r=warning.getBoundingClientRect();tip.style.left=`${Math.max(8,Math.min(innerWidth-tip.offsetWidth-8,r.left))}px`;tip.style.top=`${Math.max(8,r.top-tip.offsetHeight-10)}px`;};
   warning.onpointerenter=show;warning.onpointerleave=hide;warning.onfocus=show;warning.onblur=hide;warning.onclick=e=>{e.stopPropagation();warning.focus();};warning.onpointerdown=e=>e.stopPropagation();warning.onkeydown=e=>e.stopPropagation();card.append(warning);
  }
  card.onclick=e=>{e.stopPropagation();select(entry.id,true);};
  card.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();select(entry.id,true);}};
  card.ondragstart=e=>{if(trimGesture){e.preventDefault();return;}e.dataTransfer.setData(TYPE,JSON.stringify({entry:entry.id}));e.dataTransfer.effectAllowed='move';};
  card.ondragover=e=>{e.preventDefault();e.stopPropagation();card.classList.toggle('drop-before',e.clientX<card.getBoundingClientRect().left+card.clientWidth/2);card.classList.toggle('drop-after',!card.classList.contains('drop-before'));};
  card.ondragleave=()=>card.classList.remove('drop-before','drop-after');
  card.ondrop=e=>{const after=e.clientX>=card.getBoundingClientRect().left+card.clientWidth/2;drop(e,index+(after?1:0));};
  for(const edge of ['left','right']){const handle=document.createElement('div');handle.className=`trim-handle ${edge}`;handle.title=`Trim ${edge==='left'?'start':'end'}`;handle.draggable=false;handle.onpointerdown=e=>beginTrim(e,entry,edge);handle.onclick=e=>e.stopPropagation();card.append(handle);}
  $('track-clips').append(card);start+=length(entry);
 });
 $('track-empty').hidden=timeline.entries.length>0;$('sequence-time').textContent=`${timeline.duration.toFixed(2)} / 30 s`;$('undo').disabled=!history.length;
 const e=selectedEntry();$('selected-name').textContent=e?e.clip.prompt:'Select a clip to trim or move it.';$('selected-name').title=e?.clip.prompt||'';
 for(const id of ['trim-in','trim-out','apply-trim','move-left','move-right','delete-clip'])$(id).disabled=!e;
 $('clip-downloads').replaceChildren();
 if(e?.base?.startsWith('/clips/'))for(const [label,file] of [['NPZ','motion_23dof.npz'],['CSV','motion_23dof.csv'],['JSON','motion.json'],['Audit','audit.json']]){const link=document.createElement('a');link.textContent=label;link.href=`${e.base}/${file}`;link.download=file;$('clip-downloads').append(link);}
 if(e){$('trim-in').value=(e.inFrame/e.motion.fps).toFixed(2);$('trim-out').value=(e.outFrame/e.motion.fps).toFixed(2);$('trim-in').step=$('trim-out').step=1/e.motion.fps;$('trim-in').max=$('trim-out').max=(e.motion.qpos.length-1)/e.motion.fps;}
 else{$('trim-in').value=$('trim-out').value='';}
}
function beginTrim(event,entry,edge){
 event.preventDefault();event.stopPropagation();trimGesture=true;selected=entry.id;remember();callbacks.pause?.();
 const origin=event.clientX,scale=MAX_SECONDS/$('track').getBoundingClientRect().width,original={...entry};
 const move=e=>{const delta=Math.round((e.clientX-origin)*scale*entry.motion.fps);let a=original.inFrame,b=original.outFrame;
  if(edge==='left')a=Math.max(0,Math.min(b-1,a+delta));else b=Math.min(entry.motion.qpos.length-1,Math.max(a+1,b+delta));
  try{timeline.trim(entry.id,a,b);draw();callbacks.change?.(false);message(`Trim: ${(a/entry.motion.fps).toFixed(2)}–${(b/entry.motion.fps).toFixed(2)} s`);}catch(error){message(error.message);}
 };
 const finish=()=>{window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',finish);window.removeEventListener('pointercancel',finish);trimGesture=false;changed();};
 window.addEventListener('pointermove',move);window.addEventListener('pointerup',finish,{once:true});window.addEventListener('pointercancel',finish,{once:true});
}
export async function addClip(clip,base=`/clips/${clip.id}`,index=timeline.entries.length,provided=null){
 const motion=provided||await callbacks.fetchMotion(base);const entry={id:crypto.randomUUID(),clip,base,motion,inFrame:0,outFrame:motion.qpos.length-1};
 try{timeline.validate([...timeline.entries,entry]);}catch(e){message(e.message);throw e;}
 remember();timeline.add(entry,index);selected=entry.id;changed();callbacks.select?.(entry);callbacks.activate?.();message(`Added ${clip.prompt}. Press Play timeline.`);
}
async function drop(e,index=timeline.entries.length){
 e.preventDefault();e.stopPropagation();$('track').classList.remove('drag-over');
 try{
  if(e.dataTransfer.files.length){await callbacks.importFile(e.dataTransfer.files[0],index);return;}
  const data=JSON.parse(e.dataTransfer.getData(TYPE));
  if(data.entry){const old=timeline.entries.findIndex(x=>x.id===data.entry);if(old<0)return;mutate(()=>timeline.move(data.entry,index-(old<index?1:0)));select(data.entry,true);message('Clip order updated.');}
  else if(data.clip){if(!/^[a-f0-9]{32}$/.test(data.clip.id))throw Error('Invalid saved motion ID');await addClip(data.clip,undefined,index);}
 }catch(error){message(error.message);}finally{document.querySelectorAll('.drop-before,.drop-after').forEach(e=>e.classList.remove('drop-before','drop-after'));}
}
export function draggable(button,clip){
 // Pointer dragging works across the scrollable library and WebGL viewport.
 button.draggable=false;let suppressClick=false;
 button.addEventListener('click',e=>{if(suppressClick){e.preventDefault();e.stopImmediatePropagation();suppressClick=false;}},true);
 button.onpointerdown=event=>{
  if(event.button!==0)return;
  const x=event.clientX,y=event.clientY;let ghost=null;
  const move=e=>{
   if(!ghost&&Math.hypot(e.clientX-x,e.clientY-y)>6){ghost=document.createElement('div');ghost.className='dragged-library-clip';ghost.textContent=clip.prompt;document.body.append(ghost);suppressClick=true;$('track').classList.add('drag-over');}
   if(ghost){e.preventDefault();ghost.style.left=`${e.clientX+12}px`;ghost.style.top=`${e.clientY+12}px`;}
  };
  const finish=async e=>{
   window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',finish);window.removeEventListener('pointercancel',finish);
   if(!ghost)return;ghost.remove();$('track').classList.remove('drag-over');
   if(e.type==='pointercancel')return;
   const r=$('track').getBoundingClientRect();
   if(e.clientX>=r.left&&e.clientX<=r.right&&e.clientY>=r.top&&e.clientY<=r.bottom){
    let index=timeline.entries.length,start=0;const time=(e.clientX-r.left)/r.width*MAX_SECONDS;
    for(let i=0;i<timeline.entries.length;i++){if(time<start+length(timeline.entries[i])/2){index=i;break;}start+=length(timeline.entries[i]);}
    try{await addClip(clip,undefined,index);}catch(error){message(error.message);}
   }
   setTimeout(()=>{suppressClick=false;},0);
  };
  window.addEventListener('pointermove',move,{passive:false});window.addEventListener('pointerup',finish,{once:true});window.addEventListener('pointercancel',finish,{once:true});
 };
}
export function updatePlayhead(time,active){$('playhead').style.left=`${Math.max(0,Math.min(MAX_SECONDS,time))/MAX_SECONDS*100}%`;$('playhead').style.opacity=active?'1':'.25';const now=active?timeline.resolve(time)?.entry.id:null;for(const e of $('track-clips').children)e.classList.toggle('is-playing',e.dataset.id===now);}
export async function setupTimeline(options){
 callbacks=options;
 for(let n=0;n<=30;n+=5){const tick=document.createElement('span');tick.textContent=`${n}s`;tick.style.left=`${n/30*100}%`;$('ruler').append(tick);}
 const seek=e=>{const r=$('track-area').getBoundingClientRect();callbacks.seek(Math.max(0,Math.min(timeline.duration,(e.clientX-r.left)/r.width*30)));};
 $('ruler').onpointerdown=e=>{e.preventDefault();seek(e);const move=e=>seek(e),end=()=>{window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',end);};window.addEventListener('pointermove',move);window.addEventListener('pointerup',end,{once:true});};
 $('track').onclick=seek;
 $('track').ondragover=e=>{e.preventDefault();$('track').classList.add('drag-over');};$('track').ondragleave=()=>$('track').classList.remove('drag-over');$('track').ondrop=e=>drop(e);
 $('delete-clip').onclick=()=>{if(selected){mutate(()=>timeline.remove(selected));selected=null;draw();message('Clip removed. Undo restores it.');}};
 $('clear-sequence').onclick=()=>{if(timeline.entries.length){mutate(()=>{timeline.entries=[];});selected=null;draw();}};
 $('undo').onclick=()=>{if(history.length){timeline.entries=history.pop();selected=timeline.entries.at(-1)?.id||null;changed();message('Edit undone.');}};
 $('move-left').onclick=()=>{const i=timeline.entries.findIndex(e=>e.id===selected);if(i>0)mutate(()=>timeline.move(selected,i-1));};
 $('move-right').onclick=()=>{const i=timeline.entries.findIndex(e=>e.id===selected);if(i>=0&&i<timeline.entries.length-1)mutate(()=>timeline.move(selected,i+1));};
 const applyTrim=()=>{const e=selectedEntry();if(!e)return;const a=Math.round(Number($('trim-in').value)*e.motion.fps),b=Math.round(Number($('trim-out').value)*e.motion.fps);mutate(()=>timeline.trim(e.id,a,b));draw();};$('trim-in').onchange=$('trim-out').onchange=applyTrim;$('apply-trim').onclick=applyTrim;for(const id of ['trim-in','trim-out'])$(id).onkeydown=e=>{if(e.key==='Enter')applyTrim();};
 window.addEventListener('keydown',e=>{if(e.target.closest('input,textarea,select'))return;if(e.code==='Space'){e.preventDefault();$('play').click();}else if(e.key==='Delete'){$('delete-clip').click();}else if((e.ctrlKey||e.metaKey)&&e.key==='z'){e.preventDefault();$('undo').click();}});
 $('canvas').ondragover=e=>e.preventDefault();$('canvas').ondrop=async e=>{e.preventDefault();try{if(e.dataTransfer.files.length)await callbacks.importFile(e.dataTransfer.files[0]);else{const data=JSON.parse(e.dataTransfer.getData(TYPE));if(data.clip)await callbacks.preview(data.clip);}}catch(error){message(error.message);}};
 restoring=true;
 try{const saved=JSON.parse(localStorage.getItem('ardyClipTimelineV1')||'[]');if(!Array.isArray(saved)||saved.length>100)throw Error('Invalid saved timeline');
  for(const e of saved){if(e.base!=='/builtin/stand'&&!/^\/clips\/[a-f0-9]{32}(\/stream\/\d{4})?$/.test(e.base))continue;const motion=await callbacks.fetchMotion(e.base);timeline.add({...e,motion});}
 }catch(e){message(`Some saved clips could not be restored: ${e.message}`);}finally{restoring=false;draw();}
 callbacks.change?.(false);
}
