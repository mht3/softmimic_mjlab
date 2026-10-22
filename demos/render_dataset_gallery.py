"""Render audited standing references on Mesa CPU; never launch a policy or use CUDA."""
import os
for k,v in {'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','LP_NUM_THREADS':'2'}.items():os.environ[k]=v
from pathlib import Path
import sys,json,hashlib,subprocess,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]/'softmimic_mjlab'
sys.path.insert(0,str(ROOT/'scripts'))
import render_compliance_reference_video as render
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
from OpenGL.GL import glGetString,GL_RENDERER
DATA=ROOT/'src/assets/compliant_motions/g1_23dof/stand'
OUT=ROOT/'logs/dataset_gallery_20261021';OUT.mkdir(exist_ok=True)
audit=json.loads((DATA/'audit.json').read_text())
assert len(audit)==85
for row in audit:
 assert row['self_collision_free'] and row['npz_structural_valid'] and row['audit_stride']==1
 for ext,key in [('.csv','csv_sha256'),('.npz','npz_sha256')]:
  path=(DATA/row['file']).with_suffix(ext)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==row[key],str(path)
print('Verified all 85 CSV/NPZ source hashes against the completed audit.',flush=True)
files=sorted((DATA/'forcefield').glob('*.csv')); arrays={p:np.loadtxt(p,delimiter=',') for p in files}
def window(a,i):return float(np.clip(i/30-8,0,(len(a)-1)/30-18))
p,a=max(arrays.items(),key=lambda x:np.max(x[1][:,2]-x[1][:,34]));i=int(np.argmax(a[:,2]-a[:,34]))
selected=[dict(path=p,peak=i,title='Deep downward load and recovery',name='01_downward_load')]
for body,side in [(19,'left'),(24,'right')]:
 candidates=[]
 for p,a in arrays.items():
  if p in [x['path'] for x in selected]:continue
  score=np.where((a[:,62]==body)&(np.abs(a[:,65])<np.linalg.norm(a[:,63:65],axis=1)),np.linalg.norm(a[:,63:65],axis=1),0)
  i=int(np.argmax(score));candidates.append((score[i],p,i))
 _,p,i=max(candidates)
 selected.append(dict(path=p,peak=i,title=f'{side.title()} hand lateral load and recovery',name=f'0{len(selected)+1}_{side}_hand_load'))
manifest={'dataset':str(DATA),'audit_sha256':hashlib.sha256((DATA/'audit.json').read_bytes()).hexdigest(),'type':'Kinematic CSV reference replay, not policy rollout','individual':[],'grid':[]}
for item in selected:
 a=arrays[item['path']];start=window(a,item['peak']);out=OUT/(item['name']+'.mp4')
 info=dict(file=str(item['path'].relative_to(DATA)),start_s=start,end_s=start+18,peak_s=item['peak']/30,output=out.name)
 manifest['individual'].append(info)
 print('Rendering',info,flush=True)
 if os.environ.get('GALLERY_GRID_ONLY')=='1' and out.exists():continue
 subprocess.run([sys.executable,str(ROOT/'scripts/render_compliance_reference_video.py'),str(item['path']),'--output',str(out),'--start',str(start),'--end',str(start+18),'--fps','15','--title',item['title']],check=True)
# 28 evenly spaced forcefield seeds, plus two labeled static-mode examples.
grid_files=[files[i] for i in np.linspace(0,39,28).round().astype(int)]
grid_files += [sorted((DATA/mode).glob('*.csv'))[0] for mode in ['collision-emulator','zero-wrench']]
rows=[]
for p in grid_files:
 a=arrays.get(p)
 if a is None:a=np.loadtxt(p,delimiter=',')
 score=np.linalg.norm(a[:,39:62]-a[:,7:30],axis=1)+5*np.maximum(a[:,2]-a[:,34],0)
 i=int(np.argmax(score));start=window(a,i) if np.max(score)>.01 else 0
 info=dict(file=str(p.relative_to(DATA)),start_s=start,end_s=start+18,peak_s=i/30)
 manifest['grid'].append(info); rows.append((p,a,start))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
xml=ROOT/'softmimic_deploy/src/assets/g1/g1_23dof.xml';tree=ET.parse(xml);top=tree.getroot();top.find('compiler').set('meshdir',str(xml.parent/'meshes'))
world=top.find('worldbody');ET.SubElement(world,'geom',name='floor',type='plane',size='3 3 .1',rgba='.20 .24 .29 1',group='2')
ET.SubElement(world,'light',pos='1 -2 4',dir='-1 2 -4',diffuse='.8 .8 .8',ambient='.4 .4 .4')
visual=ET.SubElement(top,'visual');ET.SubElement(visual,'global',offwidth='240',offheight='176')
model=mujoco.MjModel.from_xml_string(ET.tostring(top,encoding='unicode'));data=mujoco.MjData(model)
model.vis.quality.shadowsize=256
model.vis.quality.offsamples=0
renderer=mujoco.Renderer(model,height=176,width=240)
assert 'llvmpipe' in glGetString(GL_RENDERER).decode().lower()
cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[0,0,.54];cam.distance=2.1;cam.azimuth=135;cam.elevation=-12
opt=mujoco.MjvOption();opt.geomgroup[:]=0;opt.geomgroup[1:3]=1
fontfile='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(fontfile,12);title=ImageFont.truetype(fontfile,23)
cache={};output=OUT/'04_grid_30_motions.mp4'
with imageio.get_writer(output,fps=15,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-pix_fmt','yuv420p','-threads','2','-movflags','+faststart']) as writer:
 for n in range(270):
  canvas=Image.new('RGB',(1440,1136),(18,24,33));draw=ImageDraw.Draw(canvas)
  draw.text((18,8),'AUDITED STANDING DATASET | 30 simultaneous reference clips',font=title,fill='white')
  draw.text((18,39),f'28 forcefield clips + 2 static baselines | event-centered windows, independent CSV times | 1x speed | elapsed {n/15:.1f}s',font=font,fill='#b9cad9')
  for idx,(p,a,start) in enumerate(rows):
   i=min(round((start+n/15)*30),len(a)-1);r=a[i]
   key=(idx,i) if idx<28 else (idx,0)
   if key not in cache:
    render.pose(data,r,32);renderer.update_scene(data,cam,scene_option=opt)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
    body=int(r[62]);force=r[63:66]
    if 0<body<model.nbody:render.add_force(renderer.scene,data.xpos[body].copy(),force,.003)
    img=Image.fromarray(renderer.render().copy())
    if idx>=28:cache[key]=img
   else:img=cache[key]
   x=(idx%6)*240;y=64+(idx//6)*208
   canvas.paste(img,(x,y+18));draw=ImageDraw.Draw(canvas)
   short=('FF' if idx<28 else 'COLLISION: static' if idx==28 else 'ZERO: static')+' '+p.stem[-3:]
   draw.text((x+5,y+2),short,font=font,fill='#f4c792')
   force=np.linalg.norm(r[63:66]);drop=100*(r[2]-r[34])
   draw.text((x+5,y+193),f't {i/30:4.1f}s | F {force:5.1f}N | drop {drop:+4.1f}cm',font=font,fill='#dce5ee')
  draw.text((18,1112),'RED = applied force | Kinematic IK replay, not learned policy behavior | all source hashes match the 85-pair audit',font=font,fill='#ffb0a5')
  writer.append_data(np.asarray(canvas))
  if n==120:canvas.save(output.with_suffix('.png'))
  if n%30==0:print(f'Grid: {n}/270 frames',flush=True)
renderer.close();print('SAVED ALL FOUR VIDEOS:',OUT,flush=True)
