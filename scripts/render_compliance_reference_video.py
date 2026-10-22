"""Render direct23DOF CSV poses and world-frame force arrows (CPU graphics).

Left is the unchanged nominal actor reference; right is the adapted IK target.
These are kinematic dataset replays, not dynamically simulated policy rollouts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

# Force Mesa software graphics so rendering does not contend with GPU training.
os.environ['MUJOCO_GL']='egl'
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['EGL_PLATFORM']='surfaceless'
os.environ['LIBGL_ALWAYS_SOFTWARE']='1'
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']='/usr/share/glvnd/egl_vendor.d/50_mesa.json'
import mujoco
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from OpenGL.GL import glGetString,GL_RENDERER

ROOT=Path(__file__).resolve().parents[1]


def pose(data,row,offset):
    data.qpos[:3]=row[offset:offset+3]
    data.qpos[3:7]=row[[offset+6,offset+3,offset+4,offset+5]]
    data.qpos[7:]=row[offset+7:offset+30]
    mujoco.mj_forward(data.model,data)


def add_force(scene,start,force,scale):
    if np.linalg.norm(force)<.1:return
    geom=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.95,.15,.10,1.]))
    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.018,start,start+force*scale)
    scene.ngeom+=1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=float,required=True);p.add_argument('--end',type=float,required=True)
    p.add_argument('--title',default='Standing compliance reference')
    p.add_argument('--azimuth',type=float,default=135.)
    p.add_argument('--fps',type=int,default=30);p.add_argument('--force-scale',type=float,default=.003)
    args=p.parse_args()
    arr=np.loadtxt(args.csv,delimiter=',',ndmin=2)
    if arr.shape[1]!=83 or not np.isfinite(arr).all():raise ValueError('Expected finite83-column direct23DOF CSV')
    xml=ROOT/'softmimic_deploy/src/assets/g1/g1_23dof.xml'
    tree=ET.parse(xml);top=tree.getroot();top.find('compiler').set('meshdir',str(xml.parent/'meshes'))
    world=top.find('worldbody')
    ET.SubElement(world,'geom',name='display_floor',type='plane',size='3 3 .1',rgba='.20 .24 .29 1',group='2')
    ET.SubElement(world,'light',pos='1 -2 4',dir='-1 2 -4',diffuse='.8 .8 .8',ambient='.4 .4 .4')
    visual=ET.SubElement(top,'visual');ET.SubElement(visual,'global',offwidth='480',offheight='512')
    model=mujoco.MjModel.from_xml_string(ET.tostring(top,encoding='unicode'))
    nominal=mujoco.MjData(model);adapted=mujoco.MjData(model)
    feet=[model.body(side+'_ankle_roll_link').id for side in ('left','right')]
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=[.0,0.,.65];camera.distance=2.45;camera.azimuth=args.azimuth;camera.elevation=-12
    options=mujoco.MjvOption();options.geomgroup[:]=0;options.geomgroup[1:3]=1
    renderer=mujoco.Renderer(model,height=512,width=480)
    graphics=glGetString(GL_RENDERER).decode();print('Graphics:',graphics,flush=True)
    if 'llvmpipe' not in graphics.lower():raise RuntimeError('Expected CPU software graphics')
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,17);small=ImageFont.truetype(fontpath,14);title=ImageFont.truetype(fontpath,23)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    times=np.arange(args.start,min(args.end,(len(arr)-1)/30),1/args.fps)
    indices=np.minimum(np.rint(times*30).astype(int),len(arr)-1)
    peak=int(np.argmax(np.linalg.norm(arr[indices,63:66],axis=1)))
    cached_left=None
    static_nominal=np.allclose(arr[:, :30],arr[0,:30],atol=1e-7,rtol=0.)
    with imageio.get_writer(args.output,fps=args.fps,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-pix_fmt','yuv420p']) as writer:
        for frame,i in enumerate(indices):
            row=arr[i];pose(nominal,row,0);pose(adapted,row,32)
            if cached_left is None or not static_nominal:
                renderer.update_scene(nominal,camera,scene_option=options);cached_left=renderer.render().copy()
            left=cached_left
            renderer.update_scene(adapted,camera,scene_option=options)
            body=int(row[62]);force=row[63:66]
            if 0<body<model.nbody:add_force(renderer.scene,adapted.xpos[body].copy(),force,args.force_scale)
            right=renderer.render().copy()
            canvas=Image.new('RGB',(960,688),(18,24,33));canvas.paste(Image.fromarray(left),(0,80));canvas.paste(Image.fromarray(right),(480,80))
            draw=ImageDraw.Draw(canvas)
            draw.text((20,10),args.title,font=title,fill='#ffffff')
            draw.text((20,46),'NOMINAL INPUT | fixed standing reference',font=font,fill='#a9dbbf')
            draw.text((500,46),'IK TARGET | adapted training reference',font=font,fill='#f4c792')
            draw.line([(480,80),(480,592)],fill='#516071',width=2)
            draw.text((20,601),f'CSV time {i/30:.2f} s  |  Red arrow: applied-to-robot force direction',font=font,fill='#f2f2f2')
            draw.text((20,628),f'F = [{force[0]:+.1f}, {force[1]:+.1f}, {force[2]:+.1f}] N    |F| = {np.linalg.norm(force):.1f} N    K = {row[69]:.1f} N/m',font=font,fill='#ffb0a5')
            drop=100*(nominal.qpos[2]-adapted.qpos[2]);foot=100*np.linalg.norm(adapted.xpos[feet]-nominal.xpos[feet],axis=1).max()
            draw.text((20,656),f'Pelvis drop {drop:+.1f} cm   Foot residual {foot:.2f} cm   |Torque| {np.linalg.norm(row[66:69]):.2f} Nm   |   IK replay, not policy behavior',font=small,fill='#bdc9da')
            writer.append_data(np.asarray(canvas))
            if frame==peak:canvas.save(args.output.with_suffix('.png'))
    renderer.close()
    meta={'source_csv':str(args.csv.resolve()),'csv_sha256':hashlib.sha256(args.csv.read_bytes()).hexdigest(),
          'start_s':args.start,'end_s':args.end,'frames':len(indices),'fps':args.fps,
          'force_arrow_m_per_N':args.force_scale,'camera_azimuth_deg':args.azimuth,'renderer':graphics,'type':'kinematic reference replay; no policy or dynamics'}
    args.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Saved',args.output,flush=True)

if __name__=='__main__':main()
