"""Persistent local ARDY text-to-motion engine. CPU is the default."""
import ast
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
os.environ.setdefault('HF_HUB_CACHE',str(ROOT/'sonic23/checkpoints/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TEXT_ENCODER_DEVICE','cpu')


def presets():
    path=ROOT/'ardy/scripts/interactive_demo/common.py'
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PRESET_PROMPTS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Upstream preset prompts not found')


class Engine:
    def __init__(self,progress=lambda x:None,device='cpu'):
        import torch
        from ardy.model import load_model
        from ardy.exports.mujoco import MujocoQposConverter
        from demos.ardy_web.retarget import Retargeter
        torch.set_num_threads(2)
        manifest=json.loads((ROOT/'sonic23/ardy_text_models.json').read_text())
        models={x['repo']:x['path'] for x in manifest['models']}
        self.text_models=models
        self.encoder=None  # Load once, only when a prompt is not already cached.
        progress('Loading ARDY G1 motion model on '+device)
        self.model=load_model('g18',device=device,text_encoder=False,checkpoints_dir=str(ROOT/'sonic23/checkpoints'))
        self.converter=MujocoQposConverter(self.model.skeleton)
        self.retargeter=Retargeter()
        # Verify converter's actual XML joint order, rather than assuming the SONIC ordering.
        import xml.etree.ElementTree as ET
        names=[j.get('name') for j in ET.parse(self.converter.xml_path).find('worldbody').findall('.//joint')]
        if names!=self.retargeter.source_names:
            raise ValueError('ARDY converter and retarget source order differ')
        self.device=device
        self.embedding_cache=ROOT/'sonic23/ardy_embeddings'
        self.embedding_cache.mkdir(exist_ok=True)
        self.encoder_signature=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        progress('Ready — CPU text inference / '+device+' motion inference')

    def encode_prompt(self,prompt,progress=lambda x:None):
        import torch
        key=hashlib.sha256((self.encoder_signature+'\n'+prompt).encode()).hexdigest()
        cache=self.embedding_cache/(key+'.npy')
        if cache.exists():
            embedding=torch.from_numpy(np.load(cache,allow_pickle=False)).to(torch.bfloat16)
        else:
            if self.encoder is None:
                from demos.ardy_web.text_encoder import load_encoder
                progress('Loading Llama / LLM2Vec on CPU for this uncached prompt')
                self.encoder=load_encoder(self.text_models)
            embedding,lengths=self.encoder([prompt])
            if lengths!=[1] or embedding.shape!=(1,1,4096):
                raise ValueError('Unexpected LLM2Vec embedding contract')
            np.save(cache,embedding.float().cpu().numpy())
        if embedding.shape!=(1,1,4096) or not bool(torch.isfinite(embedding).all()):
            raise ValueError('Invalid cached text embedding')
        return embedding

    def generate(self,prompt,seconds,seed,output,progress=lambda x:None):
        import torch
        output=Path(output); output.mkdir(parents=True,exist_ok=False)
        begin=time.perf_counter()
        progress('Encoding prompt on CPU')
        embedding=self.encode_prompt(prompt,progress)
        encoding_s=time.perf_counter()-begin
        frames=int(round(seconds*float(self.model.motion_rep.fps)))
        windows=0
        def bar(items,**kwargs):
            nonlocal windows
            windows+=1
            progress(f'Generating motion: window {windows}/{int(np.ceil(frames/8))}')
            return items
        t=time.perf_counter()
        # Lazy encoder/adapter construction can consume RNG. Seed the motion
        # sampler afterwards so cache hits and first-use prompts reproduce.
        torch.manual_seed(seed)
        with torch.inference_mode():
            samples=self.model(texts=[prompt],num_frames=frames,
                num_denoising_steps=int(self.model.diffusion.num_base_steps),
                pad_mask=torch.ones((1,frames),dtype=torch.bool,device=self.device),
                first_heading_angle=torch.zeros(1,device=self.device),motion_mask=None,observed_motion=None,
                cfg_weight=(2.,2.),text_feat=embedding.to(self.device),
                text_pad_mask=torch.ones((1,1),dtype=torch.bool,device=self.device),progress_bar=bar)
            decoded=self.model.motion_rep.inverse(samples,is_normalized=True)
            q29=np.asarray(self.converter.dict_to_qpos(decoded,root_quat_w_first=True))[0]
        generation_s=time.perf_counter()-t
        if q29.shape!=(frames,36) or not np.isfinite(q29).all():
            raise ValueError('Invalid ARDY motion output')
        np.savez_compressed(output/'source_29dof.npz',qpos=q29,fps=float(self.model.motion_rep.fps),quaternion_order='wxyz',joint_names=self.retargeter.source_names)
        progress('Retargeting to 23 DOF and auditing collision paths')
        from demos.ardy_web.export import export_motion
        return export_motion(output,q29,float(self.model.motion_rep.fps),prompt,seconds,seed,self.retargeter,
                             timings=dict(encoder_seconds=encoding_s,generation_seconds=generation_s),device=self.device)
