"""Publish a retargeted preview and its provenance; never a policy/training bundle."""
import json
import hashlib
from pathlib import Path
import time
import numpy as np

def export_motion(output,q29,fps,prompt,seconds,seed,retargeter,timings=None,device='cpu',source_id=None):
    output=Path(output)
    started=time.perf_counter()
    q23,audit=retargeter.run(q29,fps)
    fps=audit['output_fps']
    retarget_seconds=time.perf_counter()-started
    np.savez_compressed(output/'motion_23dof.npz',qpos=q23,fps=fps,joint_names=retargeter.names,quaternion_order='wxyz',up_axis='z')
    np.savetxt(output/'motion_23dof.csv',q23,delimiter=',',header=','.join(['root_x','root_y','root_z','qw','qx','qy','qz',*retargeter.names]))
    (output/'motion.json').write_text(json.dumps(dict(qpos=q23.tolist(),fps=fps,joint_names=retargeter.names,quaternion_order='wxyz')))
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    timings=timings or {}
    metadata=dict(id=output.name,prompt=prompt,seconds=seconds,seed=seed,frames=len(q23),fps=fps,
        device=device,text_device='cpu',retarget_seconds=retarget_seconds,
        total_seconds=retarget_seconds+sum(timings.values()),**timings,
        audit=audit,model='ARDY-G1-RP-25FPS-Horizon8',status='kinematic_preview_only',source_id=source_id,
        timing_scope='generation_and_retarget' if timings else 'retarget_only')
    manifest=Path(__file__).resolve().parents[3]/'sonic23/ardy_text_models.json'
    metadata['text_model_revisions']=json.loads(manifest.read_text())['models']
    metadata['implementation_sha256']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                       for name in ('engine.py','retarget.py','text_encoder.py','export.py','streaming.py','constraints.py')}
    temporary=output/'metadata.partial'
    temporary.write_text(json.dumps(metadata,indent=2)+'\n');temporary.replace(output/'metadata.json')
    return metadata
