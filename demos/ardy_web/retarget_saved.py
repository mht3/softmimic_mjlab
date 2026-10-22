"""Retry retargeting a saved real ARDY source into a new immutable clip directory."""
import argparse,json,shutil,sys,uuid
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"softmimic_mjlab"))
from demos.ardy_web.retarget import Retargeter
from demos.ardy_web.export import export_motion
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);args=p.parse_args()
    source=args.source.resolve()
    if (source/'failure.json').exists():job=json.loads((source/'failure.json').read_text())['job']
    else:job=json.loads((source/'metadata.json').read_text())
    z=np.load(source/'source_29dof.npz',allow_pickle=False)
    output=ROOT/'sonic23/ardy_motions'/uuid.uuid4().hex;output.mkdir()
    shutil.copy2(source/'source_29dof.npz',output/'source_29dof.npz')
    m=export_motion(output,z['qpos'],float(z['fps']),job['prompt'],job['seconds'],job['seed'],Retargeter(),source_id=source.name)
    print(json.dumps(dict(path=str(output),audit=m['audit']),indent=2))
