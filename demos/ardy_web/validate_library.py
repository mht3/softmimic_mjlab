"""Independently re-audit published clips and cross-check JSON/CSV/NPZ contracts."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"softmimic_mjlab"))
from demos.ardy_web.retarget import Retargeter
r=Retargeter();reports=[]
for path in sorted((ROOT/'sonic23/ardy_motions').glob('*/metadata.json')):
    meta=json.loads(path.read_text());folder=path.parent
    z=np.load(folder/'motion_23dof.npz',allow_pickle=False)
    view=json.loads((folder/'motion.json').read_text());csv=np.loadtxt(folder/'motion_23dof.csv',delimiter=',')
    assert z['joint_names'].tolist()==view['joint_names']==r.names
    assert str(z['quaternion_order'])==view['quaternion_order']=='wxyz'
    assert float(z['fps'])==view['fps']==meta['fps']
    assert z['qpos'].shape==(meta['frames'],30)
    assert np.isfinite(z['qpos']).all()
    np.testing.assert_allclose(np.linalg.norm(z['qpos'][:,3:7],axis=1),1.,atol=1e-6)
    np.testing.assert_allclose(z['qpos'],view['qpos'],atol=1e-12)
    np.testing.assert_allclose(z['qpos'],csv,atol=1e-12)
    report=r.audit(z['qpos'],float(z['fps']))
    assert report['geometry_passed'],(str(folder),report)
    reports.append(dict(id=folder.name,prompt=meta['prompt'],audit=report,fidelity_review_required=meta['audit']['fidelity_review_required']))
result=dict(passed=True,clips=len(reports),total_frames=sum(x['audit']['frames'] for x in reports),reports=reports)
(ROOT/'sonic23/ardy_library_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
