"""Read-only full export and streamed-prefix consistency check."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"softmimic_mjlab"))
from demos.ardy_web.retarget import Retargeter


def validate(folder):
    folder=Path(folder)
    meta=json.loads((folder/'metadata.json').read_text())
    schedule=json.loads((folder/'schedule.json').read_text())
    assert schedule['final'], 'Only completed streams are final exports'
    source=np.load(folder/'source_29dof.npz',allow_pickle=False)['qpos']
    target=np.load(folder/'motion_23dof.npz',allow_pickle=False)
    q=target['qpos']
    view=json.loads((folder/'motion.json').read_text())
    csv=np.loadtxt(folder/'motion_23dof.csv',delimiter=',')
    r=Retargeter()
    assert source.shape==(round(meta['seconds']*25),36)
    assert q.shape==(4*(len(source)-1)+1,30)
    assert target['joint_names'].tolist()==view['joint_names']==r.names
    assert str(target['quaternion_order'])==view['quaternion_order']=='wxyz'
    assert float(target['fps'])==view['fps']==100
    np.testing.assert_array_equal(q,np.array(view['qpos']))
    np.testing.assert_array_equal(q,csv)
    np.testing.assert_allclose(np.linalg.norm(q[:,3:7],axis=1),1,atol=1e-6)
    report=r.audit(q,100)
    assert report['geometry_passed'],report
    prefixes=[]
    for path in sorted((folder/'stream').glob('*/metadata.json')):
        prefix_meta=json.loads(path.read_text())
        prefix=np.load(path.parent/'source_29dof.npz',allow_pickle=False)['qpos']
        np.testing.assert_array_equal(prefix,source[:len(prefix)])
        assert prefix_meta['audit']['geometry_passed']
        prefixes.append(dict(seconds=prefix_meta['seconds'],frames=prefix_meta['frames']))
    expected=[0]
    for seconds in np.cumsum([s['seconds'] for s in schedule['requested']])[:-1]:
        expected.append(int(np.ceil(round(seconds*25)/8))*8)
    assert [e['frame'] for e in schedule['actual_prompt_events']]==expected
    return dict(passed=True,id=folder.name,seconds=meta['seconds'],source_frames=len(source),
                target_frames=len(q),audit=report,prefixes=prefixes,
                prefix_sources_match_final=True,formats_match=True,
                prompt_events=schedule['actual_prompt_events'],
                fidelity_review_required=meta['audit']['fidelity_review_required'],
                total_seconds=meta['total_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder');parser.add_argument('--report',required=True)
    args=parser.parse_args();report=validate(args.folder)
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
