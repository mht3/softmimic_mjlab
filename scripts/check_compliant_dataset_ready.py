"""Reject stale or missing physical-feasibility audits before standing training."""
import hashlib
import json
import math
from pathlib import Path
import sys


def check(source):
    source=Path(source).resolve()
    folder=source if source.is_dir() else source.parent
    # A generation-level audit can be above a force-mode directory.
    audit_path=folder/'audit.json'
    if not audit_path.is_file() and (folder.parent/'audit.json').is_file():
        folder=folder.parent;audit_path=folder/'audit.json'
    if not audit_path.is_file():raise ValueError(f'Missing audited dataset: {audit_path}. Run the full collision-aware audit first.')
    rows=json.loads(audit_path.read_text())
    selected=[r for r in rows if ((folder/r['file']).with_suffix('.npz').resolve().is_relative_to(source)
                                 if source.is_dir() else (folder/r['file']).with_suffix('.npz').resolve()==source)]
    if not selected:raise ValueError('Audit contains no selected motion files')
    for row in selected:
        if (row.get('error') or row.get('audit_stride')!=1 or
            not row.get('self_collision_checked') or not row.get('self_collision_free') or
            not row.get('standing_recovery_checked') or
            not row.get('npz_structural_valid')):
            raise ValueError(f"Missing or failed collision-aware full audit for {row.get('file')}; do not train this dataset")
        for key,limit in {'max_joint_limit_violation_rad':1e-3,
                          'max_joint_speed_rad_s':2*math.pi+1e-5,
                          'sampled_max_foot_displacement_m':.05+1e-6,
                          'sampled_max_F_over_K_error_m':.05+1e-6}.items():
            value=row.get(key)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value>limit:
                raise ValueError(f"Failed kinematic audit for {row['file']}: {key}={value}")
        if not math.isfinite(row.get('max_settled_hand_error_m',float('nan'))) or row['max_settled_hand_error_m']>.05+1e-6:
            raise ValueError(f"Failed standing recovery audit: {row['file']}")
        for path,key in ((folder/row['file'],'csv_sha256'),((folder/row['file']).with_suffix('.npz'),'npz_sha256')):
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row.get(key):raise ValueError(f'Stale audit hash: {path}')
    actual={p.resolve() for p in source.rglob('*.npz') if not any(x.startswith('.') for x in p.relative_to(source).parts)} if source.is_dir() else {source}
    audited={(folder/r['file']).with_suffix('.npz').resolve() for r in selected}
    if actual!=audited:raise ValueError('Audit and selected motion files differ')
    return len(selected)

if __name__=='__main__':
    try:print(f'Validated {check(sys.argv[1])} collision-audited motion files',flush=True)
    except (ValueError,OSError,KeyError) as error:
        print('TRAINING BLOCKED: '+str(error),file=sys.stderr);sys.exit(1)
