"""One-shot, fail-closed dataset audit and authorized standing training handoff."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from audit_compliant_dataset import audit

ROOT = Path(__file__).resolve().parents[1]
MODES = {'forcefield': 40, 'collision-emulator': 40, 'zero-wrench': 5}


def validate_report(rows):
    errors=[]
    counts={m:0 for m in MODES}
    for row in rows:
        name=row.get('file','');mode=Path(name).parts[0] if name else ''
        if mode not in counts: errors.append(f'Unexpected mode: {name}')
        else: counts[mode]+=1
        if row.get('error') or not row.get('npz_structural_valid') or not row.get('npz_has_nominal'):
            errors.append(f'{name}: invalid CSV/NPZ: {row.get("error", "missing validated nominal channels")}')
        limits={'max_joint_limit_violation_rad':1e-3,'max_joint_speed_rad_s':2*math.pi+1e-5,
                'sampled_max_foot_displacement_m':.05+1e-6,'sampled_max_F_over_K_error_m':.05+1e-6}
        for key,limit in limits.items():
            value=row.get(key)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value>limit:
                errors.append(f'{name}: {key}={value} exceeds {limit}')
        if not row.get('self_collision_checked') or not row.get('self_collision_free'):
            errors.append(f'{name}: missing or failed self-collision validation')
        if not row.get('standing_recovery_checked') or not math.isfinite(row.get('max_settled_hand_error_m',float('nan'))) or row['max_settled_hand_error_m']>.05+1e-6:
            errors.append(f'{name}: missing or failed standing release recovery validation')
        if row.get('audit_stride')!=1: errors.append(f'{name}: audit must inspect every frame')
        if row.get('npz_joint_count')!=23 or row.get('npz_fps')!=50.:errors.append(f'{name}: incorrect training dimensions/FPS')
    if counts!=MODES:errors.append(f'Expected {MODES}; found {counts}')
    return errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-stage',required=True)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--run-name',default='corrected_stand')
    args=p.parse_args()
    args.state.parent.mkdir(parents=True,exist_ok=True)
    lock=args.state.with_suffix('.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads(args.state.read_text()) if args.state.exists() else {}
    if state.get('status') not in (None, 'waiting_generation', 'submitted'):
        raise RuntimeError(f'Handoff already attempted: {state}; refusing duplicate training')
    def save(status,**details):
        state.update(status=status,updated_at=datetime.now(timezone.utc).isoformat(),**details)
        temp=args.state.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(args.state)
    dataset=ROOT/'src/assets/compliant_motions/g1_23dof/stand'
    report=ROOT/'logs/stand_dataset_audit.json'
    save('auditing',dataset=str(dataset),audit=str(report),expected_stage=args.expected_stage)
    try:
        manifest=json.loads((dataset/'generation.json').read_text())
        generated_audit=json.loads((dataset/'audit.json').read_text())
        if not generated_audit or any(Path(r['dataset_root']).parent.name!=args.expected_stage for r in generated_audit):
            raise RuntimeError('Canonical data are not the completed generation requested by this watcher')
        if manifest['modes']!=list(MODES) or manifest['num_files']!=list(MODES.values()) or manifest['seed']!=42:
            raise RuntimeError('Unexpected generation configuration')
        for relative,digest in manifest['source_sha256'].items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Generation source changed: {relative}')
        for filename,key in (('generate_compliant_23dof.py','generator_script_sha256'),
                             ('compliant_csv_to_npz.py','converter_script_sha256')):
            if hashlib.sha256((ROOT/'scripts'/filename).read_bytes()).hexdigest()!=manifest.get(key):
                raise RuntimeError(f'Generation/conversion script changed: {filename}')
        print('Auditing all 85 installed CSV/NPZ pairs, every frame...',flush=True)
        rows=audit(dataset,ROOT/'softmimic_deploy/src/assets/g1/g1_23dof.xml',stride=1,input_fps=30.)
        report.write_text(json.dumps(rows,indent=2)+'\n')
        errors=validate_report(rows)
        prior={r['file']:r['csv_sha256'] for r in generated_audit}
        if any(prior.get(r['file'])!=r.get('csv_sha256') for r in rows):errors.append('CSV hashes changed after generation audit')
        if errors:raise RuntimeError('\n'.join(errors))
        summary={'files':len(rows),'max_foot_error_m':max(r['sampled_max_foot_displacement_m'] for r in rows),
                 'max_hand_F_over_K_error_m':max(r['sampled_max_F_over_K_error_m'] for r in rows),
                 'max_joint_speed_rad_s':max(r['max_joint_speed_rad_s'] for r in rows),
                 'downward_frames':sum(r['downward_frames_over10N'] for r in rows)}
        print('AUDIT PASSED: '+json.dumps(summary),flush=True)
    except Exception as error:
        save('audit_failed',error=str(error));print('TRAINING NOT STARTED: '+str(error),flush=True);raise
    cmd=['bash',str(ROOT/'train_configs/compliance/stand.sh'),'--agent.run-name',args.run_name]
    env=os.environ.copy();env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env['PATH'];env['WANDB_MODE']='online'
    env['MPLCONFIGDIR']=env.get('MPLCONFIGDIR','/tmp/compliance-mpl')
    save('training',audit_summary=summary,command=cmd,wandb_project='g1_compliance')
    print('Starting fresh GPU training; W&B project: g1_compliance',flush=True)
    try:
        code=subprocess.call(cmd,cwd=ROOT,env=env)
    except BaseException as error:
        save('training_failed',error=repr(error));raise
    save('training_finished' if code==0 else 'training_failed',return_code=code)
    sys.exit(code)

if __name__=='__main__':main()
