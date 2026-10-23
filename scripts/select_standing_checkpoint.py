"""Reproducible CPU selection; writes evidence, never installs or starts hardware.

Basic tests use modest forces below the Stable training cap (140 N / 10 Nm),
not a claim of survival at the full cap. Delays are reported separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_browser_actuation import run, LOADS


LIMITS = dict(root_height_error_m=.05, root_xy_drift_m=.15,
              maximum_palm_error_m=.08, mean_final_tilt_rad=.2)


def passes(row):
    return (not row['fallen'] and row['seconds'] >= 30-1e-6 and
            row['recovery'] is not None and
            all(np.isfinite(row['recovery'][key]) and row['recovery'][key] <= value
                for key, value in LIMITS.items()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundles',type=Path,nargs='+',required=True)
    p.add_argument('--baseline',type=int,default=15000)
    p.add_argument('--minimum-iteration',type=int,default=15000)
    p.add_argument('--maximum-iteration',type=int,default=19999)
    p.add_argument('--previous-report',type=Path,help='Reuse identical-protocol trials only when the ONNX hash matches')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists; keep previous comparisons immutable')
    report=dict(protocol=dict(duration_s=30,noise_seeds=[42,43,44],contacts='frictional-body',
                body_friction=.5,foot_friction=.6,forces_world_N=LOADS,load_window_s=[5,9],
                ramp_s=1,hold_s=2,compliance_N_m=200,rotational_compliance_Nm_rad=1,
                recovery_baseline_s=[3,5],recovery_final_s=[25,30],recovery_limits=LIMITS,
                fall_tilt_rad=1.25,collapse_height_m=.3,
                selection='All 16 basic trials must pass survival and recovery; rank mean noisy unforced wrist RMS, then action-rate RMS. Compare to baseline.',
                caveat='Finite native CPU MuJoCo tests; no hardware validation or guarantee for stronger/different loads.'),
                candidates=[],selected_iteration=None,hardware_validated=False)
    cached={}
    if a.previous_report:
        previous=json.loads(a.previous_report.read_text())
        if not previous.get('complete') or previous['protocol']!=report['protocol']:
            p.error('Cached report is incomplete or uses a different evaluation protocol')
        cached={c['iteration']:c for c in previous['candidates']}
        report['reused_report_sha256']=hashlib.sha256(a.previous_report.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for bundle in a.bundles:
        meta=json.loads((bundle/'manifest.json').read_text())
        candidate=dict(iteration=meta['checkpoint_iteration'],onnx_sha256=hashlib.sha256((bundle/'exported/policy.onnx').read_bytes()).hexdigest(),basic=[],stress=[],parity=[])
        old=cached.get(candidate['iteration'])
        if old and old['onnx_sha256']==candidate['onnx_sha256']:
            report['candidates'].append(old)
            a.output.write_text(json.dumps(report,indent=2)+'\n')
            print('REUSED',candidate['iteration'],flush=True)
            continue
        for load in LOADS:
            for noise,seed in [(False,42),(True,42),(True,43),(True,44)]:
                row=run(bundle,30,'implicit',sensor_noise=noise,seed=seed,contact_profile='frictional-body',load=load)
                row['passed']=passes(row);candidate['basic'].append(row)
        for delay in (1,2):
            candidate['stress'].append(run(bundle,30,'implicit',sensor_noise=True,seed=42,delay_steps=delay,contact_profile='frictional-body'))
        candidate['parity'].append(run(bundle,30,'implicit',sensor_noise=True,seed=42,contact_profile='training'))
        noisy=[r for r in candidate['basic'] if r['load']=='none' and r['sensor_noise']]
        candidate.update(basic_passes=sum(r['passed'] for r in candidate['basic']),
                         noisy_standing_wrist_rms=float(np.mean([r['wrist_speed_rms_rad_s'] for r in noisy])),
                         noisy_standing_action_rate_rms=float(np.mean([r['wrist_action_rate_rms'] for r in noisy])))
        report['candidates'].append(candidate)
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in candidate.items() if k not in ('basic','stress','parity')}),flush=True)
    baseline=next(c for c in report['candidates'] if c['iteration']==a.baseline)
    eligible=[c for c in report['candidates'] if a.minimum_iteration<=c['iteration']<=a.maximum_iteration and c['basic_passes']==16]
    eligible.sort(key=lambda c:(c['noisy_standing_wrist_rms'],c['noisy_standing_action_rate_rms']))
    if eligible and (baseline['basic_passes']<16 or eligible[0]['noisy_standing_wrist_rms']<=baseline['noisy_standing_wrist_rms']):
        report['selected_iteration']=eligible[0]['iteration']
    report['complete']=True
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('SELECTED',report['selected_iteration'],flush=True)


if __name__=='__main__':main()
