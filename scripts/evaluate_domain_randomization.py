"""Paired CPU noise ablations and seeded training-parameter stress tests.

Writes an immutable report; never modifies a bundle, training or hardware.
"""
import argparse
import hashlib
import json
from pathlib import Path
from evaluate_browser_actuation import run
from select_standing_checkpoint import passes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundles',nargs='+',type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('Report exists; choose a new output')
    report={'protocol':{'duration_s':30,'seeds':a.seeds,'parameter_rng':'LCG matching browser draws',
              'sensor_rng':'numpy; same bounds, not browser bitwise trajectory',
              'parameter_draw':'all groups enabled; startup ranges, one draw per trial',
              'contacts':'training for randomized trials; frictional-body 0.5/feet 0.6 for noise ablation',
              'loads':'downward: 30 N each palm, 1 s ramp / 2 s hold / 1 s release, starts 5 s',
              'recovery_gate':'same standing-checkpoint limits; relative to 3–5 s baseline'},
            'candidates':[],'hardware_validated':False,'complete':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for bundle in a.bundles:
        meta=json.loads((bundle/'manifest.json').read_text())
        c={'iteration':meta['checkpoint_iteration'],'bundle':str(bundle.resolve()),
           'onnx_sha256':hashlib.sha256((bundle/'exported/policy.onnx').read_bytes()).hexdigest(),'trials':[]}
        for seed in a.seeds:
            for case in ('velocity_noise_off','training_draw_stand','training_draw_downward'):
                randomized=case.startswith('training_draw')
                row=run(bundle,30,'implicit',sensor_noise=True,seed=seed,
                        contact_profile='training' if randomized else 'frictional-body',
                        load='downward' if case.endswith('downward') else 'none',
                        noise_scales=None if randomized else {'joint_vel_rel':0.},
                        randomization={'seed':seed} if randomized else None)
                row.update(case=case,passed=passes(row));c['trials'].append(row)
                print(c['iteration'],case,seed,'pass',row['passed'],'wrist',round(row['wrist_speed_rms_rad_s'],3),flush=True)
        report['candidates'].append(c);a.output.write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
