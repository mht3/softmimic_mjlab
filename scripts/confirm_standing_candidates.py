"""Held-out CPU trials for candidates selected by the fixed standing benchmark."""
import argparse,json
from pathlib import Path
import numpy as np
from evaluate_browser_actuation import run
from select_standing_checkpoint import passes

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--bundles',type=Path,nargs='+',required=True)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():p.error('Preserve existing reports')
 report=dict(protocol='30 s; frictional-body contacts; unforced seeds 100-104; each bounded load seeds 100-102; separate 20/40ms delays seeds 43/44; thresholds unchanged',candidates=[],hardware_validated=False)
 for b in a.bundles:
  m=json.loads((b/'manifest.json').read_text());c=dict(iteration=m['checkpoint_iteration'],bundle=str(b),onnx_sha256=m['bundle_onnx_sha256'],standing=[],loads=[],stress=[])
  for seed in range(100,105):
   r=run(b,30,'implicit',sensor_noise=True,seed=seed,contact_profile='frictional-body');r['passed']=passes(r);c['standing'].append(r)
  for load in ('left_pull','right_pull','downward'):
   for seed in range(100,103):
    r=run(b,30,'implicit',sensor_noise=True,seed=seed,contact_profile='frictional-body',load=load);r['passed']=passes(r);c['loads'].append(r)
  for delay in (1,2):
   for seed in (43,44):c['stress'].append(run(b,30,'implicit',sensor_noise=True,seed=seed,contact_profile='frictional-body',delay_steps=delay))
  c.update(mean_wrist_rms=float(np.mean([r['wrist_speed_rms_rad_s'] for r in c['standing']])),standing_passes=sum(r['passed'] for r in c['standing']),load_passes=sum(r['passed'] for r in c['loads']),stress_survivals=sum(not r['fallen'] for r in c['stress']))
  report['candidates'].append(c);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in c.items() if k not in ('standing','loads','stress')},flush=True)
 report['complete']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
