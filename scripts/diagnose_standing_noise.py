"""Paired sensor-channel ablations on frozen policies; CPU evaluation only."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_browser_actuation import run
from select_standing_checkpoint import passes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundles', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output exists; preserve previous diagnostics')
    channels = ('base_ang_vel', 'projected_gravity', 'joint_pos_rel', 'joint_vel_rel')
    conditions = {'all': {}}
    conditions.update({f'without_{channel}': {channel: 0.} for channel in channels})
    conditions['clean'] = dict.fromkeys(channels, 0.)
    report = dict(protocol='30 s unforced; frictional-body contacts; identical RNG draws and seeds 42/43/44 across channel ablations',
                  hardware_validated=False, candidates=[])
    for bundle in a.bundles:
        meta = json.loads((bundle/'manifest.json').read_text())
        candidate = dict(iteration=meta['checkpoint_iteration'], onnx_sha256=meta['bundle_onnx_sha256'], conditions={})
        for name, scales in conditions.items():
            rows = [run(bundle, 30, 'implicit', sensor_noise=True, seed=seed,
                        contact_profile='frictional-body', noise_scales=scales) for seed in (42,43,44)]
            result = dict(wrist_speed_rms_mean=float(np.mean([r['wrist_speed_rms_rad_s'] for r in rows])),
                          action_rate_rms_mean=float(np.mean([r['wrist_action_rate_rms'] for r in rows])),
                          passes=sum(passes(r) for r in rows), rows=rows)
            candidate['conditions'][name] = result
            print(candidate['iteration'], name, result['wrist_speed_rms_mean'], flush=True)
        report['candidates'].append(candidate)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2)+'\n')
    report['complete'] = True
    a.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
