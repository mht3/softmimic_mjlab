"""Shared local demo entry point. CPU is the default; no hardware transport."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOFT = ROOT / 'softmimic_mjlab'
STATE = SOFT / 'logs/compliance_recovery_handoff_20261021.json'


def policy_command(checkpoint, motion, device):
    checkpoint = Path(checkpoint).resolve()
    motion = Path(motion).resolve()
    if not checkpoint.is_file():
        raise ValueError(f'Missing checkpoint: {checkpoint}')
    # Refuse silently selecting an old, incompatible standing policy.
    import yaml
    config = yaml.safe_load((checkpoint.parent / 'params/deploy.yaml').read_text())
    if config.get('observation_dim') != 327 or config.get('reference_mode') != 'constant_nominal':
        raise ValueError('Expected the corrected 327-input constant-nominal standing export')
    sys.path.insert(0, str(SOFT / 'scripts'))
    from check_compliant_dataset_ready import check
    check(motion)
    return [sys.executable, str(SOFT / 'scripts/play.py'),
            'Unitree-G1-23Dof-Compliant-Tracking-Stand',
            '--checkpoint-file', str(checkpoint), '--motion-file', str(motion),
            '--viewer', 'viser', '--num-envs', '1', '--device', device]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    replay = sub.add_parser('reference', help='Kinematic dataset replay with force arrows')
    replay.add_argument('--csv', type=Path, required=True)
    replay.add_argument('--port', type=int, default=8080)
    play = sub.add_parser('softmimic', help='Actual mjlab policy simulation and interactive forces')
    play.add_argument('--checkpoint', type=Path, required=True)
    play.add_argument('--motion', type=Path, default=SOFT / 'src/assets/compliant_motions/g1_23dof/stand')
    play.add_argument('--device', default='cpu')
    play.add_argument('--print-command', action='store_true')
    sub.add_parser('status', help='Read the existing audit/training pipeline; never starts it')
    args = parser.parse_args()
    if args.mode == 'status':
        print(json.dumps(json.loads(STATE.read_text()), indent=2))
        return
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(SOFT), str(SOFT / 'algorithms/rsl_rl'), env.get('PYTHONPATH', '')])
    env.setdefault('WARP_CACHE_PATH', '/tmp/compliance-demo-warp')
    env.setdefault('MPLCONFIGDIR', '/tmp/compliance-demo-mpl')
    env['OMP_NUM_THREADS'] = '2'
    if args.mode == 'reference':
        env['CUDA_VISIBLE_DEVICES'] = ''
        command = [sys.executable, str(Path(__file__).with_name('replay.py')), str(args.csv.resolve()), '--port', str(args.port)]
    else:
        command = policy_command(args.checkpoint, args.motion, args.device)
        if args.device == 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = ''
        if args.print_command:
            import shlex
            print(shlex.join(command))
            return
    raise SystemExit(subprocess.call(command, cwd=SOFT, env=env))


if __name__ == '__main__':
    main()
