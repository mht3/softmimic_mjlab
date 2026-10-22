"""Run the complete standing regeneration and fail-closed training handoff."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--run-name', default='collision_aware_stand')
    args = parser.parse_args()
    args.state = args.state.resolve()
    args.state.parent.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'logs/standing_regeneration_pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.state.exists():
        raise RuntimeError(f'Run state already exists; refusing duplicate launch: {args.state}')
    def save(status, **details):
        state = dict(status=status, updated_at=datetime.now(timezone.utc).isoformat(),
                     run_name=args.run_name, wandb_project='g1_compliance', **details)
        temp = args.state.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(args.state)
    env = os.environ.copy()
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env['PATH']
    env['MPLCONFIGDIR'] = '/tmp/compliance-mpl'
    env['WANDB_MODE'] = 'online'
    generation_env = dict(env, CUDA_VISIBLE_DEVICES='', WARP_CACHE_PATH='/tmp/compliance-warp')
    command = [sys.executable, '-u', str(ROOT / 'scripts/generate_compliant_23dof.py'),
               '--motions', 'stand', '--modes', 'forcefield', 'collision-emulator', 'zero-wrench',
               '--num-files', '40', '40', '5', '--seed', '42', '--device', 'cpu', '--replace']
    save('generating', command=command)
    try:
        subprocess.run(command, cwd=ROOT, env=generation_env, check=True)
        report = json.loads((ROOT / 'src/assets/compliant_motions/g1_23dof/stand/audit.json').read_text())
        stages = {Path(row['dataset_root']).parent.name for row in report}
        if len(stages) != 1 or not next(iter(stages)).startswith('.generate-'):
            raise RuntimeError('Installed audit does not identify a unique generation stage')
        stage = stages.pop()
    except BaseException as error:
        save('generation_failed', error=repr(error))
        print(f'TRAINING NOT STARTED: {error}', flush=True)
        raise
    save('waiting_generation', expected_stage=stage)
    handoff = [sys.executable, '-u', str(ROOT / 'scripts/audit_then_train_stand.py'),
               '--expected-stage', stage, '--state', str(args.state), '--run-name', args.run_name]
    # The handoff performs a second full audit and records audit/training errors.
    result = subprocess.run(handoff, cwd=ROOT, env=env)
    if result.returncode and json.loads(args.state.read_text())['status'] == 'waiting_generation':
        save('handoff_failed', return_code=result.returncode)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
