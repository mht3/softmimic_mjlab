#!/usr/bin/env bash
# Opt-in bounded/rebalanced experiment. Never resumes or replaces the old run.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
export PYTHONPATH="$PWD:$PWD/algorithms/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
export MOTION_FILE="${MOTION_FILE:-src/assets/compliant_motions/g1_23dof/streamer_stand_smooth}"
python scripts/check_compliant_dataset_ready.py "$MOTION_FILE"
python - "$MOTION_FILE" <<'PY_CHECK'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);meta=json.loads((p/'generation.json').read_text())
if meta.get('max_joint_velocity')!=3. or meta.get('max_wrist_velocity')!=1.5:
    raise SystemExit('Generate the slower references first: --max-joint-velocity 3 --max-wrist-velocity 1.5')
PY_CHECK
export TASK=Unitree-G1-23Dof-Compliant-Tracking-Stand-Stable
export MAX_ITERATIONS=20000 WANDB_MODE=online
exec bash "$SCRIPT_DIR/stand.sh" --agent.resume False --agent.run-name stand_stable "$@"
