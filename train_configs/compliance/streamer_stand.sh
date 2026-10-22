#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export MOTION_FILE="${MOTION_FILE:-src/assets/compliant_motions/g1_23dof/streamer_stand}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/algorithms/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
python - "$MOTION_FILE" <<'PY_CHECK'
import sys
from scripts.check_compliant_dataset_ready import check
count=check(sys.argv[1])
if count != 85:
    raise SystemExit(f"TRAINING BLOCKED: expected all 85 audited standing files; found {count}")
PY_CHECK
export MAX_ITERATIONS="${MAX_ITERATIONS:-20000}" WANDB_MODE=online
exec bash "$SCRIPT_DIR/stand.sh" --agent.resume False --agent.run-name streamer_stand "$@"
