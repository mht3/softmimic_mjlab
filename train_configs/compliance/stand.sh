#!/bin/bash
# Compliant tracking training on the augmented stand dataset (Unitree G1 23dof).
#
# Requires the stand augmentations converted to NPZ first (see README
# "Compliant Tracking (SoftMimic)" section).
#
# Usage:
#   bash train_configs/compliance/stand.sh
#   bash train_configs/compliance/stand.sh --agent.run-name my_experiment

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/algorithms/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/compliance-mpl}"
export TRAIN_LAUNCH_SCRIPT="$SCRIPT_DIR/stand.sh"
export WANDB_MODE="${WANDB_MODE:-online}"

TASK="${TASK:-Unitree-G1-23Dof-Compliant-Tracking-Stand}"

# --- Environment ---
NUM_ENVS="${NUM_ENVS:-4096}"

# --- Agent ---
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
MOTION_FILE="${MOTION_FILE:-src/assets/compliant_motions/g1_23dof/stand}"
LEARNING_RATE=1e-3
NUM_STEPS_PER_ENV=24



# Require a current full audit that includes self-collision checks.
python scripts/check_compliant_dataset_ready.py "$MOTION_FILE"

python scripts/train.py "$TASK" \
  --motion_file="$MOTION_FILE" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.algorithm.learning-rate "$LEARNING_RATE" \
  --agent.num-steps-per-env "$NUM_STEPS_PER_ENV" \
  --agent.logger wandb \
  --agent.wandb-project g1_compliance \
  "$@"
