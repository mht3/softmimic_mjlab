#!/bin/bash
# Compliant tracking training on the augmented stand dataset (Unitree G1 23dof).
#
# Requires the stand augmentations converted to NPZ first (see README
# "Compliant Tracking (SoftMimic)" section).
#
# Usage:
#   bash train_configs/compliance/stand.sh
#   bash train_configs/compliance/stand.sh --agent.run-name my_experiment

set -e

TASK="${TASK:-Unitree-G1-23Dof-Compliant-Tracking-No-State-Estimation}"

# --- Environment ---
NUM_ENVS=4096

# --- Agent ---
MAX_ITERATIONS=50001
LEARNING_RATE=1e-3
NUM_STEPS_PER_ENV=24



python scripts/train.py "$TASK" \
  --motion_file=src/assets/compliant_motions/g1_23dof/stand \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.algorithm.learning-rate "$LEARNING_RATE" \
  --agent.num-steps-per-env "$NUM_STEPS_PER_ENV" \
  "$@"
