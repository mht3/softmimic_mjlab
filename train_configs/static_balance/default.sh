#!/bin/bash
# Static balance training — default parameters.
# Duplicate this file and adjust flags to run a different experiment.
#
# Usage:
#   bash train_configs/static_balance/default.sh
#   bash train_configs/static_balance/default.sh --agent.run-name my_experiment

set -e

TASK="Unitree-G1-23Dof-Balance-Flat"

# --- Environment ---
NUM_ENVS=4096

# --- Agent ---
MAX_ITERATIONS=10001
LEARNING_RATE=1e-3
NUM_STEPS_PER_ENV=24

python scripts/train.py "$TASK" \
  --init-at-random-ep-len False \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.algorithm.learning-rate "$LEARNING_RATE" \
  --agent.num-steps-per-env "$NUM_STEPS_PER_ENV" \
  "$@"
