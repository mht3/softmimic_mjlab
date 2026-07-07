#!/bin/bash
# Swallow balance training — default parameters.
# Duplicate this file and adjust flags to run a different experiment.
#
# Usage:
#   bash train_configs/swallow_balance/default.sh
#   bash train_configs/swallow_balance/default.sh --agent.run-name my_experiment

set -e

TASK="Unitree-G1-23Dof-Tracking-No-State-Estimation"

# --- Environment ---
NUM_ENVS=4096

# --- Agent ---
MAX_ITERATIONS=30001
LEARNING_RATE=1e-3
NUM_STEPS_PER_ENV=24

EVAL_ENABLED=True
EVAL_EVERY_N_ITERS=50
EVAL_NUM_EPISODES=500
EVAL_NUM_ENVS=500

python scripts/train.py "$TASK" \
  --motion_file=src/assets/motions/g1_23dof/swallow_balance_23dof.npz \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.algorithm.learning-rate "$LEARNING_RATE" \
  --agent.num-steps-per-env "$NUM_STEPS_PER_ENV" \
  --agent.eval.enabled "$EVAL_ENABLED" \
  --agent.eval.eval-every-n-iters "$EVAL_EVERY_N_ITERS" \
  --agent.eval.eval-num-episodes "$EVAL_NUM_EPISODES" \
  --agent.eval.eval-num-envs "$EVAL_NUM_ENVS" \
  "$@"
