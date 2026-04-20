#!/bin/bash
# Static balance training with push velocity curriculum.
# Push velocities start soft, then roughly double at iteration 5000.
# Logs push_vel_x_max and push_vel_y_max to wandb.
#
# Usage:
#   bash train_configs/static_balance/push_curriculum.sh
#   bash train_configs/static_balance/push_curriculum.sh --agent.run-name my_experiment

set -e

TASK="Unitree-G1-23Dof-Balance-Flat-Push-Curriculum"

# --- Environment ---
NUM_ENVS=4096

# --- Agent ---
MAX_ITERATIONS=30001
LEARNING_RATE=1e-3
NUM_STEPS_PER_ENV=24

python scripts/train.py "$TASK" \
  --init-at-random-ep-len False \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.algorithm.learning-rate "$LEARNING_RATE" \
  --agent.num-steps-per-env "$NUM_STEPS_PER_ENV" \
  "$@"
