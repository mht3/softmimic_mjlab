#!/bin/bash
# Static balance PPO with visitation-critic backtracking resets.
# Bad reward_bins episodes (bins 0/1) seed a reset buffer at terminal_idx - 25 (~0.5s at dt=0.02).

set -e

TASK="Unitree-G1-23Dof-Balance-Flat"

NUM_ENVS=4096
MAX_ITERATIONS=20001
LEARNING_RATE=1e-3
NUM_STEPS_PER_ENV=24

# Backtrack reset harvesting runs live on PPO done events after this warmup.
VC_NUM_WARMUP_ITERATIONS=5000
VC_NUM_TRAIN_STEPS=1
VC_LEARNING_RATE=5e-4
VC_BATCH_SIZE=1024
VC_MAX_TRAJECTORIES=10000
VC_MAX_NUM_TRAINS=3
VC_NUM_CLASSES=4
VC_USE_RESET_STATES=True
VC_RESET_STRATEGY=backtrack
VC_BACKTRACK_STEPS=25
VC_BACKTRACK_RESET_PROBABILITY=1.0
VC_BACKTRACK_RESET_BUFFER_SIZE=100000

EVAL_ENABLED=True
EVAL_EVERY_N_ITERS=100
EVAL_NUM_EPISODES=1000
EVAL_NUM_ENVS=1000

python scripts/train.py "$TASK" \
  --init-at-random-ep-len False \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.algorithm.learning-rate "$LEARNING_RATE" \
  --agent.num-steps-per-env "$NUM_STEPS_PER_ENV" \
  --agent.algorithm.visitation-critic-cfg.enabled True \
  --agent.algorithm.visitation-critic-cfg.conditioning-type discrete \
  --agent.algorithm.visitation-critic-cfg.label-mode reward_bins \
  --agent.algorithm.visitation-critic-cfg.num-classes "$VC_NUM_CLASSES" \
  --agent.algorithm.visitation-critic-cfg.num-warmup-iterations "$VC_NUM_WARMUP_ITERATIONS" \
  --agent.algorithm.visitation-critic-cfg.num-train-steps "$VC_NUM_TRAIN_STEPS" \
  --agent.algorithm.visitation-critic-cfg.learning-rate "$VC_LEARNING_RATE" \
  --agent.algorithm.visitation-critic-cfg.batch-size "$VC_BATCH_SIZE" \
  --agent.algorithm.visitation-critic-cfg.max-trajectories "$VC_MAX_TRAJECTORIES" \
  --agent.algorithm.visitation-critic-cfg.max-num-trains "$VC_MAX_NUM_TRAINS" \
  --agent.algorithm.visitation-critic-cfg.use-reset-states "$VC_USE_RESET_STATES" \
  --agent.algorithm.visitation-critic-cfg.reset-strategy "$VC_RESET_STRATEGY" \
  --agent.algorithm.visitation-critic-cfg.backtrack-steps "$VC_BACKTRACK_STEPS" \
  --agent.algorithm.visitation-critic-cfg.backtrack-reset-probability "$VC_BACKTRACK_RESET_PROBABILITY" \
  --agent.algorithm.visitation-critic-cfg.backtrack-reset-buffer-size "$VC_BACKTRACK_RESET_BUFFER_SIZE" \
  --agent.eval.enabled "$EVAL_ENABLED" \
  --agent.eval.eval-every-n-iters "$EVAL_EVERY_N_ITERS" \
  --agent.eval.eval-num-episodes "$EVAL_NUM_EPISODES" \
  --agent.eval.eval-num-envs "$EVAL_NUM_ENVS" \
  "$@"
