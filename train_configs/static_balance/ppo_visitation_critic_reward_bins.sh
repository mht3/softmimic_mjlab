#!/bin/bash
# Static balance training with visitation critic (CFM) enabled, using the
# 4-bin reward_bins labeling (fail-low=0, fail-high=1, succeed-low=2, succeed-high=3).
# Usage:
#   bash train_configs/static_balance/ppo_visitation_critic_reward_bins.sh
#   bash train_configs/static_balance/ppo_visitation_critic_reward_bins.sh --agent.run-name my_experiment

set -e

TASK="Unitree-G1-23Dof-Balance-Flat"

# --- Environment ---
NUM_ENVS=4096

# --- Agent ---
MAX_ITERATIONS=10001
LEARNING_RATE=1e-3
NUM_STEPS_PER_ENV=24

# --- Visitation Critic ---
VC_TRAIN_EVERY_N_ITERS=5001
VC_NUM_WARMUP_ITERATIONS=5000
VC_NUM_TRAIN_STEPS=50000
VC_LEARNING_RATE=5e-4
VC_BATCH_SIZE=1024
VC_MAX_TRAJECTORIES=10000
VC_GUIDANCE_SCALE=3.0
VC_NUM_EULER_STEPS=100
VC_CFG_DROPOUT_PROB=0.25
VC_HIDDEN_DIMS="(1024,1024,1024)"
VC_CLASS_DIM=3
VC_MAX_NUM_TRAINS=1  # -1 = unlimited; set to M to train at most M times
VC_NUM_CLASSES=4     # 4 reward bins; null/CFG class = 4
VC_RESET_BIN_PROBS="(0.1,0.4,0.4,0.1)"  # fail-low, fail-high, succeed-low, succeed-high

# --- Evaluation ---
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
  --agent.algorithm.visitation-critic-cfg.train-every-n-iters "$VC_TRAIN_EVERY_N_ITERS" \
  --agent.algorithm.visitation-critic-cfg.num-warmup-iterations "$VC_NUM_WARMUP_ITERATIONS" \
  --agent.algorithm.visitation-critic-cfg.num-train-steps "$VC_NUM_TRAIN_STEPS" \
  --agent.algorithm.visitation-critic-cfg.learning-rate "$VC_LEARNING_RATE" \
  --agent.algorithm.visitation-critic-cfg.batch-size "$VC_BATCH_SIZE" \
  --agent.algorithm.visitation-critic-cfg.max-trajectories "$VC_MAX_TRAJECTORIES" \
  --agent.algorithm.visitation-critic-cfg.guidance-scale "$VC_GUIDANCE_SCALE" \
  --agent.algorithm.visitation-critic-cfg.num-euler-steps "$VC_NUM_EULER_STEPS" \
  --agent.algorithm.visitation-critic-cfg.cfg-dropout-prob "$VC_CFG_DROPOUT_PROB" \
  --agent.algorithm.visitation-critic-cfg.hidden-dims "$VC_HIDDEN_DIMS" \
  --agent.algorithm.visitation-critic-cfg.class-dim "$VC_CLASS_DIM" \
  --agent.algorithm.visitation-critic-cfg.max-num-trains "$VC_MAX_NUM_TRAINS" \
  --agent.algorithm.visitation-critic-cfg.reset-bin-probs "$VC_RESET_BIN_PROBS" \
  --agent.eval.enabled "$EVAL_ENABLED" \
  --agent.eval.eval-every-n-iters "$EVAL_EVERY_N_ITERS" \
  --agent.eval.eval-num-episodes "$EVAL_NUM_EPISODES" \
  --agent.eval.eval-num-envs "$EVAL_NUM_ENVS" \
  "$@"
