#!/bin/bash
# Static balance training with the time-indexed DiT visitation critic, using the
# 4-bin reward_bins labeling (fail-low=0, fail-high=1, succeed-low=2, succeed-high=3).
# Trains rho(x | t_norm, label) over full 58-D relative-state trajectories.
#
# Architecture matches notebook 9 ``dit_large``: hidden=256, depth=8, n_tokens=4,
# heads=4, fourier_dim=32 — set in code as the DiT defaults, no need to override.
#
# Usage:
#   bash train_configs/static_balance/ppo_visitation_critic_dit_reward_bins.sh
#   bash train_configs/static_balance/ppo_visitation_critic_dit_reward_bins.sh \
#        --agent.run-name dit_rewbins_run1

set -e

TASK="Unitree-G1-23Dof-Balance-Flat"

# --- Environment ---
NUM_ENVS=4096

# --- Agent ---
MAX_ITERATIONS=20001
LEARNING_RATE=1e-3
NUM_STEPS_PER_ENV=24

# --- Visitation Critic (time-indexed DiT, notebook 9 dit_large) ---
VC_ARCHITECTURE=dit
VC_TRAIN_EVERY_N_ITERS=5001
VC_NUM_WARMUP_ITERATIONS=5000
VC_NUM_TRAIN_STEPS=50000        # notebook 9 setting
VC_LEARNING_RATE=1e-3           # notebook 9 setting
VC_BATCH_SIZE=1000              # notebook 9 setting
VC_MAX_TRAJECTORIES=10000
VC_GUIDANCE_SCALE=3.0
VC_NUM_EULER_STEPS=100
VC_CFG_DROPOUT_PROB=0.25
VC_MAX_NUM_TRAINS=3             # -1 = unlimited; set to M to train at most M times
VC_NUM_CLASSES=4                # 4 reward bins; null/CFG class = 4
VC_RESET_BIN_PROBS="(0.3,0.5,0.15,0.05)"  # fail-low, fail-high, succeed-low, succeed-high
VC_USE_RESET_STATES=True

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
  --agent.algorithm.visitation-critic-cfg.architecture "$VC_ARCHITECTURE" \
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
  --agent.algorithm.visitation-critic-cfg.max-num-trains "$VC_MAX_NUM_TRAINS" \
  --agent.algorithm.visitation-critic-cfg.reset-bin-probs "$VC_RESET_BIN_PROBS" \
  --agent.algorithm.visitation-critic-cfg.use-reset-states "$VC_USE_RESET_STATES" \
  --agent.eval.enabled "$EVAL_ENABLED" \
  --agent.eval.eval-every-n-iters "$EVAL_EVERY_N_ITERS" \
  --agent.eval.eval-num-episodes "$EVAL_NUM_EPISODES" \
  --agent.eval.eval-num-envs "$EVAL_NUM_ENVS" \
  "$@"
