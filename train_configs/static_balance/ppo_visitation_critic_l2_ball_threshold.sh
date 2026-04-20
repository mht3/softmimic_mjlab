#!/bin/bash
# Static balance training with visitation critic (CFM) enabled.
# Usage:
#   bash train_configs/static_balance/ppo_visitation_critic_l2_ball_threshold.sh
#   bash train_configs/static_balance/ppo_visitation_critic_l2_ball_threshold.sh --agent.run-name my_experiment

set -e

TASK="Unitree-G1-23Dof-Balance-Flat"

# --- Environment ---
NUM_ENVS=4096

# --- Agent ---
MAX_ITERATIONS=20001
LEARNING_RATE=1e-3
NUM_STEPS_PER_ENV=24

# --- Visitation Critic ---
VC_TRAIN_EVERY_N_ITERS=5001
VC_NUM_WARMUP_ITERATIONS=5000
VC_NUM_TRAIN_STEPS=50000
VC_LEARNING_RATE=5e-4
VC_BATCH_SIZE=1024
VC_MAX_TRAJECTORIES=20000
VC_GUIDANCE_SCALE=1.5
VC_NUM_EULER_STEPS=100
VC_CFG_DROPOUT_PROB=0.25
VC_L2_RADIUS=4.0
VC_HIDDEN_DIMS="(512,512,512)"
VC_CLASS_DIM=3

python scripts/train.py "$TASK" \
  --init-at-random-ep-len False \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.algorithm.learning-rate "$LEARNING_RATE" \
  --agent.num-steps-per-env "$NUM_STEPS_PER_ENV" \
  --agent.algorithm.visitation-critic-cfg.enabled True \
  --agent.algorithm.visitation-critic-cfg.conditioning-type discrete \
  --agent.algorithm.visitation-critic-cfg.label-mode l2_ball \
  --agent.algorithm.visitation-critic-cfg.l2-radius "$VC_L2_RADIUS" \
  --agent.algorithm.visitation-critic-cfg.num-classes 2 \
  --agent.algorithm.visitation-critic-cfg.train-every-n-iters "$VC_TRAIN_EVERY_N_ITERS" \
  --agent.algorithm.visitation-critic-cfg.num-warmup-iterations "$VC_NUM_WARMUP_ITERATIONS" \
  --agent.algorithm.visitation-critic-cfg.num-train-steps "$VC_NUM_TRAIN_STEPS" \
  --agent.algorithm.visitation-critic-cfg.learning-rate "$VC_LEARNING_RATE" \
  --agent.algorithm.visitation-critic-cfg.batch-size "$VC_BATCH_SIZE" \
  --agent.algorithm.visitation-critic-cfg.max-trajectories "$VC_MAX_TRAJECTORIES" \
  --agent.algorithm.visitation-critic-cfg.guidance-scale "$VC_GUIDANCE_SCALE" \
  --agent.algorithm.visitation-critic-cfg.num-euler-steps "$VC_NUM_EULER_STEPS" \
  --agent.algorithm.visitation-critic-cfg.cfg-dropout-prob "$VC_CFG_DROPOUT_PROB" \
  --agent.algorithm.visitation-critic-cfg.reset-condition-label 0 \
  --agent.algorithm.visitation-critic-cfg.hidden-dims "$VC_HIDDEN_DIMS" \
  --agent.algorithm.visitation-critic-cfg.class-dim "$VC_CLASS_DIM" \
  "$@"
