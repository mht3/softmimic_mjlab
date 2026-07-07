#!/bin/bash
# Swallow balance tracking with visitation-critic-blended PPO (SDEdit sampling).
#
# The visitation critic learns a flow-matching density model over locally
# high-MCQ state-action pairs. At each rollout step it samples candidate
# actions and the PPO action is alpha-blended with the candidate (a small
# pull toward the support set of high-value experience). The PPO objective
# is unchanged; only the executed action and its stored log_prob differ.
#
# Usage:
#   bash train_configs/swallow_balance/ppo_vc_sdedit.sh
#   bash train_configs/swallow_balance/ppo_vc_sdedit.sh --agent.run-name my_experiment

set -e

TASK="Unitree-G1-23Dof-Tracking-No-State-Estimation"

# --- Environment ---
NUM_ENVS=4096

# --- Agent ---
MAX_ITERATIONS=30001
LEARNING_RATE=1e-3
NUM_STEPS_PER_ENV=24

# --- Visitation critic (SDEdit conditional sampling) ---
VC_SAMPLE_METHOD=sdedit
VC_ALPHA=0.5
VC_WARMUP_ITERS=250
VC_NUM_SAMPLES=50
VC_TAU=0.7
VC_POLICY_TRUST_STD=3.0
VC_MODEL_TRAIN_EVERY=1
VC_MODEL_TRAIN_STEPS=80
VC_MODEL_BATCH_SIZE=256
VC_MODEL_LAMBDA_STEPS=50
VC_BUFFER_SIZE=100000
VC_Q_TOP_FRACTION=0.25
VC_Q_FILTER_K=32
VC_Q_MODE=off

# --- Fixed deterministic evaluation ---
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
  --agent.algorithm.visitation-critic-cfg.enabled True \
  --agent.algorithm.visitation-critic-cfg.sample-method "$VC_SAMPLE_METHOD" \
  --agent.algorithm.visitation-critic-cfg.alpha "$VC_ALPHA" \
  --agent.algorithm.visitation-critic-cfg.warmup-iters "$VC_WARMUP_ITERS" \
  --agent.algorithm.visitation-critic-cfg.num-samples "$VC_NUM_SAMPLES" \
  --agent.algorithm.visitation-critic-cfg.tau "$VC_TAU" \
  --agent.algorithm.visitation-critic-cfg.policy-trust-std "$VC_POLICY_TRUST_STD" \
  --agent.algorithm.visitation-critic-cfg.model-train-every "$VC_MODEL_TRAIN_EVERY" \
  --agent.algorithm.visitation-critic-cfg.model-train-steps "$VC_MODEL_TRAIN_STEPS" \
  --agent.algorithm.visitation-critic-cfg.model-batch-size "$VC_MODEL_BATCH_SIZE" \
  --agent.algorithm.visitation-critic-cfg.model-lambda-steps "$VC_MODEL_LAMBDA_STEPS" \
  --agent.algorithm.visitation-critic-cfg.buffer-size "$VC_BUFFER_SIZE" \
  --agent.algorithm.visitation-critic-cfg.q-top-fraction "$VC_Q_TOP_FRACTION" \
  --agent.algorithm.visitation-critic-cfg.q-filter-k "$VC_Q_FILTER_K" \
  --agent.algorithm.visitation-critic-cfg.q-mode "$VC_Q_MODE" \
  --agent.eval.enabled "$EVAL_ENABLED" \
  --agent.eval.eval-every-n-iters "$EVAL_EVERY_N_ITERS" \
  --agent.eval.eval-num-episodes "$EVAL_NUM_EPISODES" \
  --agent.eval.eval-num-envs "$EVAL_NUM_ENVS" \
  "$@"
