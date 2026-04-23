#!/usr/bin/env bash
set -euo pipefail

# ---------- user-editable ----------
MODEL_PATH="${MODEL_PATH:-/home/mht/research/generative_models/humanoid_visitation_critic/logs/rsl_rl/g1_23dof_static_balance/2026-04-21_13-50-25/model_4000.pt}"
NUM_ENVS="${NUM_ENVS:-1000}"

ROOT="/home/mht/research/generative_models"
HVC_DIR="$ROOT/humanoid_visitation_critic"
OUT_BASE="$ROOT/visitation_critic/examples/datasets"
OUT_GEN="$OUT_BASE/generated"
# ----------------------------------

source "/home/mht/miniconda3/etc/profile.d/conda.sh"
conda activate unitree_rl_mjlab
export PYTHONPATH=.

cd "$HVC_DIR"

python scripts/collect_dataset.py Unitree-G1-23Dof-Balance-Flat \
  --policy "$MODEL_PATH" \
  --init-perturb-mode medium \
  --num-trajectories 5000 \
  --num-envs "$NUM_ENVS" \
  --output-dir "$OUT_BASE" \
  --name model_10000_medium_init_noise
