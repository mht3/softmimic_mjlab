# Compliant standing · 20,000 iterations

Final checkpoint `model_19999.pt` from `collision_recovery_stand_20k_fresh_20261022`. [W&B run](https://wandb.ai/mht3/g1_compliance/runs/xklelhb4).

- Unitree G1 23DOF; 327 observations, 23 actions, 50 Hz; constant nominal standing reference.
- This is the completed **baseline** run. It does not contain the later bounded-action, rebalanced sampling or slower-reference training changes.
- CPU export passed 32 PyTorch/ONNX comparisons; C++/browser observation, action and reset checks passed. Exact hashes and errors are in `export_report.json` and `interface_validation.json`.
- Native CPU MuJoCo with corrected implicit PD remained upright for 20 seconds; wrist RMS speed was 0.097 rad/s. This unforced check is not comprehensive compliance or hardware validation.
- A separate six-file held-out set passed 12 deterministic 30-second CPU episodes with no early termination; see `heldout_validation.json`. Loaded tracking fidelity still needs broader evaluation.
- Policy control starts OFF. No hardware has been tested or enabled.

Run the [three-file demo command](../../../README.md#run-the-included-policy-demo). The small nominal CSV is `datasets/streamer/stand.csv`; generated assets are rebuilt at launch.

Re-export from the trusted training checkpoint:

```bash
conda activate unitree_rl_mjlab
```

```bash
python demos/export_checkpoint.py --checkpoint /path/to/run/model_19999.pt \
  --iteration 19999 --final --output /new/export/directory
```

Checkpoint SHA256: `dcd5f1046dc055597786f1fef1969bd705faf8e43501061747865483b2abd0f3`.
