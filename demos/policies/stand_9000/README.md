# Compliant standing · iteration 9000

Frozen intermediate snapshot from `collision_recovery_stand_20k_fresh_20261022`, exported from `model_9000.pt` on CPU while training continued. [Training run](https://wandb.ai/mht3/g1_compliance/runs/xklelhb4).

- `exported/policy.onnx`: self-contained deterministic actor; 327 observations → 23 actions, 50 Hz, three frames of history. Observation normalization is disabled in this training run.
- `params/deploy.yaml`: exact matching observation/action, nominal-reference and gain configuration.
- [Standing CSV](../../../datasets/streamer/stand.csv): small, versioned input for this model. Derived `params/reference.csv`, `reference.json` and `stand.npz` are generated during deployment packaging and ignored by Git.
- `export_report.json`: checkpoint/ONNX hashes and 32-input PyTorch/ONNX comparison (maximum absolute difference 1.53e-5).
- `interface_validation.json`: native C++ versus browser checks; all 327 channels match, maximum C++/ONNX-reference action difference 6.64e-6.

The browser copy at `../../sonic_web/assets/policy/policy.onnx` has the same SHA256. Both copies are explicitly included by `.gitignore`. These are intermediate trained weights, not smoke-test weights. Training was neither stopped nor resumed by the export. Final policy quality and hardware behavior are not certified. The viewer starts with policy control OFF.

Recreate an export in the training environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python demos/export_checkpoint.py --checkpoint /path/to/run/model_9000.pt \
  --iteration 9000 --output /new/export/directory
```

Use the [workflow](../../../WORKFLOW.md) to package and validate another snapshot. Machine-specific controller YAML files are intentionally kept out of this public demo snapshot. Generate a fresh deployment bundle on the target workstation using the workflow.
