# Matched 23DOF compliant-standing controller

This folder contains **Stable checkpoint 24000**, matched to the default browser preview. Selected from 28 saved checkpoints (11k–24k), it passed 30/30 basic and held-out standing/recovery tests and survived six 20/40 ms sensor-delay tests. Mean noisy wrist speed was 0.385 rad/s over eight seeds, compared with 0.443 for 15k. See `checkpoint_selection_11_24k.json`. The extended run completed at iteration 24999/25000. Its final checkpoint was also evaluated and retained as a local comparison: 26/30 recovery tests and 0.444 rad/s noisy wrist RMS, so 24000 remains selected (see `final_25k_comparison.json`). Older comparison files in this folder are historical.

- `exported/policy.onnx`: 327 actor inputs, 23 joint actions, 50 Hz.
- `params/deploy.yaml`: exact observation history, nominal channels, gains, joint map, scaling and ±5 raw-action bound from this checkpoint.
- `params/reference.csv`: the matching 30-second standing reference, included in Git.
- `controller_config.yaml`: portable, standing-only FSM; identical to the default G1 23DOF configuration.
- `manifest.json`: hashes of the three matched input files and source checkpoint provenance.

The standing controller reads its constant nominal channels directly from `deploy.yaml`. It does **not** require an ignored IK NPZ, motion capture, a root-position estimator, foot-force readings or external-force sensors. The CSV records the same nominal pose and supplies the browser timeline; the offline check verifies it against the YAML.

From the repository root:

```bash
conda activate unitree_rl_mjlab
```

```bash
python deploy/check_bundle.py
```

See [deployment instructions](../../../../../../README.md). This snapshot passed interface and bounded standing/load-recovery simulation checks; it has **not** been tested on a physical robot. Preserve the matched files together when replacing the controller.
