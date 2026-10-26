# Matched 23DOF compliant-standing controller

This folder contains **checkpoint 34000** from the fresh 50k run, matched to the default browser preview. It passed 16/16 basic standing/recovery tests, 6/6 seeded training-dynamics standing/downward-load tests and two sensor-delay stress tests. Noisy standing wrist RMS was 0.313 rad/s. See `selection_34000.json`. The run is still ongoing; this is an intermediate checkpoint selected by simulation, not a final or hardware-validated controller. Older comparison files in this folder are historical.

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
