# Compliant Humanoid · fixed reference policy player

The maintained browser player uses the published [GEAR-SONIC Three.js/MuJoCo-WASM demo source](https://nvlabs.github.io/GEAR-SONIC/demo.html), adapted for G1 23DOF and SoftMimic. It preserves physics, camera controls, body picking and spring drag forces. The local UI adds a fixed bottom reference timeline and a separate cyan ghost. Original source files and URL/SHA256 provenance remain in `upstream/` and `upstream_manifest.json`; retain their notices and upstream terms.

## Run the included snapshot

From the repository root, after the [conda setup](../../README.md#installation-and-environment):

```bash
conda activate unitree_rl_mjlab
```

```bash
python demos/sonic_web/prepare.py --reference datasets/streamer/stand.csv \
  --deploy demos/policies/stand_9000/params/deploy.yaml \
  --policy demos/policies/stand_9000/exported/policy.onnx
```

```bash
python demos/sonic_web/serve.py --port 8088
```

Open http://127.0.0.1:8088. The included policy is the [iteration-9000 intermediate snapshot](../policies/stand_9000/README.md), with its matching deployment YAML and bundled browser runtimes. Generated assets are ignored; the command above rebuilds them from the small standing CSV and source meshes. For another controller, use the training environment and follow the [bundle and validation commands](../../WORKFLOW.md#4-package-and-visualize-the-trained-policy), then supply all three matched files:

```bash
python demos/sonic_web/prepare.py \
  --reference "$BUNDLE/params/reference.csv" \
  --deploy "$BUNDLE/params/deploy.yaml" \
  --policy "$BUNDLE/exported/policy.onnx"
```

Refresh after preparing. Control starts OFF. Enable the policy and press Play; enabling it also enables physics. Space toggles playback, R resets. The player never starts hardware. Do not install an early diagnostic checkpoint as a completed policy.

The reference is required. Save CSV metadata specifies named joints, XYZW quaternion order and frame rate; headerless prepared 32-column references use 30 Hz unless `--reference-fps` is supplied. Moving references can be previewed by the ghost, but this standing policy requires its matching constant nominal reference. Invalid pairs cannot enable control.

The fixed track supports play/pause, speed, click-to-seek and a slider. Seeking pauses and resets physics/history to that pose; it is not a recording of past simulation. The reference ghost has separate FK data and no collision/force contribution. Toggle its offset to overlay it. The physical G1 uses its MJCF black/silver materials and original mesh detail; these STL assets do not contain image texture maps.

## Simulation and interface limits

The actor has 327 inputs and 23 actions at 50 Hz. Per-term histories contain three frames, oldest first, with reset backfill. Local angular velocity, body-frame gravity, reference channels, normalization, action scaling/offsets and PD gains match the exported contract. Stiffness defaults to 200 N/m and 1 Nm/rad; these are policy inputs, not motor gains.

MuJoCo runs at 5 ms with four substeps per control update. Joint order, limits, armatures and collision masks follow the training robot. Robot-ground collisions are enabled; robot self contacts are disabled as in the training configuration. The offline IK collision audit is a separate gate. Each physics substep recomputes drag force and its moment at the grabbed body anchor; release/reset clears it. Physics with policy OFF falls under gravity.

Explicit PD torques in this browser are not bitwise equivalent to mjlab's implicit position actuators, contact solver or domain randomization. Interface parity does not establish balance, compliant behavior or hardware readiness. Background browser throttling can also reduce playback speed.

```bash
node demos/sonic_web/test_drag.mjs
```

```bash
node demos/ardy_web/test_csv_export.mjs
```

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  python demos/sonic_web/test_policy_contract.py --deploy "$BUNDLE/params/deploy.yaml"
```

`adapt_from_upstream.py` is the historical initial-port script. Do not run it over the maintained player: it would replace subsequent timeline, materials and ghost changes. `prepare.py` only rebuilds runtime assets. The custom SoftMimic adapter is maintained separately from the original SONIC controller.
