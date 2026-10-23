# Compliant Humanoid · fixed reference policy player

The maintained browser player uses the published [GEAR-SONIC Three.js/MuJoCo-WASM demo source](https://nvlabs.github.io/GEAR-SONIC/demo.html), adapted for G1 23DOF and SoftMimic. It preserves physics, camera controls, body picking and spring drag forces. The local UI adds a fixed bottom reference timeline and a separate cyan ghost. Original source files and URL/SHA256 provenance remain in `upstream/` and `upstream_manifest.json`; retain their notices and upstream terms.

## Run the included policy

From the repository root, after the [conda setup](../../README.md#installation-and-environment):

```bash
conda activate unitree_rl_mjlab
```

```bash
python demos/sonic_web/serve.py --port 8088 \
  --reference deploy/robots/g1_23dof/config/policy/compliant_mimic/stand/params/reference.csv \
  --deploy deploy/robots/g1_23dof/config/policy/compliant_mimic/stand/params/deploy.yaml \
  --policy deploy/robots/g1_23dof/config/policy/compliant_mimic/stand/exported/policy.onnx
```

Open http://127.0.0.1:8088. The included policy is the [selected Stable checkpoint 24,000](../../deploy/robots/g1_23dof/config/policy/compliant_mimic/stand/README.md), with its matching deployment YAML and bundled browser runtimes. Generated assets are ignored; the command above rebuilds them from the small standing CSV and source meshes. For another controller, use the training environment and follow the [bundle and validation commands](../../WORKFLOW.md#4-package-and-visualize-the-trained-policy), then supply all three matched files:

```bash
python demos/sonic_web/serve.py --port 8088 \
  --reference "$BUNDLE/params/reference.csv" \
  --deploy "$BUNDLE/params/deploy.yaml" \
  --policy "$BUNDLE/exported/policy.onnx"
```

The launcher validates all three inputs before serving; there is no in-page upload. Control starts OFF. Enable the policy and press Play; enabling it also enables physics. Space toggles playback. Reset or R restores the reference pose, turns the matched policy and physics ON, and pauses playback; press Play to run. The player never starts hardware. Do not install an early diagnostic checkpoint as a completed policy.

The reference is required. Save CSV metadata specifies named joints, XYZW quaternion order and frame rate; headerless prepared 32-column references use 30 Hz unless `--reference-fps` is supplied. This standing policy requires its matching constant nominal reference. A moving-policy export additionally requires `--nominal-motion` with prepared nominal channels; see [motion support](../../doc/STABILITY_AND_MOTION.md#moving-reference-support). Invalid pairs fail at launch.

The fixed track supports play/pause, speed, click-to-seek and a slider. Seeking pauses and resets physics/history to that pose; it is not a recording of past simulation. The reference ghost has separate FK data and no collision/force contribution. Toggle its offset to overlay it. The physical G1 uses its MJCF black/silver materials and original mesh detail; these STL assets do not contain image texture maps.

## Simulation and interface limits

The actor has 327 inputs and 23 actions at 50 Hz. Per-term histories contain three frames, oldest first, with reset backfill. Local angular velocity, body-frame gravity, reference channels, normalization, action scaling/offsets and PD gains match the exported contract. Stiffness defaults to 200 N/m and 1 Nm/rad; these are policy inputs, not motor gains.

MuJoCo runs at 5 ms with four substeps per control update. Joint order, limits, armatures and collision masks follow the training robot. Robot-ground collisions are enabled; robot self contacts are disabled as in the training configuration. The default `--contact-profile frictional-body` adds sliding friction to body-ground contacts (assumed coefficient 0.5); feet retain the training coefficient 0.6. This prevents frictionless sliding after falls. `--body-friction` adjusts the assumption; `--contact-profile training` reproduces the original frictionless non-foot contacts. These coefficients are not hardware calibrated. The offline IK collision audit is a separate gate. Each physics substep recomputes drag force and its moment at the grabbed body anchor; release/reset clears it. Physics with policy OFF falls under gravity.

The browser uses implicit PD actuators with the exported gains and force limits. MuJoCo-WASM precision, contact solving and the absence of training domain randomization can still differ from mjlab. Interface parity does not establish balance, compliant behavior or hardware readiness. Background browser throttling can also reduce playback speed.

```bash
node demos/sonic_web/test_drag.mjs
```

```bash
node demos/ardy_web/test_csv_export.mjs
```

```bash
python demos/sonic_web/test_policy_contract.py --deploy "$BUNDLE/params/deploy.yaml"
```

`adapt_from_upstream.py` is the historical initial-port script. Do not run it over the maintained player: it would replace subsequent timeline, materials and ghost changes. `prepare.py` only rebuilds runtime assets. The custom SoftMimic adapter is maintained separately from the original SONIC controller.

### Fall cutoff

At the matched hardware tilt limit (default 1.25 rad / 71.6°), policy control latches OFF and simulation continues with damping only. Press **Reset** (or **R**) to restore the pose and turn the matched policy and physics ON; playback stays paused until **Play**. Seeking to a reference pose after a fall also clears the latch, leaving the actor OFF and available to enable. Late inference results/errors from before reset are discarded. Returning upright does not restart inference. The detector uses pelvis orientation, not world height, so deliberate squats remain allowed. See [checkpoint comparison and sensor limitations](../../deploy/SIM2REAL.md).
