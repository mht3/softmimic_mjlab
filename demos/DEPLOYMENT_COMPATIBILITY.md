# Browser and native deployment contract

Use one matched bundle: ONNX policy, deployment YAML and nominal reference. The [main README](../README.md#run-the-included-policy-demo) identifies the published checkpoint; the [workflow](../WORKFLOW.md#4-package-and-visualize-the-trained-policy) covers packaging a new run.

## Shared interface

The actor has 327 inputs and 23 outputs at 50 Hz. Its 13 observation terms each retain three frames, oldest first, with reset backfill. Both runtimes use exported observation scaling/clipping, action scaling/offsets, optional raw-action bounds, PD gains and named hardware joint order. Normalization, when enabled during training, remains inside the ONNX graph.

The six nominal channels are joint pose, local root XY velocity, local yaw velocity, root height, gravity and foot contacts. Standing exports use `constant_nominal`; moving-reference exports use `nominal_motion` and a prepared 50 Hz nominal NPZ. See [motion support](../doc/STABILITY_AND_MOTION.md#moving-reference-support). A moving reference does not turn a standing policy into a motion tracker.

The supported native runtime is `deploy/robots/g1_23dof`, with `State_Mimic`. Use the bundle's generated controller configuration after simulation validation. Do not mix this contract with the separate legacy 29DOF Python deployment path. Packaging never starts hardware or replaces the active robot configuration.

## Validate a bundle

From the repository root:

```bash
conda activate unitree_rl_mjlab
```

```bash
BUNDLE=/absolute/path/to/matched/bundle
```

```bash
python demos/check_cpp_policy_contract.py --bundle "$BUNDLE"
```

For a standing bundle, also compare the browser adapter against mjlab's quaternion/history utilities:

```bash
python demos/sonic_web/test_policy_contract.py --deploy "$BUNDLE/params/deploy.yaml"
```

The C++ check exercises all 327 observation entries, history rollover/reset, ONNX actions and motor targets against the JavaScript adapter. The moving-reference variant also changes nominal frames during the fixture. These are interface tests, not learned-behavior tests.

## Simulation

```bash
python demos/sonic_web/serve.py --port 8088 \
  --policy "$BUNDLE/exported/policy.onnx" \
  --deploy "$BUNDLE/params/deploy.yaml" \
  --reference "$BUNDLE/params/reference.csv"
```

The launcher requires all three files and validates the pair before serving. Policy control starts OFF. The reference timeline is fixed; there is no browser upload. For a moving policy, additionally supply its prepared `--nominal-motion` NPZ and matching CSV.

The browser uses implicit PD with exported gains and force limits. WASM precision, contact solving and the lack of training domain randomization can still differ from mjlab. `scripts/evaluate_browser_actuation.py` compares explicit and implicit PD using the same ONNX weights in native CPU MuJoCo; it does not itself test browser WASM or hardware.

Evaluate unforced standing, hand pulls, downward loads, release recovery, repeated resets and commanded stiffness changes. Dataset geometry checks and interface parity do not certify balance, compliant behavior or hardware readiness. Follow [real-robot deployment](../deploy/README.md) only after behavioral evaluation.
