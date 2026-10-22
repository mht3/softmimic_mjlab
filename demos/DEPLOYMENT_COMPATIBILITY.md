# Standing policy: browser and G1 23DOF deployment

The current fresh run is `2026-10-22_08-06-12_collision_recovery_stand_20k_fresh_20261022`, with 4096 environments and online W&B project `g1_compliance`: https://wandb.ai/mht3/g1_compliance/runs/xklelhb4. The user requested a restart from scratch; the prior continuation was intentionally stopped. This run loads no checkpoint and initializes new actor, critic and optimizer state, starting at iteration 0 and ending at 19999 (20,000 updates). Command: `softmimic_mjlab/logs/fresh_20k_20261022.sh`. Log: `softmimic_mjlab/logs/compliance_20k_fresh_20261022.log`. The reward, dataset and reset semantics are unchanged; per-mode health metrics remain enabled.

Frozen-checkpoint evaluation before continuation covered all 85 unique training clips. Unforced modes reached the full 30 seconds in every completed episode. Forcefield completion was 60.7% with deterministic actions and 56.9% with exploration/noise; deep crouch compliance is still undertrained. This justified continued optimization, **not deployment approval**. Full reports and the continuation rationale are in `softmimic_mjlab/logs/training_audit_20k_20261022/`. No early checkpoint was installed into the browser or hardware configuration.

## Included intermediate browser snapshot

At the user's request, exact checkpoint `model_9000.pt` from this fresh run was exported separately on CPU and packaged in `demos/policies/stand_9000/`. Its identical ONNX is installed in the browser with the matching YAML-derived configuration and standing reference. Native C++ and browser contract checks pass; see the bundle's `export_report.json` and `interface_validation.json`. This supersedes the earlier browser-without-policy state described below. Training continues unchanged; no hardware configuration was replaced.

## Verified contract

The actor has 327 inputs (13 ordered observation terms, three frames per term) and 23 outputs. Per-term histories are oldest to newest, with reset backfill. It runs at 50 Hz. Both browser and native deployment use the exported nominal standing reference, action scales/offsets, PD gains and hardware joint order. Normalization is inside the ONNX graph. Compliance commands default to 200 N/m and 1 Nm/rad in both bundles; they are distinct from motor PD gains.

The supported real-robot runtime is `softmimic_mjlab/deploy/robots/g1_23dof`, with `State_Mimic` and the `Mimic_Compliant_Stand` state. Do not mix this export with the older checked-in policy YAML or the separate legacy `softmimic_deploy` Python policy interface.

Validation completed on an immutable early-run diagnostic export:

- Native C++ observation/action managers versus browser JavaScript: all 327 observations across six frames, including history rollover and reset.
- Native ONNX Runtime versus CPU ONNX ReferenceEvaluator: maximum action error 8.96e-8; motor targets match exported scaling and offsets.
- Malformed observation/action dimensions and non-finite observations are rejected.
- Seven standing-export/control-parity regression tests pass.
- Full G1 23DOF C++ controller compiles in `/tmp/compliance-g1-23dof-build` without running it or opening hardware channels.
- Browser adapter versus actual mjlab quaternion/history utilities passes. Separate browser WASM ONNX diagnostic previously matched CPU within 1.2e-7.

Fixed during this review: stale processed motor targets on reset; unchecked ONNX observation lengths; missing non-finite observation/action checks; unsynchronized action reads/writes; non-atomic compliance-thread flags and stiffness commands. Policy inference exceptions in the 23DOF Mimic thread now latch a fault requesting the existing Passive state. Browser installer checks hardware ordering and embeds external ONNX weights into a single file.

`softmimic_mjlab/logs/deployment_interface_test_20k` is a diagnostic snapshot, **not** a behaviorally validated controller. No hardware has been commanded. These checks establish software interface compatibility, not balancing/compliance performance or hardware readiness. Browser physics also differs from training's implicit actuators, domain randomization and contact solver implementation.

## After training finishes

From `/home/mht/research/rl/humanoid/compliance`, package the final ONNX and its own deployment YAML together. The tool refuses an existing output directory and creates a self-contained ONNX, exact YAML, constant nominal `stand.npz`, hashes and an FSM configuration patch:

```bash
COMPLIANCE_RUN=/home/mht/research/rl/humanoid/compliance/softmimic_mjlab/logs/rsl_rl/g1_23dof_compliant_tracking/2026-10-22_08-06-12_collision_recovery_stand_20k_fresh_20261022
COMPLIANCE_BUNDLE=/home/mht/research/rl/humanoid/compliance/softmimic_mjlab/deploy/robots/g1_23dof/config/policy/compliant_mimic/stand_20k_final
/home/mht/miniconda3/envs/unitree_rl_mjlab/bin/python humanoid_demo/package_standing_policy.py \
  --run "$COMPLIANCE_RUN" --output "$COMPLIANCE_BUNDLE" \
  --reference softmimic_mjlab/datasets/streamer/stand.csv \
  --dataset softmimic_mjlab/src/assets/compliant_motions/g1_23dof/stand
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/mht/miniconda3/envs/unitree_rl_mjlab/bin/python humanoid_demo/check_cpp_policy_contract.py \
  --bundle "$COMPLIANCE_BUNDLE"
```

For the browser, install those **same bundled files** and refresh http://127.0.0.1:8088:

```bash
/home/mht/miniconda3/envs/unitree_rl_mjlab/bin/python humanoid_demo/sonic_web/prepare.py \
  --reference "$COMPLIANCE_BUNDLE/params/reference.csv" \
  --deploy "$COMPLIANCE_BUNDLE/params/deploy.yaml" \
  --policy "$COMPLIANCE_BUNDLE/exported/policy.onnx" \
  --stiffness 200 --rotational-stiffness 1
```

For the native 23DOF controller, merge the generated `fsm_patch.yaml` fields into `deploy/robots/g1_23dof/config/config.yaml` after validating the policy in simulation. This selects the same bundle, static reference and compliance commands without substituting the old policy gains/layout. The package does not enable hardware or modify the controller's active configuration.

Evaluate standing without forces, both-hand pulls, bounded downward loads, release recovery, repeated resets and different commanded stiffnesses before a hardware trial. A kinematic dataset replay is not this evaluation. After completion, verify `model_19999.pt`, package its matching export and run both interface tests before installing it. The earlier interrupted-run monitor was deleted; this continuation does not itself install a controller or start hardware.
