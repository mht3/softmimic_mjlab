# Streamer → standing workflow verification (2026-10-22)

- Save CSV exercised in the browser with the fixed standing reference. The file uses 23 named joints, XYZW root quaternions and 100 Hz. Tests cover trims, quaternion round trip, aligned joins, malformed input and incompatible references.
- Browser verified black/silver MJCF materials, fixed timeline, playback, physics fall with no controller, independent cyan ghost, and reset. Screenshots are under `doc/images/`.
- Three CPU reference/import tests pass; four JavaScript test files pass (timeline, blend, CSV and drag forces).
- Direct IK and converted all-frame audit pass for one file in each of the three modes, under `src/assets/compliant_motions/g1_23dof/.streamer-cltrsst7/verification_streamer_stand/audit.json`. The original verification process ended with SIGTERM after conversion of the first two modes; the remaining zero-wrench conversion and complete three-file audit were run separately. This is not the full 85-file generation or a training approval.
- Training launcher tested with a stub replacing the trainer: full 85-file audited dataset accepted; 3-file verification rejected; arguments specify fresh 20k training and online W&B `g1_compliance`. No new training was launched.
- Diagnostic bundle packaging, standing-only FSM and native C++ versus JavaScript parity passed. All 327 observations matched; maximum ONNX action difference 8.96e-8. Browser adapter also matched mjlab's actual quaternion/history utilities. Temporary diagnostic bundle: `/tmp/streamer-package-diagnostic-tbjyumpk/bundle`. It was not installed into viewer assets or hardware configuration.
- Existing SoftMimic training was not restarted or modified. No GPU inference or hardware control was launched. The active player contains a reference only and awaits a completed trained policy.

The supported end-to-end deployment contract is constant standing. Moving edits can be exported, but require motion-specific task/export/deployment work; the standing pipeline rejects them. Geometry checks, software interface parity and learned physical behavior are separate validations.

## Native ARDY constraint authoring

- Six native constraint tracks tested: root XY, full-body, both hands and both feet. The tests exercise named 23→29 joint mapping, native coordinate/rotation round trips, normalized masks, bounded future context, history masking, padding, duplicate goals and conflicting targets. These checks verify the adapter; they do not establish every prompt/goal combination's motion quality.
- Browser exercised floor-click placement, spatial sphere dragging, keyframe time dragging (2.00 → 2.52 s), captured hand offset editing, full-body pose capture and deletion. The user's saved-motion edit was preserved independently.
- Real four-second CPU waypoint generation: `../sonic23/ardy_motions/835394730ea34ef1ad24525b3b5cf26e/`. Geometry passed; requested root position error 0.00453 m after 23DOF retargeting.
- Real 12-second CPU circle: `../sonic23/ardy_motions/302e2e83ada048f2a7b42ea4d0997193/`. All 1,197 emitted frames and interpolated path samples passed geometry checks. Maximum error across 13 timed root goals: 0.03660 m. Total CPU time: 127.0 s. Conservative palm tracking-error bound 0.12427 m exceeds the 0.12 m fidelity-review threshold, so the clip correctly retains a fit warning. Root-goal fit is below its review threshold. No dynamics validation is implied.
- Example plans reproduce the mechanisms in ARDY's published examples, using its native condition classes and history-aware generation; they are not the original demo's saved motions or its SONIC controller.
- The 16-second side-step → jog → jump → stand test (seed 42, `../sonic23/ardy_motions/3598cd5d8f764bda824dccf7e0fd9618/`) passed its first 4.16-second prefix, then failed the 8.32-second retarget with `Retarget requires excessive ground correction: 0.0374 m`. The 3 cm gate was retained, and no full clip was published. This preset is an authoring example, not a validated motion asset.

## Published intermediate policy snapshot

- Exact fresh-run `model_9000.pt` exported on CPU with the native RSL-RL MLP export wrapper. Strict actor-state loading includes the saved normalization setting (disabled for this run). Checkpoint SHA256: `16d842deb58ac9df94d6b586a6a7c3221959490e1136cc96648e1371efa19d7a`.
- The immutable snapshot is `demos/policies/stand_9000/`; the identical ONNX bytes are installed in `demos/sonic_web/assets/policy/policy.onnx`. Both are explicitly unignored. Export/PyTorch comparison passed 32 inputs; native C++ and browser observation/action contract checks passed. Reports are in the snapshot directory.
- Browser loaded the actual ONNX, confirmed its reference and completed a four-second unforced physics rollout while remaining upright. This short smoke check does not validate external-force response. It starts with policy control OFF. This is an intermediate trained controller, distinct from the earlier diagnostic snapshot, and not a completed behavioral or hardware evaluation. Ongoing training and active hardware configuration were unchanged.
