# Workflow verification (2026-10-22)

- Streamer CSV export preserves 23 named joints, XYZW root quaternions, frame rate, trims, world alignment and editor transitions. Timeline, blend, CSV and drag-force tests pass.
- The new slower standing dataset passed the complete 85-pair audit: 76,670 CSV frames and 127,670 converted frames. No sampled self-penetration; all converted arrays finite. Original collision, foot, force/displacement and recovery gates remain unchanged. See [stability settings and audit limits](../doc/STABILITY_AND_MOTION.md).
- A separate six-file dataset uses new forcefield seeds. The final baseline iteration-19999 policy completed all 12 deterministic 30-second evaluation episodes without early termination. Loaded tracking error remains higher than unforced standing; this is a small held-out check, not arbitrary-force validation.
- The bounded, rebalanced training profile passed a CPU rollout smoke check. No new GPU training was started. It requires freshly audited 3 rad/s joint / 1.5 rad/s wrist references and logs to W&B `g1_compliance`.
- 109 repository regression tests passed before the additional CSV/nominal-pair checks; the focused nominal suite then passed all three tests. The native 23DOF controller builds without opening hardware channels.
- A synthetic moving-reference fixture passed all 327 C++/JavaScript observations, history/reset behavior, raw-action bounds and ONNX target comparisons; maximum action difference was 4.77e-6. This verifies the moving-reference adapter, not a trained walking controller.
- Browser actuation now uses implicit PD with the exported gains/force limits. Native CPU MuJoCo comparison found much lower wrist velocity than the former explicit-PD integration. Browser and hardware behavior require separate evaluation.
- Launch requires `--policy`, `--deploy` and `--reference`; CSV upload is removed. Missing files, incompatible pairs and mismatched CSV/nominal NPZ segments are rejected.

Generated IK data, audits and local diagnostic reports remain ignored. The published policy bundle contains its export and interface reports. No active hardware configuration was replaced and no robot controller was started.

## Native ARDY constraint authoring

- Six native constraint tracks tested: root XY, full-body, both hands and both feet. The tests exercise named 23→29 joint mapping, native coordinate/rotation round trips, normalized masks, bounded future context, history masking, padding, duplicate goals and conflicting targets. These checks verify the adapter; they do not establish every prompt/goal combination's motion quality.
- Browser exercised floor-click placement, spatial sphere dragging, keyframe time dragging (2.00 → 2.52 s), captured hand offset editing, full-body pose capture and deletion. The user's saved-motion edit was preserved independently.
- Real four-second CPU waypoint generation: `../sonic23/ardy_motions/835394730ea34ef1ad24525b3b5cf26e/`. Geometry passed; requested root position error 0.00453 m after 23DOF retargeting.
- Real 12-second CPU circle: `../sonic23/ardy_motions/302e2e83ada048f2a7b42ea4d0997193/`. All 1,197 emitted frames and interpolated path samples passed geometry checks. Maximum error across 13 timed root goals: 0.03660 m. Total CPU time: 127.0 s. Conservative palm tracking-error bound 0.12427 m exceeds the 0.12 m fidelity-review threshold, so the clip correctly retains a fit warning. Root-goal fit is below its review threshold. No dynamics validation is implied.
- Example plans reproduce the mechanisms in ARDY's published examples, using its native condition classes and history-aware generation; they are not the original demo's saved motions or its SONIC controller.
- The 16-second side-step → jog → jump → stand test (seed 42, `../sonic23/ardy_motions/3598cd5d8f764bda824dccf7e0fd9618/`) passed its first 4.16-second prefix, then failed the 8.32-second retarget with `Retarget requires excessive ground correction: 0.0374 m`. The 3 cm gate was retained, and no full clip was published. This preset is an authoring example, not a validated motion asset.

## Published final policy

The baseline run completed all 20,000 iterations and synced to W&B. Exact checkpoint `model_19999.pt` was exported on CPU, packaged as `demos/policies/stand_20000`, and passed both interface checks. The final ONNX is installed byte-for-byte in the browser assets. The former intermediate demo bundle was removed. This completed baseline does not contain the later bounded-action, rebalanced-sampling or slower-reference training changes.

The final policy completed the four-second browser reference while remaining upright. Reset returned the timeline to zero; policy and physics were left OFF. The updated screenshot is `doc/images/policy_player.png`. This browser smoke check is separate from the 20-second native MuJoCo PD comparison and 12 held-out training-environment episodes.
