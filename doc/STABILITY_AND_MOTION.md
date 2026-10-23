# Standing stability and nominal-motion playback

The baseline task retains released SoftMimic reward weights and hybrid sampling. `Unitree-G1-23Dof-Compliant-Tracking-Stand-Stable` is an explicit new experiment, not a claim of exact release parity:

- 50% forcefield / 50% zero-wrench sampling. The static collision-emulator references contain no events and are omitted from sampling; all generated files still require the original full audit.
- Raw actions bounded to ±5 **before** physics, last-action observations and action-rate rewards. The same bound is exported to browser/C++. Observation bounds are also exported. Rewards are not clipped or smoothed.
- Applied force/torque bounded to 140 N / 10 Nm; divergent simulated velocities terminate an episode. Non-finite actor outputs stop the run. Raw-action peaks and outlier records expose saturation rather than hiding it.
- New IK references use maximum joint/wrist speeds of 3 / 1.5 rad/s, inside the constrained solver and whole-configuration step bound. Collision, foot-displacement, force/compliance and release-recovery thresholds remain unchanged. No post-filtering of collision-certified poses.
- Log per-update completion fractions alongside cumulative fractions. Compare frozen checkpoints on identical seeds with `scripts/evaluate_stand_checkpoint.py`; use `--task` to select the matching experiment. Wrist velocity and action-delta metrics supplement reward. Do not expect a perfectly smooth raw PPO curve.

Generate a new dataset; existing versions are preserved:

```bash
python scripts/generate_standing_from_csv.py --input datasets/streamer/stand.csv --name streamer_stand_smooth --max-joint-velocity 3 --max-wrist-velocity 1.5 --workers 4
```

Only after the complete dataset audit passes:

```bash
bash train_configs/compliance/stand_stable.sh
```

This starts a fresh 20,000-iteration run in W&B project `g1_compliance`; it does not resume the old policy. The currently published policy was trained with the baseline configuration, not these new settings.

The full `streamer_stand_smooth` generation passed on 2026-10-22: 85 pairs, 76,670 CSV frames and 127,670 converted frames, all finite. The audit found no sampled self-penetration; maximum joint speed was 3 rad/s, foot displacement 4.98 cm, force/displacement residual 4.97 cm, and settled-hand recovery error 2.06 mm. The foot and force/displacement maxima are close to their unchanged 5 cm gates. This is a kinematic dataset audit, not a dynamics or hardware validation.

A separate six-file dataset uses new forcefield seeds for frozen-checkpoint comparisons and is excluded from training. The final baseline iteration-19999 checkpoint completed all 12 evaluated 30-second episodes on that set without early termination. Loaded tracking remains less accurate than unforced standing; this small deterministic evaluation does not establish performance over arbitrary pulls.

Recreate that held-out set and evaluate a baseline checkpoint:

```bash
python scripts/generate_standing_from_csv.py --input datasets/streamer/stand.csv \
  --name streamer_stand_validation --num-files 4 1 1 --seed 100 \
  --max-joint-velocity 3 --max-wrist-velocity 1.5 --workers 2
```

```bash
python scripts/evaluate_stand_checkpoint.py --checkpoint PATH_TO_CHECKPOINT.pt \
  --motion-file src/assets/compliant_motions/g1_23dof/streamer_stand_validation \
  --device cpu --threads 2 --num-envs 12 --steps 1800 --seed 123 \
  --output logs/heldout_evaluation.json
```

For a checkpoint trained with the new profile, add `--task Unitree-G1-23Dof-Compliant-Tracking-Stand-Stable`. Keep the dataset, seed and episode settings fixed across checkpoints.

## Browser actuator correction

The browser now uses MuJoCo implicit PD actuators with the exported gains and force limits, matching training. Previously it computed explicit PD torques at 200 Hz; the small wrist inertia made this numerically problematic. `scripts/evaluate_browser_actuation.py` compares both implementations with identical ONNX weights on CPU. Physics-only mode disables actuator gains/biases. This does not change the hardware servo gains.

## Moving reference support

Both training tasks observe six nominal channels: joint pose, local root XY velocity, local yaw velocity, root height, gravity and foot contacts. The generic `Unitree-G1-23Dof-Compliant-Tracking` task exports `nominal_motion`; the standing task exports `constant_nominal`. The C++ and browser adapters now support both. A standing policy is still rejected when paired with a changing CSV; changing its YAML mode does not train motion tracking.

For a policy actually trained on moving references, preserve the converter's **nominal** velocity/contact channels in a deployment NPZ. Never substitute adapted IK targets:

```bash
python demos/sonic_web/nominal_motion.py --source-npz PATH_TO_CONVERTED_TRAINING_CLIP.npz --output /tmp/nominal.npz --duration 30
```

```bash
python demos/package_motion_policy.py --run PATH_TO_MOTION_RUN --reference /tmp/nominal.npz --output /tmp/motion_bundle
```

```bash
python demos/check_cpp_policy_contract.py --bundle /tmp/motion_bundle
```

```bash
python demos/sonic_web/serve.py --policy /tmp/motion_bundle/exported/policy.onnx --deploy /tmp/motion_bundle/params/deploy.yaml --reference /tmp/motion_bundle/params/reference.csv --nominal-motion /tmp/motion_bundle/params/nominal.npz --port 8088
```

The packager writes a matching nominal CSV; launch validates it against the NPZ segment. Native `State_Mimic` uses that same NPZ and selects the same frame for all six channels. The moving-reference path has interface tests, not a newly trained/evaluated walking controller. Test moving-policy behavior separately before deployment.
