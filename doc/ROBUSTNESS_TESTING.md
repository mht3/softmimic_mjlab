# Standing-policy robustness checks

The browser's **Robustness lab** separates clean nominal physics, **Sensors only**, and a seeded **Training draw**. Apply/reset pauses the simulation and enables the matched policy; press Play. Reset repeats the draw, Next seed samples another. Individual groups and sensor channels can be disabled for paired comparisons. Nominal restores the launcher's contact profile. No control changes the running trainer or native deployment configuration.

Ranges were checked against the fresh 50k run's saved `params/env.yaml`, not inferred from the paper:

| Group | Uniform training distribution |
|---|---|
| Sensors at 50 Hz | Gyro ±0.2 rad/s; projected gravity ±0.01; joint position ±0.01 rad; joint velocity ±1.5 rad/s |
| Mass | Torso payload −2 to +2 kg; pelvis and six hip links ×0.7–1.3; inertia scales with mass |
| Torso center of mass | Each axis ±0.02 m |
| Actuator/joint dynamics | PD damping +0–2 N m s/rad; armature +0.01–0.1 kg m²; joint friction loss +0–0.01 N m |
| Joint limits | Each lower/upper bound ±0.02 rad |
| Foot friction | Shared sliding coefficient 0.5–2.0 |

Training draw also selects the training frictionless body-ground contacts. The prepared demo normally uses body friction 0.5 and feet 0.6. Self contacts remain disabled, matching training. Parameter draws are held until reset: they reproduce distributions, not training's GPU RNG or startup/reset schedule (training damping resets are gated at 720 steps). The policy retains its nominal offsets; standing references are interior to the randomized limits. The lab does not reproduce initial-pose randomization, motion sampling, or the IK-conditioned training forcefield.

Extra observation delay (20/40 ms) is a **stress test**, not a trained or measured delay. Palm-load buttons apply world-frame outward 20 N or downward 30 N per palm with 1 s ramp, 2 s hold, and 1 s release. They are modest reproducible perturbations, not claims about survival at the 140 N training force cap. Reset/fall clears loads. Drag forces remain available. The live wrist RMS measures simulated physical joint velocity, including transients since reset; it is not noise added to an encoder and is not a hardware measurement.

## Fresh 50k run: interim comparison, 26 October 2026

Native CPU MuJoCo, 30 s per trial, matched ONNX/YAML/reference, implicit PD. Basic evaluation: four load cases (none, left, right, downward), each clean seed 42 and noisy seeds 42/43/44. A pass requires survival plus final root-height error ≤5 cm, root XY drift ≤15 cm, palm drift ≤8 cm and mean tilt ≤0.2 rad, relative to the 3–5 s pre-load baseline. This penalizes failure to return after release, not just falling.

| Checkpoint | Basic passes /16 | Noisy standing wrist RMS | Velocity-noise-off RMS | Training-draw standing RMS | Training-draw passes /6 |
|---|---:|---:|---:|---:|---:|
| 34,000 | 16 | 0.313 rad/s | 0.141 rad/s | 0.085 rad/s | 6 |
| 39,000 | 15 | 0.335 rad/s | 0.162 rad/s | 0.084 rad/s | 6 |
| 40,000 | 13 | 0.405 rad/s | 0.189 rad/s | 0.108 rad/s | 5 |

The training-draw tests use seeds 42/43/44 for standing and downward loading. Noise ablation disables only joint-velocity noise and retains the same draws for other channels. All basic cases stayed upright, but 40k missed three recovery checks: right pull seed 44 (39.9 cm palm drift), downward seed 43 (12.1 cm), and downward seed 44 (18.9 cm). It also fell in the separate 40 ms delay stress test; 20 ms survived. 39k survived both delays. These finite tests favor 34k among these three, not a claim that it is globally best or hardware-ready. Checkpoint 34k is now the published default for both the demo and native deployment. The 40k bundle remains a local inspection snapshot.

The large reduction with randomized dynamics is a reason to test **both** nominal and randomized physics. Training adds armature even to wrists with nominal armature 0.001 kg m². A quiet randomized trial can hide sensitivity at nominal inertia/damping. The joint-velocity-noise ablation implicates that observation channel, but does not justify disabling real encoders or adding an unvalidated deployment filter.

## Reproduce

```bash
conda activate unitree_rl_mjlab
```

```bash
python demos/sonic_web/test_randomization.py --assets PATH_TO_PREPARED_ASSETS --training-config PATH_TO_RUN/params/env.yaml
```

```bash
python scripts/select_standing_checkpoint.py --bundles BUNDLE_40000 BUNDLE_39000 --baseline 39000 --minimum-iteration 40000 --maximum-iteration 40000 --output logs/basic_comparison.json
```

```bash
python scripts/evaluate_domain_randomization.py --bundles BUNDLE_40000 BUNDLE_39000 BUNDLE_34000 --seeds 42 43 44 --output logs/randomization_comparison.json
```

Use new output paths; reports refuse to overwrite existing evidence. Evaluation is CPU-only. The parameter test checks actual WASM mutations against native MuJoCo, not just UI labels. Other interface/fall tests are described in [deployment compatibility](../demos/DEPLOYMENT_COMPATIBILITY.md).

## Closing the hardware gap

No simulator can establish shake-free hardware behavior alone. First measure the actual G1's stationary and moving IMU/encoder noise, bias, timestamp jitter, latency and missing packets. Fit test distributions and replay measured sequences through the same observation/history path. Check actuator response, gains, torque limits, effective inertia and foot slip against controlled measurements. Independent white uniform training noise is not a substitute for these recordings.

Then evaluate held-out seeds, load directions/magnitudes and long rollouts, recording survival, recovery drift, loaded/unloaded wrist speed, action/torque rate, saturation and latency failures. Finally validate with a supported robot, low-force interactions, an operator and the existing fall/fresh-sensor shutdown. Gradually expand the tested envelope. The browser controls are diagnostic; they do not certify hardware operation or change its operator-controlled entry sequence.
