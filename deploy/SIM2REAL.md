# Simulation and hardware contract

The actor runs at **50 Hz** with three historical samples per observation term. `deploy.yaml` supplies term order, clipping/scaling, nominal commands, motor gains, action scaling and hardware joint indices. The Stable policies clip raw actions to ±5 in both runtimes, before last-action history is recorded. Do not smooth motor targets or observations only on the robot: that introduces a distribution change.

| Actor input | Hardware | Browser |
|---|---|---|
| Body angular velocity | Pelvis `rt/lowstate.imu_state.gyroscope` | Body-frame MuJoCo angular velocity |
| Projected gravity | Normalized WXYZ pelvis IMU quaternion, inverse-rotated gravity | Same calculation from pelvis orientation |
| Joint position/velocity | `motor_state.q/dq`, gathered by the 23-joint hardware map | Corresponding joint coordinates/velocities |
| Previous action | Previous bounded actor output | Same |
| Nominal pose, root XY/yaw velocity, height, gravity, contacts | Exported reference commands | Same reference commands |
| Desired compliance | Remote d-pad stiffness command | Stiffness sliders |

No global position, actual base linear velocity, actual contact forces or applied hand wrench is fed to this actor. A tug changes proprioception; the policy responds to that change. Reference height/contact values are desired quantities, not unavailable sensor measurements. Unitree's [G1 low-level example](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/g1/low_level/g1_ankle_swing_example.cpp) identifies the low-state IMU as the pelvis IMU; do not substitute the secondary torso IMU.

## Changes in this deployment review

- Replaced the stale compliant-mimic policy/configuration with the matched Stable 24k export and included CSV. Portable paths and a matching FixStand pose replace the unrelated velocity/static-balance route.
- Constant-standing policies consume nominal channels from the YAML without a generated training NPZ. Moving policies still require their own prepared nominal-motion NPZ.
- Reject nonfinite encoders/gyro and invalid IMU quaternion norms; normalize accepted quaternions before gravity projection.
- Hold measured joint positions until the first inference completes. Inference exceptions and a 100 ms policy deadline fault request Passive; the output loop immediately removes proportional drive and uses damping on a fault. Low-state timeout is reduced from the wrapper's 1 s default to 100 ms. This is a software fallback, not an independent emergency stop.
- Use a monotonic policy clock and avoid rapid catch-up inference after an overrun. The 1 kHz command loop checks a fresh locked pelvis-IMU snapshot before publishing targets, independently of inference.
- Refuse startup if another process owns the low-command channel; do not shut down another controller automatically.

## What the browser models

The physics model uses exported PD gains, effort saturation, the training armatures and foot contacts, 5 ms implicit integration and 20 ms target updates. Non-foot body-ground contacts default to sliding friction 0.5, instead of the frictionless `condim=1` training contacts; feet remain at 0.6. The body coefficient is an assumption, not a measurement. Use `--contact-profile training` for the original contact model, or `--body-friction` to vary the body coefficient. The browser samples a pelvis IMU and joint encoders before each actor step. **Training-range sensor noise** defaults ON: uniform ±0.2 rad/s gyro, ±0.01 projected gravity, ±0.01 rad position and ±1.5 rad/s velocity. These bounds reproduce the training actor corruption; they are not measurements of this robot. The optional extra 20/40 ms observation delay is a stress test, defaults to zero, and does not delay reference commands or previous actions. Reset clears history and delay queues.

The viewer advances simulation time around asynchronous inference. It can run slower than wall-clock time on a slow computer; it does not demonstrate meeting the real-time controller deadline. Rendering and the reference ghost may use full simulation state, but the actor's live observations do not.

## Remaining measured work

Real motor firmware closes the PD loop, while MuJoCo approximates it using implicit actuators and configured effort limits. Firmware torque/current/temperature limits, transmission friction/backlash, battery effects, motor response delay, foot-ground friction, payload and IMU mounting/bias still require measurement. Browser collision/contact masks match training and are not a full hardware collision model. Do not equate its idealized torque saturation or contact behavior with firmware guarantees.

Validate the exact new bundle with standing, hand pulls, downward loads, release recovery and noise/delay tests. Then measure inference deadlines and sensor arrival intervals on the deployment computer, verify the pelvis IMU frame/sign and joint mapping while suspended, and perform staged physical evaluation using Unitree's procedures. Fit any new sensor filtering or delay model to those measurements and retrain if needed. No physical robot was commanded or hardware behavior validated during this review.

## Recorded 14k CPU checks

All four unforced 30-second rollouts completed using seed 42. These use native CPU MuJoCo, not physical hardware.

| Input condition | Minimum pelvis height | Wrist velocity RMS |
|---|---:|---:|
| Clean | 0.785 m | 0.155 rad/s |
| Training-range noise | 0.782 m | 0.774 rad/s |
| Noise + 20 ms delay | 0.783 m | 0.632 rad/s |
| Noise + 40 ms delay | 0.782 m | 0.634 rad/s |

Noise noticeably increases wrist jitter; standing completion alone does not validate hand compliance. The recorded results are in the bundle's `sensor_robustness.json`. Reproduce one condition from the repository root (change `--delay-steps` to 0 or 2; omit `--sensor-noise` for clean inputs):

```bash
python scripts/evaluate_browser_actuation.py \
  --bundle deploy/robots/g1_23dof/config/policy/compliant_mimic/stand \
  --mode implicit --seconds 30 --sensor-noise --delay-steps 1 \
  --output logs/deployment_sensor_check.json
```

## Checkpoint comparison and fall cutoff

The historical comparison below used **15k**; the current bundle is **24k**. In identical 30-second unforced CPU rollouts, wrist-speed RMS was:

| Checkpoint | Clean sensors | Training-range noise, mean of seeds 42/43/44 |
|---|---:|---:|
| 14k | 0.155 rad/s | 0.774 rad/s |
| 14.5k | 0.155 rad/s | 0.680 rad/s |
| 15k | 0.115 rad/s | 0.443 rad/s |

All twelve rollouts completed. At that stage, this supported choosing 15k for reduced wrist motion, not claiming that it is jitter-free or better at every interaction. Raw results are in `config/policy/compliant_mimic/stand/checkpoint_comparison.json` under `robots/g1_23dof`. The earlier delay-test table above is specifically for 14k.

The native controller already had a **1.25 rad (71.6°)** tilt-to-Passive transition. It now detects this before command publication and immediately sets `kp=0`, `kd=3`, zero desired velocity and zero feed-forward torque. The policy worker stops and the FSM transitions to Passive; re-entry requires the operator's FixStand/Mimic sequence. It also rejects an already tilted pose when entering Mimic.

The browser uses the same threshold from the bundle's controller configuration (1.25 rad fallback), checks before and after each 5 ms physics substep, and latches the actor OFF. Physics continues with damping-only actuation so the robot settles; crossing upright again cannot reactivate the policy. In the browser, **Reset** or **R** restores the reference pose and turns the matched policy and physics ON; press **Play** to resume. The native hardware controller retains its operator-controlled FixStand/Mimic sequence. Seeking after a fall also clears the latch but leaves policy control OFF. Stale inference failures cannot relatch a reset session. Invalid orientation or inference errors also disable drive. No global-height fall detector is used: the hardware has no validated height estimator, and intentional compliant squats must not be treated as falls. An upright collapse without exceeding the tilt limit is therefore not covered by this detector.

Regression tests exercise roll/pitch thresholds, yaw-only rotation, invalid IMU data, damping output, substep shutdown and manual reset/re-enable. They do not establish an independent hardware emergency-stop guarantee.

## Completed-run comparison (18–20k)

The resumed run reached iteration 19999/20000. Five late checkpoints and the 15k baseline were evaluated in native CPU MuJoCo with the current body-friction profile, 30-second trials, clean inputs and noise seeds 42/43/44. Each policy received 16 basic trials: standing, a 20 N outward pull on either palm, and 30 N downward on each palm. Loads ramped over 1 s, held for 2 s, and released over 1 s. Separate 20/40 ms delay trials and a training-contact parity trial were also recorded. These moderate tests do not establish performance at the 140 N training cap.

| Checkpoint | Basic trials passed | Clean wrist RMS | Noisy wrist RMS, mean |
|---|---:|---:|---:|
| 15000 | 16/16 | 0.115 rad/s | 0.443 rad/s |
| 18000 | 16/16 | 0.114 rad/s | 0.505 rad/s |
| 18500 | 16/16 | 0.102 rad/s | 0.497 rad/s |
| 19000 | 13/16 | 0.099 rad/s | 0.472 rad/s |
| 19500 | 16/16 | 0.104 rad/s | 0.516 rad/s |
| 19999 | 16/16 | 0.077 rad/s | 0.462 rad/s |

19k remained upright but missed three downward-load recovery checks (palm/root position drift). The best late candidate, 19999, passed all basic tests and reduced clean wrist motion by 33%, but increased noisy wrist motion by 4.3% relative to 15k. No default policy replacement was made at that stage. All six policies remained upright in the separate delay tests. Full results and predeclared recovery limits are in the published bundle’s `late_checkpoint_comparison.json`; reproduce with `scripts/select_standing_checkpoint.py --help`. These results are not hardware validation.

## Full 15–20k review

All eleven saved checkpoints have now been evaluated, including 15.5k–17.5k, alongside the W&B curves and five fresh noise seeds for the four leading candidates. That review selected 15k; the subsequent 11k–24k review selected 24k. See [full comparison and training curves](../doc/CHECKPOINT_SELECTION.md); the small differences among 15k/16.5k/17k are not a calibrated hardware ranking.


## Current default: 24k

Checkpoint 24,000 is installed in both the browser and native deployment bundle. Across 28 candidates it had the lowest noisy wrist-speed RMS among candidates passing the recovery gates. It passed 30/30 standing/recovery trials and survived six separate delay stress trials; eight-seed wrist RMS was 0.385 rad/s (15k: 0.443). The ONNX, YAML, CSV and interface checks are matched. See [noise investigation and selection](../doc/WRIST_NOISE_REVIEW.md) and the bundle’s `checkpoint_selection_11_24k.json`. Existing fall latch, fresh-sensor checks, FixStand pose and portable controller paths are preserved. No hardware validation is claimed.


The final 25k checkpoint (`model_24999.pt`) was evaluated after completion. It passed 26/30 standing/recovery trials versus 24k’s 30/30, and had eight-seed wrist RMS 0.444 versus 0.385 rad/s. Four downward-load trials missed palm-position recovery tolerance without falling. Both survived six delay stress tests. The default remains 24k; see the bundle’s `final_25k_comparison.json`.
