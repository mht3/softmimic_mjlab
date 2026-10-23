# Wrist motion and checkpoint review

CPU review of 28 saved checkpoints from 11,000 through 24,000, including 19,999.
The current recommendation is **24,000**; it is now installed as the demo and deployment default.
Training settings and data were not changed. Installation updated the matched policy bundle and started the browser preview; no hardware controller was run.

| Candidate | Noisy wrist speed RMS, 8 seeds (rad/s) | Basic + held-out standing/recovery tests | Delay stress survival |
|---|---:|---:|---:|
| **24,000** | **0.385** | **30/30** | **6/6** |
| 13,000 | 0.408 | 30/30 | 6/6 |
| 11,500 | 0.415 | 30/30 | 6/6 |
| 15,000 | 0.443 | 30/30 | 6/6 |

24k reduces this wrist-motion measure by 13.1% and wrist action-rate RMS by 14.3%
relative to 15k. 13k has lower wrist action-rate RMS than 24k (10.40 versus 10.84),
but 24k has lower physical wrist speed; the ranking prioritizes the latter after
passing recovery checks. 13k and 11.5k are alternatives; this finite comparison does not
establish a universal hardware ranking. Wrist speed RMS includes transient and
posture motion, rather than isolating a particular vibration frequency.

The same 30-second tests use body friction 0.5, feet 0.6, identical nominal CSV,
PD gains and control rates, and training-range sensor corruption. Moderate loads
are 20 N outward at one hand or 30 N downward at each hand, applied from 5–9 s
with one-second ramps. Recovery thresholds were unchanged. Delay stress is
reported separately: 20/40 ms, seeds 42/43/44. Basic noisy seeds are 42/43/44;
held-out standing seeds are 100–104 and load seeds are 100–102.

## Why later checkpoints can shake more

The saved environment has fixed noise bounds and no curriculum. Joint velocity
noise is uniform ±1.5 rad/s before observation scaling. In paired ablations with
all remaining random draws preserved, disabling only joint-velocity noise reduced
wrist RMS from 0.443 to 0.215 at 15k, and from 0.648 to 0.328 at 23k. Disabling
all noise gave 0.115 and 0.147 respectively. Velocity observation noise is a major
contributor, while the remaining difference also reflects the learned controller.

The active rewards include whole-robot squared action differences (weight −0.01)
and joint velocity (−0.00028), without a dedicated wrist smoothness term. Training
averages improve across many joints, force modes, randomized environments and
stochastic actions; they do not directly measure deterministic, unforced wrist
sensitivity. This objective/validation mismatch is consistent with the observed
regression, but does not prove that changing one reward weight would fix it.

For the next training experiment, evaluate checkpoints on these fixed tests and
log wrist-specific RMS/action changes separately for loaded and unloaded episodes.
Test wrist-specific regularization and measured sensor filtering/noise as controlled
ablations. Do not silently change the active policy's sensor processing or gains.

## Artifacts and reproduction

The installed bundle is `deploy/robots/g1_23dof/config/policy/compliant_mimic/stand`; the preserved source bundle is `logs/policy_previews/stand_stable_24000`.
Its PyTorch/ONNX export parity, C++/browser contract, and portable-bundle checks pass.
No robot was commanded; the noise model is not calibrated to measured G1 sensors.

Local evidence: `logs/policy_previews/checkpoint_search_11000_24000.json`,
`heldout_11500_13000_15000.json`, `heldout_24000.json`, `noise_channel_ablation_15_23k.json`, and
`training_log_trailing_means_through_23500.json` in that same directory.
The selected policy and [full comparison report](../deploy/robots/g1_23dof/config/policy/compliant_mimic/stand/checkpoint_selection_11_24k.json) are included in the deployment bundle. Other generated previews and raw diagnostic files remain local and ignored by Git.

Use `scripts/select_standing_checkpoint.py`, `scripts/confirm_standing_candidates.py`,
and `scripts/diagnose_standing_noise.py` with their `--help` options. The last script
disables each sensor channel in turn while preserving paired noise draws.


## Final 25k follow-up

The run completed at iteration 24,999/25,000. The final checkpoint passed 26/30 standing/recovery trials and survived 6/6 delay stress tests, with eight-seed noisy wrist RMS 0.444 rad/s. Four downward-load trials missed the unchanged 8 cm palm-recovery tolerance (8.3–11.9 cm residual), without falls. Keep 24k (30/30, 0.385 rad/s). Full results: [final-checkpoint comparison](../deploy/robots/g1_23dof/config/policy/compliant_mimic/stand/final_25k_comparison.json).
