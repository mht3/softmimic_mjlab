# Standing compliance: validation and retraining

> **Collision validation required:** The original 2026-10-21 standing batch failed self-collision checks in all 40 forcefield clips. Its prior passing audit was incomplete. A replacement must pass the current full audit, including resampled-target collisions and standing release recovery, before training. `stand.sh` rejects old or failed audits. See `../research_notes/SELF_COLLISION_FIX_20261021.md`.

Updated2026-10-21. The comparison target is the **released SoftMimic Hybrid implementation**, adapted to mjlab and23DOF. The paper and released implementation differ; see the evidence and remaining simulator differences in [the parity audit](../research_notes/SOFTMIMIC_PARITY_AUDIT.md). Passing software checks is not proof that a trained robot will be compliant.

## Environment and retraining

Use `unitree_rl_mjlab`. Its `rsl-rl-lib` editable installation pointed at a moved checkout (the old path lacked `/rl/`). It has been repaired to this repository. To repair the same issue in another installation:

```bash
conda activate unitree_rl_mjlab
cd /home/mht/research/rl/humanoid/compliance/softmimic_mjlab
python -m pip install --no-deps --no-build-isolation -e algorithms/rsl_rl
python -c 'import rsl_rl; print(rsl_rl.__file__)'
```

**Train from scratch.** The restored actor has327 inputs and the critic1951, with three-frame term histories. The former231-input standing actor and checked-in378-input ONNX are incompatible. Observations, rewards, force scheduling, reset semantics, default pose, dynamics and dataset weighting have also changed. Re-exporting an old policy cannot apply these training fixes.

Standing deployment now embeds the actual constant nominal reference into the run's generated `deploy.yaml`. Changing nominal references explicitly fail deployment export until dynamic playback is implemented. Use `policy.onnx` and `params/deploy.yaml` from the same run; rebuild the C++ controller.

## Dataset findings and corrections

The failed generation produced40CSV files but no paired NPZs. The canonical training directory was not replaced. `logs/stand_dataset_audit.json` left from an earlier smoke test did not describe that failed stage. The full new-stage report is [new_dataset_audit_20261021.json](../research_notes/new_dataset_audit_20261021.json), which now explicitly reports all40 missing NPZs.

Solving directly on the23DOF model is correct. Hand IDs19/24 are resolved by body name during conversion. The40 staged CSVs are finite, respect joint limits, and contain downward-load examples. However, clip008 had two release frames with foot errors7.18/6.78cm: joint-only rate clipping broke the coordinated floating-base/leg update. The limiter now scales the entire configuration tangent displacement. Regenerating that seed reduced those errors to0.636/0.233cm and its maximum foot error to3.97cm, maintaining the2πrad/s joint-speed limit. See [the reproduction](../research_notes/release_bound_fix_verification_20261021.json).

**Regenerate the staged CSVs; do not resume their conversion.** Further fixes include coherent release recovery, rejection rewind without duplicated frames, import-path setup, seven nominal-reference NPZ channels, body-ID mapping and fail-fast conversion/auditing.

For released Hybrid parity generate40forcefield,40collision-emulator and5zero-wrench clips. The loader repeats zero-wrench files to give equal effective40/40/40 file weights. Standing collision-emulator clips are unforced because their velocity gate never activates on a static reference; retain them to reproduce the release's mixture, not as evidence of collision coverage. Missing Hybrid modes fail explicitly.

## Reproducible generation and training

```bash
cd /home/mht/research/rl/humanoid/compliance/softmimic_mjlab
conda activate unitree_rl_mjlab
export PYTHONPATH="$PWD:$PWD/algorithms/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=/tmp/compliance-mpl

python scripts/generate_compliant_23dof.py \
  --motions stand --modes forcefield collision-emulator zero-wrench \
  --num-files 40 40 5 --seed 42 --device cpu --replace
```

This stages all85CSV/NPZ pairs, audits every CSV frame, and only then replaces `src/assets/compliant_motions/g1_23dof/stand`. It retains the previous data under `.backups/`. Acceptance requires finite arrays, all seven nominal channels,23joints, correct body counts/quaternions/frame timing, joint-limit error≤1e-3rad, joint speed≤2πrad/s, and foot/hand-F-over-K residuals≤5cm. The5cm criterion is inherited kinematic feasibility, not exact planted feet or dynamic stability. A failure retains the old canonical data and identifies the staged audit to inspect.

Wait for `Done`. Review `logs/stand_dataset_audit.json` and `src/assets/compliant_motions/g1_23dof/stand/generation.json` before training. The audit records source paths/hashes; the generation manifest records parameters and code/reference hashes. To audit the installed paths again:

```bash
python scripts/audit_compliant_dataset.py \
  src/assets/compliant_motions/g1_23dof/stand \
  --stride 1 --output logs/stand_dataset_audit.json
```

After the audit is accepted, run:

```bash
bash train_configs/compliance/stand.sh --agent.run-name corrected_stand
```

The script defaults to4096environments and10000updates, logs online to W&B project **g1_compliance**, and preserves the launch command/script, configs and dataset metadata with the run. `NUM_ENVS` and `MAX_ITERATIONS` override the budget. Ensure `WANDB_MODE` is not set to `disabled` or `offline` if you want online logging. These settings are an initial budget, not a guaranteed convergence time.

## Held-out validation

Generate a separate seed without replacing training data:

```bash
python scripts/generate_compliant_23dof.py \
  --motions stand --modes forcefield collision-emulator zero-wrench \
  --num-files 5 5 1 --seed 1042 --device cpu \
  --output-root src/assets/compliant_motions/g1_23dof_validation --replace
```

Set `CHECKPOINT` to a real checkpoint from your new run:

```bash
CHECKPOINT=logs/rsl_rl/g1_23dof_compliant_tracking/ACTUAL_RUN/model_9999.pt
python scripts/validate_compliant_policy.py \
  --checkpoint "$CHECKPOINT" \
  --motion-file src/assets/compliant_motions/g1_23dof_validation/stand \
  --num-envs 64 --steps 3000 --output logs/stand_validation.json

python scripts/play.py Unitree-G1-23Dof-Compliant-Tracking-Stand \
  --motion-file src/assets/compliant_motions/g1_23dof_validation/stand \
  --checkpoint-file "$CHECKPOINT" --num-envs 1 --viewer viser
```

Test no load, lateral hand loads, downward loads, release and several stiffness values. Measure falls, hand displacement versus force, foot drift, pelvis/knee response and oscillation. Enable dataset forces to replay training springs; disable them before using independent Force/Push controls. Simultaneous two-hand manipulation needs separate testing because the dataset has one forced body per frame. A policy that runs or earns high reward has not yet passed these behavioral checks.

No physical robot validation or full retraining was performed during this repair. The GPU was left available to the user; integration checks used CPU, including an actual short PPO update and deployment export.
