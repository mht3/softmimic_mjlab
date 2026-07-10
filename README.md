# SoftMimic Mjlab


## Overview
Custom implementation of [SoftMimic](https://github.com/Improbable-AI/softmimic) using mjlab in place of IsaacLab to make setup, training, and deployment simpler. Currently supports compliant motion generation and learning for the Unitree G1 robot (both 29 and 23 dof).

Feel free to open a PR for any issues. Trained using CUDA 12.9 and mujoco 3.8.1 on an NVIDIA RTX 5090 GPU. Real world deployment has been tested on the Unitree G1 23dof only.


<div align="center">

| <div align="center">  MuJoCo </div>                                                                                                                                           | <div align="center"> Physical </div>                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> | <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity-real.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> |

</div>


## Installation and Configuration

Please refer to [setup.md](doc/setup_en.md) for installation and configuration steps.


## Process Overview

The basic workflow for using reinforcement learning to achieve motion control is:

`Train` → `Play` → `Sim2Real`

- **Train**: The agent interacts with the MuJoCo simulation and optimizes policies through reward maximization.
- **Play**: Replay trained policies to verify expected behavior.
- **Sim2Real**: Deploy trained policies to physical Unitree robots for real-world execution.


## Usage Guide

### 1. Velocity Tracking Training

Run the following command to train a velocity tracking policy:

```bash
python scripts/train.py Unitree-G1-23Dof-Flat --env.scene.num-envs=4096
```

Multi-GPU Training: Scale to multiple GPUs using --gpu-ids:

```bash
python scripts/train.py Unitree-G1-23Dof-Flat \
  --gpu-ids 0 1 \
  --env.scene.num-envs=4096
```

- The first argument (e.g., Unitree-G1-23Dof-Flat) specifies the training task.
Available velocity tracking tasks:
  - Unitree-G1-Flat
  - Unitree-G1-23Dof-Flat

> [!NOTE]
> For more details, refer to the mjlab documentation:
> [mjlab documentation](https://mujocolab.github.io/mjlab/index.html).

### 2. Motion Imitation Training

Train a Unitree G1 to mimic reference motion sequences.

<div style="margin-left: 20px;">

#### 2.1 Prepare Motion Files

Prepare csv motion files in mjlab/motions/g1/ and convert them to npz format:

```bash
python scripts/csv_to_npz.py \
--input-file src/assets/motions/g1/dance1_subject2.csv \
--output-name dance1_subject2.npz \
--input-fps 30 \
--output-fps 50 \
--robot g1 # g1 or g1_23dof
```

**npz files will be stored at:**：`src/motions/g1/...`

#### 2.2 Training

After generating the NPZ file, launch imitation training:

```bash
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --env.scene.num-envs=4096
```

Available tasks:
  - Unitree-G1-Tracking-No-State-Estimation
  - Unitree-G1-23Dof-Tracking-No-State-Estimation

</div>

> [!NOTE]
> For detailed motion imitation instructions, refer to the BeyondMimic documentation:
> [BeyondMimic documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup).

#### Parameter Description
- `--env.scene`: simulation scene configuration (e.g., num_envs, dt, ground type, gravity, disturbances)
- `--env.observations`: observation space configuration (e.g., joint state, IMU, commands, etc.)
- `--env.rewards`: reward terms used for policy optimization
- `--env.commands`: task commands (e.g., velocity, pose, or motion targets)
- `--env.terminations`: termination conditions for each episode
- `--agent.seed`: random seed for reproducibility
- `--agent.resume`: resume from the last saved checkpoint when enabled
- `--agent.policy`: policy network architecture configuration
- `--agent.algorithm`: reinforcement learning algorithm configuration (PPO, hyperparameters, etc.)

**Training results are stored at**：`logs/rsl_rl/<robot>_(velocity | tracking)/<date_time>/model_<iteration>.pt`

### 3. Compliant Motion Tracking (SoftMimic)

Train a compliant whole-body policy on SoftMimic augmented motions (Unitree G1 23dof).

<div style="margin-left: 20px;">

#### 3.1 Generate the augmented motion datasets

The pipeline has three stages: (1) `generate_all.sh` runs the mink IK solver over the
reference motion CSVs to produce 29dof augmented CSVs under
`compliant_motion_augmentation/release_examples/<mode>/<task>/`; (2)
`scripts/csv_29dof_to_23dof.py` drops the 6 unused joints; (3)
`scripts/compliant_csv_to_npz.py` replays the adapted motion through MuJoCo to produce
the training NPZs.

The reference motion CSVs (`stand.csv`, `walk.csv`, …) live in the sibling `softmimic`
repo, **not** in this repo. `generate_all.sh` resolves them relative to itself
(`../../softmimic/datasets/motions_csv`); if your layout differs, point it there
explicitly:

```bash
# Stage 1: 29dof augmented CSVs (needs `mink`; ~5 min for stand+walk on a GPU box).
export SOFTMIMIC_MOTIONS_DIR=/path/to/softmimic/datasets/motions_csv   # optional override
cd compliant_motion_augmentation && bash generate_all.sh && cd ..
```

> **Note:** a missing reference CSV now raises an error. Earlier revisions silently fell
> back to a static stand pose, which produced a "walk" dataset that never moved (every
> file was identical to `stand`). `tests/test_compliant_motion_datasets.py` guards against
> this — a walk clip must translate the root, a stand clip must not.

Then convert each task to 23dof and to training NPZs (shown for `stand` and `walk`):

```bash
# Stages 2 + 3.
for task in stand walk; do
  for mode in forcefield collision-emulator zero-wrench; do
    python scripts/csv_29dof_to_23dof.py \
      --input-dir compliant_motion_augmentation/release_examples/$mode/$task \
      --output-dir src/assets/compliant_motions_csv/$task/$mode
  done
  python scripts/compliant_csv_to_npz.py \
    --input-dir src/assets/compliant_motions_csv/$task \
    --output-dir src/assets/compliant_motions/$task
done
```

#### 3.2 Training

```bash
bash train_configs/compliance/stand.sh    # or walk.sh
```

To train a **steerable** walking policy, use the velocity-conditioned task
(`Unitree-G1-23Dof-Compliant-Tracking-Velocity`). It adds the reference root velocity
(heading frame) to the observations so the policy follows a velocity command; at play
time the GUI velocity joystick (Section 3.3) overrides it to steer the robot in x/y and
yaw. Train it against the walk dataset:

```bash
python scripts/train.py Unitree-G1-23Dof-Compliant-Tracking-Velocity \
  --motion_file=src/assets/compliant_motions/walk --env.scene.num-envs 4096
```

#### 3.3 Visualize a trained policy

```bash
python scripts/play.py Unitree-G1-23Dof-Compliant-Tracking-No-State-Estimation \
  --motion_file=src/assets/compliant_motions/stand \
  --checkpoint_file=logs/rsl_rl/g1_23dof_compliant_tracking/2026-xx-xx_xx-xx-xx/model_xx.pt
```

The viser viewer's **Motion** panel plays the reference: the time slider tracks the
current playback position (drag it to scrub), "Use augmented motions" switches to a random
augmented adapted-reference on the spot, and "Apply dataset forces" replays the augmented
motion's baked-in forcefield wrench each step — exactly what the policy saw during training
(it auto-enables augmented motions, since only those carry force events). Only the
forcefield/collision-emulator motions contain force events, so on the `stand` zero-wrench
motion nothing is applied — switch to augmented and enable dataset forces to see them.
While augmented motions are playing, the interactive **Push** panel is disabled (the
augmented reference / dataset forces are the perturbation source); uncheck "Use augmented
motions" to drive pushes manually again.

It also exposes a **Desired Stiffness** panel (log-scale sliders over the SoftMimic training
ranges) and a **Velocity Joystick** panel. The joystick only steers a policy trained on the
velocity-conditioned task — check "Enable" and drag the vel x / vel y / yaw rate sliders to
command a body-frame velocity (they override the reference velocity the policy observes):

```bash
python scripts/play.py Unitree-G1-23Dof-Compliant-Tracking-Velocity \
  --motion_file=src/assets/compliant_motions/walk \
  --checkpoint_file=logs/rsl_rl/g1_23dof_compliant_tracking/2026-xx-xx_xx-xx-xx/model_xx.pt
```

Headless episode statistics / video (`scripts/test.py`) and perturbation sweeps (`scripts/evaluate.py`) take the same task, `--motion_file`, and checkpoint arguments.

For real deployment, copy the run's `exported/policy.onnx` + `params/deploy.yaml` and a nominal motion NPZ (e.g. `src/assets/compliant_motions/stand/zero-wrench/stand_augmented_mink_001.npz`) into `deploy/robots/g1_23dof/config/policy/compliant_mimic/stand`, then deploy as in Section 5. The commanded stiffness is set per-policy in `deploy/robots/g1_23dof/config/config.yaml` (`desired_stiffness`, `desired_rotational_stiffness`) and can be adjusted online with the d-pad (up/down: translational, left/right: rotational).

</div>

### 4. Simulation Validation

To visualize policy behavior in MuJoCo:

Velocity tracking:
```bash
python scripts/play.py Unitree-G1-23Dof-Flat --viewer viser --checkpoint_file=logs/rsl_rl/g1_23dof_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

Motion imitation:
```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --checkpoint_file=logs/rsl_rl/g1_tracking/2026-xx-xx_xx-xx-xx/model_xx.pt
```

**Note**：

- During training, policy.onnx and policy.onnx.data are also exported for deployment onto physical robots.


### 5. Real Deployment

Before deployment, install the required communication tools:
- [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git)
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)

<div style="margin-left: 20px;">

#### 5.1 Power On the Robot
Start the robot in suspended state and wait until it enters `zero-torque` mode.

#### 5.2 Enable Debug Mode
While in `zero-torque` mode, press `L2 + R2` on the controller. The robot will enter `debug mode` with joint damping enabled.

#### 5.3 Connect to the Robot
Connect your PC to the robot via Ethernet. Configure the network as:
- Address：`192.168.123.222`
- Netmask：`255.255.255.0`

Use `ifconfig` to determine the Ethernet device name for deployment.

#### 5.4 Compilation

Example: Unitree G1 velocity control.
Place `policy.onnx` and `policy.onnx.data` into: `deploy/robots/g1/config/policy/velocity/v0/exported`.
Then compile:

```bash
cd deploy/robots/g1
mkdir build && cd build
cmake .. && make
```

#### 5.5 Deployment

## 5.5.1 Simulation Deployment

Before deploying on the real robot, it is recommended to perform simulation deployment using [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
to prevent abnormal behaviors on the physical robot. This framework has already integrated it.

Build unitree_mujoco：

```bash
cd simulate
mkdir build && cd build
cmake .. && make -j8
```

Launch the simulator (note that a gamepad must be connected):

```bash
./simulate/build/unitree_mujoco
```

You can select the corresponding robot in `simulate/config`

Launch the simulation control program:

```bash
cd deploy/robots/g1/build
./g1_ctrl --network=lo
```

## 5.5.2 Real-Robot Deployment

Launch the control program on the real robot:

```bash
cd deploy/robots/g1/build
./g1_ctrl --network=enp5s0
```

**Arguments**：
- `network`: The network interface used to connect to the robot. Use `lo` for simulation deployment, and `enp5s0` for the real robot(You can check it using the `ifconfig` command) 


## Acknowledgements

This project would not be possible without the contributions of the following repositories:

- [mjlab](https://github.com/mujocolab/mjlab.git): training and execution framework
- [Unitree RL Mjlab Repository](https://github.com/unitreerobotics/unitree_rl_mjlab.git): adaptation for Unitree robots
- [SoftMimic](https://github.com/Improbable-AI/softmimic): IsaacLab implementation for SoftMimic RL sim, training, and deployment.
- [Mink](https://github.com/kevinzakka/mink): Inverse kinematics solver in Mujoco
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl): Policy training framework.
