# G1 23DOF standing deployment

A fresh clone includes the matched **Stable checkpoint 24000** ONNX, YAML and reference CSV in `robots/g1_23dof/config/policy/compliant_mimic/stand`. The default `config/config.yaml` uses this folder with relative paths and enables only Passive, FixStand and Mimic_Compliant_Stand. No IK dataset or generated NPZ is required for this standing policy. The 25k run is complete; checkpoint 24k remains selected after the final 25k policy showed worse wrist motion and recovery. See the [comparison](../doc/WRIST_NOISE_REVIEW.md). The older completed 20k baseline remains a separate demo bundle.

```bash
conda activate unitree_rl_mjlab
```

```bash
python deploy/check_bundle.py
```

This offline check verifies file hashes, nominal CSV/YAML agreement, ONNX dimensions, joint mapping and the FixStand pose. It never opens a robot connection. See [sensor and actuator details](SIM2REAL.md). The maintained runtime is `robots/g1_23dof`; the legacy `softmimic_deploy` Python configuration uses a different interface.

Install [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2) and its Cyclone DDS dependencies using their upstream installation instructions. The C++ build also requires Boost program_options, yaml-cpp, Eigen3, fmt and zlib. ONNX Runtime and cnpy are under `thirdparty/`. The SDK and DDS must be visible to CMake/the linker on the deployment machine.

```bash
# From the repository root; compilation does not connect to the robot.
cmake -S deploy/robots/g1_23dof -B deploy/robots/g1_23dof/build
```

```bash
cmake --build deploy/robots/g1_23dof/build -j2
```

Before a physical trial, complete simulation evaluation of that exact bundle. Prepare the robot using Unitree's procedure: start suspended, wait for zero-torque mode, then L2 + R2 enters debug mode with joint damping. Connect Ethernet; the original repository setup uses PC address `192.168.123.222/24` on the robot subnet. Select the actual Ethernet interface, not a guessed default. Check the manufacturer's instructions for the installed robot/firmware.

After evaluating the included bundle in simulation and preparing the robot, launch from the repository root:

```bash
./deploy/robots/g1_23dof/build/g1_ctrl --network=YOUR_ROBOT_NETWORK_INTERFACE
```

This command opens the hardware control channel. LT + Up enters FixStand; RB + A enters Mimic_Compliant_Stand; LT + B returns to Passive. These bindings belong to the generated standing-only configuration. A pelvis tilt beyond 1.25 rad (71.6°) cuts policy drive and enters Passive; it does not automatically recover. The bundle's FixStand pose, nominal motion, ONNX and gains must stay matched. No velocity policy is required.

For a newly trained policy, use the [matched bundle workflow](../WORKFLOW.md#5-prepare-real-robot-deployment), validate it, and install its generated controller configuration. Do not copy just the ONNX over a different YAML/reference. Passing C++/browser contract tests is not hardware validation.
