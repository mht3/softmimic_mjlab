# G1 23DOF standing deployment

A fresh clone includes **checkpoint 34000** ONNX, matching YAML and reference CSV in `robots/g1_23dof/config/policy/compliant_mimic/stand`. The default `config/config.yaml` uses portable relative paths and enables only Passive, FixStand and Mimic_Compliant_Stand. No IK dataset, training log or generated NPZ is required. The intermediate 34k checkpoint passed the finite simulated checks in the [comparison](../doc/ROBUSTNESS_TESTING.md); it has not been validated on hardware.

Run these commands from the repository root after pulling the latest code:

```bash
conda activate unitree_rl_mjlab
```

```bash
python deploy/check_bundle.py
```

This offline check verifies file hashes (including the controller config), ONNX checkpoint metadata, nominal CSV/YAML agreement, dimensions, joint mapping, FixStand pose, fall/watchdog settings and the default portable paths. It never opens a robot connection. See [sensor and actuator details](SIM2REAL.md). The maintained runtime is `robots/g1_23dof`; the legacy `softmimic_deploy` Python configuration uses a different interface.

Install [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2) and its Cyclone DDS dependencies using their upstream installation instructions. The C++ build also requires Boost program_options, yaml-cpp, Eigen3, fmt and zlib. ONNX Runtime and cnpy are under `thirdparty/`. The SDK and DDS must be visible to CMake/the linker on the deployment machine.

```bash
# From the repository root; compilation does not connect to the robot.
cmake -S deploy/robots/g1_23dof -B deploy/robots/g1_23dof/build -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build deploy/robots/g1_23dof/build -j2
```

Before a physical trial, complete simulation evaluation of that exact bundle. Prepare the robot using Unitree's procedure: start suspended, wait for zero-torque mode, then L2 + R2 enters debug mode with joint damping. Connect Ethernet; the original repository setup uses PC address `192.168.123.222/24` on the robot subnet. Select the actual Ethernet interface, not a guessed default. Check the manufacturer's instructions for the installed robot/firmware.

Identify the cabled adapter and its IPv4 address (replace the placeholder below with its actual name):

```bash
ip -br addr
```

```bash
python deploy/preflight.py --network=YOUR_ROBOT_NETWORK_INTERFACE
```

Preflight checks the included bundle/default config, loads the native executable with `--help`, and checks the chosen Ethernet adapter's link/carrier and IPv4 address. It never initializes DDS, contacts the robot or changes networking. Confirm the displayed subnet matches the robot's configured subnet. A cable alone is not sufficient: SDK/DDS dependencies, correct subnet, compatible 23DOF firmware/mode and the robot preparation procedure are prerequisites.

After simulation evaluation, preflight and robot preparation, launch from the repository root:

```bash
./deploy/robots/g1_23dof/build/g1_ctrl --network=YOUR_ROBOT_NETWORK_INTERFACE
```

This command opens the hardware control channel. It rejects a missing/invalid adapter or loopback before DDS starts, checks for another low-level controller and requires the 23DOF machine mode. It waits for robot state before starting in Passive. LT + Up enters FixStand; RB + A enters Mimic_Compliant_Stand; LT + B returns to Passive. These bindings belong to the generated standing-only configuration. Invalid IMU/encoder data, a stale robot-state stream or policy output timeout stop policy drive; the stream and policy watchdogs are 100 ms. A pelvis tilt beyond 1.25 rad (71.6°) cuts policy drive and enters Passive; it does not automatically recover. The bundle's FixStand pose, nominal motion, ONNX and gains must stay matched. No velocity policy is required.

For a newly trained policy, use the [matched bundle workflow](../WORKFLOW.md#5-prepare-real-robot-deployment), validate it, and install its generated controller configuration. Do not copy just the ONNX over a different YAML/reference. Passing C++/browser contract tests is not hardware validation.
