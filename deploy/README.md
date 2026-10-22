# G1 23DOF standing deployment

Use the [main workflow](../README.md#5-prepare-real-robot-deployment) to package and install matching policy, YAML and nominal reference files. The maintained runtime is `robots/g1_23dof`; the legacy `softmimic_deploy` Python policy configuration is a different interface.

Install [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2) and its Cyclone DDS dependencies using their upstream installation instructions. The C++ build also requires Boost program_options, yaml-cpp, Eigen3, fmt and zlib. ONNX Runtime and cnpy are under `thirdparty/`. The SDK and DDS must be visible to CMake/the linker on the deployment machine.

```bash
# From the repository root; compilation does not connect to the robot.
cmake -S deploy/robots/g1_23dof -B deploy/robots/g1_23dof/build
cmake --build deploy/robots/g1_23dof/build -j2
```

Before a physical trial, complete simulation evaluation of that exact bundle. Prepare the robot using Unitree's procedure: start suspended, wait for zero-torque mode, then L2 + R2 enters debug mode with joint damping. Connect Ethernet; the original repository setup uses PC address `192.168.123.222/24` on the robot subnet. Select the actual Ethernet interface, not a guessed default. Check the manufacturer's instructions for the installed robot/firmware.

After installing the generated standing-only `controller_config.yaml` as `robots/g1_23dof/config/config.yaml`, launch from the repository root:

```bash
./deploy/robots/g1_23dof/build/g1_ctrl --network=YOUR_ROBOT_NETWORK_INTERFACE
```

This command opens the hardware control channel. LT + Up enters FixStand; RB + A enters Mimic_Compliant_Stand; LT + B returns to Passive. These bindings belong to the generated standing-only configuration. The bundle's FixStand pose, nominal motion, ONNX and gains must stay matched. No velocity policy is required.

The packager does not modify the active configuration or launch this controller. Passing C++/browser contract tests is not hardware validation.
