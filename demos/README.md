# G1 23DOF demos

Both viewers are maintained here in the SoftMimic mjlab repository. Start with the [end-to-end workflow](../README.md).

| App | Purpose | Local address |
| --- | --- | --- |
| [Unitree_G1_23dof Motion Streamer](ardy_web/README.md) | CPU ARDY generation, 30-second timeline editing and Save CSV | http://127.0.0.1:8089 |
| [Compliant Humanoid](sonic_web/README.md) | Required fixed reference, cyan ghost, physics and matching ONNX policy | http://127.0.0.1:8088 |

The shared renderer is adapted from the [GEAR-SONIC web demo source](https://nvlabs.github.io/GEAR-SONIC/demo.html). ARDY generates kinematic motions; SoftMimic supplies the separately trained compliance controller. Original source and provenance remain in `sonic_web/upstream/` and `sonic_web/upstream_manifest.json`.

`package_standing_policy.py` creates a matched immutable bundle for the browser and C++ deployment. `check_cpp_policy_contract.py` checks the exported C++ observation/action contract. No viewer or packaging command starts hardware.

The old workspace path `humanoid_demo` is a compatibility symlink on this machine. Use the repository's `demos/` paths for new commands. Legacy Viser/scaffold utilities are retained for research; they are not the current demo entry points.
