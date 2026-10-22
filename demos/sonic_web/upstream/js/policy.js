/**
 * policy.js – SONIC encoder/decoder policy inference via ONNX Runtime Web.
 *
 * Architecture:
 *   Encoder : obs_dict [1, 1762] → encoded_tokens [1, 64]
 *   Decoder : obs_dict [1, 994]  → action [1, 29]
 *
 * Joint ordering:
 *   Policy (IsaacLab) and MuJoCo use different joint orderings.
 *   IL_TO_MJ[il] = MuJoCo index for IsaacLab/hardware joint il.
 *
 * Requires: window.ort (ONNX Runtime Web loaded as a global script).
 */

const N_JOINTS = 29;

// ── Debug controls ────────────────────────────────────────────────────────────
// Scale factor applied to all policy actions (1.0 = full, 0.1 = 10%, etc.)
const DEBUG_POLICY_SCALE = 1.0;

// Set true to bypass PD and apply zero torques (pure physics test).
const DEBUG_ZERO_CTRL = false;

// Set true to replace policy with a sine wave on one joint (PD controller test).
// All other joints hold their default angles.
const DEBUG_SINE_TEST      = false;
const DEBUG_SINE_JOINT     = 3;    // MuJoCo index (3 = left_knee)
const DEBUG_SINE_AMPLITUDE = 0.1;  // radians (~5.7 °)
const DEBUG_SINE_FREQ      = 0.5;  // Hz

// ── Joint ordering ───────────────────────────────────────────────────────────

// IL_TO_MJ[il] = mj  →  for each IsaacLab joint il, gives the MuJoCo index mj.
// Used in encoder: obs[il] = jp[IL_TO_MJ[il]]  (MuJoCo-ordered jp → IsaacLab-ordered encoder input).
// Matches C++ mujoco_to_isaaclab[] (despite the name, that array maps IsaacLab→MuJoCo).
const IL_TO_MJ = new Int32Array([
  0, 6, 12,  1, 7, 13,  2, 8, 14,  3, 9, 15, 22,  4, 10, 16, 23,  5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]);

// MJ_TO_IL[mj] = il  →  inverse of IL_TO_MJ; for each MuJoCo joint mj, gives IsaacLab index il.
// Used in actionToTargets: targets[mj] = action[MJ_TO_IL[mj]]  (IsaacLab-ordered action → MuJoCo order).
// Matches C++ isaaclab_to_mujoco[] (despite the name, that array maps MuJoCo→IsaacLab).
const MJ_TO_IL = new Int32Array([
  0, 3, 6, 9, 13, 17,  1, 4, 7, 10, 14, 18,  2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
]);

// ── Robot parameters (all arrays in MuJoCo joint order) ─────────────────────

// Proportional gains [Nm/rad] — from policy_parameters.hpp kps[] (MuJoCo order).
// kp = armature × ω²  (ω = 10 Hz × 2π, ζ = 2).  Ankle/waist_roll/waist_pitch use 2× multiplier.
const KP_MJ = new Float32Array([
  99.0984, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012,  // left leg  (pitch, roll, yaw, knee, ank_p, ank_r)
  99.0984, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012,  // right leg
  40.1792, 28.5012, 28.5012,                              // waist (yaw, roll, pitch)
  14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783,  // left arm
  14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783,  // right arm
]);

// Derivative gains [Nm·s/rad] — from policy_parameters.hpp kds[] (MuJoCo order).
// kd = 2 × ζ × armature × ω.  Ankle/waist_roll/waist_pitch use 2× multiplier.
const KD_MJ = new Float32Array([
  6.3088, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144,  // left leg
  6.3088, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144,  // right leg
  2.5579, 1.8144, 1.8144,                           // waist
  0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,  // left arm
  0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,  // right arm
]);

// Action scaling (policy output → joint position delta, rad)
const ACTION_SCALE_MJ = new Float32Array([
  0.3507, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
  0.3507, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
  0.5475, 0.4386, 0.4386,
  0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
  0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
]);

// Default standing pose [rad], MuJoCo joint order — from policy_parameters.hpp default_angles[].
export const DEFAULT_ANGLES_MJ = new Float32Array([
  -0.312,  0.0,   0.0,   0.669, -0.363,  0.0,   // left leg  (pitch, roll, yaw, knee, ank_p, ank_r)
  -0.312,  0.0,   0.0,   0.669, -0.363,  0.0,   // right leg
   0.0,    0.0,   0.0,                           // waist
   0.2,    0.2,   0.0,   0.6,   0.0,   0.0, 0.0, // left arm
   0.2,   -0.2,   0.0,   0.6,   0.0,   0.0, 0.0, // right arm
]);

// Default standing root height (metres).
// FK with default joint angles puts foot contact sphere centres at z=-0.0086 when
// pelvis=0.77.  Sphere radius=0.005, so we need pelvis≥0.77+0.0086+0.005=0.7836.
// Using 0.789 gives ~5 mm clearance above the contact threshold.
export const DEFAULT_ROOT_HEIGHT = 0.789;

// Motor effort limits [Nm], MuJoCo joint order (from g1_29dof_sonic_model12.yaml)
const EFFORT_LIMITS_MJ = new Float32Array([
   88,  88,  88, 139,  50,  50,   // left leg  (mj  0-5)
   88,  88,  88, 139,  50,  50,   // right leg (mj  6-11)
   88,  50,  50,                  // waist     (mj 12-14)
   25,  25,  25,  25,  25,   5,  5,  // left arm  (mj 15-21)
   25,  25,  25,  25,  25,   5,  5,  // right arm (mj 22-28)
]);

// ── Quaternion math (convention: [w, x, y, z]) ───────────────────────────────

function quatMul(aw, ax, ay, az, bw, bx, by, bz) {
  return [
    aw*bw - ax*bx - ay*by - az*bz,
    aw*bx + ax*bw + ay*bz - az*by,
    aw*by - ax*bz + ay*bw + az*bx,
    aw*bz + ax*by - ay*bx + az*bw,
  ];
}

// Rotate vector (vx,vy,vz) by quaternion q
function quatRotate(w, x, y, z, vx, vy, vz) {
  const tx = 2 * (y*vz - z*vy);
  const ty = 2 * (z*vx - x*vz);
  const tz = 2 * (x*vy - y*vx);
  return [
    vx + w*tx + (y*tz - z*ty),
    vy + w*ty + (z*tx - x*tz),
    vz + w*tz + (x*ty - y*tx),
  ];
}

// Rotate vector by conjugate of q (world → body frame transform)
function quatRotateInv(w, x, y, z, vx, vy, vz) {
  return quatRotate(w, -x, -y, -z, vx, vy, vz);
}

// Heading (yaw) angle of a quaternion: atan2(y, x) of the rotated +X axis
function calcHeading(w, x, y, z) {
  const [rx, ry] = quatRotate(w, x, y, z, 1, 0, 0);
  return Math.atan2(ry, rx);
}

// Pure Z-rotation quaternion for a given yaw [w, x, y, z]
function yawQuat(yaw) {
  const h = yaw * 0.5;
  return [Math.cos(h), 0, 0, Math.sin(h)];
}

// 3×3 rotation matrix from quaternion (returned as [[r0,r1,r2],[r3,r4,r5],[r6,r7,r8]])
function quatToRotMat(w, x, y, z) {
  const x2=2*x, y2=2*y, z2=2*z;
  const xx=x*x2, xy=x*y2, xz=x*z2;
  const yy=y*y2, yz=y*z2, zz=z*z2;
  const wx=w*x2, wy=w*y2, wz=w*z2;
  return [
    [1-(yy+zz), xy-wz,     xz+wy    ],
    [xy+wz,     1-(xx+zz), yz-wx    ],
    [xz-wy,     yz+wx,     1-(xx+yy)],
  ];
}

// ── SonicPolicy ──────────────────────────────────────────────────────────────

export class SonicPolicy {
  constructor() {
    this.encoderSession = null;
    this.decoderSession = null;

    // 10-frame rolling history buffers, oldest=[0], newest=[9]
    this._hAngVel  = Array.from({length: 10}, () => new Float32Array(3));
    this._hJointP  = Array.from({length: 10}, () => new Float32Array(N_JOINTS));
    this._hJointV  = Array.from({length: 10}, () => new Float32Array(N_JOINTS));
    this._hActions = Array.from({length: 10}, () => new Float32Array(N_JOINTS));
    this._hGravity = Array.from({length: 10}, () => new Float32Array([0, 0, -1]));

    // Last raw policy action (IsaacLab order), used in history
    this._lastAction = new Float32Array(N_JOINTS);

    // Heading-alignment delta quaternion [w,x,y,z]:
    //   apply_delta = yaw(robot_init) * inv_yaw(motion_init)
    // Set via initHeading() before each episode.
    this._headingDelta = [1, 0, 0, 0];  // identity
  }

  // ── Load ONNX sessions ─────────────────────────────────────────────────────

  async load(encoderPath, decoderPath, onProgress) {
    const ort = window.ort;
    if (!ort) throw new Error('ONNX Runtime Web (window.ort) not available');

    // Explicit per-file URLs so ORT doesn't try to guess paths.
    ort.env.wasm.wasmPaths = {
      'ort-wasm-simd-threaded.wasm':      '/lib/ort-wasm-simd-threaded.wasm',
      'ort-wasm-simd-threaded.mjs':       '/lib/ort-wasm-simd-threaded.mjs',
      'ort-wasm-simd-threaded.jsep.wasm': '/lib/ort-wasm-simd-threaded.jsep.wasm',
      'ort-wasm-simd-threaded.jsep.mjs':  '/lib/ort-wasm-simd-threaded.jsep.mjs',
    };
    // Multi-threading: safe now that server sets COOP/COEP headers (SharedArrayBuffer available).
    ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 8);
    // Try WebGPU first (huge speedup if supported); fall back to multi-threaded WASM.
    // Note: WebGL was excluded for int64 ops — WebGPU handles them correctly.
    const providers = ['webgpu', 'wasm'];

    onProgress?.('Compiling encoder (may take ~30 s)…');
    this.encoderSession = await ort.InferenceSession.create(encoderPath, {
      executionProviders: providers,
    });

    onProgress?.('Compiling decoder…');
    this.decoderSession = await ort.InferenceSession.create(decoderPath, {
      executionProviders: providers,
    });
  }

  // ── Initialise heading alignment (call once per episode, before reset) ──────
  // Computes the delta quaternion that rotates the reference motion's initial
  // heading to match the robot's current heading.  Matches the C++ logic:
  //   apply_delta = calc_heading_quat(robot_init) * calc_heading_quat_inv(motion_init)
  // motionData: loaded motion JSON; qpos: live MuJoCo qpos at episode start.

  initHeading(qpos, motionData) {
    const qw = qpos[3], qx = qpos[4], qy = qpos[5], qz = qpos[6];
    const robotYaw  = calcHeading(qw, qx, qy, qz);

    const irq        = motionData.root_quat[0];  // motion initial root quat [w,x,y,z]
    const motionYaw  = calcHeading(irq[0], irq[1], irq[2], irq[3]);

    // apply_delta = yaw(robot_init) * inv_yaw(motion_init)
    // = R_z(robotYaw) * R_z(-motionYaw) = R_z(robotYaw - motionYaw)
    this._headingDelta = yawQuat(robotYaw - motionYaw);
  }

  // ── Reset history (call when starting a new episode) ──────────────────────

  reset() {
    for (let i = 0; i < 10; i++) {
      this._hAngVel[i].fill(0);
      this._hJointP[i].fill(0);
      this._hJointV[i].fill(0);
      this._hActions[i].fill(0);
      this._hGravity[i].set([0, 0, -1]);
    }
    this._lastAction.fill(0);
    this._logStep = 0;
  }

  // ── Push current robot state into rolling history ──────────────────────────
  // Call this BEFORE infer() each policy tick.
  // qpos/qvel are WASM-backed MuJoCo arrays:
  //   qpos[0:3]  = root position
  //   qpos[3:7]  = root quaternion [w,x,y,z]
  //   qpos[7+j]  = joint angle j (MuJoCo order)
  //   qvel[3:6]  = root angular velocity (world frame)
  //   qvel[6+j]  = joint velocity j (MuJoCo order)

  pushHistory(qpos, qvel) {
    const qw = qpos[3], qx = qpos[4], qy = qpos[5], qz = qpos[6];

    // Body-frame angular velocity and gravity direction
    const [avx, avy, avz] = [qvel[3], qvel[4], qvel[5]];
    const [gx,  gy,  gz ] = quatRotateInv(qw, qx, qy, qz, 0, 0, -1);

    // Joint state in IsaacLab order — matches C++ body_q[i] = state[mujoco_to_isaaclab[i]].q().
    // IL_TO_MJ[il] gives the MuJoCo joint index for IsaacLab slot il.
    const jp = new Float32Array(N_JOINTS);
    const jv = new Float32Array(N_JOINTS);
    for (let il = 0; il < N_JOINTS; il++) {
      const mj = IL_TO_MJ[il];
      jp[il] = qpos[7 + mj] - DEFAULT_ANGLES_MJ[mj];
      jv[il] = qvel[6 + mj];
    }

    // Shift rolling buffers (drop oldest, append newest)
    this._hAngVel.shift();  this._hAngVel.push(new Float32Array([avx, avy, avz]));
    this._hJointP.shift();  this._hJointP.push(jp);
    this._hJointV.shift();  this._hJointV.push(jv);
    this._hActions.shift(); this._hActions.push(new Float32Array(this._lastAction));
    this._hGravity.shift(); this._hGravity.push(new Float32Array([gx, gy, gz]));
  }

  // ── Build 1762D encoder input (G1 mode, mode_id = 0) ──────────────────────
  // Encoder sees 10 future reference frames at step5 (every 5 policy ticks).
  // Active fields: mode, joint_pos, joint_vel, anchor_orientation.
  // All other fields (VR, SMPL, lower-body only) are zero for G1.

  _buildEncoderInput(motionData, refFi, qpos) {
    const obs    = new Float32Array(1762);  // zero-initialised
    const N      = motionData.frames;
    const fps    = motionData.fps;
    // Step in motion frames = 5 policy ticks @ 50 Hz converted to motion frames
    const stride = Math.max(1, Math.round(5 * fps / 50));

    const qw = qpos[3], qx = qpos[4], qy = qpos[5], qz = qpos[6];

    // Heading-alignment delta (set once per episode via initHeading)
    const [dhw, dhx, dhy, dhz] = this._headingDelta;

    // [0:4]   encoder_mode_4 = [0,0,0,0] (g1 mode_id = 0, already zero)

    for (let k = 0; k < 10; k++) {
      const fi      = Math.min(refFi + k * stride, N - 1);
      const fi_next = Math.min(fi + 1, N - 1);
      const jp      = motionData.joint_pos[fi];
      const jp1     = motionData.joint_pos[fi_next];

      // [4:294]   motion_joint_positions_10frame_step5
      // C++ provides raw CSV data in IsaacLab order (JointPositions() is unmodified CSV).
      // motionData.joint_pos is in MuJoCo order → use IL_TO_MJ[il] to look up MuJoCo
      // position for each IsaacLab slot, giving the same IsaacLab-ordered sequence C++ sends.
      // No default subtraction (matches C++ GatherMotionJointPositionsMultiFrame).
      for (let il = 0; il < N_JOINTS; il++) {
        obs[4 + k * N_JOINTS + il] = jp[IL_TO_MJ[il]];
      }

      // [294:584] motion_joint_velocities_10frame_step5 (finite difference)
      for (let il = 0; il < N_JOINTS; il++) {
        obs[294 + k * N_JOINTS + il] = (jp1[IL_TO_MJ[il]] - jp[IL_TO_MJ[il]]) * fps;
      }

      // [601:661] motion_anchor_orientation_10frame_step5 (6D per frame)
      // C++ applies heading correction before computing the relative rotation:
      //   new_ref_rot = apply_delta_heading * motion_root_quat[frame]
      //   base_to_ref = conj(base_quat) * new_ref_rot
      // This aligns the reference motion's initial heading with the robot's.
      const rq = motionData.root_quat[fi];
      const [nw, nx, ny, nz] = quatMul(dhw, dhx, dhy, dhz, rq[0], rq[1], rq[2], rq[3]);
      const [bw, bx, by, bz] = quatMul(qw, -qx, -qy, -qz, nw, nx, ny, nz);
      const R  = quatToRotMat(bw, bx, by, bz);
      const o  = 601 + k * 6;
      obs[o+0] = R[0][0]; obs[o+1] = R[0][1];
      obs[o+2] = R[1][0]; obs[o+3] = R[1][1];
      obs[o+4] = R[2][0]; obs[o+5] = R[2][1];

      // [584:594]  motion_root_z_position_10frame_step5  → zero (g1)
      // [594:595]  motion_root_z_position               → zero (g1)
      // [595:601]  motion_anchor_orientation             → zero (g1)
      // [661:901]  lower-body positions/velocities       → zero (g1)
      // [901:1762] VR / SMPL / wrist fields              → zero (g1)
    }

    return obs;
  }

  // ── Build 994D decoder input from encoder token + rolling history ──────────

  _buildDecoderInput(token) {
    const obs = new Float32Array(994);

    // [0:64]    token_state
    for (let i = 0; i < 64; i++) obs[i] = token[i];

    // [64:94]   his_base_angular_velocity_10frame_step1  (10×3, oldest-first)
    for (let k = 0; k < 10; k++) {
      obs[64 + k*3+0] = this._hAngVel[k][0];
      obs[64 + k*3+1] = this._hAngVel[k][1];
      obs[64 + k*3+2] = this._hAngVel[k][2];
    }

    // [94:384]  his_body_joint_positions_10frame_step1   (10×29)
    for (let k = 0; k < 10; k++)
      for (let i = 0; i < N_JOINTS; i++)
        obs[94 + k*N_JOINTS + i] = this._hJointP[k][i];

    // [384:674] his_body_joint_velocities_10frame_step1  (10×29)
    for (let k = 0; k < 10; k++)
      for (let i = 0; i < N_JOINTS; i++)
        obs[384 + k*N_JOINTS + i] = this._hJointV[k][i];

    // [674:964] his_last_actions_10frame_step1            (10×29)
    for (let k = 0; k < 10; k++)
      for (let i = 0; i < N_JOINTS; i++)
        obs[674 + k*N_JOINTS + i] = this._hActions[k][i];

    // [964:994] his_gravity_dir_10frame_step1             (10×3)
    for (let k = 0; k < 10; k++) {
      obs[964 + k*3+0] = this._hGravity[k][0];
      obs[964 + k*3+1] = this._hGravity[k][1];
      obs[964 + k*3+2] = this._hGravity[k][2];
    }

    return obs;
  }

  // ── Run one inference step ─────────────────────────────────────────────────
  // Returns Float32Array[29] action in IsaacLab order.
  // Call pushHistory() BEFORE this each tick.

  async infer(motionData, refFi, qpos) {
    const ort = window.ort;

    // Encoder
    const encObs    = this._buildEncoderInput(motionData, refFi, qpos);
    const encTensor = new ort.Tensor('float32', encObs, [1, 1762]);
    const t0 = performance.now();
    const encOut    = await this.encoderSession.run({ obs_dict: encTensor });
    const t1 = performance.now();
    const token     = encOut['encoded_tokens'].data;  // Float32Array[64]

    // Decoder
    const decObs    = this._buildDecoderInput(token);
    const decTensor = new ort.Tensor('float32', decObs, [1, 994]);
    const decOut    = await this.decoderSession.run({ obs_dict: decTensor });
    const t2 = performance.now();
    const action    = decOut['action'].data;          // Float32Array[29], IsaacLab order

    // Cache for history
    for (let i = 0; i < N_JOINTS; i++) this._lastAction[i] = action[i];

    return action;
  }

  // ── Convert action → target joint positions ────────────────────────────────
  // action: Float32Array[29] in IsaacLab order
  // Returns Float32Array[29] target positions in MuJoCo order.
  // Call once per policy tick; then feed into targetsToCtrl() each sub-step.

  actionToTargets(action) {
    // Sine-wave test: hold all joints at default, sweep one joint.
    if (DEBUG_SINE_TEST) {
      const targets = new Float32Array(DEFAULT_ANGLES_MJ);
      const t = Date.now() / 1000;
      targets[DEBUG_SINE_JOINT] += DEBUG_SINE_AMPLITUDE * Math.sin(2 * Math.PI * DEBUG_SINE_FREQ * t);
      return targets;
    }

    const targets = new Float32Array(N_JOINTS);
    // Model outputs action in IsaacLab order; MJ_TO_IL[mj] gives the IsaacLab slot for MuJoCo joint mj.
    // Matches C++: action_value = floatarr[isaaclab_to_mujoco[i]] * g1_action_scale[i]
    for (let mj = 0; mj < N_JOINTS; mj++) {
      targets[mj] = action[MJ_TO_IL[mj]] * ACTION_SCALE_MJ[mj] * DEBUG_POLICY_SCALE + DEFAULT_ANGLES_MJ[mj];
    }

    return targets;
  }

  // ── Evaluate PD torques from current robot state ───────────────────────────
  // targets: Float32Array[29] target positions in MuJoCo order (from actionToTargets)
  // qpos/qvel: live MuJoCo arrays (re-read each sub-step for fresh damping)
  // Returns Float32Array[29] ctrl torques in MuJoCo order, clipped to effort limits.
  // Matches base_sim.py compute_body_torques(), called at sim_frequency (200 Hz).

  targetsToCtrl(targets, qpos, qvel) {
    if (DEBUG_ZERO_CTRL) return new Float32Array(N_JOINTS);
    const ctrl = new Float32Array(N_JOINTS);
    for (let mj = 0; mj < N_JOINTS; mj++) {
      const torque = KP_MJ[mj] * (targets[mj] - qpos[7 + mj])
                   - KD_MJ[mj] * qvel[6 + mj];
      const lim    = EFFORT_LIMITS_MJ[mj];
      ctrl[mj]     = torque < -lim ? -lim : torque > lim ? lim : torque;
    }
    return ctrl;
  }
}
