"""Evaluate trained policies across perturbation levels.

Measures success rate, mean joint L2, and mean base L2 for each
(policy, perturbation) pair. Outputs per-policy JSON/PNG and an
optional combined comparison plot.

Usage:
  python scripts/evaluate.py Unitree-G1-23Dof-Balance-Flat \
    --policies logs/.../model_10000.pt:push_curriculum \
               logs/.../model_5000.pt:baseline \
    --num-episodes 200
"""

import argparse
import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from src.utils.vecenv_wrapper import RslRlVecEnvSpecialResetWrapper as RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul
from mjlab.utils.torch import configure_torch_backends
from src.utils.mjlab_on_policy_runner_with_eval import MjlabOnPolicyRunnerWithEval


# ---------------------------------------------------------------------------
# Perturbation level definitions
# ---------------------------------------------------------------------------
# Only init_lin_vel (and matching push x/y) changes across small/medium/hard.
# All other parameters are constant across non-none levels.
#
# | Level  | init_joint | init_lin_vel | init_z_vel | init_ang_vel | init_pos_z | init_rp | push x/y | push z | push ang |
# |--------|------------|--------------|------------|--------------|------------|---------|----------|--------|----------|
# | none   | 0          | 0            | 0          | 0            | 0          | 0       | 0        | 0      | 0        |
# | small  | 0.1        | 0.5          | 0.2        | 0.78         | 0.01       | 0.1     | 0.5      | 0.2    | 0.78     |
# | medium | 0.2        | 1.25         | 0.2        | 0.78         | 0.01       | 0.3     | 1.25     | 0.2    | 1.18     |
# | hard   | 0.4        | 2.0          | 0.2        | 0.78         | 0.01       | 0.4     | 2.0      | 0.2    | 1.18     |
# ---------------------------------------------------------------------------

@dataclass
class PerturbSpec:
  init_joint_range: float = 0.0
  init_lin_vel_range: float = 0.0
  init_z_vel_range: float = 0.0
  init_ang_vel_range: float = 0.0
  init_pos_z_range: float = 0.0
  init_rp_range: float = 0.0
  push_velocity_range: dict = field(default_factory=dict)  # empty = no push
  # Ring sampling: when True, magnitudes are drawn from N(r, sigma^2) with a
  # random sign, concentrating mass near ±r. Sigmas are fixed across levels so
  # the ring width is the same regardless of the ring radius.
  ring: bool = False
  ring_lin_vel_sigma: float = 0.10   # [m/s]  — lin vel init & interval
  ring_ang_vel_sigma: float = 0.10   # [rad/s] — ang vel interval
  ring_joint_sigma: float = 0.02     # [rad]   — joint init noise
  ring_rp_sigma: float = 0.02        # [rad]   — roll/pitch init noise


_LIN, _Z, _ANG = 0.5, 0.2, 0.78

PERTURB_LEVELS: dict[str, PerturbSpec] = {
  # "none" is kept for opt-in use (pass --levels explicitly) but excluded from
  # the default sweep because zero perturbation is uninformative.
  "none": PerturbSpec(),
  "small": PerturbSpec(
    init_joint_range=0.1, init_lin_vel_range=_LIN, init_z_vel_range=_Z,
    init_ang_vel_range=_ANG, init_pos_z_range=0.01, init_rp_range=0.1,
    push_velocity_range={
      "x": (-_LIN, _LIN), "y": (-_LIN, _LIN), "z": (-_Z, _Z),
      "roll": (-_ANG, _ANG), "pitch": (-_ANG, _ANG), "yaw": (-_ANG, _ANG),
    },
  ),
  "medium": PerturbSpec(
    init_joint_range=0.2, init_lin_vel_range=1.25, init_z_vel_range=_Z,
    init_ang_vel_range=_ANG, init_pos_z_range=0.01, init_rp_range=0.3,
    push_velocity_range={
      "x": (-1.25, 1.25), "y": (-1.25, 1.25), "z": (-_Z, _Z),
      "roll": (-1.18, 1.18), "pitch": (-1.18, 1.18), "yaw": (-1.18, 1.18),
    },
  ),
  # Ring variants: same magnitudes as uniform counterparts but sampled from a
  # boundary-biased Gaussian ring so the robot starts near the failure surface.
  "medium_ring": PerturbSpec(
    init_joint_range=0.2, init_lin_vel_range=1.25, init_z_vel_range=_Z,
    init_ang_vel_range=_ANG, init_pos_z_range=0.01, init_rp_range=0.3,
    push_velocity_range={
      "x": (-1.25, 1.25), "y": (-1.25, 1.25), "z": (-_Z, _Z),
      "roll": (-1.18, 1.18), "pitch": (-1.18, 1.18), "yaw": (-1.18, 1.18),
    },
    ring=True,
  ),
  "hard_ring": PerturbSpec(
    init_joint_range=0.3, init_lin_vel_range=1.8, init_z_vel_range=_Z,
    init_ang_vel_range=_ANG, init_pos_z_range=0.01, init_rp_range=0.3,
    push_velocity_range={
      "x": (-1.8, 1.8), "y": (-1.8, 1.8), "z": (-_Z, _Z),
      "roll": (-1.18, 1.18), "pitch": (-1.18, 1.18), "yaw": (-1.18, 1.18),
    },
    ring=True,
  ),
}

# Levels used when --levels isn't passed. Excludes "none" (trivial).
_DEFAULT_LEVELS = ("small", "medium", "medium_ring", "hard_ring")


def _sample_ring(
  size: tuple[int, ...], r: float, std: float, device: torch.device,
) -> torch.Tensor:
  """Signed-Gaussian ring sampling for a scalar range (-r, r).

  Draws sign uniformly from {-1, +1} and magnitude from N(r, std^2), clamping
  magnitudes to >= 0. Fixed std means the ring width is the same across levels.
  """
  sign = (torch.randint(0, 2, size, device=device).float() * 2.0 - 1.0)
  mag = torch.normal(mean=r, std=std, size=size, device=device).abs()
  return sign * mag


def _sample_range(
  size: tuple[int, ...], lo: float, hi: float, ring: bool, std: float,
  device: torch.device,
) -> torch.Tensor:
  """Sample from (lo, hi): uniform when ring=False, ring-Gaussian when ring=True.

  Ring mode assumes a symmetric range and uses r = (hi - lo) / 2.
  """
  if ring:
    r = 0.5 * (hi - lo)
    return _sample_ring(size, r, std, device)
  return torch.empty(size, device=device).uniform_(lo, hi)


def make_ring_push_fn(
  lin_vel_sigma: float, ang_vel_sigma: float, time_cutoff_s: float = 17.0
):
  """Build a push event function that samples pushes from a Gaussian ring.

  Drop-in replacement for ``push_by_setting_velocity_with_cutoff`` that biases
  push samples toward the boundary of the per-axis range. Linear velocity axes
  (x, y, z) use ``lin_vel_sigma``; angular axes (roll, pitch, yaw) use
  ``ang_vel_sigma``. Sigmas are fixed so ring width is constant across levels.
  """
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  _DEFAULT_CFG = SceneEntityCfg("robot")
  _SIGMA = {
    "x": lin_vel_sigma, "y": lin_vel_sigma, "z": lin_vel_sigma,
    "roll": ang_vel_sigma, "pitch": ang_vel_sigma, "yaw": ang_vel_sigma,
  }

  def ring_push(
    env,
    env_ids,
    velocity_range: dict,
    time_cutoff_s: float = time_cutoff_s,
    asset_cfg: SceneEntityCfg = _DEFAULT_CFG,
  ) -> None:
    cutoff_step = int(time_cutoff_s / env.step_dt)
    mask = env.episode_length_buf[env_ids] < cutoff_step
    if not bool(mask.any()):
      return
    active_ids = env_ids[mask]
    asset = env.scene[asset_cfg.name]
    vel_w = asset.data.root_link_vel_w[active_ids]  # (K, 6)
    device = env.device
    delta = torch.zeros_like(vel_w)
    for i, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
      lo, hi = velocity_range.get(key, (0.0, 0.0))
      r = 0.5 * (hi - lo)
      if r <= 0.0:
        continue
      delta[:, i] = _sample_ring((vel_w.shape[0],), r, _SIGMA[key], device)
    asset.write_root_link_velocity_to_sim(vel_w + delta, env_ids=active_ids)

  return ring_push

TRAJ_LABELS = [
  "\u0394x (m)", "\u0394y (m)", "\u0394z (m)",
  "\u0394roll (deg)", "\u0394pitch (deg)", "\u0394yaw (deg)",
  "vx (m/s)", "vy (m/s)", "vz (m/s)",
  "\u03c9x (rad/s)", "\u03c9y (rad/s)", "\u03c9z (rad/s)",
]


def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
  """Convert (w, x, y, z) quaternion to (roll, pitch, yaw) in radians."""
  w, x, y, z = quat.unbind(dim=-1)
  sinr_cosp = 2.0 * (w * x + y * z)
  cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
  roll = torch.atan2(sinr_cosp, cosr_cosp)
  sinp = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
  pitch = torch.asin(sinp)
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  yaw = torch.atan2(siny_cosp, cosy_cosp)
  return torch.stack([roll, pitch, yaw], dim=-1)


def parse_policy_arg(arg: str) -> tuple[Path, str]:
  """Parse 'path:name' into (Path, name).  Falls back to stem as name."""
  if ":" not in arg:
    path = Path(arg)
    return path, path.stem
  path_str, name = arg.rsplit(":", 1)
  return Path(path_str), name


def _apply_init_noise_balance(
  robot,
  env_id: int,
  device: torch.device,
  init_joint_range: float,
  init_lin_vel_range: float,
  init_z_vel_range: float,
  init_ang_vel_range: float,
  init_pos_z_range: float,
  init_rp_range: float,
  ring: bool = False,
  ring_lin_vel_sigma: float = 0.10,
  ring_ang_vel_sigma: float = 0.10,
  ring_joint_sigma: float = 0.02,
  ring_rp_sigma: float = 0.02,
) -> None:
  """Apply init noise to joint positions and base velocity for one env.

  After a balance reset the robot is at rest, so we add noise around that state.
  Velocities are read in body frame, noised, then transformed to world frame
  before writing (write_root_state_to_sim expects world-frame velocities).

  When ``ring=True`` each scalar range (-r, r) is sampled from a signed Gaussian
  concentrated near the boundary — biases eval toward failure-prone states.
  Sigmas are fixed so ring width is constant regardless of the ring radius.
  """
  env_ids = torch.tensor([env_id], device=device)

  if init_joint_range > 0.0:
    n_j = robot.data.joint_pos.shape[1]
    noise_j = _sample_range(
      (1, n_j), -init_joint_range, init_joint_range, ring, ring_joint_sigma, device,
    )
    noisy_jpos = robot.data.joint_pos[env_ids] + noise_j
    if hasattr(robot.data, "soft_joint_pos_limits"):
      limits = robot.data.soft_joint_pos_limits[env_ids]  # (1, N, 2)
      noisy_jpos = torch.clamp(noisy_jpos, limits[:, :, 0], limits[:, :, 1])
    robot.write_joint_state_to_sim(
      noisy_jpos, robot.data.joint_vel[env_ids], env_ids=env_ids
    )

  apply_pose_or_vel = (
    init_lin_vel_range > 0.0 or init_z_vel_range > 0.0 or init_ang_vel_range > 0.0
    or init_pos_z_range > 0.0 or init_rp_range > 0.0
  )
  if apply_pose_or_vel:
    pos = robot.data.root_link_pos_w[env_ids].clone()
    quat = robot.data.root_link_quat_w[env_ids].clone()

    if init_pos_z_range > 0.0:
      pos[:, 2:3] += _sample_range(
        (1, 1), -init_pos_z_range, init_pos_z_range, ring, ring_lin_vel_sigma, device,
      )
    if init_rp_range > 0.0:
      roll = _sample_range(
        (1,), -init_rp_range, init_rp_range, ring, ring_rp_sigma, device,
      )
      pitch = _sample_range(
        (1,), -init_rp_range, init_rp_range, ring, ring_rp_sigma, device,
      )
      delta = quat_from_euler_xyz(roll, pitch, torch.zeros(1, device=device))
      quat = quat_mul(delta, quat)

    lin_vel_b = robot.data.root_link_lin_vel_b[env_ids].clone()
    ang_vel_b = robot.data.root_link_ang_vel_b[env_ids].clone()

    if init_lin_vel_range > 0.0:
      lin_vel_b[:, :2] += _sample_range(
        (1, 2), -init_lin_vel_range, init_lin_vel_range, ring, ring_lin_vel_sigma, device,
      )
    if init_z_vel_range > 0.0:
      lin_vel_b[:, 2:3] += _sample_range(
        (1, 1), -init_z_vel_range, init_z_vel_range, ring, ring_lin_vel_sigma, device,
      )
    if init_ang_vel_range > 0.0:
      ang_vel_b += _sample_range(
        (1, 3), -init_ang_vel_range, init_ang_vel_range, ring, ring_ang_vel_sigma, device,
      )

    lin_vel_w = quat_apply(quat, lin_vel_b)
    ang_vel_w = quat_apply(quat, ang_vel_b)

    root_state = torch.cat([pos, quat, lin_vel_w, ang_vel_w], dim=-1)
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)

  robot.clear_state(env_ids=env_ids)


def _apply_tracking_init_noise_to_cfg(
  motion_cmd: MotionCommand,
  spec: PerturbSpec,
) -> None:
  """Update MotionCommandCfg init noise fields for the given perturbation spec."""
  cfg = motion_cmd.cfg
  cfg.joint_position_range = (-spec.init_joint_range, spec.init_joint_range)
  if spec.init_lin_vel_range == 0.0:
    cfg.velocity_range = {}
    cfg.pose_range = {}
  else:
    lv = spec.init_lin_vel_range
    zv = spec.init_z_vel_range
    av = spec.init_ang_vel_range
    rp = spec.init_rp_range
    pz = spec.init_pos_z_range
    cfg.velocity_range = {
      "x": (-lv, lv), "y": (-lv, lv), "z": (-zv, zv),
      "roll": (-av, av), "pitch": (-av, av), "yaw": (-av, av),
    }
    cfg.pose_range = {
      "x": (-0.05, 0.05), "y": (-0.05, 0.05),
      "z": (-pz, pz),
      "roll": (-rp, rp), "pitch": (-rp, rp),
      "yaw": (-rp * 2, rp * 2),
    }


@torch.no_grad()
def rollout(
  env: RslRlVecEnvWrapper,
  policy,
  num_episodes: int,
  device: torch.device,
  motion_cmd: "MotionCommand | None" = None,
  perturb_spec: "PerturbSpec | None" = None,
  is_tracking: bool = False,
) -> dict:
  """Run episodes and collect aggregate metrics.

  Epos/Evel/Eacc are always computed.  For tracking tasks the reference
  comes from *motion_cmd*; for balance tasks the reference is the initial
  body positions (equilibrium) with zero velocity / acceleration.
  """
  robot = env.unwrapped.scene["robot"]
  num_envs = env.num_envs
  has_motion = motion_cmd is not None

  joint_l2_sum = torch.zeros(num_envs, device=device)
  base_l2_sum = torch.zeros(num_envs, device=device)
  base_z_sum = torch.zeros(num_envs, device=device)
  step_count = torch.zeros(num_envs, device=device)

  action_rate_sum = torch.zeros(num_envs, device=device)
  action_rate_count = torch.zeros(num_envs, device=device)
  has_prev_action = torch.zeros(num_envs, dtype=torch.bool, device=device)
  prev_actions: torch.Tensor | None = None

  obs, _ = env.reset()
  base_origin = robot.data.root_link_pos_w[:, :2].clone()
  default_joint_pos = robot.data.default_joint_pos.clone()

  # Epos / Evel / Eacc -- always computed.
  # For tracking: ref = motion_cmd body positions (subset of bodies).
  # For balance: ref = initial body positions (all bodies, constant).
  epos_sum = torch.zeros(num_envs, device=device)
  evel_sum = torch.zeros(num_envs, device=device)
  eacc_sum = torch.zeros(num_envs, device=device)
  epos_count = torch.zeros(num_envs, device=device)
  evel_count = torch.zeros(num_envs, device=device)
  eacc_count = torch.zeros(num_envs, device=device)

  if has_motion:
    prev_actual_bp = motion_cmd.robot_body_pos_w.clone()
    prev_ref_bp = motion_cmd.body_pos_w.clone()
  else:
    prev_actual_bp = robot.data.body_link_pos_w.clone()
    prev_ref_bp = robot.data.body_link_pos_w.clone()
    ref_body_origin = robot.data.body_link_pos_w.clone()

  prev_actual_vel = torch.zeros_like(prev_actual_bp)
  prev_ref_vel = torch.zeros_like(prev_ref_bp)
  vel_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
  acc_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)

  successes: list[bool] = []
  joint_l2s: list[float] = []
  base_l2s: list[float] = []
  base_zs: list[float] = []
  action_rates: list[float] = []
  epos_list: list[float] = []
  evel_list: list[float] = []
  eacc_list: list[float] = []

  pbar = tqdm(total=num_episodes, desc="Episodes", unit="ep")
  while len(successes) < num_episodes:
    joint_l2_sum += torch.norm(
      robot.data.joint_pos - default_joint_pos, dim=-1,
    )
    base_l2_sum += torch.norm(
      robot.data.root_link_pos_w[:, :2] - base_origin, dim=-1,
    )
    base_z_sum += robot.data.root_link_pos_w[:, 2]
    step_count += 1

    # --- Epos / Evel / Eacc ---
    if has_motion:
      actual_bp = motion_cmd.robot_body_pos_w
      ref_bp = motion_cmd.body_pos_w
    else:
      actual_bp = robot.data.body_link_pos_w
      ref_bp = ref_body_origin  # constant equilibrium

    epos_sum += torch.norm(ref_bp - actual_bp, dim=-1).mean(dim=-1)
    epos_count += 1

    actual_vel = actual_bp - prev_actual_bp
    ref_vel = ref_bp - prev_ref_bp  # zero for balance (ref is constant)
    if vel_valid.any():
      evel_sum[vel_valid] += torch.norm(
        ref_vel[vel_valid] - actual_vel[vel_valid], dim=-1,
      ).mean(dim=-1)
      evel_count[vel_valid] += 1

    actual_acc = actual_vel - prev_actual_vel
    ref_acc = ref_vel - prev_ref_vel  # zero for balance
    if acc_valid.any():
      eacc_sum[acc_valid] += torch.norm(
        ref_acc[acc_valid] - actual_acc[acc_valid], dim=-1,
      ).mean(dim=-1)
      eacc_count[acc_valid] += 1

    prev_actual_vel = actual_vel.clone()
    prev_ref_vel = ref_vel.clone()
    prev_actual_bp = actual_bp.clone()
    prev_ref_bp = ref_bp.clone()
    acc_valid = vel_valid.clone()
    vel_valid[:] = True

    actions = policy(obs)
    obs, _, dones, extras = env.step(actions)

    if prev_actions is not None and has_prev_action.any():
      valid = has_prev_action
      action_rate_sum[valid] += torch.norm(
        actions[valid] - prev_actions[valid], dim=-1,
      )
      action_rate_count[valid] += 1
    prev_actions = actions.clone()
    has_prev_action[:] = True

    done_mask = dones.bool()
    if not done_mask.any():
      continue

    done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
    if done_ids.dim() == 0:
      done_ids = done_ids.unsqueeze(0)

    time_outs = extras["time_outs"]
    for i in done_ids:
      if len(successes) >= num_episodes:
        break
      idx = i.item()
      sc = step_count[idx].item()
      if sc > 0:
        successes.append(bool(time_outs[idx]))
        joint_l2s.append((joint_l2_sum[idx] / sc).item())
        base_l2s.append((base_l2_sum[idx] / sc).item())
        base_zs.append((base_z_sum[idx] / sc).item())
        ar_c = action_rate_count[idx].item()
        action_rates.append((action_rate_sum[idx] / max(ar_c, 1)).item())
        epos_list.append((epos_sum[idx] / max(epos_count[idx].item(), 1)).item())
        evel_list.append((evel_sum[idx] / max(evel_count[idx].item(), 1)).item())
        eacc_list.append((eacc_sum[idx] / max(eacc_count[idx].item(), 1)).item())
        pbar.update(1)

    joint_l2_sum[done_ids] = 0.0
    base_l2_sum[done_ids] = 0.0
    base_z_sum[done_ids] = 0.0
    step_count[done_ids] = 0.0
    base_origin[done_ids] = robot.data.root_link_pos_w[done_ids, :2].clone()
    action_rate_sum[done_ids] = 0.0
    action_rate_count[done_ids] = 0.0
    has_prev_action[done_ids] = False

    epos_sum[done_ids] = 0.0
    evel_sum[done_ids] = 0.0
    eacc_sum[done_ids] = 0.0
    epos_count[done_ids] = 0.0
    evel_count[done_ids] = 0.0
    eacc_count[done_ids] = 0.0
    vel_valid[done_ids] = False
    acc_valid[done_ids] = False
    if has_motion:
      prev_actual_bp[done_ids] = motion_cmd.robot_body_pos_w[done_ids].clone()
      prev_ref_bp[done_ids] = motion_cmd.body_pos_w[done_ids].clone()
    else:
      prev_actual_bp[done_ids] = robot.data.body_link_pos_w[done_ids].clone()
      prev_ref_bp[done_ids] = robot.data.body_link_pos_w[done_ids].clone()
      ref_body_origin[done_ids] = robot.data.body_link_pos_w[done_ids].clone()
    prev_actual_vel[done_ids] = 0.0
    prev_ref_vel[done_ids] = 0.0

    # Apply balance init noise for the new episode (after ref is captured).
    if not is_tracking and perturb_spec is not None:
      for i in done_ids:
        _apply_init_noise_balance(
          robot, i.item(), device,
          perturb_spec.init_joint_range, perturb_spec.init_lin_vel_range,
          perturb_spec.init_z_vel_range, perturb_spec.init_ang_vel_range,
          perturb_spec.init_pos_z_range, perturb_spec.init_rp_range,
          ring=perturb_spec.ring,
          ring_lin_vel_sigma=perturb_spec.ring_lin_vel_sigma,
          ring_ang_vel_sigma=perturb_spec.ring_ang_vel_sigma,
          ring_joint_sigma=perturb_spec.ring_joint_sigma,
          ring_rp_sigma=perturb_spec.ring_rp_sigma,
        )

  pbar.close()
  return {
    "success_rate": float(np.mean(successes)),
    "mean_joint_l2": float(np.mean(joint_l2s)),
    "mean_base_l2": float(np.mean(base_l2s)),
    "mean_base_z": float(np.mean(base_zs)),
    "mean_action_rate": float(np.mean(action_rates)),
    "mean_epos": float(np.mean(epos_list)),
    "mean_evel": float(np.mean(evel_list)),
    "mean_eacc": float(np.mean(eacc_list)),
    "num_episodes": len(successes),
  }


@torch.no_grad()
def collect_trajectories(
  env: RslRlVecEnvWrapper,
  policy,
  num_trajectories: int,
  device: torch.device,
  motion_cmd: "MotionCommand | None" = None,
  record_env_id: int = 0,
  perturb_spec: "PerturbSpec | None" = None,
  is_tracking: bool = False,
) -> list[dict]:
  """Collect trajectory data from a fixed env for reproducible comparison.

  Always records from *record_env_id* so that ep0, ep1, ... correspond
  to the same initial conditions across policies (given the same seed).

  For balance tasks the reference (equilibrium) is captured from the clean
  reset state immediately after env.step() triggers a reset — before any
  init noise is applied. This is policy-independent: it is the nominal
  standing pose the environment always resets to, regardless of which policy
  is running or how large the perturbation is.
  """
  robot = env.unwrapped.scene["robot"]
  dt = env.unwrapped.step_dt
  eid = record_env_id

  def _capture_ref():
    """Capture equilibrium from the current (clean reset) robot state."""
    return (
      robot.data.root_link_pos_w[eid, :3].clone(),
      torch.rad2deg(
        quat_to_euler_xyz(robot.data.root_link_quat_w[eid : eid + 1])
      ).squeeze(0).clone(),
    )

  def _snapshot():
    """Return (x,y,z,roll,pitch,yaw,vx,vy,vz,wx,wy,wz) for the recorded env."""
    pos = robot.data.root_link_pos_w[eid, :3]
    rpy = torch.rad2deg(
      quat_to_euler_xyz(robot.data.root_link_quat_w[eid : eid + 1])
    ).squeeze(0)
    lin_vel = robot.data.root_link_lin_vel_w[eid]
    ang_vel = robot.data.root_link_ang_vel_w[eid]
    return torch.cat([pos, rpy, lin_vel, ang_vel], dim=-1).cpu().numpy()

  obs, _ = env.reset()

  traj_buf: list[np.ndarray] = []
  ref_buf: list[np.ndarray] = []
  trajectories: list[dict] = []
  ep_steps = 0

  zeros6 = torch.zeros(6, device=device)

  # Capture reference from the clean reset state (policy-independent equilibrium).
  if motion_cmd is None:
    ref_origin, ref_rpy = _capture_ref()
  else:
    ref_origin, ref_rpy = None, None

  while len(trajectories) < num_trajectories:
    traj_buf.append(_snapshot())

    if motion_cmd is not None:
      ref_pos = motion_cmd.anchor_pos_w[eid]
      ref_rpy_deg = torch.rad2deg(
        quat_to_euler_xyz(motion_cmd.anchor_quat_w[eid : eid + 1])
      ).squeeze(0)
      ref_lin_vel = motion_cmd.anchor_lin_vel_w[eid]
      ref_ang_vel = motion_cmd.anchor_ang_vel_w[eid]
      ref_buf.append(
        torch.cat([ref_pos, ref_rpy_deg, ref_lin_vel, ref_ang_vel], dim=-1).cpu().numpy()
      )
    else:
      ref_buf.append(
        torch.cat([ref_origin, ref_rpy, zeros6], dim=-1).cpu().numpy()
      )

    ep_steps += 1

    actions = policy(obs)
    obs, _, dones, extras = env.step(actions)

    if dones[eid]:
      success = bool(extras["time_outs"][eid])
      if len(traj_buf) >= 2:
        trajectories.append({
          "actual": np.array(traj_buf),
          "reference": np.array(ref_buf),
          "dt": dt,
          "success": success,
          "duration_s": ep_steps * dt,
        })

      traj_buf = []
      ref_buf = []
      ep_steps = 0

      # After env.step() the done env has been reset to a clean state.
      # Capture reference now (policy-independent equilibrium), then apply noise.
      if motion_cmd is None:
        ref_origin, ref_rpy = _capture_ref()

      if not is_tracking and perturb_spec is not None:
        _apply_init_noise_balance(
          robot, eid, device,
          perturb_spec.init_joint_range, perturb_spec.init_lin_vel_range,
          perturb_spec.init_z_vel_range, perturb_spec.init_ang_vel_range,
          perturb_spec.init_pos_z_range, perturb_spec.init_rp_range,
          ring=perturb_spec.ring,
          ring_lin_vel_sigma=perturb_spec.ring_lin_vel_sigma,
          ring_ang_vel_sigma=perturb_spec.ring_ang_vel_sigma,
          ring_joint_sigma=perturb_spec.ring_joint_sigma,
          ring_rp_sigma=perturb_spec.ring_rp_sigma,
        )

  return trajectories


def plot_results(
  all_results: dict[str, dict[str, dict]],
  labels: list[str],
  output_path: Path,
  title_suffix: str = "",
) -> None:
  """Generate grouped bar plot with adaptive panel count."""
  policy_names = list(all_results.keys())
  n_policies = len(policy_names)
  n_levels = len(labels)

  # Row 1: success rate, joint L2, base L2, base Z height.
  # Row 2: action rate, Epos, Evel, Eacc.
  metrics = [
    ("success_rate", "Success Rate (%)", lambda v: v * 100),
    ("mean_joint_l2", "Mean Joint L2 (rad)", lambda v: v),
    ("mean_base_l2", "Mean Base L2 (m)", lambda v: v),
    ("mean_base_z", "Mean Base Z (m)", lambda v: v),
    ("mean_action_rate", "Action Rate (rad/frame)", lambda v: v),
    ("mean_epos", "Epos (m)", lambda v: v),
    ("mean_evel", "Evel (m/frame)", lambda v: v),
    ("mean_eacc", u"Eacc (m/frame\u00b2)", lambda v: v),
  ]

  n_metrics = len(metrics)
  ncols = min(4, n_metrics)
  nrows = (n_metrics + ncols - 1) // ncols

  fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
  axes = np.atleast_1d(axes).flatten()

  x = np.arange(n_levels)
  width = 0.8 / max(n_policies, 1)
  colors = plt.cm.Set2(np.linspace(0, 1, max(n_policies, 1)))

  for ax, (key, ylabel, transform) in zip(axes, metrics):
    for i, name in enumerate(policy_names):
      values = [transform(all_results[name][lbl][key]) for lbl in labels]
      offset = (i - (n_policies - 1) / 2) * width
      ax.bar(x + offset, values, width, label=name, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Perturbation Level")
    if n_policies > 1:
      ax.legend(fontsize=7)

  for ax in axes[n_metrics:]:
    ax.set_visible(False)

  fig.suptitle(f"Policy Evaluation{title_suffix}", fontsize=14)
  fig.tight_layout()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=300, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved plot -> {output_path}")


def plot_episode_trajectory(
  traj: dict,
  output_path: Path,
  policy_name: str,
  perturb_label: str,
  episode_idx: int,
) -> None:
  """Plot 12-panel figure: pose (relative) + velocity vs time."""
  actual = traj["actual"].copy()       # (T, 12)
  reference = traj["reference"].copy()  # (T, 12)
  t = np.arange(actual.shape[0]) * traj["dt"]
  outcome = "survived" if traj["success"] else "fell"

  # Unwrap orientation columns (roll=3, pitch=4, yaw=5) to avoid ±180 jumps.
  for col in (3, 4, 5):
    actual[:, col] = np.rad2deg(np.unwrap(np.deg2rad(actual[:, col])))
    reference[:, col] = np.rad2deg(np.unwrap(np.deg2rad(reference[:, col])))

  # Make position/orientation (first 6 channels) relative to initial reference.
  # Velocity channels (6-11) stay absolute.
  origin = reference[0, :6].copy()
  actual[:, :6] -= origin
  reference[:, :6] -= origin

  fig, axes = plt.subplots(4, 3, figsize=(14, 14), sharex=True)
  axes = axes.flatten()

  min_half_range = [
    0.1, 0.1, 0.1,     # m
    5.0, 5.0, 5.0,     # deg
    0.5, 0.5, 0.5,     # m/s
    1.0, 1.0, 1.0,     # rad/s
  ]

  for i, (ax, label) in enumerate(zip(axes, TRAJ_LABELS)):
    ax.plot(t, actual[:, i], linewidth=1.2, label="actual")
    ax.plot(t, reference[:, i], "--", linewidth=1.0, alpha=0.7, label="reference")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    if i == 0:
      ax.legend(fontsize=8)

    lo, hi = ax.get_ylim()
    mid = (lo + hi) / 2
    half = max((hi - lo) / 2, min_half_range[i])
    ax.set_ylim(mid - half, mid + half)

  for ax in axes[9:]:
    ax.set_xlabel("Time (s)")

  fig.suptitle(
    f"{policy_name} | push={perturb_label} | ep {episode_idx} "
    f"({outcome}, {traj['duration_s']:.1f}s)",
    fontsize=12,
  )
  fig.tight_layout()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=300, bbox_inches="tight")
  plt.close(fig)


def main():
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  all_tasks = list_tasks()

  parser = argparse.ArgumentParser(
    description="Evaluate trained policies across perturbation levels.",
  )
  parser.add_argument("task", choices=all_tasks, help="Registered task ID.")
  parser.add_argument(
    "--policies", nargs="+", required=True, metavar="PATH:NAME",
    help="One or more checkpoint_path:display_name pairs.",
  )
  parser.add_argument("--num-episodes", type=int, default=200,
                      help="Episodes per (policy, perturbation) condition.")
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument(
    "--levels", nargs="+", default=list(_DEFAULT_LEVELS),
    choices=list(PERTURB_LEVELS),
    help=f"Perturbation levels to evaluate. Default: {list(_DEFAULT_LEVELS)}. "
         f"Pass 'none' explicitly to include the zero-perturbation baseline.",
  )
  parser.add_argument("--motion-file", type=str, default=None,
                      help="NPZ motion file (required for tracking tasks).")
  parser.add_argument("--output-dir", type=str, default=None,
                      help="Dir for combined comparison output.")
  parser.add_argument("--num-trajectory-plots", type=int, default=2,
                      help="Number of example episode trajectories to plot per "
                           "(policy, perturbation) condition. Set to 0 to disable.")
  parser.add_argument("--device", type=str, default=None)
  parser.add_argument("--seed", type=int, default=42,
                      help="RNG seed for reproducibility. Each perturbation level "
                           "reseeds so re-runs are deterministic.")
  args = parser.parse_args()

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  policies = [parse_policy_arg(p) for p in args.policies]
  for path, name in policies:
    if not path.exists():
      parser.error(f"Checkpoint not found: {path}")

  # --- Build environment (training config, not play) ---
  env_cfg = load_env_cfg(args.task, play=False)
  agent_cfg = load_rl_cfg(args.task)

  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.curriculum = {}
  env_cfg.scene.num_envs = args.num_envs

  # Handle tracking tasks that require a motion file.
  is_tracking = (
    env_cfg.commands
    and "motion" in env_cfg.commands
    and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
  )
  if is_tracking:
    if not args.motion_file:
      parser.error("Tracking tasks require --motion-file path/to/motion.npz")
    motion_path = Path(args.motion_file).expanduser().resolve()
    if not motion_path.exists():
      parser.error(f"Motion file not found: {motion_path}")
    env_cfg.commands["motion"].motion_file = str(motion_path)
    env_cfg.commands["motion"].sampling_mode = "start"

  # Zero out env-level init noise — all variation comes from perturb_spec instead.
  if is_tracking:
    motion_cfg = env_cfg.commands["motion"]
    motion_cfg.pose_range = {}
    motion_cfg.velocity_range = {}
    motion_cfg.joint_position_range = (0.0, 0.0)
    print("[eval] Zeroed tracking pose/vel/joint noise (controlled per-level).")
  if hasattr(env_cfg, "events") and "reset_base" in (env_cfg.events or {}):
    rb = env_cfg.events["reset_base"]
    rb.params["pose_range"] = {k: (0.0, 0.0) for k in rb.params.get("pose_range", {})}
    rb.params["velocity_range"] = {}
    print("[eval] Zeroed balance reset_base pose/vel noise (controlled per-level).")
  if hasattr(env_cfg, "events") and "reset_robot_joints" in (env_cfg.events or {}):
    rj = env_cfg.events["reset_robot_joints"]
    rj.params["position_range"] = (0.0, 0.0)
    rj.params["velocity_range"] = (0.0, 0.0)
  if hasattr(env_cfg, "events") and "reset_rp_noise" in (env_cfg.events or {}):
    env_cfg.events["reset_rp_noise"].params["rp_range"] = 0.0

  has_push = hasattr(env_cfg, "events") and "push_robot" in (env_cfg.events or {})
  if not has_push:
    print("[WARN] Task has no push_robot event; only evaluating without perturbation.")
    args.levels = ["none"]

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # Resolve motion command for tracking reference in trajectory plots.
  motion_cmd: MotionCommand | None = None
  if is_tracking:
    motion_cmd = env.command_manager.get_term("motion")

  # --- Evaluate each policy ---
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunnerWithEval
  runner = runner_cls(env_wrapped, asdict(agent_cfg), device=device)

  all_results: dict[str, dict[str, dict]] = {}

  for ckpt_path, policy_name in policies:
    print(f"\n{'=' * 60}")
    print(f"Policy: {policy_name} ({ckpt_path})")
    print(f"{'=' * 60}")

    runner.load(
      str(ckpt_path), load_cfg={"actor": True}, strict=True, map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    policy_results: dict[str, dict] = {}
    for level_name in args.levels:
      spec = PERTURB_LEVELS[level_name]
      print(f"\n--- Perturbation: {level_name} ---")

      # Seed RNG for this perturbation level. Using the same seed per
      # perturbation level across policies ensures identical initial
      # conditions and push sequences (until episode-length divergence).
      level_seed = args.seed + hash(level_name) % (2**31)
      torch.manual_seed(level_seed)
      np.random.seed(level_seed % (2**32))

      # Update push_robot event velocity range for this level.
      if has_push:
        push_cfg = env_wrapped.unwrapped.event_manager.get_term_cfg("push_robot")
        # Cache the original push func once so non-ring levels can restore it.
        if not hasattr(push_cfg, "_original_func_cache"):
          push_cfg._original_func_cache = push_cfg.func
        if spec.push_velocity_range:
          push_cfg.params["velocity_range"] = copy.deepcopy(spec.push_velocity_range)
          push_cfg.interval_range_s = (2.0, 5.0)
          if spec.ring:
            push_cfg.func = make_ring_push_fn(spec.ring_lin_vel_sigma, spec.ring_ang_vel_sigma)
          else:
            push_cfg.func = push_cfg._original_func_cache
        else:
          push_cfg.params["velocity_range"] = {
            k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
          }
          push_cfg.func = push_cfg._original_func_cache

      # For tracking tasks, update MotionCommandCfg init noise.
      if is_tracking:
        _apply_tracking_init_noise_to_cfg(motion_cmd, spec)

      metrics = rollout(
        env_wrapped, policy, args.num_episodes, torch.device(device),
        motion_cmd=motion_cmd, perturb_spec=spec, is_tracking=is_tracking,
      )
      policy_results[level_name] = metrics

      print(
        f"  success_rate={metrics['success_rate']:.2%}  "
        f"joint_l2={metrics['mean_joint_l2']:.4f}  "
        f"base_l2={metrics['mean_base_l2']:.4f}  "
        f"base_z={metrics['mean_base_z']:.4f}"
        f"\n  action_rate={metrics['mean_action_rate']:.4f}  "
        f"epos={metrics['mean_epos']:.4f}  "
        f"evel={metrics['mean_evel']:.6f}  "
        f"eacc={metrics['mean_eacc']:.6f}"
      )

      # Separate seeded pass for trajectory plots. The seed is
      # deterministic per perturbation level (not per policy), so
      # ep0 gets identical initial conditions across all policies.
      if args.num_trajectory_plots > 0:
        traj_seed = args.seed + 7919 + hash(level_name) % (2**31)
        torch.manual_seed(traj_seed)
        np.random.seed(traj_seed % (2**32))

        traj_list = collect_trajectories(
          env_wrapped, policy, args.num_trajectory_plots,
          torch.device(device), motion_cmd=motion_cmd,
          perturb_spec=spec, is_tracking=is_tracking,
        )
        run_dir = ckpt_path.resolve().parent
        traj_dir = run_dir / "evaluation" / "trajectories"
        for ep_i, traj in enumerate(traj_list):
          plot_episode_trajectory(
            traj,
            traj_dir / f"traj_{policy_name}_{level_name}_ep{ep_i}.png",
            policy_name, level_name, ep_i,
          )
        print(f"  Saved {len(traj_list)} trajectory plots -> {traj_dir}")

    all_results[policy_name] = policy_results

    # --- Per-policy output ---
    run_dir = ckpt_path.resolve().parent
    eval_dir = run_dir / "evaluation"

    per_policy_data = {
      "task": args.task,
      "checkpoint": str(ckpt_path.resolve()),
      "policy_name": policy_name,
      "seed": args.seed,
      "episode_length_s": env_wrapped.unwrapped.max_episode_length_s,
      "num_episodes_requested": args.num_episodes,
      "levels": args.levels,
      "results": policy_results,
    }
    eval_dir.mkdir(parents=True, exist_ok=True)
    json_path = eval_dir / f"eval_{policy_name}.json"
    with open(json_path, "w") as f:
      json.dump(per_policy_data, f, indent=2)
    print(f"\nSaved JSON -> {json_path}")

    plot_results(
      {policy_name: policy_results},
      args.levels,
      eval_dir / f"eval_{policy_name}.png",
    )

  # --- Combined comparison ---
  if len(policies) > 1:
    if args.output_dir:
      combined_dir = Path(args.output_dir)
    else:
      combined_dir = policies[0][0].resolve().parent / "evaluation"
    combined_dir.mkdir(parents=True, exist_ok=True)

    combined_data = {
      "task": args.task,
      "seed": args.seed,
      "episode_length_s": env_wrapped.unwrapped.max_episode_length_s,
      "num_episodes_requested": args.num_episodes,
      "levels": args.levels,
      "results": {
        name: {"checkpoint": str(path.resolve()), **all_results[name]}
        for path, name in policies
      },
    }
    json_path = combined_dir / "eval_comparison.json"
    with open(json_path, "w") as f:
      json.dump(combined_data, f, indent=2)
    print(f"\nSaved combined JSON -> {json_path}")

    plot_results(all_results, args.levels, combined_dir / "eval_comparison.png")

  env_wrapped.close()
  print("\nDone.")


if __name__ == "__main__":
  main()
