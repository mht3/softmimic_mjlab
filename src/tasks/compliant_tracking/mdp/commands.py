"""Compliant motion command for SoftMimic-style augmented reference tracking.

Extends mjlab's ``MotionCommand`` to sample from a directory of augmented
motions (SoftMimic compliant motion augmentation). Each motion NPZ carries the
*adapted* reference (BeyondMimic format) plus per-frame force/torque targets,
desired stiffness, and forcefield metadata used to reactively compute external
wrenches, mirroring ``ComplianceAugmentedReferenceCommand`` from the SoftMimic
Isaac Lab release.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

if TYPE_CHECKING:
  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

# Force-event modes mixed for hybrid training, following the SoftMimic authors:
# all forcefield + all collision-emulator files, and the zero-wrench files
# repeated to give each mode equal weight (e.g. 40 + 40 + 5 x 8).
_HYBRID_MODES = ("forcefield", "collision-emulator", "zero-wrench")

# Stiffness threshold marking the start of a force event (SoftMimic value).
_FF_ACTIVE_THRESHOLD = 0.1

_COMPLIANT_KEYS = (
  "force_body_index",
  "force_vector",
  "torque_vector",
  "stiffness",
  "rotational_stiffness",
  "ff_stiffness",
  "ff_rotational_stiffness",
  "ff_origin",
  "ff_setpoint_rot",
  "ff_normal",
  "force_body_pos_w",
  "force_body_quat_w",
  "foot_contacts",
)

# Foot order matches the contact columns of the reference motion CSVs.
_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


def resolve_motion_files(motion_source: str) -> list[Path]:
  """Resolve a motion file/directory into the list of NPZ files to load.

  A directory containing the augmentation mode subdirectories is expanded with
  the SoftMimic hybrid mixing (zero-wrench files repeated to match the largest
  mode); any other directory loads all NPZ files uniformly; a single NPZ file
  loads as-is.
  """
  source = Path(motion_source)
  if source.is_file():
    return [source]
  if not source.is_dir():
    raise FileNotFoundError(f"Motion source not found: {source}")

  mode_dirs = [source / mode for mode in _HYBRID_MODES]
  if all(d.is_dir() for d in mode_dirs):
    per_mode = {d.name: sorted(d.glob("*.npz")) for d in mode_dirs}
    max_count = max(len(files) for files in per_mode.values())
    files: list[Path] = []
    for mode in _HYBRID_MODES:
      mode_files = per_mode[mode]
      if not mode_files:
        continue
      repeats = max(max_count // len(mode_files), 1)
      files.extend(mode_files * repeats)
    if not files:
      raise FileNotFoundError(f"No NPZ files found under {source}")
    return files

  files = sorted(source.rglob("*.npz"))
  if not files:
    raise FileNotFoundError(f"No NPZ files found under {source}")
  return files


class CompliantMotionLoader:
  """Loads a set of augmented motion NPZ files as one concatenated timeline."""

  def __init__(
    self, motion_source: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    files = resolve_motion_files(motion_source)

    stacks: dict[str, list[np.ndarray]] = {}
    starts: list[int] = []
    total = 0
    for f in files:
      data = np.load(f)
      num_frames = data["joint_pos"].shape[0]
      starts.append(total)
      total += num_frames
      for key in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        *_COMPLIANT_KEYS,
      ):
        stacks.setdefault(key, []).append(data[key])

    def cat(key: str, dtype=torch.float32) -> torch.Tensor:
      return torch.tensor(
        np.concatenate(stacks[key], axis=0), dtype=dtype, device=device
      )

    self.joint_pos = cat("joint_pos")
    self.joint_vel = cat("joint_vel")
    self._body_pos_w = cat("body_pos_w")
    self._body_quat_w = cat("body_quat_w")
    self._body_lin_vel_w = cat("body_lin_vel_w")
    self._body_ang_vel_w = cat("body_ang_vel_w")
    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
    self.root_pos_w = self._body_pos_w[:, 0]
    self.root_quat_w = self._body_quat_w[:, 0]
    self.time_step_total = self.joint_pos.shape[0]

    # Compliance channels.
    self.force_body_index = cat("force_body_index", dtype=torch.long).view(-1)
    self.force_vector = cat("force_vector")
    self.torque_vector = cat("torque_vector")
    self.stiffness = cat("stiffness").view(-1)
    self.rotational_stiffness = cat("rotational_stiffness").view(-1)
    self.ff_stiffness = cat("ff_stiffness").view(-1)
    self.ff_rotational_stiffness = cat("ff_rotational_stiffness").view(-1)
    self.ff_origin = cat("ff_origin")
    self.ff_setpoint_rot = cat("ff_setpoint_rot")  # wxyz
    self.ff_normal = cat("ff_normal")
    self.force_body_pos_w = cat("force_body_pos_w")
    self.force_body_quat_w = cat("force_body_quat_w")
    self.foot_contacts = cat("foot_contacts")

    self.motion_starts = torch.tensor(starts, dtype=torch.long, device=device)
    ends = starts[1:] + [total]
    self.motion_ends = torch.tensor(ends, dtype=torch.long, device=device)
    self.num_motions = len(files)

    # "Main" reference motion for interactive play: the first zero-wrench file
    # (adapted motion ~= nominal reference, no force events).
    self.main_motion_index = 0
    for i, f in enumerate(files):
      if "zero-wrench" in str(f):
        self.main_motion_index = i
        break

  def motion_end_of(self, time_steps: torch.Tensor) -> torch.Tensor:
    """End frame (exclusive) of the motion containing each global frame."""
    motion_idx = (
      torch.searchsorted(self.motion_starts, time_steps, right=True) - 1
    ).clamp(min=0)
    return self.motion_ends[motion_idx]


class CompliantMotionCommand(MotionCommand):
  cfg: CompliantMotionCommandCfg

  def __init__(self, cfg: CompliantMotionCommandCfg, env: ManagerBasedRlEnv):
    # Mirror MotionCommand.__init__ but with the multi-motion loader (the base
    # loader would try to np.load() the motion directory).
    CommandTerm.__init__(self, cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = CompliantMotionLoader(
      self.cfg.motion_file, self.body_indexes, device=self.device
    )
    self.foot_body_indexes = torch.tensor(
      self.robot.find_bodies(_FOOT_BODY_NAMES, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_end_steps = self.motion.motion_end_of(self.time_steps)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_force"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_torque"] = torch.zeros(self.num_envs, device=self.device)

    # Ghost model created lazily on first visualization.
    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

    # Forcefield anchoring state (SoftMimic: the forcefield trajectory is
    # anchored to the robot's root pose at the moment a force event starts).
    self._ff_anchor_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self._ff_anchor_rot = torch.zeros(self.num_envs, 4, device=self.device)
    self._ff_anchor_rot[:, 0] = 1.0
    self._ff_anchor_ref_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self._last_ff_stiffness = torch.zeros(self.num_envs, device=self.device)

    self._force_vector_w = torch.zeros(self.num_envs, 3, device=self.device)
    self._torque_vector_w = torch.zeros(self.num_envs, 3, device=self.device)
    self._active_force_mask = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._ff_origin_w = torch.zeros(self.num_envs, 3, device=self.device)
    self._ff_setpoint_rot_w = torch.zeros(self.num_envs, 4, device=self.device)
    self._ff_setpoint_rot_w[:, 0] = 1.0
    self._ff_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.force_body_pos_relative_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.force_body_quat_relative_w = torch.zeros(self.num_envs, 4, device=self.device)
    self.force_body_quat_relative_w[:, 0] = 1.0

    # Viser GUI overrides for the commanded stiffness (None = use dataset).
    self._gui_stiffness: float | None = None
    self._gui_rotational_stiffness: float | None = None

    # Viser GUI motion controls. True during training (sample everything);
    # create_gui() switches the default to the main reference motion only.
    self._gui_use_augmented: bool = True
    self._gui_time_s: float = 0.0
    self._gui_pending_scrub: bool = False
    self._gui_pending_reset: bool = False
    # Time slider handle, so playback position can be written back to it (the
    # slider is otherwise write-only and stays at 0 while the motion advances).
    self._gui_time_slider = None
    # Returns the viewer's selected env index; set by create_gui().
    self._gui_get_env_idx: Callable[[], int] | None = None
    # Suppress the slider's on_update while we set its value programmatically,
    # so a playback-driven update is not mistaken for a user scrub.
    self._gui_suppress_time_cb: bool = False
    # Whether the dataset forcefield wrench is applied (see the
    # `apply_forcefield_wrench` event). Always on during training; create_gui()
    # turns it off by default for interactive play (GUI toggle re-enables it).
    self._dataset_forces_enabled: bool = True

    # Velocity conditioning: the actor observes the reference root velocity in
    # the reference heading frame (see `reference_root_lin_vel_b` /
    # `reference_root_ang_vel_b` observations). While the GUI joystick is
    # enabled these observed values are replaced by the slider command so the
    # policy can be steered — mirroring the SoftMimic deploy joystick that
    # overrides `reference_xy_vel` / `reference_yaw_vel`.
    self._gui_velocity_command: torch.Tensor | None = None

    self._update_compliance_buffers()

  # -- Compliance properties (current frame) --

  @property
  def force_body_indexes(self) -> torch.Tensor:
    """Robot body index targeted by the external wrench (-1 when none)."""
    return self.motion.force_body_index[self.time_steps]

  @property
  def dataset_forces_enabled(self) -> bool:
    """Whether the dataset forcefield wrench should be applied this step."""
    return self._dataset_forces_enabled

  @property
  def active_force_mask(self) -> torch.Tensor:
    return self._active_force_mask

  @property
  def target_forces_w(self) -> torch.Tensor:
    """Feedforward force targets from the dataset, in the world frame."""
    return self._force_vector_w

  @property
  def target_torques_w(self) -> torch.Tensor:
    return self._torque_vector_w

  @property
  def desired_foot_contacts(self) -> torch.Tensor:
    """Reference foot contact probabilities [left, right] at the current frame."""
    return self.motion.foot_contacts[self.time_steps]

  @property
  def reference_root_lin_vel_b(self) -> torch.Tensor:
    """Reference root linear velocity in the reference heading (yaw) frame.

    This is the velocity-conditioning command channel. During play the GUI
    joystick overrides the xy components; z is always taken from the dataset.
    """
    lin_vel_b = quat_apply_inverse(
      yaw_quat(self.anchor_quat_w), self.anchor_lin_vel_w
    )
    if self._gui_velocity_command is not None:
      lin_vel_b = lin_vel_b.clone()
      lin_vel_b[:, 0] = self._gui_velocity_command[0]
      lin_vel_b[:, 1] = self._gui_velocity_command[1]
    return lin_vel_b

  @property
  def reference_root_ang_vel_b(self) -> torch.Tensor:
    """Reference root angular velocity in the reference heading (yaw) frame.

    Only the yaw rate is meaningful for steering; the GUI joystick overrides it.
    """
    ang_vel_b = quat_apply_inverse(
      yaw_quat(self.anchor_quat_w), self.anchor_ang_vel_w
    )
    if self._gui_velocity_command is not None:
      ang_vel_b = ang_vel_b.clone()
      ang_vel_b[:, 2] = self._gui_velocity_command[2]
    return ang_vel_b

  @property
  def desired_stiffness(self) -> torch.Tensor:
    if self._gui_stiffness is not None:
      return torch.full(
        (self.num_envs,), self._gui_stiffness, device=self.device
      )
    return self.motion.stiffness[self.time_steps]

  @property
  def desired_rotational_stiffness(self) -> torch.Tensor:
    if self._gui_rotational_stiffness is not None:
      return torch.full(
        (self.num_envs,), self._gui_rotational_stiffness, device=self.device
      )
    return self.motion.rotational_stiffness[self.time_steps]

  @property
  def forcefield_stiffness(self) -> torch.Tensor:
    return self.motion.ff_stiffness[self.time_steps]

  @property
  def forcefield_rotational_stiffness(self) -> torch.Tensor:
    return self.motion.ff_rotational_stiffness[self.time_steps]

  @property
  def forcefield_forces_w(self) -> torch.Tensor:
    if self.cfg.force_computation_mode == "feedforward":
      return self._force_vector_w
    return self._compute_reactive_forcefield_force_w()

  @property
  def forcefield_torques_w(self) -> torch.Tensor:
    if self.cfg.force_computation_mode == "feedforward":
      return self._torque_vector_w
    return self._compute_reactive_forcefield_torque_w()

  def _force_body_pos_w(self) -> torch.Tensor:
    """Current world position of the per-env force body on the robot."""
    body_idx = self.force_body_indexes.clamp(min=0)
    env_ids = torch.arange(self.num_envs, device=self.device)
    return self.robot.data.body_link_pos_w[env_ids, body_idx]

  def _force_body_quat_w(self) -> torch.Tensor:
    body_idx = self.force_body_indexes.clamp(min=0)
    env_ids = torch.arange(self.num_envs, device=self.device)
    return self.robot.data.body_link_quat_w[env_ids, body_idx]

  def _compute_reactive_forcefield_force_w(self) -> torch.Tensor:
    """Spring force from the anchored forcefield acting on the force body."""
    force = torch.zeros(self.num_envs, 3, device=self.device)
    active = self._active_force_mask
    if not bool(active.any()):
      return force
    body_pos = self._force_body_pos_w()
    k = self.motion.ff_stiffness[self.time_steps]
    origin = self._ff_origin_w
    normal = self._ff_normal_w

    is_plane = torch.linalg.norm(normal, dim=-1) > 0.1
    plane = active & is_plane
    if bool(plane.any()):
      penetration = -torch.sum(
        (body_pos[plane] - origin[plane]) * normal[plane], dim=-1
      )
      force[plane] = (penetration.clamp(min=0.0) * k[plane]).unsqueeze(-1) * normal[
        plane
      ]
    setpoint = active & ~is_plane
    if bool(setpoint.any()):
      force[setpoint] = k[setpoint].unsqueeze(-1) * (
        origin[setpoint] - body_pos[setpoint]
      )
    # Cap at the peak force of the augmentation pipeline (SoftMimic Table IV)
    # so tracking errors cannot grow the virtual spring force unbounded.
    magnitude = force.norm(dim=-1, keepdim=True)
    scale = (self.cfg.max_force / magnitude.clamp(min=1e-6)).clamp(max=1.0)
    return force * scale

  def _compute_reactive_forcefield_torque_w(self) -> torch.Tensor:
    """Rotational spring torque toward the anchored setpoint orientation."""
    torque = torch.zeros(self.num_envs, 3, device=self.device)
    active = self._active_force_mask
    if not bool(active.any()):
      return torque
    body_quat = self._force_body_quat_w()[active]
    k_rot = self.motion.ff_rotational_stiffness[self.time_steps][active]
    delta = quat_mul(self._ff_setpoint_rot_w[active], quat_inv(body_quat))
    torque[active] = k_rot.unsqueeze(-1) * axis_angle_from_quat(delta)
    magnitude = torque.norm(dim=-1, keepdim=True)
    scale = (self.cfg.max_torque / magnitude.clamp(min=1e-6)).clamp(max=1.0)
    return torque * scale

  # -- Sampling / update overrides --

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      if self._gui_use_augmented:
        motion_ids = torch.randint(
          0, self.motion.num_motions, (len(env_ids),), device=self.device
        )
      else:
        motion_ids = torch.full(
          (len(env_ids),),
          self.motion.main_motion_index,
          dtype=torch.long,
          device=self.device,
        )
      self.time_steps[env_ids] = self.motion.motion_starts[motion_ids]
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)
    self.motion_end_steps = self.motion.motion_end_of(self.time_steps)

    self._clear_force_state(env_ids)
    self._write_reference_state(env_ids)

  def _clear_force_state(self, env_ids: torch.Tensor):
    """Clear force-event/anchor state so the next frame re-anchors cleanly."""
    self._last_ff_stiffness[env_ids] = 0.0
    self._active_force_mask[env_ids] = False
    self._force_vector_w[env_ids] = 0.0
    self._torque_vector_w[env_ids] = 0.0
    self._ff_anchor_rot[env_ids] = 0.0
    self._ff_anchor_rot[env_ids, 0] = 1.0

  def _write_reference_state(self, env_ids: torch.Tensor):
    """Teleport envs to the reference state at their current time step."""
    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    self.robot.clear_state(env_ids=env_ids)

  def _apply_gui_requests(self):
    """Apply pending viser motion controls on the env thread."""
    if not (self._gui_pending_scrub or self._gui_pending_reset):
      return
    reset = self._gui_pending_reset
    self._gui_pending_scrub = False
    self._gui_pending_reset = False

    env_ids = torch.arange(self.num_envs, device=self.device)
    if reset and self._gui_use_augmented:
      motion_idx = torch.randint(
        0, self.motion.num_motions, (self.num_envs,), device=self.device
      )
    elif reset:
      motion_idx = torch.full(
        (self.num_envs,),
        self.motion.main_motion_index,
        dtype=torch.long,
        device=self.device,
      )
    else:
      # Scrub within whatever motion each env is currently playing.
      motion_idx = (
        torch.searchsorted(self.motion.motion_starts, self.time_steps, right=True) - 1
      ).clamp(min=0)

    starts = self.motion.motion_starts[motion_idx]
    ends = self.motion.motion_ends[motion_idx]
    offset = int(self._gui_time_s / self._env.step_dt)
    self.time_steps[:] = torch.minimum(starts + offset, ends - 1)
    self.motion_end_steps = self.motion.motion_end_of(self.time_steps)
    self._clear_force_state(env_ids)
    if reset:
      self._write_reference_state(env_ids)

  def _set_push_control_active(self, active: bool) -> None:
    """Enable/disable the interactive Push command, if present (play only)."""
    try:
      push = self._env.command_manager.get_term("push_control")
    except (KeyError, ValueError, AttributeError):
      return
    setter = getattr(push, "set_controls_active", None)
    if setter is not None:
      setter(active)

  def _sync_time_slider(self):
    """Reflect the selected env's playback position back onto the time slider.

    The slider is otherwise write-only and would sit at 0 while the motion
    advances. Uses the viewer's selected env; guarded so this programmatic
    write is not treated as a user scrub.
    """
    if self._gui_time_slider is None or self._gui_get_env_idx is None:
      return
    idx = self._gui_get_env_idx()
    motion_idx = int(
      (
        torch.searchsorted(
          self.motion.motion_starts, self.time_steps[idx : idx + 1], right=True
        )
        - 1
      ).clamp(min=0)
    )
    start = int(self.motion.motion_starts[motion_idx])
    end = int(self.motion.motion_ends[motion_idx])
    t_s = (int(self.time_steps[idx]) - start) * self._env.step_dt
    max_s = round(max((end - 1 - start) * self._env.step_dt, self._env.step_dt), 1)
    self._gui_suppress_time_cb = True
    try:
      # Match the slider range to the current motion (augmented motions vary in
      # length), then set the playback position.
      if abs(self._gui_time_slider.max - max_s) > 1e-6:
        self._gui_time_slider.max = max_s
      self._gui_time_slider.value = min(round(t_s, 1), max_s)
    finally:
      self._gui_suppress_time_cb = False

  def _update_command(self):
    self._apply_gui_requests()
    self.time_steps += 1
    env_ids = torch.where(self.time_steps >= self.motion_end_steps)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)
    self._sync_time_slider()

    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

    # Anchored relative target for the force body (may not be a tracked body).
    motion_force_body_pos = (
      self.motion.force_body_pos_w[self.time_steps] + self._env.scene.env_origins
    )
    motion_force_body_quat = self.motion.force_body_quat_w[self.time_steps]
    self.force_body_pos_relative_w = delta_pos_w[:, 0] + quat_apply(
      delta_ori_w[:, 0], motion_force_body_pos - self.anchor_pos_w
    )
    self.force_body_quat_relative_w = quat_mul(
      delta_ori_w[:, 0], motion_force_body_quat
    )

    self._update_compliance_buffers()

    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _update_compliance_buffers(self):
    """Anchor the forcefield to the robot pose at force onset and compute
    world-frame force/torque targets, following the SoftMimic release."""
    t = self.time_steps
    env_origins = self._env.scene.env_origins

    ff_stiffness = self.motion.ff_stiffness[t]
    ff_origin_w = self.motion.ff_origin[t] + env_origins
    ff_setpoint_rot_w = self.motion.ff_setpoint_rot[t].clone()
    ff_normal_w = self.motion.ff_normal[t].clone()

    event_start = (self._last_ff_stiffness < _FF_ACTIVE_THRESHOLD) & (
      ff_stiffness >= _FF_ACTIVE_THRESHOLD
    )
    event_active = ff_stiffness >= _FF_ACTIVE_THRESHOLD

    if bool(event_start.any()):
      robot_root_pos = self.robot.data.root_link_pos_w[event_start]
      robot_root_quat = self.robot.data.root_link_quat_w[event_start]
      motion_root_pos = (
        self.motion.root_pos_w[t[event_start]] + env_origins[event_start]
      )
      motion_root_quat = self.motion.root_quat_w[t[event_start]]
      self._ff_anchor_pos[event_start] = robot_root_pos
      self._ff_anchor_rot[event_start] = yaw_quat(
        quat_mul(robot_root_quat, quat_inv(motion_root_quat))
      )
      self._ff_anchor_ref_pos[event_start] = motion_root_pos

    if bool(event_active.any()):
      anchor_rot = self._ff_anchor_rot[event_active]
      origin = (
        quat_apply(
          anchor_rot,
          ff_origin_w[event_active] - self._ff_anchor_ref_pos[event_active],
        )
        + self._ff_anchor_pos[event_active]
      )
      origin[:, 2] = ff_origin_w[event_active][:, 2]
      ff_origin_w[event_active] = origin
      ff_normal_w[event_active] = quat_apply(anchor_rot, ff_normal_w[event_active])
      ff_setpoint_rot_w[event_active] = quat_mul(
        anchor_rot, ff_setpoint_rot_w[event_active]
      )

    self._ff_origin_w = ff_origin_w
    self._ff_setpoint_rot_w = ff_setpoint_rot_w
    self._ff_normal_w = ff_normal_w
    self._last_ff_stiffness = ff_stiffness.clone()

    self._force_vector_w = quat_apply(self._ff_anchor_rot, self.motion.force_vector[t])
    self._torque_vector_w = quat_apply(
      self._ff_anchor_rot, self.motion.torque_vector[t]
    )
    self._active_force_mask = self._force_vector_w.norm(dim=-1) > 1e-6

  def _update_metrics(self):
    super()._update_metrics()
    self.metrics["error_force"] = torch.norm(
      self.forcefield_forces_w - self.target_forces_w, dim=-1
    )
    self.metrics["error_torque"] = torch.norm(
      self.forcefield_torques_w - self.target_torques_w, dim=-1
    )

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Viser controls: motion selection/scrub and the desired stiffness command.

    Motion: defaults to playing only the main (zero-wrench) reference motion;
    check "Use augmented motions" to sample the full augmented set. The time
    slider scrubs the reference; "Reset motion" teleports the robot to the
    reference at the slider time.

    Stiffness sliders are log-scale over the SoftMimic training ranges
    ([40, 1000] N/m and [0.1, 10] Nm/rad). While "Override" is checked
    (default) the slider values replace the per-motion dataset stiffness.
    """
    # -- Motion selection / scrubbing --
    self._gui_get_env_idx = get_env_idx
    self._gui_use_augmented = False
    main_idx = self.motion.main_motion_index
    main_len = int(
      (self.motion.motion_ends[main_idx] - self.motion.motion_starts[main_idx]).item()
    )
    duration_s = max((main_len - 1) * self._env.step_dt, self._env.step_dt)

    # Interactive play starts with the main reference motion and dataset forces
    # off (the GUI Force/Push panels apply wrenches instead).
    self._dataset_forces_enabled = False

    with server.gui.add_folder("Motion"):
      use_aug = server.gui.add_checkbox(
        "Use augmented motions",
        initial_value=False,
        hint="Off: play only the main (zero-wrench) reference motion. On: pick "
        "a random augmented adapted-reference motion (applied immediately). "
        "Combine with 'Apply dataset forces' to replay exactly what the policy "
        "saw during training.",
      )
      apply_forces = server.gui.add_checkbox(
        "Apply dataset forces",
        initial_value=False,
        hint="Apply the augmented motion's baked-in forcefield wrench each step "
        "(as during training), instead of the interactive Force/Push panels. "
        "Requires augmented motions — enabling this turns them on. Only the "
        "forcefield/collision-emulator motions carry force events.",
      )
      time_slider = server.gui.add_slider(
        "time (s)", min=0.0, max=round(duration_s, 1), step=0.1, initial_value=0.0
      )
      reset_btn = server.gui.add_button("Reset motion")

    self._gui_time_slider = time_slider

    @use_aug.on_update
    def _(_) -> None:
      self._gui_use_augmented = use_aug.value
      # Dataset forces only exist in the augmented set; turning augmented off
      # implies turning dataset forces off.
      if not use_aug.value and apply_forces.value:
        apply_forces.value = False
      # With augmented motions playing, the augmented reference (and, optionally,
      # its dataset forces) is the perturbation source — disable the interactive
      # Push panel so it doesn't fight it.
      self._set_push_control_active(not use_aug.value)
      # Apply immediately so the change is visible without waiting for the
      # (up to ~100 s) motion to loop: reset onto a motion from the new set.
      self._gui_time_s = 0.0
      self._gui_pending_reset = True

    @apply_forces.on_update
    def _(_) -> None:
      self._dataset_forces_enabled = apply_forces.value
      # Forces need the augmented set; auto-enable it (this fires use_aug's
      # callback, which issues the reset).
      if apply_forces.value and not use_aug.value:
        use_aug.value = True

    @time_slider.on_update
    def _(_) -> None:
      # Ignore updates we triggered ourselves to reflect playback position.
      if self._gui_suppress_time_cb:
        return
      self._gui_time_s = time_slider.value
      self._gui_pending_scrub = True

    @reset_btn.on_click
    def _(_) -> None:
      self._gui_time_s = time_slider.value
      self._gui_pending_reset = True

    # Snap to the main reference motion at t=0 on startup.
    self._gui_time_s = 0.0
    self._gui_pending_reset = True

    # -- Velocity joystick --
    # While "Enable" is checked, the sliders replace the reference root velocity
    # the policy observes (see `reference_root_lin_vel_b` /
    # `reference_root_ang_vel_b`), steering a velocity-conditioned policy. Ranges
    # match the SoftMimic velocity command limits (Table V).
    from viser import Icon

    vel_axes = [
      ("vel x (m/s)", 1.0),
      ("vel y (m/s)", 0.5),
      ("yaw rate (rad/s)", 1.0),
    ]
    vel_sliders: list = []
    with server.gui.add_folder("Velocity Joystick"):
      vel_enabled = server.gui.add_checkbox(
        "Enable",
        initial_value=False,
        hint="Override the reference root velocity with these sliders. Only "
        "works for a velocity-conditioned policy.",
      )
      for label, max_val in vel_axes:
        vs = server.gui.add_slider(
          label, min=-max_val, max=max_val, step=0.05, initial_value=0.0
        )
        vel_sliders.append(vs)
      vel_zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

    @vel_zero_btn.on_click
    def _(_) -> None:
      for vs in vel_sliders:
        vs.value = 0.0

    def _apply_velocity() -> None:
      if vel_enabled.value:
        self._gui_velocity_command = torch.tensor(
          [vel_sliders[0].value, vel_sliders[1].value, vel_sliders[2].value],
          device=self.device,
        )
      else:
        self._gui_velocity_command = None

    vel_enabled.on_update(lambda _: _apply_velocity())
    for vs in vel_sliders:
      vs.on_update(lambda _: _apply_velocity())
    _apply_velocity()

    # -- Desired stiffness --
    lin_lo, lin_hi = 40.0, 1000.0
    rot_lo, rot_hi = 0.1, 10.0

    def _to_log(v: float, lo: float, hi: float) -> float:
      return (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))

    def _from_log(t: float, lo: float, hi: float) -> float:
      return math.exp(math.log(lo) + t * (math.log(hi) - math.log(lo)))

    init_lin, init_rot = 200.0, 1.0

    with server.gui.add_folder("Desired Stiffness"):
      override = server.gui.add_checkbox("Override dataset", initial_value=True)
      lin_slider = server.gui.add_slider(
        "stiffness", min=0.0, max=1.0, step=0.01,
        initial_value=_to_log(init_lin, lin_lo, lin_hi),
      )
      lin_label = server.gui.add_text(
        "N/m", initial_value=f"{init_lin:.0f}", disabled=True
      )
      rot_slider = server.gui.add_slider(
        "rot stiffness", min=0.0, max=1.0, step=0.01,
        initial_value=_to_log(init_rot, rot_lo, rot_hi),
      )
      rot_label = server.gui.add_text(
        "Nm/rad", initial_value=f"{init_rot:.2f}", disabled=True
      )

    def _apply() -> None:
      if override.value:
        lin = _from_log(lin_slider.value, lin_lo, lin_hi)
        rot = _from_log(rot_slider.value, rot_lo, rot_hi)
        self._gui_stiffness = lin
        self._gui_rotational_stiffness = rot
        lin_label.value = f"{lin:.0f}"
        rot_label.value = f"{rot:.2f}"
      else:
        self._gui_stiffness = None
        self._gui_rotational_stiffness = None

    @override.on_update
    def _(_) -> None:
      _apply()

    @lin_slider.on_update
    def _(_) -> None:
      _apply()

    @rot_slider.on_update
    def _(_) -> None:
      _apply()

    _apply()


@dataclass(kw_only=True)
class CompliantMotionCommandCfg(MotionCommandCfg):
  """Configuration for the compliant motion command.

  ``motion_file`` may point to a single NPZ, a directory of NPZ files, or a
  directory holding ``forcefield``/``collision-emulator``/``zero-wrench``
  subdirectories (hybrid mixing as in the SoftMimic release).
  """

  force_computation_mode: Literal["feedforward", "forcefield"] = "forcefield"
  # Peak wrench of the reactive forcefield (SoftMimic Table IV limits).
  max_force: float = 140.0
  max_torque: float = 10.0

  def build(self, env: ManagerBasedRlEnv) -> CompliantMotionCommand:
    return CompliantMotionCommand(self, env)
