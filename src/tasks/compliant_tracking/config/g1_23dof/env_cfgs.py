"""Unitree G1_23Dof flat compliant tracking environment configurations."""

from src.assets.robots.unitree_g1.g1_23dof_constants import (
  G1_23DOF_ACTION_SCALE,
  get_g1_23dof_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.tasks.compliant_tracking.mdp import CompliantMotionCommandCfg
from src.tasks.compliant_tracking.tracking_env_cfg import (
  make_compliant_tracking_env_cfg,
)
from src.tasks.static_balance.mdp.force_control_command import ForceControlCommandCfg
from src.tasks.static_balance.mdp.push_control_command import PushControlCommandCfg


def unitree_g1_23dof_flat_compliant_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  history_length: int = 3,
  velocity_conditioning: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1_23Dof flat terrain compliant tracking configuration."""
  cfg = make_compliant_tracking_env_cfg(
    history_length=history_length, velocity_conditioning=velocity_conditioning
  )

  cfg.scene.entities = {"robot": get_g1_23dof_robot_cfg()}

  # No self-collision sensor: the self-collision penalty is disabled to match
  # SoftMimic's reward set.
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_23DOF_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, CompliantMotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_rubber_hand",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_rubber_hand",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["payload_mass"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.sim.nconmax = None

    cfg.observations["actor"].enable_corruption = False

    # Interactive sandbox. The dataset forcefield event stays registered but is
    # gated by the command's `dataset_forces_enabled` GUI toggle (off by
    # default), so by default the viser Force/Push panels own the external
    # wrench. Step events run after the command manager, so when the toggle is
    # on the dataset wrench takes precedence — letting you replay training
    # forces exactly. `create_gui()` sets the toggle off at startup.
    cfg.commands["push_control"] = PushControlCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      # Manual mode, sliders at zero — no automatic pushes.
      manual_by_default=True,
    )
    cfg.commands["force_control"] = ForceControlCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      # SoftMimic Table IV: 140 N / 10 Nm peak force/torque at the 0.7 m
      # displacement limit.
      force_scale=200.0,
      force_max=140.0,
      torque_scale=10.0,
      torque_max=10.0,
    )

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
