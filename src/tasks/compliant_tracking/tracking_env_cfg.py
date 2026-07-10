"""Compliant motion mimic task configuration.

SoftMimic-style compliant tracking: the robot imitates *adapted* reference
motions from the compliant motion augmentation pipeline while the matching
external wrench is applied to the force-target body. Soft reward terms make
the policy reproduce the demonstrated interaction forces instead of stiffly
rejecting them.

Based on the BeyondMimic tracking task in ``src/tasks/tracking`` and the
SoftMimic Isaac Lab release (https://github.com/Improbable-AI/softmimic).
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.compliant_tracking.mdp as mdp
from src.tasks.compliant_tracking.mdp import CompliantMotionCommandCfg

def make_compliant_tracking_env_cfg(
  history_length: int = 3, velocity_conditioning: bool = False
) -> ManagerBasedRlEnvCfg:
  """Create base compliant tracking task configuration.

  Args:
    history_length: Number of past frames stacked into every observation term
      (SoftMimic Table III uses 3).
    velocity_conditioning: If True, add the reference root velocity (heading
      frame) as an actor+critic observation. This lets the policy be steered
      by a velocity command (GUI joystick at play time), mirroring SoftMimic's
      ``reference_xy_vel`` / ``reference_yaw_vel`` conditioning. It changes the
      observation dimension, so checkpoints are not compatible across the flag.
  """

  ##
  # Observations
  ##

  # SoftMimic Table III: 3-step history on proprioception, reference motion,
  # action, and the desired-stiffness command (log scale). Noise ranges follow
  # Table V. The 20-point future reference horizon is not (yet) ported.
  actor_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.25, n_max=0.25),
      history_length=history_length,
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
      history_length=history_length,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
      history_length=history_length,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      history_length=history_length,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
      history_length=history_length,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      history_length=history_length,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action, history_length=history_length),
    "desired_stiffness_log": ObservationTermCfg(
      func=mdp.desired_stiffness_log,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
    "desired_rotational_stiffness_log": ObservationTermCfg(
      func=mdp.desired_rotational_stiffness_log,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
  }

  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      history_length=history_length,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      history_length=history_length,
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, history_length=history_length),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, history_length=history_length),
    "actions": ObservationTermCfg(func=mdp.last_action, history_length=history_length),
    # Privileged compliance state (SoftMimic).
    "force_applied": ObservationTermCfg(
      func=mdp.forcefield_force_applied,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
    "torque_applied": ObservationTermCfg(
      func=mdp.forcefield_torque_applied,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
    "force_desired": ObservationTermCfg(
      func=mdp.forcefield_force_desired,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
    "torque_desired": ObservationTermCfg(
      func=mdp.forcefield_torque_desired,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
    "desired_stiffness_log": ObservationTermCfg(
      func=mdp.desired_stiffness_log,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
    "desired_rotational_stiffness_log": ObservationTermCfg(
      func=mdp.desired_rotational_stiffness_log,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
    "forcefield_stiffness_log": ObservationTermCfg(
      func=mdp.forcefield_stiffness_log,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
    "forcefield_rotational_stiffness_log": ObservationTermCfg(
      func=mdp.forcefield_rotational_stiffness_log,
      params={"command_name": "motion"},
      history_length=history_length,
    ),
  }

  # Velocity conditioning: expose the reference root velocity (heading frame)
  # to both actor and critic so the policy learns to follow a velocity command
  # (steerable via the GUI joystick at play time).
  if velocity_conditioning:
    for terms in (actor_terms, critic_terms):
      terms["reference_root_lin_vel_b"] = ObservationTermCfg(
        func=mdp.reference_root_lin_vel_b,
        params={"command_name": "motion"},
        history_length=history_length,
      )
      terms["reference_root_ang_vel_b"] = ObservationTermCfg(
        func=mdp.reference_root_ang_vel_b,
        params={"command_name": "motion"},
        history_length=history_length,
      )

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": CompliantMotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={},
      velocity_range={},
      joint_position_range=(-0.2, 0.2),
      sampling_mode="uniform",
      force_computation_mode="forcefield",
      # Override in robot cfg.
      motion_file="",
      anchor_body_name="",
      body_names=(),
    )
  }

  ##
  # Events
  ##

  # No random pushes during training: the dataset force events are the only
  # perturbations (SoftMimic). Random pushes fight the onset-anchored
  # forcefield spring and blow up the applied force.
  events: dict[str, EventTermCfg] = {
    "apply_forcefield": EventTermCfg(
      func=mdp.apply_forcefield_wrench,
      mode="step",
      params={"command_name": "motion"},
    ),
    # Payload mass and link mass scale follow SoftMimic Table V.
    "payload_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
        "operation": "add",
        "ranges": (-2.0, 2.0),
      },
    ),
    "link_mass_scale": EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        # Mass/inertia scale = e^alpha, so this is a [0.7, 1.3] mass scale.
        "alpha_range": (math.log(0.7), math.log(1.3)),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    # Joint parameter randomization (added), SoftMimic Table V.
    "joint_damping": EventTermCfg(
      mode="startup",
      func=dr.joint_damping,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "add",
        "ranges": (0.0, 2.0),
      },
    ),
    "joint_armature": EventTermCfg(
      mode="startup",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "add",
        "ranges": (0.01, 0.1),
      },
    ),
    "joint_friction": EventTermCfg(
      mode="startup",
      func=dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "add",
        "ranges": (0.0, 0.01),
      },
    ),
    # Foot geoms have contact priority over the terrain, so this IS the
    # ground friction; range follows SoftMimic Table V.
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.5, 2.0),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
  }

  ##
  # Rewards
  ##

  # Terms and weights follow SoftMimic Table VI (identical across tasks).
  rewards: dict[str, RewardTermCfg] = {
    # -- Compliance --
    "force_link_pos": RewardTermCfg(
      func=mdp.force_link_position_tracking_exp,
      weight=3.0,
      params={"command_name": "motion", "sigma": 0.1},
    ),
    "force_link_ori": RewardTermCfg(
      func=mdp.force_link_orientation_tracking_exp,
      weight=3.0,
      params={"command_name": "motion", "sigma": 0.1},
    ),
    "force_command_tracking": RewardTermCfg(
      func=mdp.force_command_tracking,
      weight=2.0,
      params={"command_name": "motion", "sigma": 20.0},
    ),
    "torque_command_tracking": RewardTermCfg(
      func=mdp.torque_command_tracking,
      weight=2.0,
      params={"command_name": "motion", "sigma": 2.0},
    ),
    # -- Motion tracking --
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=2.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=2.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_base_ori": RewardTermCfg(
      func=mdp.motion_base_gravity_error_exp,
      weight=0.5,
      params={"command_name": "motion"},
    ),
    "motion_base_lin_vel": RewardTermCfg(
      func=mdp.motion_base_lin_vel_local_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.5},
    ),
    "motion_base_ang_vel": RewardTermCfg(
      func=mdp.motion_base_ang_vel_local_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 2.0},
    ),
    # -- Stability --
    "alive": RewardTermCfg(func=mdp.is_alive, weight=1.5),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.feet_slide_proportional,
      weight=-0.005,
      params={"sensor_name": "feet_ground_contact", "command_name": "motion"},
    ),
    "joint_vel_l2": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=-2.8e-4,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "foot_contact_schedule": RewardTermCfg(
      func=mdp.foot_contact_schedule_mismatch,
      weight=-0.4,
      params={"command_name": "motion", "sensor_name": "feet_ground_contact"},
    ),
    # No self-collision penalty, matching SoftMimic (they disable
    # self-collision physics; here MuJoCo still resolves the contacts).
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": (),  # Set per-robot.
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=350,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    # SoftMimic uses 30 s episodes.
    episode_length_s=30.0,
  )
