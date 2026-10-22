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
from src.tasks.compliant_tracking.mdp.parity_events import make_softmimic_parameter_events
from src.tasks.compliant_tracking.mdp import CompliantMotionCommandCfg

# SoftMimic critic future-reference horizon: 20 points subsampled from a
# 100-step (2 s at 50 Hz) window.
_FUTURE_STEPS = (1, *range(5, 100, 5))


def make_compliant_tracking_env_cfg(
  history_length: int = 3,
) -> ManagerBasedRlEnvCfg:
  """Create base compliant tracking task configuration.

  Args:
    history_length: Number of past frames stacked into every observation term
      (SoftMimic Table III uses 3).
  """

  ##
  # Observations
  ##

  # Released actor: original reference + proprioception + stiffness. Each
  # term stacks its own history (327 inputs at23DOF and history3).
  def obs(func, *, params=None, noise=None, scale=None, history=True):
    return ObservationTermCfg(
      func=func, params=params or {}, noise=noise, scale=scale,
      history_length=history_length if history else 0,
    )

  refs = {
    "reference_dof_pos": mdp.reference_dof_pos,
    "reference_xy_vel": mdp.reference_xy_vel,
    "reference_yaw_vel": mdp.reference_ang_vel,
    "reference_root_height": mdp.reference_root_height,
    "reference_gravity_vector": mdp.reference_gravity_vector,
    "reference_foot_contact": mdp.reference_foot_contacts,
  }
  actor_terms = {
    "base_ang_vel": obs(mdp.base_ang_vel, noise=Unoise(n_min=-.2, n_max=.2), scale=.25),
    "projected_gravity": obs(mdp.projected_gravity, noise=Unoise(n_min=-.01, n_max=.01)),
    "joint_pos": obs(mdp.joint_pos_rel, noise=Unoise(n_min=-.01, n_max=.01)),
    "joint_vel": obs(mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=.05),
    "actions": obs(mdp.last_action),
  }
  for name, func in refs.items():
    actor_terms[name] = obs(func, params={"command_name": "motion", "nominal": True})
  actor_terms["desired_stiffness_log"] = obs(mdp.desired_stiffness_log, params={"command_name": "motion"})
  actor_terms["desired_rotational_stiffness_log"] = obs(mdp.desired_rotational_stiffness_log, params={"command_name": "motion"})

  # Critic contains nominal AND adapted references, with both future horizons.
  # Its force vectors have distinct left/right hand slots in body-local frames.
  critic_terms = {
    "base_ang_vel": obs(mdp.base_ang_vel, scale=.25),
    "projected_gravity": obs(mdp.projected_gravity),
    "joint_pos": obs(mdp.joint_pos_rel),
    "joint_vel": obs(mdp.joint_vel_rel, scale=.05),
    "actions": obs(mdp.last_action),
    "base_lin_vel": obs(mdp.base_lin_vel),
    "root_height": obs(mdp.root_height),
    "foot_contact": obs(mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}),
    "foot_contact_forces": obs(mdp.foot_contact_forces, params={"sensor_name": "feet_ground_contact"}, history=False),
    "foot_velocities": obs(mdp.foot_velocities, params={"command_name": "motion"}, history=False),
  }
  for name, func in refs.items():
    critic_terms[name] = obs(func, params={"command_name": "motion", "nominal": True})
  critic_terms["keypoint_error"] = obs(mdp.keypoint_error_local_shifted, params={"command_name": "motion"})
  for name, func in (
    ("force_applied", mdp.forcefield_force_applied),
    ("torque_applied", mdp.forcefield_torque_applied),
    ("desired_stiffness_log", mdp.desired_stiffness_log),
    ("desired_rotational_stiffness_log", mdp.desired_rotational_stiffness_log),
    ("forcefield_stiffness_log", mdp.forcefield_stiffness_log),
    ("forcefield_rotational_stiffness_log", mdp.forcefield_rotational_stiffness_log),
    ("forcefield_force_desired", mdp.forcefield_force_desired),
    ("forcefield_torque_desired", mdp.forcefield_torque_desired),
    ("all_adapted_keypoints_pos_error_local", mdp.all_adapted_keypoints_pos_error_local),
    ("all_adapted_keypoints_quat_error_local", mdp.all_adapted_keypoints_quat_error_local),
  ):
    critic_terms[name] = obs(func, params={"command_name": "motion"})
  for name, func in refs.items():
    critic_terms["adapted_" + name] = obs(func, params={"command_name": "motion", "nominal": False})
  for nominal, prefix in ((True, ""), (False, "adapted_")):
    for name, func in refs.items():
      critic_terms[prefix + name + "_future"] = obs(func, params={
        "command_name": "motion", "nominal": nominal, "future_steps": _FUTURE_STEPS,
      }, history=False)

  observations = {
    "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True, enable_corruption=True),
    "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True, enable_corruption=False),
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
    ),
    # Zero-dim term rotates the body-local wrench cached at policy rate
    # into MuJoCo world coordinates each physics substep.
    "forcefield_wrench": mdp.ForcefieldWrenchActionCfg(
      entity_name="robot",
      command_name="motion",
    ),
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
  # forcefield spring and blow up the applied force. The forcefield wrench
  # itself is applied per-substep by the "forcefield_wrench" action term.
  events: dict[str, EventTermCfg] = {
    **make_softmimic_parameter_events(),
    "forcefield_cache": EventTermCfg(mode="step", func=mdp.cache_forcefield_wrench),
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
    # Retained in the released implementation (omitted from paper TableVI).
    "joint_position_tracking": RewardTermCfg(
      func=mdp.joint_deviation_from_command_exp,
      weight=2.0,
      params={"command_name": "motion", "sigma": 1.0},
    ),
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
    # self-collision physics in the SoftMimic robot preset).
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "bad_keypoints": TerminationTermCfg(
      func=mdp.bad_motion_body_pos,
      params={"command_name": "motion", "threshold": .5},
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
