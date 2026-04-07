"""Static balance task configuration.

Base config for balancing in place against environment perturbations.
No velocity command — the robot must simply remain upright and still.
Robot-specific configurations call make_balance_env_cfg() and customise as needed.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.static_balance.mdp as mdp


def make_balance_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base static balance task configuration."""

  ##
  # Observations
  ##

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.0),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 5.0),
      params={
        "velocity_range": {
          "x": (-1.25, 1.25),
          "y": (-1.25, 1.25),
          "z": (-0.5, 0.5),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-2.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "base_lin_vel_penalty": RewardTermCfg(
      func=mdp.base_lin_vel_penalty,
      weight=-0.5,
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.05,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "base_height": RewardTermCfg(
      func=mdp.base_height_penalty,
      weight=-2.0,
      params={"target_height": 0.78},  # Override per-robot.
    ),
    "joint_deviation_waist": RewardTermCfg(
      func=mdp.default_pose_penalty,
      weight=-0.5,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*waist.*"), "std": 0.05},
    ),
    "joint_deviation_hips": RewardTermCfg(
      func=mdp.default_pose_penalty,
      weight=-0.3,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*hip.*"), "std": 0.05},
    ),
    "joint_deviation_lower": RewardTermCfg(
      func=mdp.default_pose_penalty,
      weight=-0.5,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=r".*knee.*|.*ankle.*"), "std": 0.05},
    ),
    "joint_deviation_arms": RewardTermCfg(
      func=mdp.default_pose_penalty,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=r".*shoulder.*|.*elbow.*|.*wrist.*"), "std": 0.05},
    ),
    "feet_in_contact": RewardTermCfg(
      func=mdp.feet_in_contact,
      weight=0.5,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "is_alive": RewardTermCfg(func=mdp.is_alive, weight=0.15),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={
      "push_velocity": CurriculumTermCfg(
        func=mdp.push_velocity_curriculum,
        params={
          "velocity_stages": [
            # Stage 0: soft pushes for the first 5k iterations.
            {"step": 0, "velocity_range": {
              "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
              "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
            }},
            # Stage 1: harder pushes after iteration 5000 (step = 5000 * num_steps_per_env).
            {"step": 2500 * 24, "velocity_range": {
              "x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-0.5, 0.5),
              "roll": (-0.78, 0.78), "pitch": (-0.78, 0.78), "yaw": (-1.0, 1.0),
            }},
            # Stage 2
            {"step": 7500 * 24, "velocity_range": {
              "x": (-1.5, 1.5), "y": (-1.5, 1.5), "z": (-0.5, 0.5),
              "roll": (-1.57, 1.57), "pitch": (-1.57, 1.57), "yaw": (-1.57, 1.57),
            }},
            # Stage 3
            {"step": 12500 * 24, "velocity_range": {
              "x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-0.5, 0.5),
              "roll": (-1.57, 1.57), "pitch": (-1.57, 1.57), "yaw": (-1.57, 1.57),
            }},
          ],
        },
      ),
    },
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
