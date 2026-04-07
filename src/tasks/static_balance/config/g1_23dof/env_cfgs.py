"""Unitree G1-23DOF static balance environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.robots import G1_23DOF_ACTION_SCALE, get_g1_23dof_robot_cfg
from src.tasks.static_balance.balance_env_cfg import make_balance_env_cfg
import src.tasks.static_balance.mdp as mdp
from src.tasks.static_balance.mdp.push_control_command import PushControlCommandCfg
from src.tasks.static_balance.mdp.force_control_command import ForceControlCommandCfg


def unitree_g1_23dof_balance_push_curriculum_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1-23DOF balance with push velocity curriculum (base configuration).

  """
  cfg = make_balance_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  cfg.scene.entities = {"robot": get_g1_23dof_robot_cfg()}

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

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
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_23DOF_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.commands["push_control"] = PushControlCommandCfg(
      entity_name="robot", resampling_time_range=(1.0e9, 1.0e9)
    )
    cfg.commands["force_control"] = ForceControlCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
    )

  return cfg


def unitree_g1_23dof_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1-23DOF flat balance env — no push curriculum."""
  cfg = unitree_g1_23dof_balance_push_curriculum_env_cfg(play=play)
  cfg.curriculum.pop("push_velocity", None)
  return cfg
