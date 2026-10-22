"""Unitree G1_23DOF constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from src.assets.robots.deploy_entity_cfg import DeployEntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

G1_23DOF_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1_23dof.xml"
)
assert G1_23DOF_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, G1_23DOF_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_23DOF_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Motor specs (from Unitree).
ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_7520_22 = (
  1,
  4.5,
  5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020,
  velocity_limit=37.0,
  effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=32.0,
  effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22,
  velocity_limit=20.0,
  effort_limit=139.0,
)
ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=22.0,
  effort_limit=5.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)

# Ankles are 4-bar linkages with 2 5020 actuators.
# Due to the parallel linkage, the effective armature at the ankle and waist joints
# is configuration dependent. Since the exact geometry of the linkage is unknown, we
# assume a nominal 1:1 gear ratio. Under this assumption, the joint armature in the
# nominal configuration is approximated as the sum of the 2 actuators' armatures.
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

##
# SoftMimic actuator config.
#
# PD gains of the working 29-DoF SoftMimic stand policy, identical in sim and
# on the real robot (softmimic assets/robots/unitree.py == deploy
# g1_config.py). Actuators are split by motor type x gain value because
# BuiltinPositionActuatorCfg takes scalar gains. Effort limits and armatures
# match the retained joints in SoftMimic's assets/robots/unitree.py.
##

G1_SOFTMIMIC_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
  ),
  stiffness=40.0,
  damping=5.0,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=0.01,
)
G1_SOFTMIMIC_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_roll_joint",),
  stiffness=5.0,
  damping=1.0,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=0.001,
)
G1_SOFTMIMIC_ACTUATOR_HIP_PITCH_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", "waist_yaw_joint"),
  stiffness=200.0,
  damping=5.0,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=0.01,
)
G1_SOFTMIMIC_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint",),
  stiffness=150.0,
  damping=5.0,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=0.01,
)
G1_SOFTMIMIC_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=150.0,
  damping=5.0,
  effort_limit=88.0,
  armature=0.01,
)
G1_SOFTMIMIC_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=0.01,
)
G1_SOFTMIMIC_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=20.0,
  damping=2.0,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=0.01,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.8),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
  },
  joint_vel={".*": 0.0},
)

# Action offsets and relative joint observations depend on this pose, even
# when resets initialize from a dataset reference rather than the keyframe.
SOFTMIMIC_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.74),
  joint_pos={
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_pitch_joint": 0.35,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.78),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

G1_23DOF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)

G1_23DOF_SOFTMIMIC_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_SOFTMIMIC_ACTUATOR_ARM,
    G1_SOFTMIMIC_ACTUATOR_WRIST,
    G1_SOFTMIMIC_ACTUATOR_HIP_PITCH_WAIST,
    G1_SOFTMIMIC_ACTUATOR_HIP_YAW,
    G1_SOFTMIMIC_ACTUATOR_HIP_ROLL,
    G1_SOFTMIMIC_ACTUATOR_KNEE,
    G1_SOFTMIMIC_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


G1_23DOF_JOINT_IDS_MAP: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]

# Hardware PD gains (action-space order, not sim actuator values).
G1_23DOF_HARDWARE_STIFFNESS: list[float] = [
  40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
  40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
  40.2,
  14.3, 14.3, 14.3, 14.3, 14.3,
  14.3, 14.3, 14.3, 14.3, 14.3,
]
G1_23DOF_HARDWARE_DAMPING: list[float] = [
  2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
  2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
  2.6,
  0.9, 0.9, 0.9, 0.9, 0.9,
  0.9, 0.9, 0.9, 0.9, 0.9,
]

# SoftMimic gains in action-space order; identical to the sim gains above so
# the trained policy runs against the exact same PD loop on hardware.
G1_23DOF_SOFTMIMIC_STIFFNESS: list[float] = [
  200.0, 150.0, 150.0, 200.0, 20.0, 20.0,
  200.0, 150.0, 150.0, 200.0, 20.0, 20.0,
  200.0,
  40.0, 40.0, 40.0, 40.0, 5.0,
  40.0, 40.0, 40.0, 40.0, 5.0,
]
G1_23DOF_SOFTMIMIC_DAMPING: list[float] = [
  5.0, 5.0, 5.0, 5.0, 2.0, 2.0,
  5.0, 5.0, 5.0, 5.0, 2.0, 2.0,
  5.0,
  5.0, 5.0, 5.0, 5.0, 1.0,
  5.0, 5.0, 5.0, 5.0, 1.0,
]


def get_g1_23dof_robot_cfg(gains: str = "natural_freq") -> EntityCfg:
  """Get a fresh G1_23DOF robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.

  Args:
    gains: "natural_freq" for the armature-derived gains, "softmimic" for the
      proven SoftMimic hardware gains (sim == deploy).
  """
  if gains == "softmimic":
    articulation = G1_23DOF_SOFTMIMIC_ARTICULATION
    hardware_stiffness = G1_23DOF_SOFTMIMIC_STIFFNESS
    hardware_damping = G1_23DOF_SOFTMIMIC_DAMPING
    init_state = SOFTMIMIC_HOME_KEYFRAME
    collisions = (FULL_COLLISION_WITHOUT_SELF,)
  elif gains == "natural_freq":
    articulation = G1_23DOF_ARTICULATION
    hardware_stiffness = G1_23DOF_HARDWARE_STIFFNESS
    hardware_damping = G1_23DOF_HARDWARE_DAMPING
    init_state = HOME_KEYFRAME
    collisions = (FULL_COLLISION,)
  else:
    raise ValueError(f"Unknown gains preset: {gains!r}")
  return DeployEntityCfg(
    init_state=init_state,
    collisions=collisions,
    spec_fn=get_spec,
    articulation=articulation,
    joint_ids_map=G1_23DOF_JOINT_IDS_MAP,
    hardware_stiffness=hardware_stiffness,
    hardware_damping=hardware_damping,
  )


G1_23DOF_ACTION_SCALE: dict[str, float] = {}
for a in G1_23DOF_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_23DOF_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_23dof_robot_cfg())

  viewer.launch(robot.spec.compile())
