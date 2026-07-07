FORCEABLE_LINKS = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

DOWNWARD_ONLY_FORCEABLE_LINKS = [
    "torso_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
]

KEYPOINT_BODY_NAMES = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "torso_link",
    "pelvis",
]

FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

MAX_RELEASE_LINEAR_VEL = 1.0  # m/s
MAX_RELEASE_ANGULAR_VEL = 2.0  # rad/s
TELEPORT_THRESHOLD = 1.0  # m

IK_CHECK_MAX_IK_TRACKING_ERROR = 0.05
IK_CHECK_MAX_FOOT_DISP = 0.05
IK_CHECK_MAX_COM_TRACKING_ERROR = 0.15
IK_CHECK_MAX_FORCE_MAGNITUDE = 140.0
IK_CHECK_MAX_DISPLACEMENT_MAGNITUDE = 0.7
IK_CHECK_MAX_TORQUE_MAGNITUDE = 10.0
IK_CHECK_MAX_ROTATIONAL_DISPLACEMENT_MAGNITUDE = 2.0

MIN_ROBOT_STIFFNESS = 10.0
MAX_ROBOT_STIFFNESS = 1000.0
MIN_ROBOT_ROTATIONAL_STIFFNESS = 0.1
MAX_ROBOT_ROTATIONAL_STIFFNESS = 10.0
MIN_FORCEFIELD_STIFFNESS = 10.0
MAX_FORCEFIELD_STIFFNESS = 1000.0
MIN_FORCEFIELD_ROTATIONAL_STIFFNESS = 0.1
MAX_FORCEFIELD_ROTATIONAL_STIFFNESS = 10.0
