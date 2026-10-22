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

# The 23-DOF G1 drops the wrist pitch/yaw joints, so the arm ends in the
# wrist-roll hand rather than the wrist-yaw link. Map the 29-DOF end-effector
# names onto their nearest 23-DOF equivalents so the same force/keypoint tasks
# work on either model.
_29DOF_TO_23DOF_LINK = {
    "left_wrist_yaw_link": "left_wrist_roll_rubber_hand",
    "right_wrist_yaw_link": "right_wrist_roll_rubber_hand",
}

# Pristine copies of the link lists, so configure_links_for_model() can be
# called for different models in one process (e.g. a 23-DOF solver followed by
# a 29-DOF one): remapping always starts from the 29-DOF default names rather
# than compounding on a previous in-place rewrite, which was irreversible.
_DEFAULT_FORCEABLE_LINKS = list(FORCEABLE_LINKS)
_DEFAULT_DOWNWARD_ONLY_FORCEABLE_LINKS = list(DOWNWARD_ONLY_FORCEABLE_LINKS)
_DEFAULT_KEYPOINT_BODY_NAMES = list(KEYPOINT_BODY_NAMES)
_DEFAULT_FOOT_NAMES = list(FOOT_NAMES)


def configure_links_for_model(model) -> None:
    """Retarget the module-level link lists to bodies that exist in ``model``.

    The augmentation modules read ``FORCEABLE_LINKS`` / ``KEYPOINT_BODY_NAMES``
    at task-construction time, so calling this once (before the IK tasks are
    built) is enough to make the whole pipeline work on the 23-DOF model. It is
    a no-op on the 29-DOF model, where every default link already exists.
    """
    import mujoco

    def _exists(name: str) -> bool:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) != -1

    def _remap(name: str) -> str:
        if _exists(name):
            return name
        alt = _29DOF_TO_23DOF_LINK.get(name)
        if alt is not None and _exists(alt):
            return alt
        return name

    FORCEABLE_LINKS[:] = [_remap(n) for n in _DEFAULT_FORCEABLE_LINKS]
    KEYPOINT_BODY_NAMES[:] = [_remap(n) for n in _DEFAULT_KEYPOINT_BODY_NAMES]
    DOWNWARD_ONLY_FORCEABLE_LINKS[:] = [
        _remap(n) for n in _DEFAULT_DOWNWARD_ONLY_FORCEABLE_LINKS
    ]
    FOOT_NAMES[:] = [_remap(n) for n in _DEFAULT_FOOT_NAMES]

MAX_RELEASE_LINEAR_VEL = 1.0  # m/s
MAX_RELEASE_ANGULAR_VEL = 2.0  # rad/s
# Per-joint rate limit for returning leftover joint-space displacement to the
# reference after a force event. The Cartesian release above only tracks the
# forced link's pose, so null-space displacement (e.g. a wound-up shoulder yaw
# whose hand barely moved) survives the release and used to be teleported into
# the data by a hard configuration reset.
MAX_IK_JOINT_VEL = 2.0 * 3.141592653589793  # whole-frame joint rate, rad/s
MAX_RELEASE_JOINT_VEL = 3.141592653589793  # rad/s
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
