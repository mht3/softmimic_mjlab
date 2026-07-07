from .airexo_sensor import AirexoSensor
from .base_sensor import BaseSensor
from .gravity_sensor import GravitySensor
from .joint_pos_sensor import JointPosSensor
from .joint_pos_error_sensor import JointPosErrorSensor
from .joint_vel_sensor import JointVelSensor
from .joystick_velocity_sensor import JoystickVelocitySensor
from .joystick_standing_sensor import JoystickStandingSensor
from .last_action_sensor import LastActionSensor
from .object_pose_sensor import ObjectPoseSensor
from .root_ang_vel_sensor import RootAngVelSensor
from .wholeexo_sensor import WholeexoSensor
from .reference_gravity_sensor import ReferenceGravitySensor
from .reference_root_height_sensor import ReferenceRootHeightSensor
from .reference_foot_contact_sensor import ReferenceFootContactSensor
from .reference_ang_vel_sensor import ReferenceAngVelSensor
from .reference_xy_vel_sensor import ReferenceXYVelSensor
# from .motionvae_sensor import MotionVAESensor

from .desired_stiffness_sensor import DesiredStiffnessSensor
from .desired_rotational_stiffness_sensor import DesiredRotationalStiffnessSensor
from .desired_stiffness_log_sensor import DesiredStiffnessLogSensor
from .desired_rotational_stiffness_log_sensor import DesiredRotationalStiffnessLogSensor
from .desired_stiffness_inv_sensor import DesiredStiffnessInvSensor
from .desired_rotational_stiffness_inv_sensor import DesiredRotationalStiffnessInvSensor
from .external_force_sensor import ExternalForceSensor
from .external_torque_sensor import ExternalTorqueSensor

from .root_lin_vel_sensor import RootLinVelSensor
from .root_height_sensor import RootHeightSensor
from .foot_contact_sensor import FootContactSensor
from .future_reference_dof_pos_sensor import FutureReferenceDofPosSensor
from .future_reference_xy_vel_sensor import FutureReferenceXYVelSensor
from .future_reference_ang_vel_sensor import FutureReferenceAngVelSensor
from .future_reference_root_height_sensor import FutureReferenceRootHeightSensor
from .future_reference_gravity_sensor import FutureReferenceGravitySensor
from .future_reference_foot_contact_sensor import FutureReferenceFootContactSensor

REFERENCE_SENSORS = [
    ReferenceAngVelSensor,
    ReferenceXYVelSensor,
    ReferenceGravitySensor,
    ReferenceRootHeightSensor,
    ReferenceFootContactSensor,
    FutureReferenceAngVelSensor,
    FutureReferenceXYVelSensor,
    FutureReferenceGravitySensor,
    FutureReferenceRootHeightSensor,
    FutureReferenceFootContactSensor,
    FutureReferenceDofPosSensor,
]

ISAACLAB_FUNCTION_MAP = {
    # Proprioceptive sensors
    "isaaclab.envs.mdp.observations:last_action": LastActionSensor,
    "isaaclab.envs.mdp.observations:joint_pos_rel": JointPosSensor,
    "isaaclab.envs.mdp.observations:joint_pos_error": JointPosErrorSensor,
    "isaaclab.envs.mdp.observations:joint_vel_rel": JointVelSensor,
    "isaaclab.envs.mdp.observations:projected_gravity": GravitySensor,
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:projected_gravity_bodies": GravitySensor,
    "isaaclab.envs.mdp.observations:base_ang_vel": RootAngVelSensor,
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:base_ang_vel": RootAngVelSensor,
    
    # Reference motion quantities
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:reference_dof_pos": {
        "reference_dof_pos": WholeexoSensor,
        "reference_dof_pos_future": FutureReferenceDofPosSensor
    },
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:reference_gravity_vector": {
        "reference_gravity_vector": ReferenceGravitySensor,
        "reference_gravity_vector_future": FutureReferenceGravitySensor
    },
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:reference_root_height": {
        "reference_root_height": ReferenceRootHeightSensor,
        "reference_root_height_future": FutureReferenceRootHeightSensor
    },
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:reference_foot_contacts": {
        "reference_foot_contact": ReferenceFootContactSensor,
        "reference_foot_contact_future": FutureReferenceFootContactSensor
    },
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:reference_ang_vel": {
        "reference_yaw_vel": ReferenceAngVelSensor,
        "reference_yaw_vel_future": FutureReferenceAngVelSensor
    },
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:reference_xy_vel": {
        "reference_xy_vel": ReferenceXYVelSensor,
        "reference_xy_vel_future": FutureReferenceXYVelSensor
    },

    # Compliance-related commands
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:desired_stiffness": DesiredStiffnessSensor, # differentiate by command_name
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:desired_rotational_stiffness": DesiredRotationalStiffnessSensor, # differentiate by command_name
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:desired_stiffness_log": DesiredStiffnessLogSensor, # differentiate by command_name
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:desired_rotational_stiffness_log": DesiredRotationalStiffnessLogSensor, # differentiate by command_name
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:desired_stiffness_inv": DesiredStiffnessInvSensor, # differentiate by command_name
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:desired_rotational_stiffness_inv": DesiredRotationalStiffnessInvSensor, # differentiate by command_name
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:forcefield_force_applied": ExternalForceSensor, # differentiate by command_name
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:forcefield_torque_applied": ExternalTorqueSensor, # differentiate by command_name

    # Privileged sensors
    "isaaclab.envs.mdp.observations:base_lin_vel": RootLinVelSensor,
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:root_height": RootHeightSensor,
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:foot_contacts": FootContactSensor,

    # # Future reference motion quantities
    # "isaaclab.envs.mdp.observations:future_reference_dof_pos": FutureReferenceDofPosSensor,
    # "isaaclab.envs.mdp.observations:future_reference_xy_vel": FutureReferenceXYVelSensor,
    # "isaaclab.envs.mdp.observations:future_reference_ang_vel": FutureReferenceAngVelSensor,
    # "isaaclab.envs.mdp.observations:future_reference_root_height": FutureReferenceRootHeightSensor,
    # "isaaclab.envs.mdp.observations:future_reference_gravity_vector": FutureReferenceGravitySensor,
    # "isaaclab.envs.mdp.observations:future_reference_foot_contact": FutureReferenceFootContactSensor,

    "isaaclab.envs.mdp.observations:generated_commands": 
        { # differentiate by command_name
            "base_velocity": JoystickVelocitySensor,
            "velocity_commands": JoystickVelocitySensor,
            # "upper_body_joints": AirexoSensor,
            "reference_motion_commands": WholeexoSensor,
            "torso_roll_pitch": None,
            "body_height": None,
            "reference_dof_pos": WholeexoSensor,
            "reference_gravity_vector": ReferenceGravitySensor,
            "reference_root_height": ReferenceRootHeightSensor,
            "reference_foot_contact": ReferenceFootContactSensor,
            "reference_ang_vel": ReferenceAngVelSensor,
            "reference_xy_vel": ReferenceXYVelSensor,
        },
    "softmimic_gym.tasks.locomotion.tracking.mdp.observations:generated_commands_slice": 
        { # differentiate by command_name
            "base_velocity": JoystickVelocitySensor,
            # "upper_body_joints": AirexoSensor,
            "reference_motion_commands": WholeexoSensor,
            "torso_roll_pitch": None,
            "body_height": None,
            "reference_dof_pos": WholeexoSensor,
            "reference_gravity_vector": ReferenceGravitySensor,
            "reference_root_height": ReferenceRootHeightSensor,
            "reference_foot_contact": ReferenceFootContactSensor,
            "reference_ang_vel": ReferenceAngVelSensor,
            "reference_xy_vel": ReferenceXYVelSensor,
        },
    "isaaclab.envs.mdp.observations:generated_commands_norm_leq": 
        { # differentiate by command_name
            "base_velocity": JoystickStandingSensor,
        },
    "isaaclab.envs.mdp.observations:dummy_object_pos_quat_in_robot_root_frame": ObjectPoseSensor,
    "isaaclab.envs.mdp.observations:object_pose": ObjectPoseSensor,
}