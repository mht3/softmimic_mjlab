import numpy as np
import os
import time
import mujoco
from scipy.spatial.transform import Rotation as R

from softmimic_deploy.src.interfaces.base_interface import BaseInterface
from softmimic_deploy.src.utils.math_utils import get_rotation_matrix_from_rpy, get_rpy_from_quaternion, get_rotation_matrix_from_quaternion
from softmimic_deploy.src.utils.zmq_utils import PosePublisher

class MujocoInterface(BaseInterface):
    def __init__(self, robot_configuration, cfg, headless=False, task_name=None):
        self.robot_configuration = robot_configuration
        self.cfg = cfg
        
        self.init_time = time.time()
        self.step_counter = 0

        self.task_name = task_name
        
        self.key_state = [
            ["R1", 0], ["L1", 0], ["start", 0], ["select", 0],
            ["R2", 0], ["L2", 0], ["F1", 0], ["F2", 0],
            ["A", 0], ["B", 0], ["X", 0], ["Y", 0],
            ["up", 0], ["right", 0], ["down", 0], ["left", 0],
        ]
        
        self.joint_idxs = [i for i in range(19)]
        self.last_action = None
        self.base_lock_pos = None

        # Add ghost joint mappings
        # self.ghost_joint_map = {
        #     'torso_joint': 'ghost_torso_joint',
        #     'left_hip_yaw_joint': 'ghost_left_hip_yaw_joint',
        #     'left_hip_roll_joint': 'ghost_left_hip_roll_joint',
        #     'left_hip_pitch_joint': 'ghost_left_hip_pitch_joint',
        #     'left_knee_joint': 'ghost_left_knee_joint',
        #     'left_ankle_joint': 'ghost_left_ankle_joint',
        #     'right_hip_yaw_joint': 'ghost_right_hip_yaw_joint',
        #     'right_hip_roll_joint': 'ghost_right_hip_roll_joint',
        #     'right_hip_pitch_joint': 'ghost_right_hip_pitch_joint',
        #     'right_knee_joint': 'ghost_right_knee_joint',
        #     'right_ankle_joint': 'ghost_right_ankle_joint',
        #     'left_shoulder_pitch_joint': 'ghost_left_shoulder_pitch_joint',
        #     'left_shoulder_roll_joint': 'ghost_left_shoulder_roll_joint',
        #     'left_shoulder_yaw_joint': 'ghost_left_shoulder_yaw_joint',
        #     'left_elbow_joint': 'ghost_left_elbow_joint',
        #     'right_shoulder_pitch_joint': 'ghost_right_shoulder_pitch_joint',
        #     'right_shoulder_roll_joint': 'ghost_right_shoulder_roll_joint',
        #     'right_shoulder_yaw_joint': 'ghost_right_shoulder_yaw_joint',
        #     'right_elbow_joint': 'ghost_right_elbow_joint',
        # }
        
        # self.ghost_joint_map = {
        #     'torso_joint': 'ghost_torso_joint',
        #     'left_hip_yaw_joint': 'ghost_left_hip_yaw_joint',
        #     'left_hip_roll_joint': 'ghost_left_hip_roll_joint',
        #     'left_hip_pitch_joint': 'ghost_left_hip_pitch_joint',
        #     'left_knee_joint': 'ghost_left_knee_joint',
        #     'left_ankle_joint': 'ghost_left_ankle_joint',
        #     'right_hip_yaw_joint': 'ghost_right_hip_yaw_joint',
        #     'right_hip_roll_joint': 'ghost_right_hip_roll_joint',
        #     'right_hip_pitch_joint': 'ghost_right_hip_pitch_joint',
        #     'right_knee_joint': 'ghost_right_knee_joint',
        #     'right_ankle_joint': 'ghost_right_ankle_joint',
        #     'left_shoulder_pitch_joint': 'ghost_left_shoulder_pitch_joint',
        #     'left_shoulder_roll_joint': 'ghost_left_shoulder_roll_joint',
        #     'left_shoulder_yaw_joint': 'ghost_left_shoulder_yaw_joint',
        #     'left_elbow_joint': 'ghost_left_elbow_joint',
        #     'right_shoulder_pitch_joint': 'ghost_right_shoulder_pitch_joint',
        #     'right_shoulder_roll_joint': 'ghost_right_shoulder_roll_joint',
        #     'right_shoulder_yaw_joint': 'ghost_right_shoulder_yaw_joint',
        #     'right_elbow_joint': 'ghost_right_elbow_joint',
        # }

        # Cache joint IDs for faster lookup
        self.joint_ids = {}
        self.ghost_joint_ids = {}

        self.accumulated_pos = np.zeros(2)

        # Spoof object pose from perception system
        # self.pose_publisher = PosePublisher(port=5555)
        
        self.headless = headless
    def initialize(self, fix_base=False, render=False, keyframe_name="home"):
        import mujoco
        import mujoco.viewer
        # Load MuJoCo model
        pwd = os.path.dirname(os.path.realpath(__file__))
        
        self.fix_base = fix_base
        self.with_table_and_object = (self.task_name is not None) and ("Manip" in self.task_name)
        if self.with_table_and_object:
            if self.fix_base:
                self.model = mujoco.MjModel.from_xml_path(self.robot_configuration._mujoco_xml_paths["fixed_base_with_table"])
                self.jpos_idx = 0
                self.jvel_idx = 0
                self.objects = {
                    "marker": {"idx": 28, "vel_idx": 27, "initialized": False},
                    "physics": {"idx": 35, "vel_idx": 33, "initialized": False},
                    "table_marker": {"idx": 42, "vel_idx": 39, "initialized": False}
                }
            else:
                self.model = mujoco.MjModel.from_xml_path(self.robot_configuration._mujoco_xml_paths["with_table"])
                self.jpos_idx = 7
                self.jvel_idx = 6
                self.objects = {
                    "marker": {"idx": 52, "vel_idx": 51, "initialized": False},
                    "physics": {"idx": 59, "vel_idx": 57, "initialized": False},
                    "table_marker": {"idx": 66, "vel_idx": 63, "initialized": False}
                }
        else:
            if self.fix_base:
                self.model = mujoco.MjModel.from_xml_path(self.robot_configuration._mujoco_xml_paths["fixed_base"])
                # self.model = mujoco.MjModel.from_xml_path(self.robot_configuration._mujoco_xml_paths["fixed_base_flipped"])
                # self.jpos_idx = 0
                # self.jvel_idx = 0
                self.jpos_idx = 0
                self.jvel_idx = 0
                self.objects = {
                    "marker": {"idx": 28, "vel_idx": 27, "initialized": False},
                    "physics": {"idx": 35, "vel_idx": 33, "initialized": False}
                }
            else:
                self.model = mujoco.MjModel.from_xml_path(self.robot_configuration._mujoco_xml_paths["default"])
                # input(self.robot_configuration._mujoco_xml_paths["default"])
                self.jpos_idx = 7
                self.jvel_idx = 6
                self.objects = {
                    "marker": {"idx": 52, "vel_idx": 51, "initialized": False},
                    "physics": {"idx": 59, "vel_idx": 57, "initialized": False}
                }
            
        self.data = mujoco.MjData(self.model)
        
        # Physics engine parameters
        # self.model.opt.timestep = 0.0005     # Simulation timestep (smaller = more accurate but slower)
        self.model.opt.timestep = 0.002     # Simulation timestep (smaller = more accurate but slower)
        self.model.opt.integrator = 1        # 0: semi-implicit Euler, 1: RK4
        self.model.opt.iterations = 10       # Maximum number of constraint solver iterations
        self.model.opt.solver = 0            # 0: PGS, 1: CG, 2: Newton
        self.model.opt.tolerance = 1e-10     # Solver convergence tolerance
        
        # self.disable_self_collision()
          
        # # Contact parameters
        self.model.opt.noslip_iterations = 10    # Number of iterations for non-slip constraints
        # self.model.opt.mpr_iterations = 50      # Maximum number of iterations for MPR
        # self.model.opt.gravity = [0, 0, -9.81]  # Gravity vector
        
        # # Numerical stability parameters
        # self.model.opt.viscosity = 0.001        # Global viscosity
        # self.model.opt.density = 1.2            # Global density
        # self.model.opt.o_margin = 0.002         # Global margin for contact detection
        
        # # Energy parameters
        # self.model.opt.impratio = 1.0           # Ratio of implicit to explicit integration
        # self.model.opt.wind = [0, 0, 0]         # Wind vector
        # self.model.opt.magnetic = [0, 0, 0]     # Magnetic flux vector
        
        # # Warm start parameters
        # self.model.opt.jacobian = 2             # 2: dense Jacobian for warm start
        # self.model.opt.cone = 0                 # 0: pyramidal cone, 1: elliptic cone
        
        # # Diagnostics and debug
        # self.model.opt.disableflags = 0         # Disable standard flags
        # self.model.opt.enableflags = 0          # Enable standard flags
        
        # Friction parameters (global defaults)
        # self.model.opt.o_solref = [0.02, 1]     # Global contact solver reference
        # self.model.opt.o_solimp = [0.9, 0.95, 0.001]  # Global contact solver impedance
        
        # # Time parameters
        # self.model.opt.timerate = 1.0           # Simulation time rate (1.0 = real-time)
        # self.model.opt.apirate = 100.0          # API update rate

        # Initialize state from keyframe if specified
        if keyframe_name is not None:
            self.load_keyframe(keyframe_name)

        from softmimic_deploy.src.joysticks.mujoco_joystick import MujocoJoystick
        self.joystick = MujocoJoystick()
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.joystick.key_callback)
            self.viewer._hide_overlay = True
            self.viewer._render_every_frame = True
        else:
            self.viewer=None

        # Initialize state values
        self.joint_pos = np.array(self.data.qpos[self.jpos_idx:self.jpos_idx+19])
        self.joint_vel = np.array(self.data.qvel[self.jvel_idx:self.jvel_idx+19])
        self.gravity_vector = self._calculate_gravity_vector()
        self.root_ang_vel = np.array(self.data.qvel[3:6])

        # Cache all joint IDs
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        # print(joint_names)
        self.joint_names = [name for name in joint_names if name is not None and not name.startswith("ghost")]
        for joint_name in self.joint_names:
            self.joint_ids[joint_name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            # print(f"ghost_{joint_name}")
            self.ghost_joint_ids[joint_name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"ghost_{joint_name}")

    def disable_self_collision(self):
        # Disable self-collision
        print("[MujocoInterface] Disabling self-collision for robot geoms")
        # Known environment geom names
        env_geom_names = ['floor', 'wall', 'table', 'ground']
        env_geom_ids = []

        for name in env_geom_names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                env_geom_ids.append(geom_id)

        # All geoms except environment
        robot_geom_ids = [i for i in range(self.model.ngeom) if i not in env_geom_ids]
        
        for geom_id in robot_geom_ids:
            self.model.geom_contype[geom_id] = 0  # Robot collision group
            self.model.geom_conaffinity[geom_id] = 2  # Only collides with group 1 (ground)
      

    def get_joint_pos(self, joint_order):
        self.joint_pos = np.array(self.data.qpos[self.jpos_idx:self.jpos_idx+self.robot_configuration.num_joints])
        if joint_order == "isaaclab":
            raise ValueError("Conversion from hw to isaaclab must be implemented in deploy_policy_interface")
        elif joint_order == "hw":
            hw_joint_pos = self.robot_configuration.remap_joint_array(self.joint_pos, from_order="mujoco", to_order="hw")
            return hw_joint_pos
        else:
            raise ValueError("Invalid joint order")
    
    def get_joint_vel(self, joint_order):
        self.joint_vel = np.array(self.data.qvel[self.jvel_idx:self.jvel_idx+self.robot_configuration.num_joints])
        if joint_order == "isaaclab":
            raise ValueError("Conversion from hw to isaaclab must be implemented in deploy_policy_interface")
        elif joint_order == "hw":
            hw_joint_vel = self.robot_configuration.remap_joint_array(self.joint_vel, from_order="mujoco", to_order="hw")
            return hw_joint_vel
        else:
            raise ValueError("Invalid joint order")
        
    def get_joint_acc(self, joint_order):
        self.joint_acc = np.array(self.data.qacc[self.jvel_idx:self.jvel_idx+self.robot_configuration.num_joints])
        if joint_order == "isaaclab":
            raise ValueError("Conversion from hw to isaaclab must be implemented in deploy_policy_interface")
        elif joint_order == "hw":
            hw_joint_acc = self.robot_configuration.remap_joint_array(self.joint_acc, from_order="mujoco", to_order="hw")
            return hw_joint_acc
        else:
            raise ValueError("Invalid joint order")
        
    def get_joint_torque(self, joint_order):
        self.joint_torque = np.array(self.data.ctrl[:self.robot_configuration.num_joints])
        if joint_order == "isaaclab":
            raise ValueError("Conversion from hw to isaaclab must be implemented in deploy_policy_interface")
        elif joint_order == "hw":
            hw_joint_torque = self.robot_configuration.remap_joint_array(self.joint_torque, from_order="mujoco", to_order="hw")
            return hw_joint_torque
        else:
            raise ValueError("Invalid joint order")
    
    def get_gravity_vector(self):
        return self.gravity_vector

    def get_root_ang_vel(self):
        return self.root_ang_vel
    
    def get_root_lin_vel(self):
        return self.root_lin_vel
    
    def get_foot_contact_states(self):
        return self.foot_contact_states
    
    def get_root_height(self):
        return self.root_height
    
    def get_obs_timestamp(self):
        return 0
    
    def get_pub_timestamp(self):
        return 0
    
    def get_write_timestamp(self):
        return 0
    
    def get_buttons(self):
        abxy = self.joystick.get_abxy()
        switches = self.joystick.get_buttons()
        
        if 1 in abxy:
            print(abxy)
        
        # Update key state
        self.key_state[8][1] = abxy[0]
        self.key_state[9][1] = abxy[1]
        self.key_state[10][1] = abxy[2]
        self.key_state[11][1] = abxy[3]
        
        # # Emulate L1 press after 1 second
        # if time.time() - self.init_time > 1 and self.key_state[1][1] == 0:
        # if switches[1] and self.key_state[1][1] == 0:
        #     self.key_state[1][1] = 1
        #     print("[MujocoInterface] Emulating L1 press!")
        self.key_state[1][1] = switches[1]

        # # # Emulate X press after 7 seconds
        # if time.time() - self.init_time > 3 and self.key_state[10][1] == 0:
        #     self.key_state[10][1] = 1
        #     # print("[MujocoInterface] Emulating X press!")

        return self.key_state
    
    def set_base_pose(self, pose, vel=None, adjust_camera=False, ghost_only=False):
        """
        Set the base pose using IMU data.
        
        Args:
            accel (np.ndarray): 3D acceleration vector from IMU
            gyro (np.ndarray): 3D angular velocity vector from IMU
        """
        # Skip if base is fixed
        if self.fix_base:
            return

        # R = get_rotation_matrix_from_rpy(euler_angles)
        R = pose[:3, :3]

        if adjust_camera:
            # Transform camera to body frame
            camera_rot = get_rotation_matrix_from_rpy(np.array([0, 0.88, 0]))
            R = R @ camera_rot.T
        
        # Convert to quaternion [x, y, z, w]
        trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / S
                x = 0.25 * S
                y = (R[0, 1] + R[1, 0]) / S
                z = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / S
                x = (R[0, 1] + R[1, 0]) / S
                y = 0.25 * S
                z = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / S
                x = (R[0, 2] + R[2, 0]) / S
                y = (R[1, 2] + R[2, 1]) / S
                z = 0.25 * S
        
        # Update base position and orientation
        # In MuJoCo's free joint, first 3 values are position, next 4 are quaternion [w x y z]
        # if ghost_only:
            # if base_vel is not None:
            #     self.data.qvel[25:27] = base_vel[:2]
            #     self.data.qvel[27:31] = 0
            #     self.accumulated_pos += base_vel[:2] * 0.02
            #     self.data.qpos[26:28] = self.accumulated_pos
            #     self.data.qpos[28] = self.data.qpos[2]
            #     self.data.qpos[29:33] = self.data.qpos[3:7]
            # else:
        # self.data.qpos[self.robot_configuration.num_joints+7:self.robot_configuration.num_joints+10] = pose[:3, 3]
        # self.data.qpos[self.robot_configuration.num_joints+10:self.robot_configuration.num_joints+14] = [w, x, y, z]
        # self.data.qvel[self.robot_configuration.num_joints+6:self.robot_configuration.num_joints+12] = 0
        # if not ghost_only:
        self.data.qpos[0:3] = pose[:3, 3]# + np.array([0, 0, 1.1])  # Set position
        self.data.qpos[3:7] = [w, x, y, z]  # Set orientation
        self.data.qvel[0:6] = 0 #gyro  # Angular velocity
            
        # Forward kinematics to update all positions
        mujoco.mj_forward(self.model, self.data)

    def lock_base(self):
        self.base_lock_pos = self.data.qpos[:7].copy()

    def unlock_base(self):
        self.base_lock_pos = None

    def set_object_pose(self, pose, object_name="marker", initialize_only=False, debug=False):
        """
        Sets the pose of a specified object in the MuJoCo simulation.
        
        Args:
            pose (numpy.ndarray): A 4x4 transformation matrix in head camera frame
            object_name (str): Name of the object to update ("marker" or "physics")
            initialize_only (bool): If True, only set the pose if object hasn't been initialized
            debug (bool): If True, print debug information about transforms
        """
        if not self.with_table_and_object:
            return

        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise ValueError("Pose must be a 4x4 numpy array transformation matrix")
        
        if object_name not in self.objects:
            raise ValueError(f"Unknown object name: {object_name}")
            
        # Skip if we're only initializing and the object is already initialized
        if initialize_only and self.objects[object_name]["initialized"]:
            return
            
        # Get camera transform
        camera_pos = np.array([0.1, 0.0175, 0.693])
        # camera_pos = np.array([0.1, 0.0175, 0.55])
        camera_rot = get_rotation_matrix_from_rpy(np.array([(3.14-2.45), 3.14, 1.57]))
        
        camera_transform = np.eye(4)
        camera_transform[:3, :3] = camera_rot
        camera_transform[:3, 3] = camera_pos
        
        # Get torso transform
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "torso_joint")
        torso_qpos_idx = self.model.jnt_qposadr[joint_id]
        torso_angle = self.data.qpos[torso_qpos_idx]
        
        torso_transform = np.eye(4)
        torso_transform[:3, :3] = get_rotation_matrix_from_rpy(np.array([0, 0, torso_angle]))
        
        
        # Get pelvis transform
        if self.fix_base:
            pelvis_pos = np.array([0, 0, 1.1])
            pelvis_quat = np.array([1, 0, 0, 0])
        else:
            pelvis_pos = self.data.qpos[:3]
            pelvis_quat = self.data.qpos[3:7]

        pelvis_rot = get_rotation_matrix_from_quaternion(pelvis_quat)
        
        pelvis_transform = np.eye(4)
        pelvis_transform[:3, :3] = pelvis_rot
        pelvis_transform[:3, 3] = pelvis_pos

        # Transform object pose to world frame
        world_transform = pelvis_transform @ torso_transform @ camera_transform @ pose
        
        # Extract position and rotation
        position = world_transform[:3, 3]
        rotation = world_transform[:3, :3]
        
        # Convert rotation matrix to quaternion [w, x, y, z]
        trace = np.trace(rotation)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (rotation[2, 1] - rotation[1, 2]) / S
            qy = (rotation[0, 2] - rotation[2, 0]) / S
            qz = (rotation[1, 0] - rotation[0, 1]) / S
        else:
            if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
                S = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
                qw = (rotation[2, 1] - rotation[1, 2]) / S
                qx = 0.25 * S
                qy = (rotation[0, 1] + rotation[1, 0]) / S
                qz = (rotation[0, 2] + rotation[2, 0]) / S
            elif rotation[1, 1] > rotation[2, 2]:
                S = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
                qw = (rotation[0, 2] - rotation[2, 0]) / S
                qx = (rotation[0, 1] + rotation[1, 0]) / S
                qy = 0.25 * S
                qz = (rotation[1, 2] + rotation[2, 1]) / S
            else:
                S = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
                qw = (rotation[1, 0] - rotation[0, 1]) / S
                qx = (rotation[0, 2] + rotation[2, 0]) / S
                qy = (rotation[1, 2] + rotation[2, 1]) / S
                qz = 0.25 * S
                
        quaternion = np.array([qw, qx, qy, qz])
        
        if debug:
            print(f"Setting {object_name} position to:", position)
            obj_idx = self.objects[object_name]["idx"]
            current_pos = self.data.qpos[obj_idx:obj_idx + 3]
            print("Distance moved:", np.linalg.norm(position - current_pos))
        
        # Get object index
        obj_idx = self.objects[object_name]["idx"]
        obj_vel_idx = self.objects[object_name]["vel_idx"]
        
        # Set velocities to zero
        self.data.qvel[obj_vel_idx:obj_vel_idx+6] = 0
        
        # Update position and orientation
        self.data.qpos[obj_idx:obj_idx + 3] = position
        self.data.qpos[obj_idx + 3:obj_idx + 7] = quaternion
        
        # Mark as initialized
        self.objects[object_name]["initialized"] = True
        
        # Forward kinematics to update positions of all bodies
        mujoco.mj_forward(self.model, self.data)
    
    def update_ghost_target(self, q, base_vel=None):
        """
        Update ghost robot joint positions to show target configuration
        
        Args:
            q (numpy.ndarray): Target joint positions in hw order
        """
        # Convert hw order to mujoco order
        q_mujoco = np.zeros(self.robot_configuration.num_joints)
        # q_mujoco[-8:] = q_arm
        q_mujoco = self.robot_configuration.remap_joint_array(q, from_order="hw", to_order="mujoco")
        
        # # Get the joint positions for upper body only
        # torso_idx = self.joint_ids['torso_joint']
        # left_shoulder_pitch_idx = self.joint_ids['left_shoulder_pitch_joint']
        # right_elbow_idx = self.joint_ids['right_elbow_joint']
        
        # Update ghost joint positions
        for joint_name in self.joint_names:
            orig_id  = self.joint_ids[joint_name]
            ghost_id = self.ghost_joint_ids[joint_name]
            if orig_id == -1 or ghost_id == -1:
                continue

            # copy position
            orig_addr = self.model.jnt_qposadr[orig_id]
            self.data.qpos[self.model.jnt_qposadr[ghost_id]] = \
                q_mujoco[orig_addr - self.jpos_idx]

            # zero velocity
            dof_addr  = self.model.jnt_dofadr[ghost_id]
            self.data.qvel[dof_addr] = 0.0

        if not self.fix_base:
            # Update ghost base position
            if base_vel is not None:
                self.data.qvel[self.robot_configuration.num_joints+6:self.robot_configuration.num_joints+8] = base_vel[:2]
                self.data.qvel[self.robot_configuration.num_joints+8:self.robot_configuration.num_joints+12] = 0
                self.accumulated_pos += base_vel[:2] * 0.02
                self.data.qpos[self.robot_configuration.num_joints+7:self.robot_configuration.num_joints+9] = self.accumulated_pos
                self.data.qpos[self.robot_configuration.num_joints+9] = self.data.qpos[2]
                self.data.qpos[self.robot_configuration.num_joints+10:self.robot_configuration.num_joints+14] = self.data.qpos[3:7]
            else:
                self.data.qpos[self.robot_configuration.num_joints+7:self.robot_configuration.num_joints+14] = self.data.qpos[0:7]
                self.data.qvel[self.robot_configuration.num_joints+6:self.robot_configuration.num_joints+12] = self.data.qvel[0:6]

            # self.data.qpos[33] = self.data.qpos[17]
            # self.data.qvel[33] = self.data.qvel[17]
        # else:
        #     self.data.qpos[26] = self.data.qpos[17]
        #     self.data.qvel[26] = self.data.qvel[17]

        # Forward kinematics to update ghost positions
        mujoco.mj_forward(self.model, self.data)

    def apply_command(self, kp, kd, q, dq, tau, torque_mode=False):
        # Update ghost target position
        # self.update_ghost_target(q)

        if torque_mode:
            q_meas = self.get_joint_pos("hw")
            dq_meas = self.get_joint_vel("hw")
            tau_comp = kp * (q - q_meas) + kd * (dq - dq_meas)
            # for j in range(20):
            #     q[j] = 0.0
            #     dq[j] = 0.0
            #     kp[j] = 0.0
            #     kd[j] = 0.0
            #     tau[j] = tau_comp[j] + tau[j]
            q[:] = 0.0
            dq[:] = 0.0
            kp[:] = 0.0
            kd[:] = 0.0
            tau[:] = tau_comp + tau
        
        for _ in range(int(self.robot_configuration.dt / self.model.opt.timestep)):
            joint_pos = self.get_joint_pos("hw")
            joint_vel = self.get_joint_vel("hw")

            # Apply PD control
            torques = kp * (q - joint_pos) - kd * joint_vel + tau
            torques = self.robot_configuration.remap_joint_array(torques, from_order="hw", to_order="mujoco")
            self.data.ctrl = torques

            if self.base_lock_pos is not None:
                self.data.qpos[:7] = self.base_lock_pos
                self.data.qvel[:6] = 0
                #ghost
                try:
                    self.data.qpos[self.robot_configuration.num_joints+7:self.robot_configuration.num_joints+14] = self.base_lock_pos
                    self.data.qvel[self.robot_configuration.num_joints+6:self.robot_configuration.num_joints+12] = 0
                except AttributeError:
                    raise ValueError("The ghost robot is missing or incorrectly indexed!")

            mujoco.mj_step(self.model, self.data)


        steps_per_frame = int(0.04 / self.robot_configuration.dt)
        if self.viewer is not None and self.step_counter % steps_per_frame == 0:
            self.viewer.sync()
        
        self.last_action = q
        
        # Update state values
        self.joint_pos = np.array(self.data.qpos[self.jpos_idx:self.jpos_idx+19])
        self.joint_vel = np.array(self.data.qvel[self.jvel_idx:self.jvel_idx+19])
        self.gravity_vector = self._calculate_gravity_vector()
        self.root_ang_vel = np.array(self.data.qvel[3:6])
        
        # privileged state information
        root_quat_wxyz = self.data.qpos[3:7]
        root_quat_xyzw = root_quat_wxyz[[1, 2, 3, 0]]
        r = R.from_quat(root_quat_xyzw)
        self.root_lin_vel = r.apply(np.array(self.data.qvel[0:3]), inverse=True)
        
        # Get the geom IDs for each foot. Using sets is more efficient for lookups.
        left_foot_geom_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"left_ankle_roll_{idx}") for idx in [1, 2, 3, 4]}
        right_foot_geom_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"right_ankle_roll_{idx}") for idx in [1, 2, 3, 4]}
        # left_foot_geom_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"left_ankle_roll")}
        # right_foot_geom_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"right_ankle_roll")}

        # Initialize contact flags
        left_contact = False
        right_contact = False

        # Iterate through all active contacts in the current simulation step
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Check if the contact involves a geom from the left foot
            if contact.geom1 in left_foot_geom_ids or contact.geom2 in left_foot_geom_ids:
                left_contact = True
                
            # Check if the contact involves a geom from the right foot
            if contact.geom1 in right_foot_geom_ids or contact.geom2 in right_foot_geom_ids:
                right_contact = True
                
            # Optimization: if both feet have contact, we can stop checking
            if left_contact and right_contact:
                break

        # Update the foot contact states
        # self.foot_contact_states = np.array([left_contact, right_contact])
        self.foot_contact_states = np.array([right_contact, left_contact])

        # print(f"Left foot IDs: {left_foot_geom_ids}, Right foot IDs: {right_foot_geom_ids}")
        # print(f"Foot contact states: {self.foot_contact_states}")

        self.root_height = self.data.qpos[2:3]

        self.step_counter += 1

        # self.spoof_object_pose_pub()

    def get_joystick_commands(self):        
        return self.joystick.get_command()[0:3]
    
    def get_height_commands(self):
        return self.joystick.get_command()[3] + 0.75
    
    def get_stiffness_commands(self):
        min_stiffness, max_stiffness = 40.0, 800.0
        log_stiffness_commands = min(max(np.log(40.0) + (np.log(800.0) - np.log(40.0)) * (self.joystick.get_command()[3] + 1) / 2, np.log(min_stiffness)), np.log(max_stiffness))
        stiffness_commands = np.exp(log_stiffness_commands)
        return stiffness_commands
    
    def get_pitch_commands(self):
        return 0.0

    def get_frequency_command(self):
        return 0.9

    def check_comms_safety(self):
        return True
    
    def damping_mode(self):
        print("============================")
        print("ENTER DAMPING MODE")
        print("============================")

    def shutdown(self):
        if self.viewer is not None:
            self.viewer.close()

    def _calculate_gravity_vector(self):
        # Get orientation

        # torso_body_id = self.model.body("torso_link").id
        # quat = np.array(self.data.xquat[torso_body_id])
        # rpy = get_rpy_from_quaternion(quat)
        quat = np.array(self.data.qpos[3:7])
        rpy = get_rpy_from_quaternion(quat)

#         pelvis_body_id = self.model.body("pelvis").id
#         quat = np.array(self.data.xquat[pelvis_body_id])
#         rpy = get_rpy_from_quaternion(quat)

        rotation_matrix = get_rotation_matrix_from_rpy(np.array(rpy))
        
        # Transform gravity vector from world to body frame
        gravity_world = np.array([0, 0, -1])
        gravity_body = rotation_matrix.T @ gravity_world
        
        return gravity_body

    def repeat_last_command(self):
        # if self.last_action is not None:
        #     self.apply_command(None, None, self.last_action, None, None)
        pass

    def load_keyframe(self, keyframe_name):
        """
        Load a named keyframe from the model.
        
        Args:
            keyframe_name (str): Name of the keyframe to load
            
        Returns:
            bool: True if keyframe was found and loaded, False otherwise
        """
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
        if key_id >= 0:
            print(f"Loading keyframe '{keyframe_name}'")
            # Copy keyframe state into qpos
            self.data.qpos[:] = self.model.key_qpos[key_id * self.model.nq:(key_id + 1) * self.model.nq]
            # Reset velocities to zero
            self.data.qvel[:] = 0
            # Forward kinematics to update all positions
            mujoco.mj_forward(self.model, self.data)
            return True
        else:
            print(f"Warning: Keyframe '{keyframe_name}' not found")
            return False

    def get_available_keyframes(self):
        """
        Get list of available keyframe names in the model.
        
        Returns:
            list: List of keyframe names
        """
        keyframes = []
        for i in range(self.model.nkey):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
            if name:
                keyframes.append(name)
        return keyframes

    def spoof_object_pose_pub(self):
        current_policy = self.joystick.get_current_policy()
        if current_policy == 2: # press 2 to spoof an object detection
            pose = np.eye(4)
            pose[:3, 3] = np.array([0.13, 0.29, 0.73])
            self.pose_publisher.send_pose(pose)
        elif current_policy == 1: # press 1 to spoof no object detection
            pose = np.eye(4)
            self.pose_publisher.send_pose(pose)
        else:
            pass
