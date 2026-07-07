MEASURE_FREQUENCY = True
SAVE_POLICY_DATA = True

# Import necessary libraries
import time
import numpy as np
import math
from datetime import datetime
from tabulate import tabulate

from scipy.signal import filtfilt, butter

# For handshake with controller and PC
import os, pathlib
import multiprocessing

# Import torch for policy execution
import torch
from softmimic_deploy.src.utils.yaml_loader import load_yaml

# Policy Weights Verification
import hashlib


from softmimic_deploy.src.sensors import *
from softmimic_deploy.src.sensors import ISAACLAB_FUNCTION_MAP
from softmimic_deploy.src.utils.logger_h5 import H5Logger as Logger
from scipy.spatial.transform import Rotation

def parse_env_cfg(cfg, robot_configuration):
    # Handles the parsing of environment configurations from IsaacLab
    dict_cfg = {}

    dict_cfg["command_ranges"] = {}
    if "upper_body_joints" in cfg["env_cfg"]["value"]["commands"].keys() and \
                cfg["env_cfg"]["value"]["commands"]["upper_body_joints"] is not None:                       
        dict_cfg["command_ranges"]["lin_vel_x"] = [
            cfg["env_cfg"]["value"]["commands"]["upper_body_joints"]["proc_ranges"]["min_vel"],
            cfg["env_cfg"]["value"]["commands"]["upper_body_joints"]["proc_ranges"]["max_vel"],
        ]
        dict_cfg["command_ranges"]["lin_vel_y"] = [0, 0]
        dict_cfg["command_ranges"]["ang_vel_yaw"] = [
            cfg["env_cfg"]["value"]["commands"]["upper_body_joints"]["proc_ranges"]["min_yaw_vel"],
            cfg["env_cfg"]["value"]["commands"]["upper_body_joints"]["proc_ranges"]["max_yaw_vel"],
        ]
    elif "base_velocity" in cfg["env_cfg"]["value"]["commands"].keys() and \
                cfg["env_cfg"]["value"]["commands"]["base_velocity"] is not None:
        dict_cfg["command_ranges"]["lin_vel_x"] = [
            cfg["env_cfg"]["value"]["commands"]["base_velocity"]["ranges"]["lin_vel_x"][0],
            cfg["env_cfg"]["value"]["commands"]["base_velocity"]["ranges"]["lin_vel_x"][1],
        ]
        dict_cfg["command_ranges"]["lin_vel_y"] = [
            cfg["env_cfg"]["value"]["commands"]["base_velocity"]["ranges"]["lin_vel_y"][0],
            cfg["env_cfg"]["value"]["commands"]["base_velocity"]["ranges"]["lin_vel_y"][1],
        ]
        dict_cfg["command_ranges"]["ang_vel_yaw"] = [
            cfg["env_cfg"]["value"]["commands"]["base_velocity"]["ranges"]["ang_vel_z"][0],
            cfg["env_cfg"]["value"]["commands"]["base_velocity"]["ranges"]["ang_vel_z"][1],
        ]
    else:
        raise ValueError("No velocity commands found in the configuration.")

    dict_cfg["control"] = {}
    dict_cfg["control"]["action_scale"] = cfg["env_cfg"]["value"]["actions"]["joint_pos"]["scale"]

    dict_cfg["obs_scales"] = {}
    if "velocity_commands" in cfg["env_cfg"]["value"]["observations"]["policy"] and \
         cfg["env_cfg"]["value"]["observations"]["policy"]["velocity_commands"] is not None:
        dict_cfg["obs_scales"]["lin_vel"] = cfg["env_cfg"]["value"]["observations"]["policy"]["velocity_commands"]["scale"] if cfg["env_cfg"]["value"]["observations"]["policy"]["velocity_commands"]["scale"] is not None else 1.0
        dict_cfg["obs_scales"]["ang_vel"] = cfg["env_cfg"]["value"]["observations"]["policy"]["velocity_commands"]["scale"] if cfg["env_cfg"]["value"]["observations"]["policy"]["velocity_commands"]["scale"] is not None else 1.0
    if "joint_vel" in cfg["env_cfg"]["value"]["observations"]["policy"] and \
        cfg["env_cfg"]["value"]["observations"]["policy"]["joint_vel"] is not None:
        dict_cfg["obs_scales"]["dof_vel"] = cfg["env_cfg"]["value"]["observations"]["policy"]["joint_vel"]["scale"] if cfg["env_cfg"]["value"]["observations"]["policy"]["joint_vel"]["scale"] is not None else 1.0

    dict_cfg["env"] = {}
    # dict_cfg["env"]["num_observation_history"] = 3 #cfg["env_cfg"]["value"]["observations"]["policy"]["history_length"]
    dict_cfg["env"]["num_observation_history"] = cfg["env_cfg"]["value"]["observations"]["policy"]["joint_vel"]["history_length"]
    # dict_cfg["env"]["num_observation_history"] = 10

    dict_cfg["actions"] = {}
    dict_cfg["actions"]["use_default_offset"] = cfg["env_cfg"]["value"]["actions"]["joint_pos"]["use_default_offset"]
    dict_cfg["actions"]["clip_to_limits"] = False #cfg["env_cfg"]["value"]["actions"]["joint_pos"]["clip_to_limits"]
    # dict_cfg["actions"]["clip_to_limits"] = True #cfg["env_cfg"]["value"]["actions"]["joint_pos"]["clip_to_limits"]


    # Collect observations
    table_data = []
    sensors = []
    future_sensors = []
    zero_obs = []
    zero_future_obs = []
    num_observations = 0
    for observation_name, observation in cfg["env_cfg"]["value"]["observations"]["policy"].items():
        if type(observation) is not dict:
            continue
        # print(observation)
        func = observation["func"]
        sensor = ISAACLAB_FUNCTION_MAP[func]
        if isinstance(sensor, dict):
            # differentiate by command_name
            # sensor = sensor[observation["params"]["command_name"]]
            # differentiate by observation name
            sensor = sensor[observation_name]
            # print(func, sensor, sensor.dim)
        if sensor is not None:
            # print(func, sensor, sensor.dim)
            # sensors.append(sensor)
            if sensor.dim == "nj":
                sensor_dim = robot_configuration.num_joints
            elif "nj" in str(sensor.dim):
                # parse e.g. "nj+3"
                sensor_dim = robot_configuration.num_joints + int(sensor.dim.split("+")[1])
            else:
                sensor_dim = sensor.dim

            table_data.append([observation_name, sensor_dim])
            num_observations += sensor_dim

            if "future" not in observation_name:
                zero_obs.append(np.zeros(sensor_dim))
                sensors.append(sensor)
            else:
                zero_future_obs.append(np.zeros(sensor_dim))
                future_sensors.append(sensor)
                
    print("[DeployPolicyInterface] Loaded sensors:")
    headers = ["Sensor Name", "Dimension"]
    print("--- Observation Space ---")
    print(tabulate(table_data, headers=headers, tablefmt="psql"))
    print(f"Total Observation Dimension: {num_observations}")
    print(f"Observation History Length: {dict_cfg['env']['num_observation_history']}")
    
    dict_cfg["env"]["num_observations"] = num_observations

    dict_cfg["control"]["stiffnesses"] = {}
    for actuator_group in cfg["env_cfg"]["value"]["scene"]["robot"]["actuators"].keys():
        for joint_names_expr in cfg["env_cfg"]["value"]["scene"]["robot"]["actuators"][actuator_group]["joint_names_expr"]:
            dict_cfg["control"]["stiffnesses"][joint_names_expr] = cfg["env_cfg"]["value"]["scene"]["robot"]["actuators"][actuator_group]["stiffness"][joint_names_expr]

    dict_cfg["control"]["dampings"] = {}
    for actuator_group in cfg["env_cfg"]["value"]["scene"]["robot"]["actuators"].keys():
        for joint_names_expr in cfg["env_cfg"]["value"]["scene"]["robot"]["actuators"][actuator_group]["joint_names_expr"]:
            dict_cfg["control"]["dampings"][joint_names_expr] = cfg["env_cfg"]["value"]["scene"]["robot"]["actuators"][actuator_group]["damping"][joint_names_expr]

    return dict_cfg, zero_obs, zero_future_obs, sensors, future_sensors

class RobotController:
    def __init__(
            self, 
            interface_type="dummy", 
            policy_name=None,
            motion_path=None,
            task_name=None,
            fix_base=False,
            render=False,
            timestamp=None,
    ):
        self.interface_type = interface_type
        self.policy_name = policy_name
        self.task_name = task_name

        if self.policy_name is None:
            raise ValueError("Policy name must be provided.")
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        
        policy_path = os.path.join(os.path.dirname(__file__), self.policy_name)
        policy_dir = os.path.dirname(policy_path)

        ## Load config
        
        env_cfg = load_yaml(os.path.join(policy_dir, "params/env.yaml"))
        agent_cfg = load_yaml(os.path.join(policy_dir, "params/agent.yaml"))

        cfg = {
            "env_cfg": {"value": env_cfg},
            "agent_cfg": {"value": agent_cfg},
        }
            
        self.iteration_count = 0
        self.control_dt = 0.02
        self.decimation = 1
        # self.control_dt = 0.01
        # self.decimation = 2
        # self.control_dt = 0.005
        # self.decimation = 4
            
        # Robot configuration (G1 only)
        expected_robot_type = "g1"
        command_cfg = cfg["env_cfg"]["value"]["commands"].get("upper_body_joints")
        if command_cfg is not None:
            robot_type = command_cfg.get("joint_config", {}).get("robot_type", expected_robot_type)
            if robot_type != expected_robot_type:
                raise ValueError(f"Unsupported robot type '{robot_type}'. This deployment only targets '{expected_robot_type}'.")

        from softmimic_deploy.src.robots.g1_config import G1Config as RobotConfig
        self.robot_configuration = RobotConfig(dt=self.control_dt)
            
        self.cfg, zero_obs, zero_future_obs, sensors, future_sensors = parse_env_cfg(cfg, self.robot_configuration)
        self.env = self.cfg["env"]
        self.robot_configuration.load_from_cfg(self.cfg)
        self.zero_obs_dim = np.concatenate(zero_obs).shape[0]
        if len(zero_future_obs) == 0:
            self.zero_future_obs_dim = 0
        else:
            self.zero_future_obs_dim = np.concatenate(zero_future_obs).shape[0]
        
        # Load policy
        self.policy = self.load_policy(policy_path)

        # State variables
        # self.obs_history_buf = torch.zeros(
        #     (self.env["num_observation_history"], self.env["num_observations"]),
        #     dtype=torch.float,
        #     device='cpu'
        # )
        self.obs_history_buf = [zero_obs for _ in range(self.env["num_observation_history"])]
        
        self.velocity_commands = np.zeros(3)
        self.prev_action = np.zeros(self.robot_configuration.num_joints)

        # Check velocity reference limits
        self.check_vel_ref_limits()
        
        # Robot interface
        if self.interface_type == "dummy":
            from softmimic_deploy.src.interfaces.dummy_interface import DummyInterface as RobotInterface
        elif self.interface_type == "lcm":
            from softmimic_deploy.src.interfaces.lcm_interface import G1LCMInterface as RobotInterface
        elif self.interface_type == "mujoco":
            from softmimic_deploy.src.interfaces.mujoco_interface import MujocoInterface as RobotInterface
        else:
            raise ValueError(f"Unsupported interface type: {self.interface_type}")
        
        self.robot = RobotInterface(robot_configuration=self.robot_configuration, cfg=self.cfg, task_name=self.task_name)
        self.robot.initialize(fix_base=fix_base, render=render)

        datasets_root = os.environ.get(
            "SOFTMIMIC_DATA_ROOT",
            str(pathlib.Path(__file__).absolute().parents[2] / "datasets" / "motions_csv"),
        )
        candidate_paths = [
            pathlib.Path(motion_path),
            pathlib.Path(datasets_root) / motion_path,
            pathlib.Path(datasets_root) / pathlib.Path(motion_path).name,
        ]
        demo_recording_path = next((p for p in candidate_paths if p.is_file()), None)
        if demo_recording_path is None:
            available = ", ".join(str(p) for p in candidate_paths)
            raise FileNotFoundError(f"Unable to resolve motion CSV '{motion_path}'. Checked: {available}")
        demo_recording_path = str(demo_recording_path.resolve())

        demo_terminal_frame_path = pathlib.Path(demo_recording_path).with_name(
            pathlib.Path(demo_recording_path).stem + "_micro_cycle_frame_mapping.npy"
        )
        if os.path.exists(demo_terminal_frame_path):
            print(f"Loading demo terminal frame mapping from {demo_terminal_frame_path}")
            self.demo_terminal_frame_mapping = np.load(demo_terminal_frame_path, allow_pickle=True)
        else:
            self.demo_terminal_frame_mapping = None
            print(f"No demo terminal frame mapping found at {demo_terminal_frame_path}")
        self.was_reset = False

        demo_start_time = 0.0

        scales = [
            0.25 if sensor == RootAngVelSensor else
            self.cfg["obs_scales"]["dof_vel"] if sensor == JointVelSensor else
            1.0 for sensor in sensors
        ]
        if WholeexoSensor in sensors:
            wholeexo_sensor = WholeexoSensor(
                self.robot, 
                scale=1.0,
                demo_recording_path=demo_recording_path,
                demo_start_time=demo_start_time,
                # upper_demo_only=True,
            )
        input(sensors)
        self.sensors = [
            wholeexo_sensor if sensor == WholeexoSensor else
            sensor(self.robot, scale=scale, wholeexo_sensor=wholeexo_sensor) if sensor in REFERENCE_SENSORS else
            sensor(self.robot, scale=scale) 
            for sensor, scale in zip(sensors, scales)
        ]
        self.future_sensors = [
            future_sensor(self.robot, scale=1.0, wholeexo_sensor=wholeexo_sensor) if future_sensor in REFERENCE_SENSORS else
            future_sensor(self.robot, scale=1.0) 
            for future_sensor in future_sensors
        ]
        self.ACTION_MODE = "pos"
        self.offset_actions = self.cfg["actions"]["use_default_offset"]
        self.limit_pos_targets = self.cfg["actions"]["clip_to_limits"]
        
        self.wholeexo_sensor_id = None
        for i, sensor in enumerate(self.sensors):
            if isinstance(sensor, WholeexoSensor):
                self.wholeexo_sensor_id = i
                self.wholeexo_sensor = sensor
                break
        if self.wholeexo_sensor_id is None:
            self.wholeexo_sensor = WholeexoSensor(
                self.robot, 
                scale=1.0,
                demo_recording_path=os.path.join(pathlib.Path(__file__).absolute().parents[2], "datasets/motions_csv/stand.csv"),
                demo_start_time=3.0,
            )
            print("WARNING: WholeExoSensor not found in sensors list")

        if SAVE_POLICY_DATA:
            log_dir = os.path.join(policy_dir, "logs", self.interface_type, self.timestamp)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
           
            self.logger = Logger(
                log_dir=log_dir,
                num_joints=self.robot_configuration.num_joints,
                # max_length=200 * 30, # 30 seconds
                max_length=200 * 60, # 60 seconds
                # max_length=200 * 60 * 5, # 5 minutes
            )
            
        self.is_mujoco = type(self.robot).__module__ == "softmimic_deploy.src.interfaces.mujoco_interface"
        

    ####################################################
    # Policy Loading and Observation/Action Processing #
    ####################################################

    def compute_observation(self):

        obs_buf = []
        for sensor in self.sensors:
            if isinstance(sensor, LastActionSensor):
                data = sensor.get_data(self.prev_action)
            elif isinstance(sensor, AirexoSensor):
                data = sensor.get_data()
            else:
                data = sensor.get_data()
            obs_buf += [torch.tensor(data).clone()]

        future_obs_buf = []
        for sensor in self.future_sensors:
            data = sensor.get_data()
            future_obs_buf += [torch.tensor(data).clone()]
            
    
        return obs_buf, future_obs_buf

    def get_action(self, obs):
        with torch.no_grad():
            action = self.policy(obs).squeeze(0).detach().numpy()
        return action
    
    def compute_actor_hash(self, actor):
        # Compute hash of the actor network weights
        hash_md5 = hashlib.md5()
        for param in actor.parameters():
            hash_md5.update(param.data.cpu().numpy().tobytes())
        return hash_md5.hexdigest()

    def load_policy(self, policy_path: str):
        # Load the policy on the CPU
        torch.set_num_threads(1)

        policy = torch.jit.load(policy_path, map_location='cpu')
        policy.eval()
        torch.jit.optimize_for_inference(policy)
        
        # load any state estimators if present
        # policy name is like path/to/model_0.jit_actor.jit
        # look for files like model_0.jit_estimator_force_applied.jit or model_0.jit_estimator_torque_applied.jit etc in the same folder
        
        policy_dir = os.path.dirname(policy_path)
        policy_base_name = os.path.basename(policy_path).split(".jit_actor.jit")[0]
        estimator_files = [f for f in os.listdir(policy_dir) if f.startswith(policy_base_name) and f.endswith(".jit") and "estimator" in f]
        self.estimators = {}
        for estimator_file in estimator_files:
            estimator_path = os.path.join(policy_dir, estimator_file)
            estimator = torch.jit.load(estimator_path, map_location='cpu')
            estimator.eval()
            torch.jit.optimize_for_inference(estimator)
            if "force_applied" in estimator_file:
                self.estimators["force_applied"] = estimator
                print(f"Loaded state estimator for force applied from {estimator_file}")
            elif "torque_applied" in estimator_file:
                self.estimators["torque_applied"] = estimator
                print(f"Loaded state estimator for torque applied from {estimator_file}")
            else:
                print(f"Warning: Unknown estimator type in file {estimator_file}. Skipping.")
        
        self.ob_buffer = torch.zeros(1, self.zero_obs_dim * self.env["num_observation_history"] + self.zero_future_obs_dim, dtype=torch.float, device='cpu') 
        
        # Define the mapped policy
        def mapped_policy(obs, info={}):
            self.ob_buffer.copy_(obs["obs_history"].cpu().reshape(1, -1))
            with torch.inference_mode():
                policy_input = self.ob_buffer
                action = policy.forward(policy_input)
            return action

        return mapped_policy

    #####################
    # Control Sequences #
    #####################

    def initialize_pose(self, duration=2.0):
        """Gradually moves all joints to the initial pose over the specified duration, with an option to switch to damping mode."""
        total_steps = int(duration / 0.005)  # Calculate the number of steps based on the time interval
        bar_length = 40  # Length of the progress bar
        
        initial_q = self.robot.get_joint_pos(joint_order="hw")
        final_q = self.robot_configuration.joint_default_positions(joint_order="hw")

        # # if the motion lib is loaded, use it to initialize the pose
        self.wholeexo_sensor.update_command()
        motion_lib_init_pos = self.wholeexo_sensor.get_command()[:self.robot_configuration.num_joints]
        final_q = motion_lib_init_pos.numpy()
        final_q = self.robot_configuration.remap_joint_array(final_q, from_order="isaaclab", to_order="hw")
        
        for step in range(total_steps):
            t = step / total_steps
            
            q_target = t * final_q + (1 - t) * initial_q

            # print(q_target)
            if self.is_mujoco:
                self.robot.update_ghost_target(q_target)
            
            self.robot.apply_command(
                kp = self.robot_configuration.joint_stiffnesses(joint_order="hw"), # Gradual kp increase
                kd = self.robot_configuration.joint_dampings(joint_order="hw"), # Gradual kd increase
                q = q_target,  # Interpolate towards the initial pose
                dq = np.zeros(self.robot_configuration.num_joints_hw),  # Zero velocity
                tau = np.zeros(self.robot_configuration.num_joints_hw),  # Zero torque
                torque_mode=False,
            )
            # Update progress bar
            progress = (step + 1) / total_steps
            blocks = int(bar_length * progress)
            bar = '=' * blocks + ' ' * (bar_length - blocks)
            percentage = int(progress * 100)

            time.sleep(0.005)  # Time step for gradual movement
        # print(q_target)  # Move to the next line after the progress bar is complete
        print("Pose initialized. Holding position. Press L2 to apply sine wave motion or X to deploy policy directly.")

    def execute_policy(self, previous_hold):
        """Executes the policy and applies the actions to the robot."""
        
        obs_buf, future_obs_buf = self.compute_observation()
        # print(obs_buf)

        # Get the action from the policy
        if previous_hold:
            action = self.prev_action
        else:
            # Update history buffer
            if self.iteration_count == 0:
                # On the first step, fill the entire history with the current observation
                self.obs_history_buf = [obs_buf] * self.env["num_observation_history"]
            else:
                # For all subsequent steps, use the standard sliding window update
                self.obs_history_buf = self.obs_history_buf[1:] + [obs_buf]
            
            obs_history_buf_flattened = [torch.cat([torch.tensor(obs_term[i]) for obs_term in self.obs_history_buf], dim=0) for i in range(len(obs_buf))]
            self.policy_input_obs = torch.cat(obs_history_buf_flattened, dim=0).unsqueeze(0)

            # concatenate the future observations
            if len(future_obs_buf) == 0:
                future_obs_buf = torch.zeros(1, 0, dtype=torch.float, device='cpu')
            else:
                future_obs_buf = [torch.tensor(future_obs_buf[i]).flatten() for i in range(len(future_obs_buf))]
                future_obs_buf = torch.cat(future_obs_buf, dim=0).unsqueeze(0)

            # concatenate the proprio observations
            self.policy_input_obs = torch.cat([self.policy_input_obs, future_obs_buf], dim=1) 

            obs = {"obs_history": self.policy_input_obs}
            action = self.get_action(obs)

        # Apply scaling to the action
        scaled_action = action * self.cfg["control"]["action_scale"]

        # Add residual
        if self.ACTION_MODE == "residual":
            wholeexo_command = self.wholeexo_sensor.get_command()
            reference = wholeexo_command[:self.robot_configuration.num_joints]
            target_vel = wholeexo_command[self.robot_configuration.num_joints:]
            
            scaled_action = scaled_action + reference.numpy()
           
            if self.is_mujoco:
                reference_isaaclab = reference.numpy()
                reference_hw = self.robot_configuration.remap_joint_array(reference_isaaclab, from_order="isaaclab", to_order="hw")
                self.robot.update_ghost_target(reference_hw)
                root_pose = np.eye(4)
                # root_pose[:3, 3] = np.array([0, 0, 1])
                root_pose[:3, 3] = self.robot.data.qpos[:3]
                root_quat = self.robot.data.qpos[3:7]
                root_rpy = Rotation.from_quat(root_quat[[1, 2, 3, 0]]).as_euler('xyz')
                # root_pitch = self.wholeexo_sensor.root_pitch
                # print(root_pitch, root_rpy[2])
                root_quat_yaw = Rotation.from_euler('xyz', [root_rpy[0], root_rpy[1], root_rpy[2]]).as_quat()
                root_pose[:3, :3] = Rotation.from_quat(root_quat_yaw).as_matrix()
                # root_pose[:3, 3] = self.sensors[3].root_pos
                # self.robot.set_base_pose(root_pose, ghost_only=True)
            target_vel = torch.zeros(3)
        else:
            reference = torch.zeros(self.robot_configuration.num_joints)
            target_vel = torch.zeros(3)
            
        # Reindex the action from URDF order to hardware order before applying
        hw_action = self.robot_configuration.remap_joint_array(scaled_action, from_order="isaaclab", to_order="hw")

        # Pre-compute constants and get arrays
        # joint_indices = np.arange(20) != 9  # Create boolean mask for valid joints
        joint_indices = np.arange(self.robot_configuration.num_joints)
        
        # Get all configuration arrays at once
        default_positions = self.robot_configuration.joint_default_positions(joint_order="hw")
        lower_limits = self.robot_configuration.joint_lower_limits(joint_order="hw")
        upper_limits = self.robot_configuration.joint_upper_limits(joint_order="hw")
        max_torques = self.robot_configuration.joint_saturated_torques(joint_order="hw")
        max_velocities = self.robot_configuration.joint_max_velocities(joint_order="hw")
        joint_kps = self.robot_configuration.joint_stiffnesses(joint_order="hw")
        joint_kds = self.robot_configuration.joint_dampings(joint_order="hw")
        
        # Get current state
        q = self.robot.get_joint_pos(joint_order="hw")
        # dq = self.filter_velocity_observations(self.robot.get_joint_vel(joint_order="hw"))
        dq = self.robot.get_joint_vel(joint_order="hw")
        gravity = self.robot.get_gravity_vector()
        timestamp_obs = self.robot.get_obs_timestamp()

        # Initialize output array
        q_des_clipped = np.zeros(self.robot_configuration.num_joints)
        
        # Apply offset if needed
        q_des = hw_action.copy()
        if self.offset_actions:
            q_des += default_positions
            
        # Debug printing for clipped actions
        # clipped_mask = np.abs(q_des - q_des_limited)[joint_indices] > 0.0
        # if np.any(clipped_mask):
        #     for j in np.where(clipped_mask)[0]:
        #         print(f"Clipping action for joint {j} from {q_des[j]} to {q_des_limited[j]}")
        
        if self.limit_pos_targets:
            # Soft joint limit clipping
            soft_joint_limit = 0.9
            q_des_limited = np.clip(
                q_des,
                lower_limits * soft_joint_limit,
                upper_limits * soft_joint_limit
            )
            q_des = q_des_limited
        
        # Compute torque-based clipping
        ideal_torque = joint_kps * (q_des - q) - joint_kds * dq
        clip_torque = max_torques * np.maximum(1 - np.abs(dq / max_velocities), 0)
        
        # Vectorized computation of clipped desired positions
        positive_direction = q_des > q
        q_des_clipped[joint_indices] = np.where(
            positive_direction[joint_indices],
            np.minimum(q_des[joint_indices], 
                    q[joint_indices] + (clip_torque[joint_indices] + joint_kds[joint_indices] * dq[joint_indices]) / joint_kps[joint_indices]),
            np.maximum(q_des[joint_indices], 
                    q[joint_indices] + (-clip_torque[joint_indices] + joint_kds[joint_indices] * dq[joint_indices]) / joint_kps[joint_indices])
        )
    
        
        if SAVE_POLICY_DATA:
            # record the joint states before applying the command
            dof_pos = self.robot.get_joint_pos(joint_order="hw")
            dof_vel = self.robot.get_joint_vel(joint_order="hw")
            dof_acc = self.robot.get_joint_acc(joint_order="hw")

        
        self.robot.apply_command(
            kp = self.robot_configuration.joint_stiffnesses(joint_order="hw"),
            kd = self.robot_configuration.joint_dampings(joint_order="hw"),
            q = q_des_clipped,
            dq = np.zeros(self.robot_configuration.num_joints),
            tau = np.zeros(self.robot_configuration.num_joints),
            torque_mode=False,
        )
        
        # Update prev_action to the current action
        # If we provide the scaled action here by mistake, things don't work!
        if not previous_hold:
            self.prev_action[:] = action

        if SAVE_POLICY_DATA:

            # record the torque after applying the command
            dof_torque = self.robot.get_joint_torque(joint_order="hw")

            dof_pos = self.robot_configuration.remap_joint_array(dof_pos, from_order="hw", to_order="isaaclab")
            dof_vel = self.robot_configuration.remap_joint_array(dof_vel, from_order="hw", to_order="isaaclab")
            dof_acc = self.robot_configuration.remap_joint_array(dof_acc, from_order="hw", to_order="isaaclab")
            dof_torque = self.robot_configuration.remap_joint_array(dof_torque, from_order="hw", to_order="isaaclab")

            q_des_clipped = self.robot_configuration.remap_joint_array(q_des_clipped, from_order="hw", to_order="isaaclab")

            timestamp_act = self.robot.get_write_timestamp()
            timestamp_pub = self.robot.get_pub_timestamp()
            timestamps = np.array([timestamp_obs, timestamp_act, timestamp_pub])

            kp = self.robot_configuration.joint_stiffnesses(joint_order="isaaclab")
            kd = self.robot_configuration.joint_dampings(joint_order="isaaclab")
            error_pos = q_des_clipped - dof_pos
            ideal_torque = kp * error_pos - kd * dof_vel
            self.logger.log(reference, dof_pos, dof_vel, dof_acc, target_vel)#, self.policy_input_obs)
    
    def shutdown(self):
        """Exiting by engaging damping mode and closing channels."""
        print("Exiting by engaging damping mode and closing channels...")
        self.robot.damping_mode()
        time.sleep(0.1)
        self.robot.shutdown()
        self.logger.close()
        print("Shutdown complete. Exiting.")

    #############
    # Main Loop #
    #############

    def run(self, time_limit_s=None, callback=None):
        print("Waiting for L1 button press to assume initial pose...")
        self.pose_initialised = False
        self.iteration_count = 0
        start_time = time.time()
        last_log_time = time.time()

        while True:
            if not self.pose_initialised:
                buttons = self.robot.get_buttons()
                if buttons[1][1] == 1:
                    print("L1 button detected. Assuming initial pose...")
                    if self.interface_type in ["mujoco"] and not self.robot.fix_base:
                        self.robot.lock_base()
                    self.initialize_pose()
                    if self.interface_type in ["mujoco"] and not self.robot.fix_base:
                        self.robot.unlock_base()
                    self.pose_initialised = True
                elif buttons[11][1] == 1:
                    print("Y button detected. Entering damping mode...")
                    self.robot.damping_mode()
                else:
                    #print("L1 not detected")
                    time.sleep(0.0001)
                    continue
            else:
                #time.sleep(0.01)
                #continue
                # Repeat the last command until a button press
                self.robot.repeat_last_command()
                
                buttons = self.robot.get_buttons()

                if buttons[5][1] == 1:
                    print("L2 button detected. Applying sine wave motion...")
                    self.apply_sine_wave()
                elif buttons[10][1] == 1:
                    print("X button detected. Deploying policy...")

                    loop_start_time = time.time()
                    if self.demo_terminal_frame_mapping is not None:
                        self.wholeexo_sensor.reset_motion_time(len(self.demo_terminal_frame_mapping) / 30.0 if self.demo_terminal_frame_mapping is not None else 0)
                    else:
                        self.wholeexo_sensor.lock_motion_time(0)

                    while True:
                        # if time_limit_s is not None and time.time() - start_time > time_limit_s:
                        if time_limit_s is not None and self.iteration_count / (1/self.control_dt) > time_limit_s:
                            print(f"Time limit of {time_limit_s} seconds reached. Exiting...")
                            self.robot.damping_mode()
                            return

                        previous_hold = self.iteration_count % self.decimation != 0
                        self.execute_policy(previous_hold=previous_hold)
                        self.iteration_count += 1
                        
                        # if callback is not None and not previous_hold:
                        if callback is not None:# and self.iteration_count % 4 == 0:
                            callback(self)

                        ctrl_time = time.time()
                        elapsed_time = ctrl_time - loop_start_time
                        # print(elapsed_time)
                        sleep_time = self.control_dt - elapsed_time
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                            loop_start_time = ctrl_time + sleep_time
                        else:
                            time.sleep(0.0001)
                            # print(f"Control loop took too long: {elapsed_time} s")
                            loop_start_time = ctrl_time - 0.0001
                            pass

                        buttons = self.robot.get_buttons()

                        if buttons[11][1] == 1:
                            print("Y button detected. Stopping policy execution...")
                            self.robot.damping_mode()
                            break
                        elif buttons[1][1] == 1:
                            print("L1 button detected. Stopping policy execution and calibrating...")
                            
                            # reset the motion library to the start
                            # MANUAL TRANSITION
                            if self.demo_terminal_frame_mapping is not None and not self.was_reset:
                                # left_hip_pitch_idx = 0
                                # left_hip_pitch_angle = self.wholeexo_sensor.get_command()[left_hip_pitch_idx]
                                # if left_hip_pitch_angle > 0:
                                # print(left_hip_pitch_angle)
                                motion_count = self.wholeexo_sensor.motion_count
                                motion_time = motion_count * self.robot.robot_configuration.dt
                                current_frame_index = int(motion_time * 30) # assuming the demo is at 30 Hz
                                current_frame_index = min(current_frame_index, len(self.demo_terminal_frame_mapping)-1)
                                terminal_frame = self.demo_terminal_frame_mapping[current_frame_index]
                                # self.wholeexo_sensor.lock_motion_time(72) 
                                print(terminal_frame, terminal_frame / 30.0)
                                self.wholeexo_sensor.reset_motion_time(terminal_frame / 30.0)
                                self.was_reset = True
                            # self.initialize_pose()
                            elif self.demo_terminal_frame_mapping is None:
                                self.wholeexo_sensor.reset_motion_time(0)
                                self.initialize_pose()
                                break
                        elif buttons[8][1] == 1:
                            print("A button detected. calibrating...")
                            self.wholeexo_sensor.reset_motion_time(0)
                            self.was_reset = False
                            self.initialize_pose()
                            break
                        elif buttons[10][1] == 1:
                            print("X button detected. Deploying policy...")
                            if self.demo_terminal_frame_mapping is not None:
                                if self.wholeexo_sensor.motion_count > len(self.demo_terminal_frame_mapping) / 30.0 / self.robot.robot_configuration.dt - 10:
                                    self.wholeexo_sensor.reset_motion_time(0)
                                    self.was_reset = False
                            else:
                                self.wholeexo_sensor.unlock_motion_time()
                           
                           
                        if self.iteration_count % 100 == 0:
                            if MEASURE_FREQUENCY:
                                print(f"Frequency: {100 / (time.time() - last_log_time)} Hz")
                                last_log_time = time.time()
                    
    ####################
    # Helper Functions #
    ####################
    def filter_velocity_observations(self, velocity_observations, order=5, cutoff=0.01):
        """Applies a 5th-order Butterworth low-pass filter to velocity observations."""
        return filtfilt(*butter(order, cutoff, btype='low'), velocity_observations) 
    
    def check_vel_ref_limits(self):
        # Construct arrays for min and max limits
        self.velocity_cmd_min_limits = np.array([
            self.cfg["command_ranges"]["lin_vel_x"][0],
            self.cfg["command_ranges"]["lin_vel_y"][0],
            self.cfg["command_ranges"]["ang_vel_yaw"][0]
        ])

        self.velocity_cmd_max_limits = np.array([
            self.cfg["command_ranges"]["lin_vel_x"][1],
            self.cfg["command_ranges"]["lin_vel_y"][1],
            self.cfg["command_ranges"]["ang_vel_yaw"][1]
        ])

        # Assert that all min limits are <= corresponding max limits
        assert np.all(self.velocity_cmd_min_limits <= self.velocity_cmd_max_limits), (
            "Velocity reference limits are wrong: some min limits are greater than max limits."
        )

if __name__ == '__main__':
    # ==================== Start of New Configuration Management ====================
    import argparse

    parser = argparse.ArgumentParser(description="Deploy a trained policy on a robot.")
    parser.add_argument(
        "--interface",
        type=str,
        required=True,
        choices=["dummy", "mujoco", "lcm"],
        help="The type of interface to use for the robot (e.g., 'mujoco', 'lcm')."
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to the *.jit policy file."
    )
    parser.add_argument(
        "--motion_path",
        type=str,
        required=True,
        help="Path to the motion csv file."
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="Isaac-G1-Natural-Walk",
        help="The name of the task, used for specific interface configurations (e.g., Isaac Lab)."
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering/video recording (only for 'mujoco')."
    )
    parser.add_argument(
        "--fix-base",
        action="store_true",
        help="Fix the robot's base in the simulation."
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Set a time limit in seconds for the policy execution."
    )
    args = parser.parse_args()
    # ===================== End of New Configuration Management =====================

    try:
        # Generate a unique timestamp for this run's logs.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Instantiate the controller with parameters from the command line.
        controller = RobotController(
            interface_type=args.interface,
            policy_name=args.policy,
            motion_path=args.motion_path,
            task_name=args.task_name,
            fix_base=args.fix_base,
            render=args.render,
            timestamp=timestamp,
        )

        # Set up rendering/video recording if enabled.
        if args.render:
            assert args.interface in ["mujoco"], "Rendering is only supported for 'mujoco' interface."
            import imageio

            # Create a log directory for the video.
            policy_path = os.path.join(os.path.dirname(__file__), args.policy)
            log_dir = os.path.join(os.path.dirname(policy_path), "logs", args.interface, timestamp)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            video_path = os.path.join(log_dir, "video.mp4")
            mp4_writer = imageio.get_writer(video_path, fps=50)
            print(f"Recording video to: {video_path}")

            if args.interface == "mujoco":
                import mujoco
                renderer = mujoco.Renderer(controller.robot.model, height=480, width=360)
                print('MuJoCo renderer initialized')

                def render_callback(controller):
                    side_cam_id = mujoco.mj_name2id(
                        controller.robot.model,
                        mujoco.mjtObj.mjOBJ_CAMERA,
                        'track'
                    )
                    renderer.update_scene(controller.robot.data, camera=side_cam_id)
                    rgb_image = renderer.render()
                    if controller.iteration_count > 0:
                        mp4_writer.append_data(rgb_image)
        else:
            # If rendering is disabled, provide a dummy callback.
            def render_callback(controller):
                pass

        # Run the main control loop.
        controller.run(time_limit_s=args.time_limit, callback=render_callback)

        # Clean up the video writer if it was used.
        if args.render:
            mp4_writer.close()

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Exiting gracefully...")
    finally:
        try:
            # Ensure the controller shuts down cleanly.
            controller.robot.damping_mode()
            controller.robot.shutdown()
            controller.logger.close()
        except Exception as e:
            print(f"Error during shutdown: {e}")
            pass
