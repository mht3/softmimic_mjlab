# airexo_sensor_zmq.py
from softmimic_deploy.src.sensors.base_sensor import BaseSensor
import numpy as np
import torch
import cv2
# import zmq
import json
import threading
import time
from typing import Dict, Optional

class WholeexoSensor(BaseSensor):
    '''
    Enhanced WholeexoSensor with support for multiple CSV files and stunt sequences.
    Compatible with motion libraries that return multiple future time steps.
    '''
    dim = 'nj'

    def __init__(self, interface, scale=1.0, demo_recording_path=None, demo_start_time=0.0, zmq_port=5555, upper_demo_only=False, stunt_configs=None, n_future_steps=100):
        super().__init__(interface, scale)

        self.num_envs = 1
        self.device = "cpu"
        self.upper_demo_only = upper_demo_only
        self.n_future_steps = n_future_steps # How many future steps to request
        
        # Store stunt configurations
        self.stunt_configs = stunt_configs or {}
        self.current_stunt_config = None
        self.stunt_motion_libs = {}  # Cache for different motion libraries
        
        # Cache for default motion library
        self.default_motion_lib = None
        self.default_motion_config = None
        
        # Current state
        self.current_csv_file = demo_recording_path
        self.stunt_start_motion_count = None
        self.stunt_start_offset = 0.0
        self.stunt_duration_steps = None
        
        # Initialize robot-specific configurations
        self.setup_robot_config()
        
        # Default motion library setup (cached)
        self.setup_default_motion_lib(demo_recording_path, demo_start_time)
        
        # State variables
        self.motion_count = np.zeros(self.num_envs)
        self.motion_lengths = np.array([self.motion_length] * self.num_envs)
        self.total_count = 0
        self.standing_timer = 0
        self.transition_time = 0
        self.locked_time = False

    def setup_robot_config(self):
        """Initialize robot-specific joint configurations"""
        # This part remains unchanged
        from softmimic_deploy.src.motion_lib.motion_lib_from_multi_csv import JointConfig
        
        robot_configs = {
            "G1Config": {
                "joint_config": JointConfig(
                    num_joints=29,
                    left_leg_indices={"hip_yaw": 6, "hip_roll": 3, "hip_pitch": 0, "knee": 9, "ankle_pitch": 13},
                    right_leg_indices={"hip_yaw": 7, "hip_roll": 4, "hip_pitch": 1, "knee": 10, "ankle_pitch": 14},
                    left_arm_indices={"shoulder_pitch": 11, "shoulder_roll": 15, "shoulder_yaw": 19, "elbow": 21, "wrist_pitch": 25, "wrist_roll": 23, "wrist_yaw": 27},
                    right_arm_indices={"shoulder_pitch": 12, "shoulder_roll": 16, "shoulder_yaw": 20, "elbow": 22, "wrist_pitch": 26, "wrist_roll": 24, "wrist_yaw": 28},
                    thigh_length=0.3,
                    calf_length=0.3,
                ),
                "frequency": 1.1,
                "height": 0.8,
                "reindex_mapping": [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
            }
        }
        
        robot_config_class = self.interface.robot_configuration.__class__.__name__
        if robot_config_class not in robot_configs:
            raise ValueError(f"Unknown robot configuration class: {robot_config_class}. WholeexoSensor supports only G1Config.")
        
        config = robot_configs[robot_config_class]
        self.joint_config = config["joint_config"]
        self.frequency = config["frequency"]
        self.height = config["height"]
        self.reindex_mapping = config["reindex_mapping"]

    def setup_default_motion_lib(self, demo_recording_path, demo_start_time=0.0):
        """Setup the default motion library, passing n_future_steps."""
        default_key = f"default_{demo_recording_path}_{self.upper_demo_only}"
        
        if self.default_motion_lib is not None and self.default_motion_config == default_key:
            self.motion_lib = self.default_motion_lib
            self.motion_length = getattr(self, '_cached_motion_length', 2000.0)
            return
        
        self.motion_length = 2000.0
        
        if demo_recording_path and demo_recording_path.endswith(".csv"):
            from softmimic_deploy.src.motion_lib.motion_lib_from_multi_csv import ProceduralMotionLibFromDemo
            self.motion_lib = ProceduralMotionLibFromDemo(
                demo_recording_path,
                motion_dt=0.02,
                start_range=[demo_start_time, demo_start_time],
                n_future_steps=self.n_future_steps, # Pass the parameter here
                demo_playback_mode="references",
                joint_config=self.joint_config,
                reindex_mapping=self.reindex_mapping,
                feet_contacts=True,
                upper_demo_only=self.upper_demo_only,
            )
        else:
            raise ValueError("WholeexoSensor requires a CSV motion file when running with the G1 robot.")
        
        self.default_motion_lib = self.motion_lib
        self.default_motion_config = default_key
        self._cached_motion_length = self.motion_length
        self.setup_motion_parameters()
        self.load_initial_motion()
    
    # Unchanged methods: setup_motion_parameters, load_initial_motion
    def setup_motion_parameters(self):
        self.pitch = 0.0
        self.dynamic_standing_only = False
        self.stance_duration = 0.6
        self.swing_height = 0.1
        self.motion_lib.set_motion_parameters(freq_range=[self.frequency, self.frequency],height_range=[self.height, self.height],vel_range=[0.0, 0.0],yaw_vel_range=[0, 0],standing_prob=1.0,stance_duration=self.stance_duration,swing_height=self.swing_height,dynamic_standing_only=self.dynamic_standing_only,)

    def load_initial_motion(self):
        self.motion_lib.load_motions(env_ids=torch.arange(0, self.num_envs),durations=torch.ones(self.num_envs, device=self.device) * self.motion_length,limb_weights=[np.zeros(10)] * self.num_envs,random_sample=False,start_idx=0,)

    def configure_for_stunt(self, stunt_name, csv_file, start_time, duration):
        """Configure sensor for a stunt, passing n_future_steps."""
        print(f"Configuring WholeExoSensor for stunt: {stunt_name}")
        try:
            if csv_file not in self.stunt_motion_libs:
                try:
                    from softmimic_motions.lafan.motion_lib_from_csv_optimized import ProceduralMotionLibFromDemo  # type: ignore[attr-defined]
                except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                    raise ImportError(
                        "Stunt playback requires the optional 'softmimic_motions' package. "
                        "Install it or disable stunt configurations."
                    ) from exc
                self.stunt_motion_libs[csv_file] = ProceduralMotionLibFromDemo(
                    csv_file,
                    n_future_steps=self.n_future_steps, # Pass the parameter here
                    demo_playback_mode="references",
                    joint_config=self.joint_config,
                    reindex_mapping=self.reindex_mapping,
                    upper_demo_only=self.upper_demo_only,
                )
            
            self.motion_lib = self.stunt_motion_libs[csv_file]
            dt = self.interface.robot_configuration.dt
            duration_steps = int(duration / dt) if duration else None
            
            self.current_stunt_config = {"name": stunt_name, "csv_file": csv_file, "start_time": start_time, "duration": duration}
            self.stunt_start_motion_count = self.total_count
            self.stunt_start_offset = start_time
            self.stunt_duration_steps = duration_steps
            
            self.motion_lib.set_motion_parameters(freq_range=[self.frequency, self.frequency], height_range=[self.height, self.height], vel_range=[0.0, 0.0], yaw_vel_range=[0, 0], standing_prob=0.0, stance_duration=self.stance_duration, swing_height=self.swing_height, dynamic_standing_only=False,)
            start_idx = int(start_time / dt)
            self.motion_lib.load_motions(env_ids=torch.arange(0, self.num_envs), durations=torch.ones(self.num_envs, device=self.device) * (duration or 10.0), limb_weights=[np.zeros(10)] * self.num_envs, random_sample=False, start_idx=start_idx,)
            self.motion_count = np.zeros(self.num_envs)
            
        except Exception as e:
            print(f"  Failed to configure for stunt {stunt_name}: {e}")
            self.reset_to_default()

    def reset_to_default(self):
        print("Resetting WholeExoSensor to default locomotion")
        self.current_stunt_config = None
        self.stunt_start_motion_count = None
        self.stunt_start_offset = 0.0
        self.stunt_duration_steps = None
        if self.default_motion_lib is not None:
            self.motion_lib = self.default_motion_lib
            self.motion_length = self._cached_motion_length
            self.setup_motion_parameters()
            self.motion_count = np.zeros(self.num_envs)
            self.motion_lengths = np.array([self.motion_length] * self.num_envs)
            self.load_initial_motion()
        else:
            self.setup_default_motion_lib(self.current_csv_file)

    def switch_to_cached_stunt(self, csv_file):
        if csv_file in self.stunt_motion_libs:
            self.motion_lib = self.stunt_motion_libs[csv_file]
            return True
        return False

    def is_stunt_complete(self):
        stunt_config = self.current_stunt_config
        start_motion_count = self.stunt_start_motion_count  
        duration_steps = self.stunt_duration_steps
        if not stunt_config or start_motion_count is None or not duration_steps: return False
        elapsed_steps = self.total_count - start_motion_count
        return elapsed_steps >= duration_steps

    def get_stunt_progress(self):
        if (not self.current_stunt_config or self.stunt_start_motion_count is None or not self.stunt_duration_steps): return 0.0
        elapsed_steps = self.total_count - self.stunt_start_motion_count
        return min(elapsed_steps / self.stunt_duration_steps, 1.0)

    def update_command(self):
        """
        Update motion command, correctly parsing multi-timestep data.
        Separates current state from future predicted states.
        """
        if not self.locked_time:
            self.motion_count += 1
            self.total_count += 1
        
        # Motion reloading logic (unchanged)
        if not self.current_stunt_config:
            update_ids = np.where(self.motion_count % (self.motion_lengths / self.interface.robot_configuration.dt) == 0)[0]
            if len(update_ids) > 0:
                self.motion_lib.load_motions(
                    env_ids=torch.tensor(update_ids, device=self.device),
                    durations=torch.tensor(self.motion_lengths[update_ids], device=self.device, dtype=torch.float32),
                    limb_weights=[np.zeros(10)] * len(update_ids),
                    random_sample=False, start_idx=0,
                )
                self.motion_count[update_ids] = 0

        # Get motion state (time calculation is unchanged)
        motion_time = self.motion_count * self.interface.robot_configuration.dt
        if self.current_stunt_config:
            motion_time += self.stunt_start_offset
        
        motion_res = self.motion_lib.get_motion_state(
            torch.arange(0, self.num_envs),
            torch.tensor(motion_time, device=self.device, dtype=torch.float32),
            offset=None,
            # future_frame_dt=0.02,
        )

        # print(motion_res["dof_pos"][:, 0, :])

        # --- MODIFIED SECTION: Parse multi-timestep output ---
        motion_id = 0 # We only have one environment

        # Robustness: Check if the output has the time dimension.
        # This handles the case where the fallback procedural lib is used.
        if motion_res["dof_pos"].ndim == 2:
            for key, val in motion_res.items():
                if isinstance(val, torch.Tensor):
                    motion_res[key] = val.unsqueeze(1) # Add a time-step dimension of size 1

        # Extract the CURRENT state (at time_step = 0)
        self.root_vel = motion_res["root_vel"][motion_id, 0]
        self.root_ang_vel = motion_res["root_ang_vel"][motion_id, 0]
        self.root_pos = motion_res["root_pos"][motion_id, 0]
        self.root_pitch = motion_res["root_pitch"][motion_id, 0]
        self.root_gravity = motion_res["gravity_vec"][motion_id, 0]
        # Use .get() for optional keys like foot_contacts
        self.foot_contacts = motion_res["foot_contacts"][motion_id, 0]

        # Extract FUTURE states (from time_step = 1 onwards)
        future_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        # input(motion_res["root_vel"].shape)
        self.future_root_vel = motion_res["root_vel"][motion_id, future_steps]
        self.future_root_ang_vel = motion_res["root_ang_vel"][motion_id, future_steps]
        self.future_root_pos = motion_res["root_pos"][motion_id, future_steps]
        self.future_root_gravity = motion_res["gravity_vec"][motion_id, future_steps]
        self.future_foot_contacts = motion_res["foot_contacts"][motion_id, future_steps]
        self.future_dof_pos = motion_res["dof_pos"][motion_id, future_steps]

        # Command is based on the CURRENT dof positions
        current_dof_pos = motion_res["dof_pos"][motion_id, 0, :]

        self._command = torch.cat([
            current_dof_pos.unsqueeze(0), # Ensure it has a batch dimension
            ], dim=1
        )
        
        self._command = self._command * self.scale

    # get_data, get_command, get_stunt_status remain unchanged.
    def get_data(self):
        self.update_command()
        return self.get_command()

    def get_command(self):
        return self._command[0]

    def get_stunt_status(self):
        stunt_config = self.current_stunt_config
        start_motion_count = self.stunt_start_motion_count
        duration_steps = self.stunt_duration_steps
        if not stunt_config: return {"active": False}
        elapsed_steps = self.total_count - start_motion_count if start_motion_count is not None else 0
        elapsed_time = elapsed_steps * self.interface.robot_configuration.dt
        duration_time = duration_steps * self.interface.robot_configuration.dt if duration_steps else 0
        progress = min(elapsed_steps / duration_steps, 1.0) if duration_steps and duration_steps > 0 else 0.0
        return {"active": True, "name": stunt_config["name"], "csv_file": stunt_config["csv_file"], "progress": progress, "elapsed_time": elapsed_time, "duration": duration_time, "elapsed_steps": elapsed_steps, "duration_steps": duration_steps, "complete": duration_steps and elapsed_steps >= duration_steps}
    
    def reset_motion_time(self, time=0.0):
        # self.motion_count = np.zeros(self.num_envs)
        # self.total_count = 0
        self.motion_count = np.zeros(self.num_envs) + int(time / self.interface.robot_configuration.dt)
        self.total_count = int(time / self.interface.robot_configuration.dt)
        self.update_command()

    def lock_motion_time(self, time):
        self.motion_count = np.zeros(self.num_envs) + int(time / self.interface.robot_configuration.dt)
        self.total_count = int(time / self.interface.robot_configuration.dt)
        self.update_command()
        self.locked_time = True

    def unlock_motion_time(self):
        self.locked_time = False
