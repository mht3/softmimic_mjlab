import torch
import numpy as np
import time
import math
import os
# Keep scipy for Rotation if needed for the test script, but main lib is pure torch
from scipy.spatial.transform import Rotation # Used in __main__ for test data

_GRAVITY_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _gravity_vector(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    cached = _GRAVITY_CACHE.get(key)
    if cached is None:
        cached = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype)
        _GRAVITY_CACHE[key] = cached
    return cached

# --- Helper functions for pitch and roll (torch-based) ---
def pitch_from_gravity_torch(g_base):
    gx, gy, gz = g_base[..., 0], g_base[..., 1], g_base[..., 2]
    norm_g = torch.linalg.norm(g_base, dim=-1)
    near_zero_mask = norm_g < 1e-8
    safe_norm_g = torch.where(near_zero_mask, torch.ones_like(norm_g), norm_g)
    gx_n = gx / safe_norm_g
    gy_n = gy / safe_norm_g
    gz_n = gz / safe_norm_g
    denominator_pitch = torch.sqrt(gy_n**2 + gz_n**2)
    pitch = torch.atan2(-gx_n, denominator_pitch)
    pitch = torch.where(near_zero_mask, torch.zeros_like(pitch), pitch)
    return pitch

def roll_from_gravity_torch(g_base):
    gx, gy, gz = g_base[..., 0], g_base[..., 1], g_base[..., 2]
    norm_g = torch.linalg.norm(g_base, dim=-1)
    near_zero_mask = norm_g < 1e-8
    safe_norm_g = torch.where(near_zero_mask, torch.ones_like(norm_g), norm_g)
    gy_n = gy / safe_norm_g
    gz_n = gz / safe_norm_g
    roll = torch.atan2(gy_n, -gz_n)
    roll = torch.where(near_zero_mask, torch.zeros_like(roll), roll)
    return roll


def angular_velocity_from_quats(q1, q2, dt):
    """
    Compute angular velocity from two quaternions (xyzw format)
    
    Args:
        q1: Initial quaternion (N, 4) in xyzw format
        q2: Final quaternion (N, 4) in xyzw format  
        dt: Time step (scalar or tensor)
    
    Returns:
        Angular velocity (N, 3) in rad/s
    """
    # Ensure shortest path (handle quaternion double cover)
    # If dot product is negative, negate one quaternion
    # dot_product = torch.sum(q1 * q2, dim=-1, keepdim=True)
    dot_product = torch.sum(q1.double() * q2.double(), dim=-1, keepdim=True).float()
    q2 = torch.where(dot_product < 0, -q2, q2)
    
    # Compute quaternion difference: q_diff = q2 * q1_inverse
    q1_inverse = quat_conjugate(q1)  # For unit quats, inverse = conjugate
    q_diff = quat_multiply(q2, q1_inverse)
    
    # For accurate angular velocity with large rotations:
    xyz = q_diff[..., :3]
    w = q_diff[..., 3:4]
    
    # Handle the case where w might be negative (rotation > 180°)
    w = torch.abs(w)
    
    # xyz_norm = torch.norm(xyz, dim=-1, keepdim=True)
    xyz_norm = torch.norm(xyz.double(), dim=-1, keepdim=True).float()
    
    # Use atan2 for proper angle computation
    angle = 2.0 * torch.atan2(xyz_norm, w)
    
    # Avoid division by zero
    safe_norm = torch.where(xyz_norm < 1e-6, torch.ones_like(xyz_norm), xyz_norm)
    axis = xyz / safe_norm
    
    angular_vel = (angle / dt) * axis
    
    # Handle near-zero rotation case
    angular_vel = torch.where(xyz_norm < 1e-6, 
                             2.0 * xyz / dt,  # Small angle approximation
                             angular_vel)
    
    return angular_vel

def euler_to_quat(euler, order='xyz'):
    """Convert euler angles to quaternions"""
    r = Rotation.from_euler(order, euler, degrees=False)
    return r.as_quat()  # Returns [x, y, z, w]

def quat_conjugate(q):
    """Conjugate of quaternion (xyzw format)"""
    q_conj = q.clone()
    q_conj[..., :3] *= -1  # Negate xyz, keep w
    return q_conj

def quat_multiply(q1, q2):
    """Multiply two quaternions (xyzw format)"""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    
    return torch.stack([x, y, z, w], dim=-1)

class JointConfig:
    def __init__(self, num_joints, left_leg_indices, right_leg_indices, left_arm_indices, right_arm_indices, thigh_length, calf_length):
        self.num_joints = num_joints
        self.num_arm_joints = len(left_arm_indices) + len(right_arm_indices)
        self.left_leg = left_leg_indices
        self.right_leg = right_leg_indices
        self.left_arm = left_arm_indices
        self.right_arm = right_arm_indices
        self.thigh_length = thigh_length
        self.calf_length = calf_length


class ProceduralMotionLibFromDemo:
    MIN_FRAMES_PER_FILE = 3 # Minimum frames a file must have after processing (for stable velocity calcs)

    def __init__(self, input_path, n_future_steps=0, motion_dt=0.005, start_range=[0.0, 0.0], demo_playback_mode="references", device="cpu", stand_height=1.1, pitch_angle=0.0, joint_order_mode="default", joint_config=None, reindex_mapping=None, feet_contacts=False, upper_demo_only=False, speed=1.0):

        assert demo_playback_mode in ["references", "commands"], f"Invalid demo playback mode: {demo_playback_mode}"
        assert joint_order_mode in ["default", "torsoroot"], f"Invalid joint order mode: {joint_order_mode}"

        self.device = device
        self.demo_playback_mode = demo_playback_mode
        self.joint_order_mode = joint_order_mode
        # start_range is processed below after num_files is known
        self.joint_config = joint_config
        
        if self.joint_config is None:
             raise ValueError("JointConfig must be provided")
        
        if reindex_mapping is None:
            self.reindex_mapping = torch.arange(self.joint_config.num_joints, device=self.device, dtype=torch.long)
        else:
            self.reindex_mapping = torch.tensor(reindex_mapping, device=self.device, dtype=torch.long)

        self.feet_contacts = feet_contacts
        self.upper_demo_only = upper_demo_only
        self.n_future_steps = n_future_steps

        self.data_fps = 30.0 * speed
        self._motion_dt = motion_dt 
        self.dt = 1.0 / self.data_fps 

        # Handle single or multiple input_paths
        if isinstance(input_path, str):
            self.input_paths = [input_path]
        elif isinstance(input_path, list):
            self.input_paths = input_path
        else:
            raise TypeError("input_path must be a string or a list of strings.")
        self.num_files = len(self.input_paths)

        # Parse start_range to get start_times_per_file
        if isinstance(start_range, list) and len(start_range) > 0:
            if isinstance(start_range[0], list): # List of lists, e.g., [[s1, e1], [s2, e2], ...]
                if len(start_range) != self.num_files:
                    raise ValueError(f"If start_range is a list of lists, its length ({len(start_range)}) must match the number of input files ({self.num_files}).")
                self.start_times_per_file = [sr[0] for sr in start_range]
            elif isinstance(start_range[0], (int, float)): # Single list [s, e]
                if len(start_range) != 2:
                     raise ValueError("If start_range is a single list like [start_val, end_val], it must have 2 elements.")
                self.start_times_per_file = [start_range[0]] * self.num_files # Apply same start_time to all
            else:
                raise TypeError("Elements of start_range must be lists [start, end] or start_range itself must be a list [start_val, end_val_unused].")
        else:
            raise TypeError("start_range must be a list, e.g., [0.0, 0.0] or [[0.0, 0.0], [1.0, 1.0]].")

        loaded_data_list = []
        num_frames_loaded_per_file_list = [] 
        
        self.pos_cols = slice(0, 3)
        self.quat_cols = slice(3, 7) 
        self.joint_cols = slice(7, 7 + self.joint_config.num_joints)
        expected_cols = 7 + self.joint_config.num_joints
        #new thing for foot contacts
        if self.feet_contacts:
            self.feet_contacts_cols = slice(7 + self.joint_config.num_joints, 9 + self.joint_config.num_joints)
            expected_cols += 2 # Add 2 for foot contacts

        for i, path_str in enumerate(self.input_paths):
            current_start_time = self.start_times_per_file[i]
            raw_data_np = self.load_csv_data(path_str, start_time=current_start_time)
            
            current_num_frames = raw_data_np.shape[0] if raw_data_np.size > 0 else 0

            if current_num_frames < self.MIN_FRAMES_PER_FILE:
                print(f"Warning: Motion file '{path_str}' has {current_num_frames} frames after start_time={current_start_time} (min is {self.MIN_FRAMES_PER_FILE}). Padding.")
                if current_num_frames > 0: # File is short but not empty
                    fill_frame = raw_data_np[0:1, :] 
                    processed_data_np = np.repeat(fill_frame, self.MIN_FRAMES_PER_FILE, axis=0)
                else: # File is empty
                    processed_data_np = np.zeros((self.MIN_FRAMES_PER_FILE, expected_cols), dtype=np.float32)
                    # Set identity quaternion for pure zero padding qx,qy,qz,qw
                    processed_data_np[:, self.quat_cols.start + 3] = 1.0 
                
                num_frames_loaded_per_file_list.append(self.MIN_FRAMES_PER_FILE)
                loaded_data_list.append(torch.from_numpy(processed_data_np).to(self.device).float())
            else:
                num_frames_loaded_per_file_list.append(current_num_frames)
                loaded_data_list.append(torch.from_numpy(raw_data_np).to(self.device).float())
        
        if not loaded_data_list: # Should not happen due to padding, but as a safeguard
            raise RuntimeError("No motion data loaded or all files resulted in empty data even after padding attempts.")

        self.num_frames_per_file = torch.tensor(num_frames_loaded_per_file_list, device=self.device, dtype=torch.long)
        
        self.raw_data_tensor = torch.cat(loaded_data_list, dim=0)

        self.start_frame_global_indices = torch.cat((
            torch.tensor([0], device=self.device, dtype=torch.long), 
            torch.cumsum(self.num_frames_per_file[:-1], dim=0)
        ))
        
        self.num_total_frames = self.raw_data_tensor.shape[0]

        self.motion_length_per_file = self.num_frames_per_file * self.dt
        
        self.stand_height = stand_height
        self.pitch_angle = pitch_angle

        if self.upper_demo_only:
            self._apply_upper_body_only_modification()
            
        self.precompute_keypoints()

        self._cached_step_indices: torch.Tensor | None = None

    def _apply_upper_body_only_modification(self):
        """
        Modifies self.raw_data_tensor in-place for upper-body-only demos.
        This function sets a static standing leg posture based on the original motion's
        height and pitch, zeroes out horizontal root motion, and sets a fixed root pitch.
        This must be called BEFORE precomputing keypoints to ensure consistency.
        """
        # --- 1. Extract necessary data from the entire raw tensor ---
        root_pos_orig = self.raw_data_tensor[:, self.pos_cols]
        root_quat_orig = self.raw_data_tensor[:, self.quat_cols]

        # --- 2. Calculate inputs for IK from original motion's characteristics ---
        root_height = root_pos_orig[:, 2]  # Preserve original Z motion

        # Calculate gravity vector in base frame from original quaternions
        g_world = _gravity_vector(self.device, root_pos_orig.dtype).unsqueeze(0)
        qw, qx, qy, qz = root_quat_orig[:, 3], root_quat_orig[:, 0], root_quat_orig[:, 1], root_quat_orig[:, 2]
        q_inv_vec = torch.stack([-qx, -qy, -qz], dim=-1)
        v = g_world.expand(root_quat_orig.shape[0], 3)
        uv = torch.cross(q_inv_vec, v, dim=-1)
        uuv = torch.cross(q_inv_vec, uv, dim=-1)
        g_base = v + 2.0 * (qw.unsqueeze(-1) * uv + uuv)
        
        root_pitch_orig = pitch_from_gravity_torch(g_base)

        # --- 3. Approximate a comfortable standing leg posture for G1 ---
        hip_angle = torch.full_like(root_height, -0.2)
        knee_angle = torch.full_like(root_height, 0.42)
        ankle_angle = torch.full_like(root_height, -0.23)

        # --- 4. Overwrite the raw data tensor in-place ---
        dof_pos_new = self.raw_data_tensor[:, self.joint_cols] # Modify in place

        # Overwrite leg joints
        self.joint_config.left_leg = {'hip_yaw': 2, 'hip_roll': 1, 'hip_pitch': 0, 'knee': 3, 'ankle_pitch': 4, 'ankle_roll': 5}
        self.joint_config.right_leg = {'hip_yaw': 8, 'hip_roll': 7, 'hip_pitch': 6, 'knee': 9, 'ankle_pitch': 10, 'ankle_roll': 11}
        indices_and_values = [
            (self.joint_config.left_leg["hip_yaw"], 0.0),
            (self.joint_config.right_leg["hip_yaw"], 0.0),
            (self.joint_config.left_leg["hip_roll"], 0.0),
            (self.joint_config.right_leg["hip_roll"], 0.0),
            (self.joint_config.left_leg["ankle_roll"], 0.0),
            (self.joint_config.right_leg["ankle_roll"], 0.0),
            (self.joint_config.left_leg["hip_pitch"], hip_angle),
            (self.joint_config.right_leg["hip_pitch"], hip_angle),
            (self.joint_config.left_leg["knee"], knee_angle),
            (self.joint_config.right_leg["knee"], knee_angle),
            (self.joint_config.left_leg["ankle_pitch"], ankle_angle),
            (self.joint_config.right_leg["ankle_pitch"], ankle_angle),
        ]
        for idx, val in indices_and_values:
            if isinstance(val, torch.Tensor):
                dof_pos_new[:, idx] = val
            else: # is float
                dof_pos_new[:, idx] = val
        
        # Overwrite root position (keep original Z, zero out X and Y)
        self.raw_data_tensor[:, self.pos_cols.start] = 0.0
        self.raw_data_tensor[:, self.pos_cols.start + 1] = 0.0
        
        # Overwrite root orientation to a fixed pitch
        pitch_rad_half = self.pitch_angle * 0.5
        qx_new = torch.full((self.num_total_frames,), 0.0, device=self.device, dtype=root_pos_orig.dtype)
        qy_new = torch.full((self.num_total_frames,), math.sin(pitch_rad_half), device=self.device, dtype=root_pos_orig.dtype)
        qz_new = torch.full((self.num_total_frames,), 0.0, device=self.device, dtype=root_pos_orig.dtype)
        qw_new = torch.full((self.num_total_frames,), math.cos(pitch_rad_half), device=self.device, dtype=root_pos_orig.dtype)
        self.raw_data_tensor[:, self.quat_cols] = torch.stack([qx_new, qy_new, qz_new, qw_new], dim=-1)

        # Overwrite foot contacts to be always active
        if self.feet_contacts:
            self.raw_data_tensor[:, self.feet_contacts_cols] = 1.0


    def load_csv_data(self, csv_path, start_time=0.0):
        try:
            raw_data = np.genfromtxt(csv_path, delimiter=',')
            if raw_data.ndim == 1: raw_data = raw_data.reshape(1, -1)

            num_frames_file, dof = raw_data.shape
            # print(f"Loaded CSV '{csv_path}' with shape: ({num_frames_file}, {dof})", end="")

            expected_dof = 7 + self.joint_config.num_joints
            if self.feet_contacts:
                expected_dof += 2
            if dof < expected_dof:
                 print(f" -> Warning: Has {dof} cols, expected {expected_dof}. Padding joints.", end="")
                 padding_cols = expected_dof - dof
                 if padding_cols > 0:
                    padding = np.ones((num_frames_file, padding_cols))
                    raw_data = np.hstack((raw_data, padding))
            elif dof > expected_dof:
                 print(f" -> Warning: Has {dof} cols, expected {expected_dof}. Truncating.", end="")
                 raw_data = raw_data[:, :expected_dof]
            
            data_fps = self.data_fps # Assuming LAFAN dataset is 30 fps for all files
            start_frame = int(start_time * data_fps)
            
            if start_frame >= num_frames_file:
                print(f" -> Warning: Start time {start_time:.2f}s ({start_frame} frames) is beyond motion length ({num_frames_file} frames). Resulting data will be empty for this file.")
                raw_data = raw_data[0:0, :] # Return empty array with correct number of columns
            else:
                raw_data = raw_data[start_frame:]
            
            # print(f" -> Processed, returning {raw_data.shape[0]} frames.")
            return raw_data

        except Exception as e:
            raise RuntimeError(f"Failed to load CSV data from '{csv_path}': {e}") from e
        
    def precompute_keypoints(self):
        from softmimic_deploy.src.motion_lib.keypoint_extractor import KeypointExtractor

        if self.joint_config.num_joints != 29:
            raise ValueError(
                f"Unsupported joint configuration with {self.joint_config.num_joints} joints. "
                "This deployment expects 29 joints for the G1 robot."
            )
        keypoint_extractor = KeypointExtractor(robot_type="g1")
        # Precompute keypoints for the entire dataset
        batch_size = 1000
        num_batches = (self.num_total_frames + batch_size - 1) // batch_size
        all_keypoints = []
        all_keypoint_rotations = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, self.num_total_frames)
            batch_data = self.raw_data_tensor[start_idx:end_idx]#.cpu().numpy()
            root_pos = batch_data[..., self.pos_cols]#.astype(np.float32)
            root_quat = batch_data[..., self.quat_cols]#.astype(np.float32)
            joint_pos = batch_data[..., self.joint_cols][:, self.reindex_mapping]
            # print(joint_pos.shape)
            keypoints, keypoint_rotations, _ = keypoint_extractor.compute_keypoints_batch(
                root_pos, root_quat, joint_pos
            )
            keypoints_tensor = keypoints.detach() if isinstance(keypoints, torch.Tensor) else torch.tensor(keypoints, dtype=torch.float)
            keypoint_rotations_tensor = (
                keypoint_rotations.detach()
                if isinstance(keypoint_rotations, torch.Tensor)
                else torch.tensor(keypoint_rotations, dtype=torch.float)
            )

            if keypoints_tensor.device != self.device:
                keypoints_tensor = keypoints_tensor.to(self.device)
            if keypoint_rotations_tensor.device != self.device:
                keypoint_rotations_tensor = keypoint_rotations_tensor.to(self.device)

            all_keypoints.append(keypoints_tensor.contiguous())
            all_keypoint_rotations.append(keypoint_rotations_tensor.contiguous())
        self.keypoints = torch.cat(all_keypoints, dim=0).to(self.device)
        self.keypoint_rotations = torch.cat(all_keypoint_rotations, dim=0).to(self.device)

    # --- Keep these methods, implementation might be needed later ---
    def load_motions(self, **kwargs):
        # print("load_motions called but not implemented yet.")
        pass

    def set_motion_parameters(self, **kwargs):
        # print("set_motion_parameters called but not implemented yet.")
        pass

    def reset_positions(self, motion_ids):
        # This doesn't directly apply to the stateless get_motion_state,
        # but could reset internal state if the class were stateful.
        # print("reset_positions called but not implemented yet.")
        pass

    def _rotate_vector_by_quat_inverse_torch(self, v, q_xyzw):
        qw, qx, qy, qz = q_xyzw[..., 3], q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2]
        q_inv_vec = torch.stack([-qx, -qy, -qz], dim=-1)
        
        # This broadcasting logic handles cases where v might be (1, 1, 3) and q might be (B, T, 4)
        if v.dim() < q_inv_vec.dim():
            v = v.expand_as(q_inv_vec)

        uv = torch.cross(q_inv_vec, v, dim=-1)
        uuv = torch.cross(q_inv_vec, uv, dim=-1)
        rotated_v = v + 2.0 * (qw.unsqueeze(-1) * uv + uuv)
        return rotated_v

    def _quat_to_yaw_torch(self, q_xyzw):
        qx, qy, qz, qw = q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2], q_xyzw[..., 3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return yaw

    def _angle_diff_torch(self, a, b):
        diff = a - b
        return torch.atan2(torch.sin(diff), torch.cos(diff))
    
    def get_max_times(self, motion_ids):
        """
        Get the maximum times for the given motion IDs.
        """

        file_indices = motion_ids % self.num_files
        return self.motion_length_per_file[file_indices] #- self.start_times_per_file[file_indices]

    def get_motion_state(self, motion_ids, motion_times, offset=None, future_frame_dt=0.02):
        # if offset is not None: motion_times = motion_times + offset
        
        batch_size = motion_ids.shape[0]
        n_steps = self.n_future_steps + 1
        device = self.device

        # --- Step 1: Calculate all target times for current and future steps ---
        # Create a time offset for each future step: [0, dt, 2*dt, ...]
        if (
            self._cached_step_indices is None
            or self._cached_step_indices.device != motion_times.device
            or self._cached_step_indices.dtype != motion_times.dtype
        ):
            self._cached_step_indices = torch.arange(n_steps, device=motion_times.device, dtype=motion_times.dtype)
        time_steps = self._cached_step_indices * future_frame_dt
        target_motion_times = motion_times.unsqueeze(1) + time_steps.unsqueeze(0)

        # --- Step 2: Convert all target times to floating-point frame indices and interpolation alphas ---
        file_indices = motion_ids % self.num_files
        target_frame_times = target_motion_times * self.data_fps

        # print(self.data_fps)
        # print(target_frame_times)
        # print(target_frame_times.floor().long())
        
        # Get the bracketing frame indices and alpha for EACH target time
        local_frame_idx0_raw = (target_frame_times + 1e-6).floor().long()  # Shape: (batch_size, n_steps)
        lerp_alphas = target_frame_times - local_frame_idx0_raw   # Shape: (batch_size, n_steps)

        # The second bracketing frame is just the next one
        local_frame_idx1_raw = local_frame_idx0_raw + 1
        local_frame_idx2_raw = local_frame_idx1_raw + 1

        # --- Step 3: Clamp indices and gather data ---
        max_frames_for_selected_files = self.num_frames_per_file[file_indices]
        # Expand clamp bound for broadcasting across the n_steps dimension
        upper_clamp_bound = torch.maximum(
            torch.zeros_like(max_frames_for_selected_files), 
            max_frames_for_selected_files - 2 #30#2
        ).unsqueeze(1).expand(-1, n_steps)
        
        lower_clamp_bound = torch.zeros_like(upper_clamp_bound, device=device, dtype=local_frame_idx0_raw.dtype)
        local_frame_idx0 = torch.clamp(local_frame_idx0_raw, lower_clamp_bound, upper_clamp_bound)
        local_frame_idx1 = torch.clamp(local_frame_idx1_raw, lower_clamp_bound, upper_clamp_bound + 1)
        local_frame_idx2 = torch.clamp(local_frame_idx2_raw, lower_clamp_bound, upper_clamp_bound + 1)
        
        base_global_indices = self.start_frame_global_indices[file_indices].unsqueeze(1)
        global_frame_idx0 = base_global_indices + local_frame_idx0
        global_frame_idx1 = base_global_indices + local_frame_idx1
        global_frame_idx2 = base_global_indices + local_frame_idx2

        # Fetch bracketing data for all time steps at once
        data0 = self.raw_data_tensor[global_frame_idx0] # Shape: (batch_size, n_steps, num_features)
        data1 = self.raw_data_tensor[global_frame_idx1] # Shape: (batch_size, n_steps, num_features)
        data2 = self.raw_data_tensor[global_frame_idx2] # For velocity calc

        # Expand alpha for broadcasting with feature dimensions
        lerp_alphas_expanded = lerp_alphas.unsqueeze(-1) # Shape: (batch_size, n_steps, 1)

        # --- Step 4: Extract and Interpolate All States ---
        root_pos0, quat0_xyzw, dof_pos0 = data0[..., self.pos_cols], data0[..., self.quat_cols], data0[..., self.joint_cols]
        root_pos1, quat1_xyzw, dof_pos1 = data1[..., self.pos_cols], data1[..., self.quat_cols], data1[..., self.joint_cols]
        root_pos2, quat2_xyzw, dof_pos2 = data2[..., self.pos_cols], data2[..., self.quat_cols], data2[..., self.joint_cols]
        keypoints0 = self.keypoints[global_frame_idx0]
        keypoints1 = self.keypoints[global_frame_idx1]
        
        # Linear interpolation for vectors
        root_pos = torch.lerp(root_pos0, root_pos1, lerp_alphas_expanded)
        dof_pos = torch.lerp(dof_pos0, dof_pos1, lerp_alphas_expanded)
        interp_keypoints = torch.lerp(keypoints0, keypoints1, lerp_alphas_expanded.unsqueeze(-1))
        
        # NLERP for quaternions
        dot = torch.sum(quat0_xyzw * quat1_xyzw, dim=-1, keepdim=True)
        quat1_interp = torch.where(dot < 0, -quat1_xyzw, quat1_xyzw)
        interp_quat_xyzw = torch.lerp(quat0_xyzw, quat1_interp, lerp_alphas_expanded)
        norm_interp = torch.linalg.norm(interp_quat_xyzw, dim=-1, keepdim=True)
        root_rot = torch.where(norm_interp > 1e-8, interp_quat_xyzw / norm_interp, quat0_xyzw)

        # Interpolate keypoint rotations using spherical linear interpolation (SLERP)
        keypoint_rotations0 = self.keypoint_rotations[global_frame_idx0]
        keypoint_rotations1 = self.keypoint_rotations[global_frame_idx1]
        dot = torch.sum(keypoint_rotations0 * keypoint_rotations1, dim=-1, keepdim=True)
        quat1_interp = torch.where(dot < 0, -keypoint_rotations1, keypoint_rotations1)
        # print(keypoint_rotations0.shape, quat1_interp.shape, lerp_alphas_expanded.unsqueeze(-1).shape)
        interp_keypoint_rotations = torch.lerp(keypoint_rotations0, quat1_interp, lerp_alphas_expanded.unsqueeze(-1).repeat(1, 1, quat1_interp.shape[2], 1))
        norm_interp = torch.norm(interp_keypoint_rotations, dim=-1, keepdim=True)
        keypoint_rotations_interp = torch.where(norm_interp > 1e-8, interp_keypoint_rotations / norm_interp, keypoint_rotations0)
        # print('a', keypoint_rotations_interp.shape)

        # Foot contacts are discrete, use the value from the first bracketing frame.
        if self.feet_contacts:
            foot_contacts = data0[:, :, self.feet_contacts_cols]
        else:
            foot_contacts = torch.ones((batch_size, n_steps, 2), device=device, dtype=root_pos0.dtype)

        # --- Step 5: Calculate Velocities and Derived Values ---
        # Velocities are calculated using finite difference over the source data's timestep (self.dt)
        # This provides a stable estimate of the instantaneous velocity at each interpolated point.
        delta_t_data = self.dt
        
        # Calculate velocities at the bracketing frames (frame0 and frame1)
        root_vel_global_0 = (root_pos1 - root_pos0) / delta_t_data
        root_vel_global_1 = (root_pos2 - root_pos1) / delta_t_data
        dof_vel_0 = (dof_pos1 - dof_pos0) / delta_t_data
        dof_vel_1 = (dof_pos2 - dof_pos1) / delta_t_data

        # Normalize quaternions for stable angular velocity calculation
        identity_quat_fill = torch.zeros_like(quat0_xyzw); identity_quat_fill[..., 3] = 1.0
        norm0 = torch.linalg.norm(quat0_xyzw, dim=-1, keepdim=True)
        safe_quat0_xyzw = torch.where(norm0 > 1e-8, quat0_xyzw / norm0, identity_quat_fill)
        norm1 = torch.linalg.norm(quat1_xyzw, dim=-1, keepdim=True)
        safe_quat1_xyzw = torch.where(norm1 > 1e-8, quat1_xyzw / norm1, identity_quat_fill)
        norm2 = torch.linalg.norm(quat2_xyzw, dim=-1, keepdim=True)
        safe_quat2_xyzw = torch.where(norm2 > 1e-8, quat2_xyzw / norm2, identity_quat_fill)

        root_ang_vel_global_0 = angular_velocity_from_quats(safe_quat0_xyzw, safe_quat1_xyzw, delta_t_data)
        root_ang_vel_global_1 = angular_velocity_from_quats(safe_quat1_xyzw, safe_quat2_xyzw, delta_t_data)

        # Linearly interpolate the calculated velocities
        root_vel_global = torch.lerp(root_vel_global_0, root_vel_global_1, lerp_alphas_expanded)
        dof_vel = torch.lerp(dof_vel_0, dof_vel_1, lerp_alphas_expanded)
        root_ang_vel_global = torch.lerp(root_ang_vel_global_0, root_ang_vel_global_1, lerp_alphas_expanded)

        # Transform global velocities into the (interpolated) root's local frame
        root_vel = self._rotate_vector_by_quat_inverse_torch(root_vel_global, root_rot)

        root_ang_vel = self._rotate_vector_by_quat_inverse_torch(root_ang_vel_global, root_rot)

        # Other derived values are calculated from the final interpolated states
        g_world = _gravity_vector(device, root_pos.dtype).unsqueeze(0)
        g_base = self._rotate_vector_by_quat_inverse_torch(g_world, root_rot)
        root_pitch = -1.0 * pitch_from_gravity_torch(g_base).unsqueeze(-1)
        root_roll = -1.0 * roll_from_gravity_torch(g_base).unsqueeze(-1)
        root_yaw = self._quat_to_yaw_torch(root_rot).unsqueeze(-1)
        
        remap_idx = self.reindex_mapping
        gait_params = torch.zeros((batch_size, n_steps, 8), device=device, dtype=root_pos.dtype)

        # The returned dictionary now contains time-series tensors.
        # The consumer of this function can get the "current" state by indexing `[:, 0]`
        # and future states by indexing `[:, 1:]`.
        return {
            'root_pos': root_pos,
            'root_rot': root_rot,
            'root_vel': root_vel,
            'root_ang_vel': root_ang_vel,
            'dof_pos': dof_pos[:,:, remap_idx],
            'dof_vel': dof_vel[:,:, remap_idx],
            'keypoints': interp_keypoints,
            'keypoint_rotations': keypoint_rotations_interp,
            'foot_contacts': foot_contacts,
            'root_pitch': root_pitch,
            'root_roll': root_roll,
            'root_yaw': root_yaw,
            'gravity_vec': g_base,
            'root_vel_global': root_vel_global,
            'root_ang_vel_global': root_ang_vel_global,
            'gait_parameters': gait_params,
        }
