import torch
import numpy as np

# Import the base class and its dependencies
from softmimic_deploy.src.motion_lib.motion_lib_from_multi_csv import ProceduralMotionLibFromDemo, JointConfig, \
    pitch_from_gravity_torch, roll_from_gravity_torch, angular_velocity_from_quats


class AugmentedMotionLibFromDemo(ProceduralMotionLibFromDemo):
    """
    Extends ProceduralMotionLibFromDemo to load and process augmented motion data using column slices.

    This class reads CSV files that contain four contiguous blocks of data:
    1. The original reference motion (handled by the parent class).
    2. An 'adapted' motion, which has the same structure and number of columns as the original.
    3. Force and Torque information related to the adapted motion.
    4. Collision/Forcefield metadata (stiffness, setpoint/origin, plane normal).
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the motion library for augmented data.

        It first calls the parent constructor, which will load the first block of data
        (the original motion). Then, it re-loads the full data and parses the additional
        adapted, force, and collision blocks using predefined column slices.
        """
        # --- Step 1: Call the parent __init__ ---
        super().__init__(*args, **kwargs)
        print("AugmentedMotionLib: Parent class initialized. Now loading augmented data...")

        # --- Step 2: Define Column Slices for ALL data blocks ---
        num_orig_joints = self.joint_config.num_joints
        num_orig_contacts = 2 if self.feet_contacts else 0
        self.orig_block_size = 7 + num_orig_joints + num_orig_contacts
        self.orig_data_cols = slice(0, self.orig_block_size)
        
        # Block 2: Adapted Motion Data
        self.adap_block_start = self.orig_block_size
        self.adap_block_size = self.orig_block_size - num_orig_contacts
        self.adap_data_cols = slice(self.adap_block_start, self.adap_block_start + self.adap_block_size)
        # Slices relative to the start of the ADAPTED block
        self.adap_pos_cols = slice(0, 3)
        self.adap_quat_cols = slice(3, 7)
        self.adap_joint_cols = slice(7, 7 + num_orig_joints)
        # if self.feet_contacts:
        #     self.adap_feet_contacts_cols = slice(7 + num_orig_joints, self.adap_block_size)

        # Block 3: Force and Torque Data
        self.force_block_start = self.adap_block_start + self.adap_block_size
        self.force_block_size = 9 # 1 (id) + 3 (force) + 3 (torque) + 1 (stiff) + 1 (rot_stiff)
        self.force_data_cols = slice(self.force_block_start, self.force_block_start + self.force_block_size)
        # Slices relative to the start of the FORCE block
        self.force_id_cols = slice(0, 1)
        self.force_vec_cols = slice(1, 4)
        self.torque_vec_cols = slice(4, 7)
        self.stiffness_vec_cols = slice(7, 9)

        # --- Block 4: Collision/Forcefield Metadata ---
        self.collision_block_start = self.force_block_start + self.force_block_size
        # UPDATED: Size is now 12 to include rotational stiffness, setpoint rotation, etc.
        self.collision_block_size = 12 # 2 (stiffs) + 3 (origin) + 4 (quat) + 3 (normal)
        self.collision_data_cols = slice(self.collision_block_start, self.collision_block_start + self.collision_block_size)
        # Slices relative to the start of the COLLISION block
        self.ff_stiffness_cols = slice(0, 2)     # Now captures both linear and rotational stiffness
        self.ff_origin_cols = slice(2, 5)        # Shifted by 1
        self.ff_setpoint_rot_cols = slice(5, 9)  # NEW: Quaternion for setpoint orientation
        self.ff_normal_cols = slice(9, 12)       # Shifted by 5

        # --- Step 3: Re-load data using our own logic ---
        self._load_and_parse_augmented_data()

        # --- Step 4: Re-compute keypoints from the newly loaded ORIGINAL and ADAPTED data blocks ---
        self.precompute_keypoints()
        self.precompute_adapted_keypoints()
        print("AugmentedMotionLib: Initialization complete.")

    def _load_and_parse_augmented_data(self):
        """
        Loads the full CSVs and parses them into original, adapted, force, and collision tensors using slices.
        """
        loaded_orig_data_list = []
        loaded_adap_data_list = []
        loaded_force_data_list = []
        loaded_collision_data_list = [] # NEW
        num_frames_loaded_per_file_list = []

        expected_total_cols = self.orig_block_size + self.adap_block_size + self.force_block_size + self.collision_block_size

        for i, path_str in enumerate(self.input_paths):
            try:
                full_data_np = np.genfromtxt(path_str, delimiter=',')
                if full_data_np.ndim == 1:
                    full_data_np = full_data_np.reshape(1, -1)

                num_frames, num_cols = full_data_np.shape
                if num_cols != expected_total_cols:
                    raise ValueError(
                        # CHANGED: Updated error message for clarity
                        f"File '{path_str}' has {num_cols} columns, but expected {expected_total_cols} "
                        f"for an augmented file (orig={self.orig_block_size}, "
                        f"adap={self.adap_block_size}, force={self.force_block_size}, "
                        f"collision={self.collision_block_size}). Please check the data generation script. Cannot proceed."
                    )

                current_start_time = self.start_times_per_file[i]
                start_frame = int(current_start_time * self.data_fps)
                if start_frame < num_frames:
                    full_data_np = full_data_np[start_frame:, :]
                else:
                    full_data_np = full_data_np[0:0, :]

                current_num_frames = full_data_np.shape[0]
                if current_num_frames < self.MIN_FRAMES_PER_FILE:
                    print(f"Warning: Augmented motion file '{path_str}' has {current_num_frames} frames. Padding.")
                    if current_num_frames > 0:
                        full_data_np = np.repeat(full_data_np[0:1, :], self.MIN_FRAMES_PER_FILE, axis=0)
                    else:
                        full_data_np = np.zeros((self.MIN_FRAMES_PER_FILE, expected_total_cols), dtype=np.float32)
                        full_data_np[:, self.quat_cols.start + 3] = 1.0
                        full_data_np[:, self.adap_data_cols.start + self.adap_quat_cols.start + 3] = 1.0
                        full_data_np[:, self.force_data_cols.start + self.force_id_cols.start] = -1.0

                    num_frames_loaded_per_file_list.append(self.MIN_FRAMES_PER_FILE)
                else:
                    num_frames_loaded_per_file_list.append(current_num_frames)

                orig_block = full_data_np[:, self.orig_data_cols]
                adap_block = full_data_np[:, self.adap_data_cols]
                force_block = full_data_np[:, self.force_data_cols]
                collision_block = full_data_np[:, self.collision_data_cols] # NEW

                loaded_orig_data_list.append(torch.from_numpy(orig_block).to(self.device).float())
                loaded_adap_data_list.append(torch.from_numpy(adap_block).to(self.device).float())
                loaded_force_data_list.append(torch.from_numpy(force_block).to(self.device).float())
                loaded_collision_data_list.append(torch.from_numpy(collision_block).to(self.device).float()) # NEW

            except Exception as e:
                raise RuntimeError(f"Failed to load augmented CSV data from '{path_str}': {e}") from e

        self.raw_data_tensor = torch.cat(loaded_orig_data_list, dim=0)
        self.adapted_data_tensor = torch.cat(loaded_adap_data_list, dim=0)
        self.force_data_tensor = torch.cat(loaded_force_data_list, dim=0)
        self.collision_data_tensor = torch.cat(loaded_collision_data_list, dim=0) # NEW

        self.num_frames_per_file = torch.tensor(num_frames_loaded_per_file_list, device=self.device, dtype=torch.long)
        self.start_frame_global_indices = torch.cat((
            torch.tensor([0], device=self.device, dtype=torch.long),
            torch.cumsum(self.num_frames_per_file[:-1], dim=0)
        ))
        self.num_total_frames = self.raw_data_tensor.shape[0]
        self.motion_length_per_file = self.num_frames_per_file * self.dt

    def precompute_adapted_keypoints(self):
        from softmimic_deploy.src.motion_lib.keypoint_extractor import KeypointExtractor
        if self.joint_config.num_joints != 29:
            raise ValueError(f"Unsupported joint configuration with {self.joint_config.num_joints} joints. This augmented motion library expects the 29-DOF G1 configuration.")
        keypoint_extractor = KeypointExtractor(robot_type="g1")
        
        batch_size = 1000
        num_batches = (self.num_total_frames + batch_size - 1) // batch_size
        all_keypoints = []
        all_keypoint_rotations = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, self.num_total_frames)
            batch_data = self.adapted_data_tensor[start_idx:end_idx]
            root_pos = batch_data[..., self.pos_cols]
            root_quat = batch_data[..., self.quat_cols]
            joint_pos = batch_data[..., self.joint_cols][:, self.reindex_mapping]
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
        self.adapted_keypoints = torch.cat(all_keypoints, dim=0).to(self.device)
        self.adapted_keypoint_rotations = torch.cat(all_keypoint_rotations, dim=0).to(self.device)

    def get_motion_state(self, motion_ids, motion_times, offset=None, future_frame_dt=0.02):
        """
        Overrides the parent method to include augmented and collision data in the returned dictionary.
        """
        original_motion_state = super().get_motion_state(
            motion_ids, motion_times, offset=offset, future_frame_dt=future_frame_dt
        )

        n_steps = self.n_future_steps + 1
        file_indices = motion_ids % self.num_files
        if (
            self._cached_step_indices is None
            or self._cached_step_indices.device != motion_times.device
            or self._cached_step_indices.dtype != motion_times.dtype
        ):
            self._cached_step_indices = torch.arange(n_steps, device=motion_times.device, dtype=motion_times.dtype)
        target_motion_times = motion_times.unsqueeze(1) + (self._cached_step_indices * future_frame_dt)
        target_frame_times = target_motion_times * self.data_fps
        local_frame_idx0_raw = target_frame_times.floor().long()
        lerp_alphas = (target_frame_times - local_frame_idx0_raw).unsqueeze(-1)
        
        max_frames_for_files = (self.num_frames_per_file[file_indices] - 1).unsqueeze(1)
        lower_clamp_bound = torch.zeros_like(max_frames_for_files, device=self.device, dtype=local_frame_idx0_raw.dtype)
        local_frame_idx0 = torch.clamp(local_frame_idx0_raw, lower_clamp_bound, max_frames_for_files)
        local_frame_idx1 = torch.clamp(local_frame_idx0 + 1, lower_clamp_bound, max_frames_for_files)
        
        base_global_indices = self.start_frame_global_indices[file_indices].unsqueeze(1)
        global_frame_idx0 = base_global_indices + local_frame_idx0
        global_frame_idx1 = base_global_indices + local_frame_idx1
        
        adap_data0 = self.adapted_data_tensor[global_frame_idx0]
        adap_data1 = self.adapted_data_tensor[global_frame_idx1]
        
        adap_root_pos0, adap_quat0, adap_dof_pos0 = adap_data0[..., self.adap_pos_cols], adap_data0[..., self.adap_quat_cols], adap_data0[..., self.adap_joint_cols]
        adap_root_pos1, adap_quat1, adap_dof_pos1 = adap_data1[..., self.adap_pos_cols], adap_data1[..., self.adap_quat_cols], adap_data1[..., self.adap_joint_cols]
        
        adapted_keypoints0 = self.adapted_keypoints[global_frame_idx0]
        adapted_keypoints1 = self.adapted_keypoints[global_frame_idx1]
        
        adapted_root_pos = torch.lerp(adap_root_pos0, adap_root_pos1, lerp_alphas)
        adapted_dof_pos = torch.lerp(adap_dof_pos0, adap_dof_pos1, lerp_alphas)
        adapted_keypoints = torch.lerp(adapted_keypoints0, adapted_keypoints1, lerp_alphas.unsqueeze(-1))
        
        dot = torch.sum(adap_quat0 * adap_quat1, dim=-1, keepdim=True)
        quat1_interp = torch.where(dot < 0, -adap_quat1, adap_quat1)
        adapted_interp_quat_xyzw = torch.lerp(adap_quat0, quat1_interp, lerp_alphas)
        norm_interp = torch.norm(adapted_interp_quat_xyzw, dim=-1, keepdim=True)
        adapted_root_rot = torch.where(norm_interp > 1e-8, adapted_interp_quat_xyzw / norm_interp, adap_quat0)

        adapted_keypoint_rotations0 = self.adapted_keypoint_rotations[global_frame_idx0]
        adapted_keypoint_rotations1 = self.adapted_keypoint_rotations[global_frame_idx1]
        dot = torch.sum(adapted_keypoint_rotations0 * adapted_keypoint_rotations1, dim=-1, keepdim=True)
        quat1_interp = torch.where(dot < 0, -adapted_keypoint_rotations1, adapted_keypoint_rotations1)
        adapted_interp_keypoint_rotations = torch.lerp(adapted_keypoint_rotations0, quat1_interp, lerp_alphas.unsqueeze(-1).repeat(1, 1, quat1_interp.shape[2], 1))
        norm_interp = torch.norm(adapted_interp_keypoint_rotations, dim=-1, keepdim=True)
        adapted_keypoint_rotations = torch.where(norm_interp > 1e-8, adapted_interp_keypoint_rotations / norm_interp, adapted_keypoint_rotations0)

        g_world = torch.tensor([[0.0, 0.0, -1.0]], device=self.device, dtype=adapted_root_pos.dtype)
        adapted_g_base = self._rotate_vector_by_quat_inverse_torch(g_world, adapted_root_rot)

        delta_t_data = self.dt
        adapted_root_vel_global = (adap_root_pos1 - adap_root_pos0) / delta_t_data
        adapted_dof_vel = (adap_dof_pos1 - adap_dof_pos0) / delta_t_data
        adapted_root_ang_vel_global = angular_velocity_from_quats(adap_quat0, adap_quat1, delta_t_data)
        adapted_root_vel = self._rotate_vector_by_quat_inverse_torch(adapted_root_vel_global, adapted_root_rot)
        adapted_root_ang_vel = self._rotate_vector_by_quat_inverse_torch(adapted_root_ang_vel_global, adapted_root_rot)

        if self.feet_contacts:
            adapted_foot_contacts = original_motion_state['foot_contacts']
        else:
            adapted_foot_contacts = torch.ones_like(original_motion_state['foot_contacts'])

        # Gather FORCE and TORQUE Data
        force_data0 = self.force_data_tensor[global_frame_idx0]
        force_link_id = force_data0[..., self.force_id_cols].long()
        force_vector = force_data0[..., self.force_vec_cols]
        torque_vector = force_data0[..., self.torque_vec_cols]
        stiffness_vector = force_data0[..., self.stiffness_vec_cols]
        
        # --- Gather COLLISION Data ---
        collision_data0 = self.collision_data_tensor[global_frame_idx0]
        ff_stiffnesses = collision_data0[..., self.ff_stiffness_cols] 
        ff_origin = collision_data0[..., self.ff_origin_cols]
        ff_setpoint_rot = collision_data0[..., self.ff_setpoint_rot_cols]
        ff_normal = collision_data0[..., self.ff_normal_cols]

        original_motion_state.update({
            'adapted_root_pos': adapted_root_pos,
            'adapted_root_rot': adapted_root_rot,
            'adapted_dof_pos': adapted_dof_pos[:, :, self.reindex_mapping],
            'adapted_foot_contacts': adapted_foot_contacts,
            'adapted_root_vel': adapted_root_vel,
            'adapted_root_ang_vel': adapted_root_ang_vel,
            'adapted_dof_vel': adapted_dof_vel[:, :, self.reindex_mapping],
            'adapted_root_vel_global': adapted_root_vel_global,
            'adapted_root_ang_vel_global': adapted_root_ang_vel_global,
            'adapted_keypoints': adapted_keypoints,
            'adapted_keypoint_rotations': adapted_keypoint_rotations,
            'adapted_gravity_vec': adapted_g_base,
            'force_link_id': force_link_id,
            'force_vector': force_vector,
            'torque_vector': torque_vector,
            'stiffness': stiffness_vector[..., 0],
            'rotational_stiffness': stiffness_vector[..., 1],
            'ff_stiffness': ff_stiffnesses[..., 0:1],
            'ff_rotational_stiffness': ff_stiffnesses[..., 1:2],
            'ff_origin': ff_origin,
            'ff_setpoint_rot': ff_setpoint_rot,
            'ff_normal': ff_normal,
        })
        
        return original_motion_state
