import os
import h5py
import torch
import numpy as np

class H5Logger:
    def __init__(
        self,
        log_dir: str,
        num_joints: int,
        max_length:    int = 1000,
        filename:      str = "recorded_data.h5",
        # internal paths
        command_h5_path:      str = "data/demo_0/commands/reference_position",
        ref_pos_h5_path:      str = "data/demo_0/states/articulation/robot/joint_position",
        ref_vel_h5_path:      str = "data/demo_0/states/articulation/robot/joint_velocity",
        ref_acc_h5_path:      str = "data/demo_0/states/articulation/robot/joint_acceleration",
        prev_ref_pos_h5_path: str = "data/demo_0/initial_state/articulation/robot/joint_position",
        prev_ref_vel_h5_path: str = "data/demo_0/initial_state/articulation/robot/joint_velocity",
        target_vel_h5_path:   str = "data/demo_0/commands/target_velocity",
        wbc_obs_h5_path:      str = "data/demo_0/wbc_state/wbc_observation",
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)
        # remove old file
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        # open HDF5 for writing
        self.h5 = h5py.File(self.filepath, "w")

        # store shapes
        self.max_length = max_length
        self.num_joints = num_joints

        # dataset paths
        self.command_path      = command_h5_path
        self.ref_pos_path      = ref_pos_h5_path
        self.ref_vel_path      = ref_vel_h5_path
        self.ref_acc_path      = ref_acc_h5_path
        self.prev_pos_path     = prev_ref_pos_h5_path
        self.prev_vel_path     = prev_ref_vel_h5_path
        self.target_vel_path   = target_vel_h5_path
        self.wbc_obs_path      = wbc_obs_h5_path

        # helper to create each dataset
        def make_ds(path, elem_shape):
            full_shape = (max_length, *elem_shape)
            parts = path.strip("/").split("/")
            grp = self.h5
            for p in parts[:-1]:
                grp = grp.require_group(p)
            return grp.create_dataset(parts[-1], shape=full_shape, dtype="f8")
        

        def make_single_ds(path, elem_shape):
            full_shape = (1, *elem_shape)
            parts = path.strip("/").split("/")
            grp = self.h5
            for p in parts[:-1]:
                grp = grp.require_group(p)
            return grp.create_dataset(parts[-1], shape=full_shape, dtype="f8")

        # create all five
        self.ds_cmd    = make_ds(self.command_path,      (self.num_joints,))
        self.ds_rpos   = make_ds(self.ref_pos_path,      (self.num_joints,))
        self.ds_rvel   = make_ds(self.ref_vel_path,      (self.num_joints,))
        self.ds_racc   = make_ds(self.ref_acc_path,      (self.num_joints,))
        self.ds_prev_p = make_single_ds(self.prev_pos_path,     (self.num_joints,))
        self.ds_prev_v = make_single_ds(self.prev_vel_path,     (self.num_joints,))
        self.ds_target_v = make_ds(self.target_vel_path,   (3,))
        # self.ds_wbc_obs = make_ds(self.wbc_obs_path,      ((self.num_joints * 3 + 6) * 3,))  # +6 for the WBC observation
        self.ds_wbc_obs = make_ds(self.wbc_obs_path,      ((self.num_joints * 3 + 9) * 3,))  # +6 for the WBC observation

        self.index = 0

    def log(self, cmd, jpos, jvel, jacc, target_vel, wbc_obs=None):
        idx = self.index
        if idx >= self.max_length:
            # buffer full
            return

        # write current step
        self.ds_cmd   [idx, ...] = cmd
        self.ds_rpos  [idx, ...] = jpos
        self.ds_rvel  [idx, ...] = jvel
        self.ds_racc  [idx, ...] = jacc
        self.ds_target_v[idx, ...] = target_vel
        if  wbc_obs is not None:
            self.ds_wbc_obs[idx, ...] = wbc_obs

        # write "previous" step
        if idx == 0:
            # initial state: same as current
            self.ds_prev_p[0, ...] = jpos
            self.ds_prev_v[0, ...] = jvel
        # else:
        #     # copy last recorded ref_pos/ref_vel
        #     self.ds_prev_p[idx, ...] = self.ds_rpos[idx - 1, ...]
        #     self.ds_prev_v[idx, ...] = self.ds_rvel[idx - 1, ...]

        self.index += 1

    def close(self):
        """Flush and close the HDF5 file when done."""
        self.h5.close()
