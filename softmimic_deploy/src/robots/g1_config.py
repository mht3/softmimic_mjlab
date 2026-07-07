import numpy as np
import os
from tabulate import tabulate

class G1Config:
    
    num_joints = num_joints_hw = 29

    _joint_map_isaaclab = [
        "left_hip_pitch_joint",   #0
        "right_hip_pitch_joint",  #1
        "waist_yaw_joint",        #2
        "left_hip_roll_joint",    #3
        "right_hip_roll_joint",   #4
        "waist_roll_joint",       #5
        "left_hip_yaw_joint",     #6
        "right_hip_yaw_joint",    #7
        "waist_pitch_joint",      #8
        "left_knee_joint",        #9
        "right_knee_joint",       #10
        "left_shoulder_pitch_joint", #11
        "right_shoulder_pitch_joint", #12
        "left_ankle_pitch_joint",  #13
        "right_ankle_pitch_joint", #14
        "left_shoulder_roll_joint", #15
        "right_shoulder_roll_joint", #16
        "left_ankle_roll_joint",     #17
        "right_ankle_roll_joint",    #18
        "left_shoulder_yaw_joint",   #19
        "right_shoulder_yaw_joint",   #20
        "left_elbow_joint",          #21
        "right_elbow_joint",         #22
        "left_wrist_roll_joint",     #23
        "right_wrist_roll_joint",    #24
        "left_wrist_pitch_joint",    #25
        "right_wrist_pitch_joint",   #26
        "left_wrist_yaw_joint",      #27
        "right_wrist_yaw_joint",     #28
        ]
    _joint_map_hw = [
        "left_hip_pitch_joint",     # 0
        "left_hip_roll_joint",      # 1
        "left_hip_yaw_joint",       # 2
        "left_knee_joint",          # 3
        "left_ankle_pitch_joint",   # 4
        "left_ankle_roll_joint",    # 5
        "right_hip_pitch_joint",    # 6
        "right_hip_roll_joint",     # 7
        "right_hip_yaw_joint",      # 8
        "right_knee_joint",         # 9
        "right_ankle_pitch_joint",  # 10
        "right_ankle_roll_joint",   # 11
        "waist_yaw_joint",          # 12
        "waist_roll_joint",         # 13
        "waist_pitch_joint",        # 14
        "left_shoulder_pitch_joint", # 15
        "left_shoulder_roll_joint",  # 16
        "left_shoulder_yaw_joint",   # 17
        "left_elbow_joint",          # 18
        "left_wrist_roll_joint",     #19
        "left_wrist_pitch_joint",    #20
        "left_wrist_yaw_joint",      #21 
        "right_shoulder_pitch_joint",#22
        "right_shoulder_roll_joint", #23
        "right_shoulder_yaw_joint",  #24
        "right_elbow_joint",         #25
        "right_wrist_roll_joint",    #26
        "right_wrist_pitch_joint",   #27
        "right_wrist_yaw_joint",     #28      
        ]
    
    _joint_map_mujoco = [
        "left_hip_pitch_joint",     # 0
        "left_hip_roll_joint",      # 1
        "left_hip_yaw_joint",       # 2
        "left_knee_joint",          # 3
        "left_ankle_pitch_joint",   # 4
        "left_ankle_roll_joint",    # 5
        "right_hip_pitch_joint",    # 6
        "right_hip_roll_joint",     # 7
        "right_hip_yaw_joint",      # 8
        "right_knee_joint",         # 9
        "right_ankle_pitch_joint",  # 10
        "right_ankle_roll_joint",   # 11
        "waist_yaw_joint",          # 12
        "waist_roll_joint",         # 13
        "waist_pitch_joint",        # 14
        "left_shoulder_pitch_joint", # 15
        "left_shoulder_roll_joint",  # 16
        "left_shoulder_yaw_joint",   # 17
        "left_elbow_joint",          # 18
        "left_wrist_roll_joint",     #19
        "left_wrist_pitch_joint",    #20
        "left_wrist_yaw_joint",      #21
        "right_shoulder_pitch_joint",#22
        "right_shoulder_roll_joint", #23
        "right_shoulder_yaw_joint",  #24
        "right_elbow_joint",         #25
        "right_wrist_roll_joint",    #26
        "right_wrist_pitch_joint",   #27
        "right_wrist_yaw_joint",     #28
    ]

    _joint_default_positions ={
            "left_hip_pitch_joint": -0.20,
            "right_hip_pitch_joint": -0.20,
            "left_knee_joint": 0.42,
            "right_knee_joint": 0.42,
            "left_ankle_pitch_joint": -0.23,
            "right_ankle_pitch_joint": -0.23,
            "left_elbow_joint": 0.87,
            "right_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
        }
    for joint in _joint_map_hw:
        if joint not in _joint_default_positions:
            _joint_default_positions[joint]=0
    
    
    # _joint_stiffnesses = {
    #     joint_name: 40.0 for joint_name in _joint_map_isaaclab
    # }
    _joint_stiffnesses = {
        "left_hip_pitch_joint": 200,     # 0
        "left_hip_roll_joint": 150,      # 1
        "left_hip_yaw_joint": 150,       # 2
        "left_knee_joint": 200,          # 3
        "left_ankle_pitch_joint": 20,   # 4
        "left_ankle_roll_joint": 20,    # 5
        "right_hip_pitch_joint": 200,    # 6
        "right_hip_roll_joint": 150,     # 7
        "right_hip_yaw_joint": 150,      # 8
        "right_knee_joint": 200,         # 9
        "right_ankle_pitch_joint": 50,  # 10
        "right_ankle_roll_joint": 50,   # 11
        "waist_yaw_joint": 200,          # 12
        "waist_roll_joint": 200,         # 13
        "waist_pitch_joint": 200,        # 14
        "left_shoulder_pitch_joint": 40, # 15
        "left_shoulder_roll_joint":  40,  # 16
        "left_shoulder_yaw_joint":  40,   # 17
        "left_elbow_joint":  40,          # 18
        "left_wrist_roll_joint":  5,     #19
        "left_wrist_pitch_joint":  5,    #20
        "left_wrist_yaw_joint":  5,      #21 
        "right_shoulder_pitch_joint":  40,#22
        "right_shoulder_roll_joint":  40, #23
        "right_shoulder_yaw_joint":  40,  #24
        "right_elbow_joint":  40,         #25
        "right_wrist_roll_joint":  5,    #26
        "right_wrist_pitch_joint":  5,   #27
        "right_wrist_yaw_joint":  5,     #28  
    }
    # _joint_dampings = {
    #     joint_name: 1.0 for joint_name in _joint_map_isaaclab
    # }

    # The wrist gains are not aligned with isaac lab
    # in isaac lab they are =10
    _joint_dampings =  {
        "left_hip_pitch_joint": 5,     # 0
        "left_hip_roll_joint": 5,      # 1
        "left_hip_yaw_joint": 5,       # 2
        "left_knee_joint": 5,          # 3
        "left_ankle_pitch_joint": 2,   # 4
        "left_ankle_roll_joint": 2,    # 5
        "right_hip_pitch_joint": 5,    # 6
        "right_hip_roll_joint": 5,     # 7
        "right_hip_yaw_joint": 5,      # 8
        "right_knee_joint": 5,         # 9
        "right_ankle_pitch_joint": 2,  # 10
        "right_ankle_roll_joint": 2,   # 11
        "waist_yaw_joint": 5,          # 12
        "waist_roll_joint": 5,         # 13
        "waist_pitch_joint": 5,        # 14
        "left_shoulder_pitch_joint": 5, # 15
        "left_shoulder_roll_joint":  5,  # 16
        "left_shoulder_yaw_joint":  5,   # 17
        "left_elbow_joint":  5,          # 18
        "left_wrist_roll_joint":  1,     #19
        "left_wrist_pitch_joint":  1,    #20
        "left_wrist_yaw_joint":  1,      #21 
        "right_shoulder_pitch_joint":  5,#22
        "right_shoulder_roll_joint":  5, #23
        "right_shoulder_yaw_joint":  5,  #24
        "right_elbow_joint":  5,         #25
        "right_wrist_roll_joint":  1,    #26
        "right_wrist_pitch_joint":  1,   #27
        "right_wrist_yaw_joint":  1,     #28  
        }
    _joint_max_velocities = {
        joint_name: 100.0 for joint_name in _joint_map_isaaclab
    }

    _joint_saturated_torques = {
        joint_name: 120.0 for joint_name in _joint_map_isaaclab
    }

    _joint_lower_limits = {
        "left_hip_pitch_joint": -2.53,     # 0
        "left_hip_roll_joint": -0.52,      # 1
        "left_hip_yaw_joint": -2.75,       # 2
        "left_knee_joint": -0.08,          # 3
        "left_ankle_pitch_joint": -0.87,   # 4
        "left_ankle_roll_joint": -0.26,    # 5
        "right_hip_pitch_joint": -2.53,    # 6
        "right_hip_roll_joint": -2.96,     # 7
        "right_hip_yaw_joint": -2.75,      # 8
        "right_knee_joint": -0.087,         # 9
        "right_ankle_pitch_joint": -0.87,  # 10
        "right_ankle_roll_joint": -0.26,   # 11
        "waist_yaw_joint": -2.61,          # 12
        "waist_roll_joint": -0.52,         # 13
        "waist_pitch_joint": -0.52,        # 14
        "left_shoulder_pitch_joint": -3.08, # 15
        "left_shoulder_roll_joint": -1.58,  # 16
        "left_shoulder_yaw_joint": -2.61,   # 17
        "left_elbow_joint": -1.04,          # 18
        "left_wrist_roll_joint": -1.97,     #19
        "left_wrist_pitch_joint": -1.61,    #20
        "left_wrist_yaw_joint": -1.61,      #21 
        "right_shoulder_pitch_joint": -3.08,#22
        "right_shoulder_roll_joint": -2.25, #23
        "right_shoulder_yaw_joint": -2.61,  #24
        "right_elbow_joint": -1.04,         #25
        "right_wrist_roll_joint": -1.97,    #26
        "right_wrist_pitch_joint": -1.61,   #27
        "right_wrist_yaw_joint": -1.61,     #28  
    }

    _joint_upper_limits = {
        "left_hip_pitch_joint": 2.87,     # 0
        "left_hip_roll_joint": 2.96,      # 1
        "left_hip_yaw_joint": 2.75,       # 2
        "left_knee_joint": 2.87,          # 3
        "left_ankle_pitch_joint": 0.52,   # 4
        "left_ankle_roll_joint": 0.26,    # 5
        "right_hip_pitch_joint": 2.87,    # 6
        "right_hip_roll_joint": 0.52,     # 7
        "right_hip_yaw_joint": 2.75,      # 8
        "right_knee_joint": 2.87,         # 9
        "right_ankle_pitch_joint": 0.52,  # 10
        "right_ankle_roll_joint": 0.26,   # 11
        "waist_yaw_joint": 2.61,          # 12
        "waist_roll_joint": 0.52,         # 13
        "waist_pitch_joint": 0.52,        # 14
        "left_shoulder_pitch_joint": 2.67, # 15
        "left_shoulder_roll_joint": 2.25,  # 16
        "left_shoulder_yaw_joint": 2.61,   # 17
        "left_elbow_joint": 2.09,          # 18
        "left_wrist_roll_joint": 1.97,     #19
        "left_wrist_pitch_joint": 1.61,    #20
        "left_wrist_yaw_joint": 1.61,      #21 
        "right_shoulder_pitch_joint": 2.67,#22
        "right_shoulder_roll_joint": 1.58, #23
        "right_shoulder_yaw_joint": 2.61,  #24
        "right_elbow_joint": 2.09,         #25
        "right_wrist_roll_joint": 1.97,    #26
        "right_wrist_pitch_joint": 1.61,   #27
        "right_wrist_yaw_joint": 1.61,     #28  
    }

    _mujoco_xml_paths = {
        # "default": f"{os.path.dirname(os.path.realpath(__file__))}/../assets/g1/g1_29dof_w_ghost_boxfeet.xml",
        "default": f"{os.path.dirname(os.path.realpath(__file__))}/../assets/g1/g1_29dof_w_ghost.xml",
        # i think the fixed base mujoco file has sensors/ghost that mujoco warp does not like ...
        "fixed_base": f"{os.path.dirname(os.path.realpath(__file__))}/../assets/g1/g1_29dof_fixed_base.xml",
    }

    def __init__(self, dt):
        self.dt = dt
    
    def load_from_cfg(self, cfg):
        import re

        self._joint_stiffnesses = {k: None for k in self._joint_map_isaaclab}
        self._joint_dampings = {k: None for k in self._joint_map_isaaclab}

        for joint_names_expr, stiffness in cfg["control"]["stiffnesses"].items():
            # regex to match joint names
            for joint_name in self._joint_map_isaaclab:
                if re.match(joint_names_expr, joint_name):
                    self._joint_stiffnesses[joint_name] = stiffness
        for joint_names_expr, damping in cfg["control"]["dampings"].items():
            for joint_name in self._joint_map_isaaclab:
                if re.match(joint_names_expr, joint_name):
                    self._joint_dampings[joint_name] = damping

        headers = ["Joint Name", "Stiffness", "Damping"]
        table_data = [
            [joint, self._joint_stiffnesses[joint], self._joint_dampings[joint]]
            for joint in sorted(self._joint_stiffnesses.keys())
        ]
        print("[G1Config] Loaded joint stiffnesses and dampings:")
        print(tabulate(table_data, headers=headers, tablefmt="psql", floatfmt=".1f"))   

        # ensure all joints have stiffness and damping values
        for joint_name in self._joint_map_isaaclab:
            if self._joint_stiffnesses[joint_name] is None:
                raise ValueError(f"Missing stiffness value for joint {joint_name}")
            if self._joint_dampings[joint_name] is None:
                raise ValueError(f"Missing damping value for joint {joint_name}")

    def _joint_param(self, dict, joint_order):
        # warning, it might be slow to do this every time
        if joint_order == "isaaclab":
            return np.array([dict[joint] for joint in self._joint_map_isaaclab])
        elif joint_order == "hw":
            return np.array([dict[joint] for joint in self._joint_map_hw])
        else:
            raise ValueError("Unknown joint_order")
        
    def joint_names(self, joint_order):
        if joint_order == "isaaclab":
            return self._joint_map_isaaclab
        elif joint_order == "hw":
            return self._joint_map_hw
        elif joint_order == "mujoco":
            return self._joint_map_mujoco
        else:
            raise ValueError("Unknown joint_order")

    def joint_default_positions(self, joint_order):
        return self._joint_param(self._joint_default_positions, joint_order)
    
    def joint_stiffnesses(self, joint_order):
        return self._joint_param(self._joint_stiffnesses, joint_order)
    
    def joint_dampings(self, joint_order):
        return self._joint_param(self._joint_dampings, joint_order)
    
    def joint_max_velocities(self, joint_order):
        return self._joint_param(self._joint_max_velocities, joint_order)
    
    def joint_saturated_torques(self, joint_order):
        return self._joint_param(self._joint_saturated_torques, joint_order)
    
    def joint_lower_limits(self, joint_order):
        return self._joint_param(self._joint_lower_limits, joint_order)
    
    def joint_upper_limits(self, joint_order):
        return self._joint_param(self._joint_upper_limits, joint_order)
    
    def compute_reindex_mapping(self, from_order, to_order):
        if from_order == to_order:
            return list(range(self.num_joints))
        else:
            return [self.joint_names(from_order).index(joint) for joint in self.joint_names(to_order)]
        
    def remap_joint_array(self, joint_array, from_order, to_order):
        assert len(joint_array) == self.num_joints, "joint_array must have 29 elements"
        mapping = self.compute_reindex_mapping(from_order, to_order)
        return np.array([joint_array[i] for i in mapping])
    

if __name__ == "__main__":

    g1_config = G1Config(0.005)

    # print all the remappings

    print("isaaclab -> hw")
    print(g1_config.compute_reindex_mapping("isaaclab",
                                                "hw"))
    print("hw -> isaaclab")
    print(g1_config.compute_reindex_mapping("hw",
                                                "isaaclab"))
    print("mujoco -> hw")
    print(g1_config.compute_reindex_mapping("mujoco",
                                                "hw"))
    print("hw -> mujoco")
    print(g1_config.compute_reindex_mapping("hw",
                                                "mujoco"))
    print("isaaclab -> mujoco")
    print(g1_config.compute_reindex_mapping("isaaclab",
                                                "mujoco"))
    print("mujoco -> isaaclab")
    print(g1_config.compute_reindex_mapping("mujoco",
                                                "isaaclab"))