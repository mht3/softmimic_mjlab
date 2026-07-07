import numpy as np
import time
import math
import select
import threading

from softmimic_deploy.src.interfaces.lcm_types.body_control_data_lcmt import body_control_data_lcmt
from softmimic_deploy.src.interfaces.lcm_types.rc_command_lcmt import rc_command_lcmt
from softmimic_deploy.src.interfaces.lcm_types.state_estimator_lcmt import state_estimator_lcmt
from softmimic_deploy.src.interfaces.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt
from softmimic_deploy.src.interfaces.lcm_types.arm_action_lcmt import arm_action_lcmt

from softmimic_deploy.src.interfaces.base_interface import BaseInterface
import lcm
class G1LCMInterface(BaseInterface):
    def __init__(self, robot_configuration, cfg, task_name):
        self.robot_configuration = robot_configuration
        self.cfg = cfg
        self.lc = lcm.LCM()

        self.height_commands = 0.8
        self.frequency_commands = 1.1
        self.pitch_commands = 0.0
        self.stiffness_commands = 180.0
        self.velocity_commands=np.zeros((3,))
        
        self.init_time = time.time()
        self.imu_time = time.time()
        self.counter = 0
        self.received_first_bodydata = False

    ######################
    # Main API Functions #
    ######################

    def initialize(self, fix_base=False, render=False):
        self.initialize_state_variables()
        self.initialize_lcm()
        self.damping_mode_active = False
        self.zero_command()
        self.spin()

    
    def apply_command(self, kp, kd, q, dq, tau, torque_mode=False):
        msg = pd_tau_targets_lcmt()
        msg.q_des =q
        msg.tau_ff = tau

        msg_arm = arm_action_lcmt()
        msg_arm.act = q[15:]

        # Continuously check if R2 is pressed
        if self.key_state[4][1] == 0 or self.damping_mode_active == True:  # If R2 was pressed or in damping mode
            # self.cmd.crc = self.crc.Crc(self.cmd)
            # self.pub.Write(self.cmd) 
            pass
        else:
            print("R2 button pressed. Switching to damping mode.")
            self.damping_mode()
            return      
        self.lc.publish("pd_plustau_targets", msg.encode())
        self.lc.publish("arm_action", msg_arm.encode())
        

    def repeat_last_command(self):
        # Continuously check if R2 is pressed
        if self.key_state[4][1] == 0 or self.damping_mode_active == True:  # If R2 was pressed or in damping mode
            pass
        else:
            print("R2 button pressed. Switching to damping mode.")
            self.damping_mode()
            return  
    
    def get_joint_pos(self, joint_order):
        if joint_order == "hw":
            return self.joint_pos
        elif joint_order == "isaaclab":
            raise ValueError("Conversion from hw to isaaclab must be implemented in deploy_policy_interface, not hardware_interface")
        else:
            raise ValueError("Invalid joint order argument")
    
    def get_joint_vel(self, joint_order):
        if joint_order == "hw":
            return self.joint_vel
        elif joint_order == "isaaclab":
            raise ValueError("Conversion from hw to isaaclab must be implemented in deploy_policy_interface, not hardware_interface")
        else:
            raise ValueError("Invalid joint order argument")
        
    def get_joint_acc(self, joint_order):
        if joint_order == "hw":
            return np.zeros((29,))
        elif joint_order == "isaaclab":
            raise ValueError("Conversion from hw to isaaclab must be implemented in deploy_policy_interface, not hardware_interface")
        else:
            raise ValueError("Invalid joint order argument")
        
    def get_joint_torque(self, joint_order):
        if joint_order == "hw":
            return np.zeros(29,)
        elif joint_order == "isaaclab":
            raise ValueError("Conversion from hw to isaaclab must be implemented in deploy_policy_interface, not hardware_interface")
        else:
            raise ValueError("Invalid joint order argument")
        
    def get_joystick_commands(self):
        return self.velocity_commands

    def get_height_commands(self):
        return self.height_commands
    
    def get_pitch_commands(self):
        return self.pitch_commands
    
    def get_stiffness_commands(self):
        return self.stiffness_commands

    def get_frequency_command(self):
        return self.frequency_commands
    
    def get_gravity_vector(self):
        # get the gravity vector in the base frame
        return self.gravity_vec
    
    def get_root_ang_vel(self):
        return self.omegaBody
    
    def get_root_lin_vel(self):
        raise NotImplementedError("Root linear velocity is not available in LCM interface. Use a deployable observation space.")
    
    def get_foot_contact_states(self):
        raise NotImplementedError("Foot contact states are not available in LCM interface. Use a deployable observation space.")
    
    def get_root_height(self):
        raise NotImplementedError("Root height is not available in LCM interface. Use a deployable observation space.")
       
    def get_obs_timestamp(self):
        return 0
    
    def get_pub_timestamp(self):
        return 0
    
    def get_write_timestamp(self):
        return 0

    # ############
    # # Handlers #
    # ############
    def _bodydata_cb(self, channel, data):
        if not self.received_first_bodydata:
            self.received_first_bodydata = True
            print(f"First body data: {time.time() - self.init_time}")

        msg = body_control_data_lcmt.decode(data)
        self.joint_pos = np.array(msg.q)
        self.joint_vel = np.array(msg.qd)



    def _imu_cb(self, channel, data):
        msg = state_estimator_lcmt.decode(data)
        self.rpy = msg.rpy
        R = self.get_rotation_matrix_from_rpy(self.rpy)
        self.gravity_vec = np.dot(R.T, np.array([0, 0, -1]))  # Gravity vector in the world frame
        self.omegaBody = np.array(msg.omegaBody)
        # print(self.gravity_vec)
        self.counter+=1
        if self.counter%1000==0:
            print('frequency of imu state = ', 1000/(time.time()-self.imu_time))
            self.counter=0
            self.imu_time=time.time()

    def _rc_command_cb(self, channel, data):

        msg = rc_command_lcmt.decode(data)
        self.velocity_commands = np.array([msg.right_stick[1], msg.right_stick[0], -1*msg.left_stick[0]])
        self.key_state = [
            ["R1", 0], ["L1", msg.left_upper_switch], ["start", 0], ["select", 0],
            ["R2",  msg.right_lower_right_switch], ["L2", 0], ["F1", 0], ["F2", 0],
            ["A", msg.right_lower_right_switch], ["B", 0], ["X", msg.left_lower_left_switch], ["Y", msg.right_upper_switch],
            ["up", 0], ["right", 0], ["down", 0], ["left", 0],
        ]

        self.height_commands = 0.75 #min(max(0.75 + 0.35 * msg.left_stick[1], 0.4), 0.9)
        self.pitch_commands = 0.0 #min(max(0.0 + 2.0 * msg.right_stick[0], -1.5), 2.0)
        # self.frequency_commands = 1.2 + 0.3 * msg.left_stick[1]
        min_stiffness, max_stiffness = 20.0, 800.0
        self.log_stiffness_commands = min(max(np.log(min_stiffness) + (np.log(max_stiffness) - np.log(min_stiffness)) * (msg.left_stick[1] + 1) / 2, np.log(min_stiffness)), np.log(max_stiffness))
        self.stiffness_commands = np.exp(self.log_stiffness_commands)

    def initialize_state_variables(self):
        self.joint_pos = None
        self.joint_vel = np.zeros(self.robot_configuration.num_joints)
        self.joint_torque = np.zeros(self.robot_configuration.num_joints)
        self.rpy = None
        self.gravity_vec = None
        self.omegaBody = None
        self.key_state = [
            ["R1", 0], ["L1", 0], ["start", 0], ["select", 0],
            ["R2", 0], ["L2", 0], ["F1", 0], ["F2", 0],
            ["A", 0], ["B", 0], ["X", 0], ["Y", 0],
            ["up", 0], ["right", 0], ["down", 0], ["left", 0],
        ]


    def get_buttons(self):
        return self.key_state

    def zero_command(self):
        msg = pd_tau_targets_lcmt()
        msg.q_des =np.zeros((29,))
        msg.tau_ff = np.zeros((29,))
        self.lc.publish("pd_plustau_targets", msg.encode())

    def initialize_lcm(self):
        self.imu_subscription = self.lc.subscribe("state_estimator_data", self._imu_cb)
        self.bodydata_state_subscription = self.lc.subscribe("body_control_data", self._bodydata_cb)
        self.rc_command_subscription = self.lc.subscribe("rc_command", self._rc_command_cb)
        print('initialized lcm')
        
    def damping_mode(self):
        """Switch all joints to damping mode."""
        print("Switching to damping mode.")
        self.damping_mode_active = True
        msg = pd_tau_targets_lcmt()
        msg.q_des =np.zeros((29,))
        msg.tau_ff = np.zeros((29,))
        self.lc.publish("pd_plustau_targets", msg.encode())
   
    def get_rotation_matrix_from_rpy(self, rpy):
        """
        Get rotation matrix from the given quaternion.
        Args:
            q (np.array[float[4]]): quaternion [w,x,y,z]
        Returns:
            np.array[float[3,3]]: rotation matrix.
        """
        r, p, y = rpy
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(r), -math.sin(r)],
                        [0, math.sin(r), math.cos(r)]
                        ])

        R_y = np.array([[math.cos(p), 0, math.sin(p)],
                        [0, 1, 0],
                        [-math.sin(p), 0, math.cos(p)]
                        ])

        R_z = np.array([[math.cos(y), -math.sin(y), 0],
                        [math.sin(y), math.cos(y), 0],
                        [0, 0, 1]
                        ])

        rot = np.dot(R_z, np.dot(R_y, R_x))
        return rot

    ######################
    # Run lcm functions #
    ######################

    def poll(self, cb=None):
        try:
            while True:
                timeout = 0.01
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                # nrfds, nwfds, nefds = select.select([self.new_lcm.fileno()], [], [], timeout)
                if rfds:
                    self.lc.handle()
                    # print(f'Freq {1. / (time.time() - t)} Hz'); t = time.time()
                else:
                    continue
        except KeyboardInterrupt:
            pass

    def spin(self):
        self.run_thread = threading.Thread(target=self.poll, daemon=False)
        self.run_thread.start()

    def shutdown(self):
        print('exiting lcm interface')
        self.run_thread.join()
        exit(0)
        

if __name__ == "__main__":
    from imp_deploy.src.robots.g1_config import G1Config
    robot_config = G1Config()

    interface = G1LCMInterface(robot_config, {}, "test")
    interface.initialize()

    interface.poll()
