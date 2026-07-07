
class BaseInterface:
    def __init__(self, robot_configuration, dt):
        self.robot_configuration = robot_configuration

    ######################
    # Main API Functions #
    ######################

    def initialize(self, fix_base=False, render=False):
        raise NotImplementedError()

    def get_joint_pos(self):
        # get joint position in URDF order
        raise NotImplementedError()
    
    def get_joint_vel(self):
        # get joint velocity in URDF order
        raise NotImplementedError()
    
    def get_joint_acc(self):
        # get joint velocity in URDF order
        raise NotImplementedError()
    
    def get_joint_torque(self):
        # get joint velocity in URDF order
        raise NotImplementedError()
    
    def get_gravity_vector(self):
        # get the gravity vector in the base frame
        raise NotImplementedError()
    
    def get_buttons(self):
        raise NotImplementedError()
    
    def apply_command(self, kp, kd, q, dq, tau):
        raise NotImplementedError()
