from softmimic_deploy.src.sensors.base_sensor import BaseSensor

class JointVelSensor(BaseSensor):

    dim = 'nj'

    def __init__(self, interface, scale=1.0):
        super().__init__(interface, scale)

    def get_data(self):
        joint_vel_hw = self.interface.get_joint_vel(joint_order="hw")
        joint_vel_policy = self.interface.robot_configuration.remap_joint_array(joint_vel_hw, from_order="hw", to_order="isaaclab")

        return joint_vel_policy * self.scale #self.interface.cfg["obs_scales"]["dof_vel"]