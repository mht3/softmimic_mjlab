from softmimic_deploy.src.sensors.base_sensor import BaseSensor

class JointPosSensor(BaseSensor):

    dim = 'nj'

    def __init__(self, interface, scale=1.0):
        super().__init__(interface, scale)

    def get_data(self):
        joint_pos_hw = self.interface.get_joint_pos("hw")
        joint_pos_policy = self.interface.robot_configuration.remap_joint_array(joint_pos_hw, from_order="hw", to_order="isaaclab")
        joint_pos_policy = joint_pos_policy - self.interface.robot_configuration.joint_default_positions(joint_order="isaaclab")

        return joint_pos_policy * self.scale