from softmimic_deploy.src.sensors.base_sensor import BaseSensor
import numpy as np

class ExternalTorqueSensor(BaseSensor):

    dim = 3*39

    def __init__(self, interface, scale=1.0):
        super().__init__(interface, scale=scale)

        # HARDCODED FOR NOW
        self.scale = 1./25.

    def get_data(self):
        torque_field_torques = np.zeros(3 * 39)
        real_body_id = self.interface.real_body_id
        active_torque = self.interface.active_torque
        if real_body_id is not None and real_body_id < 39:
            torque_field_torques[3 * real_body_id:3 * real_body_id + 3] = active_torque
        return torque_field_torques * self.scale