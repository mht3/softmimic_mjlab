from softmimic_deploy.src.sensors.base_sensor import BaseSensor
import numpy as np

class ExternalForceSensor(BaseSensor):

    dim = 3*39

    def __init__(self, interface, scale=1.0):
        super().__init__(interface, scale=scale)

        # HARDCODED FOR NOW
        self.scale = 1./50.

    def get_data(self):
        force_field_forces = np.zeros(3 * 39)
        real_body_id = self.interface.real_body_id
        active_force = self.interface.active_force
        if real_body_id is not None and real_body_id < 39:
            force_field_forces[3 * real_body_id:3 * real_body_id + 3] = active_force
        return force_field_forces * self.scale