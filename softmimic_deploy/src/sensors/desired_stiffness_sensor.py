from softmimic_deploy.src.sensors.base_sensor import BaseSensor
import numpy as np

class DesiredStiffnessSensor(BaseSensor):

    dim = 1

    def __init__(self, interface, scale=1.0):
        super().__init__(interface, scale=scale)

        # HARDCODED FOR NOW
        # self.scale = 1./500.
        self.scale = 1./250.

    def get_data(self):
        # HARDCODED FOR NOW
        return np.array([200.0]) * self.scale