from softmimic_deploy.src.sensors.base_sensor import BaseSensor
import numpy as np

class DesiredRotationalStiffnessLogSensor(BaseSensor):

    dim = 1

    def __init__(self, interface, scale=1.0):
        super().__init__(interface, scale=scale)

        # HARDCODED FOR NOW
        self.scale = 1.

    def get_data(self):
        # HARDCODED FOR NOW
        return np.log(np.array([1.0])) * self.scale
