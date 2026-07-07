from softmimic_deploy.src.sensors.base_sensor import BaseSensor
import numpy as np

class DesiredRotationalStiffnessInvSensor(BaseSensor):

    dim = 1

    def __init__(self, interface, scale=1.0):
        super().__init__(interface, scale=scale)

        # HARDCODED FOR NOW
        self.scale = 5.

    def get_data(self):
        # HARDCODED FOR NOW
        return 1. / np.array([10.0]) * self.scale