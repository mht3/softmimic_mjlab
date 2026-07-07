from softmimic_deploy.src.sensors.base_sensor import BaseSensor
import numpy as np

class DesiredStiffnessInvSensor(BaseSensor):

    dim = 1

    def __init__(self, interface, scale=1.0):
        super().__init__(interface, scale=scale)

        # HARDCODED FOR NOW
        self.scale = 40.

    def get_data(self):
        # HARDCODED FOR NOW
        return 1. / np.array([100.0]) * self.scale