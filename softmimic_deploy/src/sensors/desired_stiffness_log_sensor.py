from softmimic_deploy.src.sensors.base_sensor import BaseSensor
import numpy as np

class DesiredStiffnessLogSensor(BaseSensor):

    dim = 1

    def __init__(self, interface, scale=1.0):
        super().__init__(interface, scale=scale)

        # HARDCODED FOR NOW
        self.scale = 1.

    def get_data(self):

        # stiffness_commands = self.interface.get_stiffness_commands()
        # return np.log(np.array([stiffness_commands])) * self.scale
        return np.log(np.array([60.0])) * self.scale
        # return np.log(np.array([1000.0])) * self.scale
