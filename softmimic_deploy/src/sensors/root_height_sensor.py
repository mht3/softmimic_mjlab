from softmimic_deploy.src.sensors.base_sensor import BaseSensor

class RootHeightSensor(BaseSensor):

    dim = 1

    def __init__(self, interface, scale=1.0):
        super().__init__(interface, scale)

    def get_data(self):
        return self.interface.get_root_height() * self.scale