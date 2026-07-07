from softmimic_deploy.src.sensors.base_sensor import BaseSensor

class FootContactSensor(BaseSensor):

    dim = 2

    def __init__(self, interface, scale=1.0):
        super().__init__(interface, scale)

    def get_data(self):
        return self.interface.get_foot_contact_states() * self.scale