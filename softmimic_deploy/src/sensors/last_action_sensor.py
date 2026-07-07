from softmimic_deploy.src.sensors.base_sensor import BaseSensor

class LastActionSensor(BaseSensor):

    dim = 'nj'

    def __init__(self, interface, scale=1.0):
        super().__init__(interface, scale=1.0)

    def get_data(self, last_action):
        # TODO(gmargo): this is kind of a hack
        return last_action