from softmimic_deploy.src.sensors.base_sensor import BaseSensor

class FutureReferenceRootHeightSensor(BaseSensor):

    dim = 1 * 20

    def __init__(self, interface, scale=1.0, wholeexo_sensor=None):
        super().__init__(interface, scale)

        assert wholeexo_sensor is not None, "wholeexo_sensor must be provided"
        self.wholeexo_sensor = wholeexo_sensor

    def get_data(self):
        # Get the reference gravity vector from the wholeexo sensor
        reference_root_height = self.wholeexo_sensor.future_root_pos[:, 2:3]
        return reference_root_height * self.scale