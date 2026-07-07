from softmimic_deploy.src.sensors.base_sensor import BaseSensor

class ReferenceAngVelSensor(BaseSensor):

    dim = 1

    def __init__(self, interface, scale=1.0, wholeexo_sensor=None):
        super().__init__(interface, scale)

        assert wholeexo_sensor is not None, "wholeexo_sensor must be provided"
        self.wholeexo_sensor = wholeexo_sensor

    def get_data(self):
        # Get the reference gravity vector from the wholeexo sensor
        reference_root_ang_vel = self.wholeexo_sensor.root_ang_vel[2:3]
        return reference_root_ang_vel * self.scale