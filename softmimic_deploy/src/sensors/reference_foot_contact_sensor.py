from softmimic_deploy.src.sensors.base_sensor import BaseSensor

class ReferenceFootContactSensor(BaseSensor):

    dim = 2

    def __init__(self, interface, scale=1.0, wholeexo_sensor=None):
        super().__init__(interface, scale)

        assert wholeexo_sensor is not None, "wholeexo_sensor must be provided"
        self.wholeexo_sensor = wholeexo_sensor

    def get_data(self):
        # Get the reference gravity vector from the wholeexo sensor
        reference_foot_contact = self.wholeexo_sensor.foot_contacts
        return reference_foot_contact * self.scale