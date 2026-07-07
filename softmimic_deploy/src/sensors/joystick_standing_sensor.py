from softmimic_deploy.src.sensors.base_sensor import BaseSensor
import numpy as np

class JoystickStandingSensor(BaseSensor):

    dim = 1

    def __init__(self, interface, scale=1.0, threshold=0.1):
        super().__init__(interface, scale=scale)

        self.velocity_cmd_min_limits = np.array([
            self.interface.cfg["command_ranges"]["lin_vel_x"][0],
            self.interface.cfg["command_ranges"]["lin_vel_y"][0],
            self.interface.cfg["command_ranges"]["ang_vel_yaw"][0]
        ])

        self.velocity_cmd_max_limits = np.array([
            self.interface.cfg["command_ranges"]["lin_vel_x"][1],
            self.interface.cfg["command_ranges"]["lin_vel_y"][1],
            self.interface.cfg["command_ranges"]["ang_vel_yaw"][1]
        ])

        self.threshold = threshold

    def get_data(self):
        base_velocity_commands = self.interface.get_joystick_commands()

        base_velocity_commands_scaled = np.array([
            np.clip(base_velocity_commands[0] * self.interface.cfg["obs_scales"]["lin_vel"], 
                    self.velocity_cmd_min_limits[0], 
                    self.velocity_cmd_max_limits[0]),  # Scale and clip linear X velocity

            np.clip(base_velocity_commands[1] * self.interface.cfg["obs_scales"]["lin_vel"], 
                    self.velocity_cmd_min_limits[1], 
                    self.velocity_cmd_max_limits[1]),  # Scale and clip linear Y velocity

            np.clip(base_velocity_commands[2] * self.interface.cfg["obs_scales"]["ang_vel"], 
                    self.velocity_cmd_min_limits[2], 
                    self.velocity_cmd_max_limits[2])  # Scale and clip angular velocity (Yaw)
        ])

        return np.array([np.linalg.norm(base_velocity_commands_scaled) < self.threshold]) * self.scale