# airexo_sensor.py
from softmimic_deploy.src.sensors.base_sensor import BaseSensor
import numpy as np
import cv2

def raw_airexo_data_to_joint_angles(raw_data):
    airexo_cal_angle = np.array([130, 80, 70, 350, 300, 20, 9, -40])
    airexo_angle = raw_data - airexo_cal_angle
    airexo_angle = airexo_angle * np.array([-1, -1, 1, -1, 1, -1, 1, 1])
    airexo_angle_rad = airexo_angle * np.pi / 180.0
    airexo_angle_rad[airexo_angle_rad > np.pi] -= 2 * np.pi
    airexo_angle_rad[airexo_angle_rad < -np.pi] += 2 * np.pi
    return airexo_angle_rad

class AirexoSensor(BaseSensor):
    '''
    A sensor interface for the Airexo exoskeleton handling both live hardware and 
    replay of recorded demonstrations. It processes raw encoder values into joint 
    angles for both arms of the exoskeleton and visualizes the teleoperated robot 
    when used with MuJoCo. It includes safety checks for initialization poses and 
    sudden movements.
    '''
    dim = 11

    def __init__(self, interface, scale=1.0, demo_recording_path=None):
        super().__init__(interface, scale)
        
        self.demo_recording_path = demo_recording_path
        self.is_initialized = False
        self.ctrl_step_counter = 0
        self.last_arm_angles = np.zeros(8)
        self.interface = interface

        # Initialize smoothing variables
        self.smoothing_factor = 0.0  # Higher = more smoothing
        self.last_pose = None

        self.is_mujoco = type(self.interface).__module__ == "softmimic_deploy.src.interfaces.mujoco_interface"

        if self.demo_recording_path is None:
            try:
                from easyrobot.encoder.angle import AngleEncoder
                self.left_arm = AngleEncoder(ids=[1, 2, 3, 4], port='/dev/ttyUSB1', baudrate=115200, streaming_freq=30, shm_name="encoder_left")
                self.left_arm.streaming()
                self.right_arm = AngleEncoder(ids=[1, 2, 3, 4], port='/dev/ttyUSB0', baudrate=115200, streaming_freq=30, shm_name="encoder_right")
                self.right_arm.streaming()
            except Exception as e:
                print("[AirexoSensor] Failed to initialize encoder reader")
                print(e)
                self.left_arm = None
                self.right_arm = None
        else:
            self.demo_index = 0
            data_dtype = [
                ('timestamp', 'f8'),
                ('color', 'u1', (480, 640, 3)),
                ('depth', 'u2', (480, 640)),
                ('encoder', 'f4', (8,))
            ]
            if "postprocessed_" in self.demo_recording_path:
                data_dtype.append(('pose', 'f8', (4, 4)))
                
            self.exo_recording = np.memmap(self.demo_recording_path, dtype=data_dtype, mode='r')
            
            if self.is_mujoco:
                self.visualization_manager = VisualizationManager()
                self.visualization_manager.setup_mujoco_renderer(self.interface.model)

    def safety_check(self, arm_angles):
        safe = True
        
        if not self.is_initialized:
            if np.all(np.abs(arm_angles) < 0.3):
                self.is_initialized = True
                print("[AirexoSensor] Initialized")
            else:
                print(arm_angles)
                safe = False
        
        if np.any(np.abs(arm_angles - self.last_arm_angles) > 0.4):
            if self.is_initialized:
                print("[AirexoSensor] Sudden movement detected, stopping")
            self.is_initialized = False
            safe = False
        
        self.last_arm_angles = arm_angles
        return safe
    
    def get_data(self, object_pose=None):
        self.update_command(object_pose)
        return self.get_command()

    def get_command(self):
        return self.upper_command

    def update_command(self, object_pose=None):
        if object_pose is not None and self.demo_recording_path is not None:
            object_position = object_pose[:3, 3]
            demo_initial_object_pose = self.exo_recording[0]['pose']
            demo_initial_object_position = demo_initial_object_pose[:3, 3]
            object_near_demo = np.linalg.norm(object_position - demo_initial_object_position) < 0.2

        if object_pose is not None and object_near_demo:
            self.ctrl_step_counter += 1
            if self.demo_recording_path is None:
                e = np.concatenate([self.left_arm.fetch_info(), self.right_arm.fetch_info()])
                airexo_angle_rad = raw_airexo_data_to_joint_angles(e)
                
                upper_command = np.zeros(11)
                if self.safety_check(airexo_angle_rad):
                    upper_command[0:8] = airexo_angle_rad
            else:
                frame_advanced = False
                while self.exo_recording[self.demo_index % len(self.exo_recording)]['timestamp'] < 0.02 * self.ctrl_step_counter:
                    self.demo_index += 1
                    frame_advanced = True
                    if self.demo_index % len(self.exo_recording) == 0:
                        self.demo_index = len(self.exo_recording) - 1
                        break
                
                frame_data = self.exo_recording[self.demo_index % len(self.exo_recording)]
                
                if self.is_mujoco:
                    self.visualization_manager.update(self, frame_data, frame_advanced)
                
                upper_command = np.zeros(11)
                upper_command[0:8] = frame_data['encoder']
        else:
            upper_command = np.zeros(11)
            self.ctrl_step_counter = 0
            self.demo_index = 0
            if self.is_mujoco:
                self.interface.objects["physics"]["initialized"] = False
        
        upper_command[-1] = self.interface.get_height_commands()
        
        # if self.is_mujoco:
        #     self.interface.update_ghost_target(upper_command[0:8])
        
        self.upper_command = upper_command * self.scale


# visualization_manager.py
import cv2
import numpy as np

class VisualizationManager:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.window_name = 'Demo Playback'
        self.last_pose = None
        self.smoothing_factor = 0.0  # Higher = more smoothing
        
        # Create window for visualization
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 160, 240)
        
    def setup_mujoco_renderer(self, model):
        import mujoco
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        
    def get_mujoco_camera_image(self, interface):
        import mujoco
        cam_id = mujoco.mj_name2id(
            interface.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            'head_camera'
        )
        self.renderer.update_scene(interface.data, camera=cam_id)
        return self.renderer.render()
    
    def smooth_pose(self, current_pose):
        """Apply exponential moving average smoothing to the pose matrix"""
        if self.last_pose is None:
            self.last_pose = current_pose
            return current_pose
        
        # Separately smooth rotation and translation
        smoothed_translation = (self.smoothing_factor * self.last_pose[:3, 3] + 
                              (1 - self.smoothing_factor) * current_pose[:3, 3])
        
        # For rotation, we use SLERP-like smoothing
        smoothed_rotation = (self.smoothing_factor * self.last_pose[:3, :3] + 
                           (1 - self.smoothing_factor) * current_pose[:3, :3])
        # Ensure the rotation matrix stays orthogonal
        U, _, Vh = np.linalg.svd(smoothed_rotation)
        smoothed_rotation = U @ Vh
        
        # Combine into final pose
        smoothed_pose = np.eye(4)
        smoothed_pose[:3, :3] = smoothed_rotation
        smoothed_pose[:3, 3] = smoothed_translation
        
        self.last_pose = smoothed_pose
        return smoothed_pose

    def draw_pose(self, image, pose):
        """Draw coordinate axes to visualize 6D pose"""
        # Camera matrix
        camera_matrix = np.array([
            [616.5984497070312, 0.0, 323.9175109863281],
            [0.0, 616.4833374023438, 239.70553588867188],
            [0.0, 0.0, 1.0]
        ])
        
        # Define coordinate axes points
        axis_points = np.float32([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Project points onto image plane
        dist_coeffs = np.zeros(4)
        axis_points_cam = np.dot(R, axis_points.T).T + t
        image_points, _ = cv2.projectPoints(axis_points_cam, np.zeros(3), np.zeros(3), 
                                          camera_matrix, dist_coeffs)
        
        # Draw coordinate axes
        origin = tuple(map(int, image_points[0].ravel()))
        x_point = tuple(map(int, image_points[1].ravel()))
        y_point = tuple(map(int, image_points[2].ravel()))
        z_point = tuple(map(int, image_points[3].ravel()))
        
        cv2.line(image, origin, x_point, (0, 0, 255), 3)  # X-axis in red
        cv2.line(image, origin, y_point, (0, 255, 0), 3)  # Y-axis in green
        cv2.line(image, origin, z_point, (255, 0, 0), 3)  # Z-axis in blue
        
        return image

    def create_visualization(self, rgb_img, mujoco_img, pose=None):
        """Create a visualization with RGB, Mujoco, and overlaid images"""
        mujoco_bgr = cv2.cvtColor(mujoco_img, cv2.COLOR_RGB2BGR)
        
        # Create mask for white background
        mujoco_hsv = cv2.cvtColor(mujoco_bgr, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.bitwise_not(cv2.inRange(mujoco_hsv, lower_white, upper_white))
        
        # Create overlay
        overlay_img = rgb_img.copy()
        overlay_img[white_mask > 0] = mujoco_bgr[white_mask > 0]
        
        # Draw pose if available
        if pose is not None:
            rgb_img = self.draw_pose(rgb_img.copy(), pose)
        
        # Stack views vertically and downsample
        combined_img = np.vstack((rgb_img, mujoco_bgr, overlay_img))[::4, ::4]
        
        return combined_img

    def update(self, sensor, frame_data, frame_advanced):
        """Update visualization with new frame data"""
        if 'pose' in frame_data.dtype.names:
            object_pose = np.array(frame_data['pose'])
            smoothed_pose = self.smooth_pose(object_pose)
            
            # Update marker and physics objects
            sensor.interface.set_object_pose(smoothed_pose, object_name="marker")
            sensor.interface.set_object_pose(smoothed_pose, object_name="physics", initialize_only=True)

            if frame_advanced:
                mujoco_img = self.get_mujoco_camera_image(sensor.interface)
                vis_img = self.create_visualization(
                    frame_data['color'],
                    mujoco_img,
                    smoothed_pose
                )
                cv2.imshow(self.window_name, vis_img)
                
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    sensor.interface.shutdown()
                    exit(0)
