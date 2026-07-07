from softmimic_deploy.src.sensors.base_sensor import BaseSensor
import numpy as np
from scipy.spatial.transform import Rotation

class ObjectPoseSensor(BaseSensor):

    dim = 7
    
    def __init__(self, interface, scale=1.0, dummy=True, listen_zmq=False):
        super().__init__(interface, scale)
        self.dummy = dummy
        self.listen_zmq = listen_zmq

        self.pose = np.eye(4)
        self.position = np.zeros(3)
        self.quaternion = np.zeros(4)

        # if self.listen_zmq:
        #     from softmimic_deploy.src.utils.zmq_utils import PoseSubscriber
        #     self.pose_subscriber = PoseSubscriber(ip="localhost", port=5555)

        self.pose = np.eye(4)

        # if not self.dummy:
        #     from softmimic_deploy.src.utils.zmq_utils import PoseSubscriber
        #     self.pose_subscriber = PoseSubscriber(ip="localhost", port=5555)

    def get_data(self):
        if self.listen_zmq:
            msg, success = self.pose_subscriber.receive_pose()
            if success:
                self.pose = msg

                self.position = self.pose[:3, 3]
                rotation = Rotation.from_matrix(self.pose[:3, :3])
                self.quaternion = rotation.as_quat()

        if self.dummy:
            return np.zeros(7) * self.scale
        else:
            return np.concatenate((self.position, self.quaternion)) * self.scale

    def get_pose(self):
        return self.pose
