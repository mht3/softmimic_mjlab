import zmq
import numpy as np
import time

class PosePublisher:
    def __init__(self, port: int = 5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        
    def send_pose(self, pose: np.ndarray):
        """Send pose matrix directly using pyobj
        
        Args:
            pose: Numpy array containing pose matrix
        """
        self.socket.send_pyobj(pose)
        
    def close(self):
        self.socket.close()
        self.context.term()

class PoseSubscriber:
    def __init__(self, ip: str, port: int = 5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{ip}:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '')
        
        # Option 1: Set socket timeout to 100ms
        self.socket.setsockopt(zmq.RCVTIMEO, 100)
        
        # Option 2: Setup poller
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
    
    def receive_pose(self):
        """Receive pose matrix with 100ms timeout using socket timeout
        
        Returns:
            Tuple of (pose_matrix, success_flag)
        """
        try:
            pose = self.socket.recv_pyobj()
            return pose, True
        except zmq.error.Again:
            return None, False
            
    def receive_pose_with_poll(self):
        """Receive pose matrix with 100ms timeout using poller
        
        Returns:
            Tuple of (pose_matrix, success_flag)
        """
        if self.poller.poll(100):  # 100ms timeout
            pose = self.socket.recv_pyobj()
            return pose, True
        return None, False
        
    def close(self):
        self.socket.close()
        self.context.term()