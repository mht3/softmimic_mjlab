"""CPU reference-buffer contract for a future ARDY -> retargeter -> tracker bridge.

Inputs are already retargeted and validated, NOT raw ARDY skeleton samples.
No physics feasibility is implied by passing these structural checks.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class ReferenceChunk:
    revision: int
    start_s: float
    fps: float
    joint_names: tuple[str, ...]
    # World XYZ + WXYZ quaternion + named joints, SI units, Z-up.
    qpos: np.ndarray


class ReferenceBuffer:
    def __init__(self, joint_names, max_seconds=4.):
        self.names = tuple(joint_names)
        if len(self.names) != 23 or len(set(self.names)) != 23:
            raise ValueError('Exactly 23 unique target joint names required')
        if not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('Positive buffer duration required')
        self.max_seconds = max_seconds
        self.revision = -1
        self.times = np.empty(0)
        self.poses = np.empty((0, 30))

    def change_prompt(self, revision):
        if not isinstance(revision, int) or revision <= self.revision:
            raise ValueError('Prompt revision must increase')
        self.revision = revision
        self.times = np.empty(0)
        self.poses = np.empty((0, 30))

    def append(self, chunk):
        if chunk.revision != self.revision:
            raise ValueError('Stale or unrequested prompt revision')
        if tuple(chunk.joint_names) != self.names:
            raise ValueError('Joint order differs; explicit retargeting is required')
        q = np.asarray(chunk.qpos, dtype=float)
        if q.ndim != 2 or q.shape[1] != 30 or len(q) < 2 or not np.isfinite(q).all():
            raise ValueError('Expected finite [T,30] poses with at least two frames')
        if not np.isfinite(chunk.fps) or chunk.fps <= 0 or not np.isfinite(chunk.start_s):
            raise ValueError('Invalid timing')
        if not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1., atol=1e-4):
            raise ValueError('Non-unit WXYZ root quaternion')
        times = chunk.start_s + np.arange(len(q))/chunk.fps
        if len(self.times) and (times[0] <= self.times[-1] or times[0]-self.times[-1] > 1/chunk.fps+1e-6):
            raise ValueError('Overlapping, reversed, or discontinuous chunks')
        all_times = np.concatenate((self.times, times))
        all_poses = np.concatenate((self.poses, q))
        keep = all_times >= all_times[-1]-self.max_seconds
        self.times, self.poses = all_times[keep], all_poses[keep]

    def sample(self, time_s):
        if not np.isfinite(time_s):
            raise ValueError('Non-finite sample time')
        if len(self.times) < 2 or time_s < self.times[0] or time_s > self.times[-1]:
            # The controller must choose its tested stop/hold behavior; no extrapolation.
            raise BufferError('Reference underrun: controller fallback required')
        result = np.array([np.interp(time_s, self.times, self.poses[:, i]) for i in range(30)])
        rotations = Rotation.from_quat(self.poses[:, [4, 5, 6, 3]])
        quat = Slerp(self.times, rotations)(time_s).as_quat()
        result[3:7] = quat[[3, 0, 1, 2]]
        return result
