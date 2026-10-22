import numpy as np
import pytest
from stream import ReferenceBuffer, ReferenceChunk

NAMES = tuple(f'joint{i}' for i in range(23))


def poses(n=8):
    q = np.zeros((n, 30)); q[:, 3] = 1
    return q


def test_ardy25_to_tracker50_and_quaternion_sign():
    buffer = ReferenceBuffer(NAMES); buffer.change_prompt(0)
    q = poses(); q[:, 0] = np.arange(8)/25; q[4:, 3] = -1
    buffer.append(ReferenceChunk(0, 0., 25., NAMES, q))
    for t in np.arange(0, .28, .02):
        sample = buffer.sample(t)
        assert sample[0] == pytest.approx(t)
        assert abs(sample[3]) == pytest.approx(1.)


def test_prompt_revision_discards_stale_work_and_requires_refill():
    buffer = ReferenceBuffer(NAMES); buffer.change_prompt(1)
    buffer.append(ReferenceChunk(1, 0., 25., NAMES, poses()))
    buffer.change_prompt(2)
    with pytest.raises(ValueError, match='Stale'):
        buffer.append(ReferenceChunk(1, .32, 25., NAMES, poses()))
    with pytest.raises(BufferError): buffer.sample(.1)


def test_missing_future_and_wrong_joint_order_fail():
    buffer = ReferenceBuffer(NAMES); buffer.change_prompt(0)
    with pytest.raises(ValueError, match='Joint order'):
        buffer.append(ReferenceChunk(0, 0., 25., NAMES[::-1], poses()))
    buffer.append(ReferenceChunk(0, 0., 25., NAMES, poses()))
    with pytest.raises(BufferError): buffer.sample(.3)
    with pytest.raises(ValueError, match='discontinuous'):
        buffer.append(ReferenceChunk(0, 1., 25., NAMES, poses()))
