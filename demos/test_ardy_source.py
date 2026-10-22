"""Contract tests with a fake generator; these do not validate ARDY weights."""
from types import SimpleNamespace
import numpy as np
import torch
from ardy_source import ArdySource


class FakeModel:
    gen_horizon_len = 8
    num_frames_per_token = 4
    diffusion = SimpleNamespace(num_base_steps=8)
    motion_rep = SimpleNamespace(fps=25, inverse=lambda x, **kw: x)

    def autoregressive_step(self, **kwargs):
        self.last = kwargs
        return torch.zeros(1, kwargs['num_frames'], 10)


class FakeConverter:
    def dict_to_qpos(self, samples, **kwargs):
        assert kwargs['root_quat_w_first']
        q = np.zeros((1, samples.shape[1], 36)); q[:, :, 3] = 1
        return q


def test_prompt_update_preserves_bounded_history_and_clock():
    model = FakeModel(); source = ArdySource(model, FakeConverter())
    first = source.step('stand')
    assert first['revision'] == 0 and first['start_s'] == 0
    second = source.step('wave')
    assert second['revision'] == 1 and second['start_s'] == 8/25
    assert model.last['init_history_sequence'].shape[1] == 8
    for _ in range(40): source.step('wave')
    assert source.revision == 1
    assert source.history.shape[1] <= source.history_limit
