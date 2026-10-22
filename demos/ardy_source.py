"""Persistent ARDY G1 generator with prompt updates; emits unretargeted29DOF chunks.

Use a separate ARDY environment. No controller/hardware connections are made.
JSON lines on stdin: {"prompt":"A person waves.","chunks":1}
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch


class ArdySource:
    def __init__(self, model, converter):
        self.model = model
        self.converter = converter
        self.fps = float(model.motion_rep.fps)
        self.horizon = int(model.gen_horizon_len)
        patch = int(model.num_frames_per_token)
        self.history_limit = ((int(10*self.fps)-self.horizon)//patch)*patch
        if self.history_limit < patch or self.horizon % patch:
            raise ValueError('Unsupported ARDY window/token dimensions')
        self.history = None
        self.frame = 0
        self.prompt = None
        self.revision = -1

    @torch.inference_mode()
    def step(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('A nonempty prompt is required')
        if prompt != self.prompt:
            self.prompt = prompt
            self.revision += 1
        length = 0 if self.history is None else self.history.shape[1]
        samples = self.model.autoregressive_step(
            num_frames=length+self.horizon,
            num_denoising_steps=int(self.model.diffusion.num_base_steps),
            motion_mask=None, observed_motion=None,
            texts=[prompt], cfg_weight=(2., 2.),
            init_history_sequence=self.history,
        )
        if samples.shape[0] != 1 or samples.shape[1] != length+self.horizon or not bool(torch.isfinite(samples).all()):
            raise ValueError('ARDY returned invalid samples')
        # Decode the full window so boundary velocities/root transforms retain history.
        decoded = self.model.motion_rep.inverse(samples, is_normalized=True)
        qpos = np.asarray(self.converter.dict_to_qpos(decoded, root_quat_w_first=True))[0, length:]
        if qpos.shape != (self.horizon, 36) or not np.isfinite(qpos).all():
            raise ValueError('Expected finite G1 29DOF qpos from ARDY')
        if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1., atol=1e-3):
            raise ValueError('Invalid ARDY root quaternion')
        self.history = samples[:, -self.history_limit:].detach().clone()
        result = dict(qpos=qpos, fps=self.fps, start_s=self.frame/self.fps,
                      prompt=prompt, revision=self.revision,
                      quaternion_order='wxyz', up_axis='z', dof=29,
                      status='requires constrained23DOF retargeting and feasibility checks')
        self.frame += self.horizon
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model', default='g18')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose an empty output directory to preserve previous streams')
    from ardy.model import load_model
    from ardy.exports.mujoco import MujocoQposConverter
    model = load_model(args.model, device=args.device)
    model.eval()
    source = ArdySource(model, MujocoQposConverter(model.skeleton))
    args.output.mkdir(parents=True, exist_ok=True)
    number = 0
    for line in sys.stdin:
        request = json.loads(line)
        count = request.get('chunks', 1)
        if not isinstance(count, int) or not 1 <= count <= 32:
            raise ValueError('chunks must be an integer in [1,32]')
        for _ in range(count):
            result = source.step(request['prompt'])
            output = args.output / f'chunk_{number:06d}.npz'
            temporary = output.with_suffix('.partial')
            with temporary.open('wb') as handle:
                np.savez(handle, **result)
            temporary.replace(output)
            print(json.dumps({'file': str(output), 'revision': source.revision,
                              'status': result['status']}), flush=True)
            number += 1


if __name__ == '__main__':
    main()
