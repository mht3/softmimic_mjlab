"""Pre-step action diagnostics. Never smooth, clip, or rewrite rewards."""
import json
from pathlib import Path
import torch


class ActionDiagnostics:
    def __init__(self, directory, clip=None):
        self.path = Path(directory) / "action_outliers.jsonl"
        self.clip = clip
        self.records = 0
        self.record_peak = 0.
        self.steps = 0
        self.peak = 0.
        self.exceeding = 0
        self.samples = 0

    def observe(self, action, command):
        self.steps += 1
        if not bool(torch.isfinite(action).all()):
            with self.path.open("a") as file:
                file.write(json.dumps(dict(step=self.steps,error="nonfinite_actor_output",environments=(~torch.isfinite(action)).any(dim=-1).nonzero().flatten().cpu().tolist()))+"\n")
            raise RuntimeError("Non-finite actor output; stopping before physics/PPO. Inspect action_outliers.jsonl")
        maximum, env_id = action.abs().amax(dim=-1).max(dim=0)
        value = float(maximum)
        self.peak = max(self.peak, value)
        limit = self.clip if self.clip is not None else 20.
        self.exceeding += int((action.abs() > limit).sum())
        self.samples += action.numel()
        if value > limit and (self.records < 64 or value > self.record_peak):
            self.record_peak = max(self.record_peak, value)
            i = int(env_id)
            robot = command.robot.data
            row = dict(step=self.steps, env=i, raw_action_max=value,
                       action=action[i].detach().cpu().tolist(),
                       global_frame=int(command.time_steps[i]), motion_id=int(command.motion_ids[i]),
                       joint_position=robot.joint_pos[i].detach().cpu().tolist(),
                       joint_velocity=robot.joint_vel[i].detach().cpu().tolist(),
                       force_N=command.forcefield_forces_w[i].detach().cpu().tolist())
            with self.path.open("a") as file:
                file.write(json.dumps(row) + "\n")
            self.records += 1

    def flush(self):
        values = {"Stability/raw_action_abs_max": self.peak,
                  "Stability/action_bound_exceedance_fraction": self.exceeding / max(1, self.samples)}
        self.peak = 0.
        self.exceeding = self.samples = 0
        return values
