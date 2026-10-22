"""Read-only, per-mode diagnostics; these never modify rewards or resets."""

import torch


class ModeHealthMetrics:
    def __init__(self, modes, device):
        self.masks = {
            mode: torch.tensor([m == mode for m in modes], device=device)
            for mode in sorted(set(modes))
        }
        # Training staggers initial episode ages. Exclude each env's first
        # completion, whose full trajectory was not observed in this run.
        self.seen_first_done = torch.zeros(len(modes), dtype=torch.bool, device=device)
        self.completed = {m: torch.zeros((), device=device) for m in self.masks}
        self.timeouts = {m: torch.zeros((), device=device) for m in self.masks}
        self.reward_sum = {m: torch.zeros((), device=device) for m in self.masks}
        self.error_sum = {m: torch.zeros((), device=device) for m in self.masks}
        self.steps = 0

    def update(self, rewards, dones, timeouts, body_error):
        done = dones.bool()
        complete = done & self.seen_first_done
        for mode, mask in self.masks.items():
            self.reward_sum[mode] += rewards[mask].mean().detach()
            self.error_sum[mode] += body_error[mask].mean().detach()
            self.completed[mode] += (complete & mask).sum()
            self.timeouts[mode] += (complete & mask & timeouts.bool()).sum()
        self.seen_first_done |= done
        self.steps += 1

    def flush(self):
        result = {}
        if not self.steps:
            return result
        for mode in self.masks:
            prefix = f"Compliance/{mode}"
            result[prefix + '/reward_per_step'] = float(self.reward_sum[mode] / self.steps)
            result[prefix + '/body_tracking_error_m'] = float(self.error_sum[mode] / self.steps)
            completed = float(self.completed[mode])
            result[prefix + '/full_episodes_since_resume'] = completed
            if completed:
                result[prefix + '/timeout_fraction_since_resume'] = float(self.timeouts[mode]) / completed
            self.reward_sum[mode].zero_()
            self.error_sum[mode].zero_()
        self.steps = 0
        return result
