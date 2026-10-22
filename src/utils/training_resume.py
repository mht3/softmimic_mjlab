"""Restore scheduler state and interpret max_iterations as a total run budget."""

import math


def prepare_training_iterations(runner, max_iterations: int, *, resumed: bool) -> int:
    if resumed:
        # RSL saves after the update with a zero-based iteration index.
        next_iteration = runner.current_learning_iteration + 1
        if next_iteration >= max_iterations:
            raise ValueError(
                f"Checkpoint already completed {next_iteration} updates; "
                f"requested total is {max_iterations}"
            )
        # PPO.load restores Adam but not PPO.learning_rate. The adaptive KL
        # scheduler would otherwise overwrite the restored LR with the CLI LR.
        rates = {float(group["lr"]) for group in runner.alg.optimizer.param_groups}
        if len(rates) != 1 or not all(math.isfinite(x) and x > 0 for x in rates):
            raise ValueError(f"Cannot restore PPO scheduler from optimizer rates: {rates}")
        runner.alg.learning_rate = rates.pop()
        runner.current_learning_iteration = next_iteration
    remaining = max_iterations - runner.current_learning_iteration
    if remaining <= 0:
        raise ValueError("No training iterations remain")
    return remaining
