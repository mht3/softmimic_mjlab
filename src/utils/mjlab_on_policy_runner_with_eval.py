"""Mjlab runner that forwards ``eval_env`` to rsl_rl's ``OnPolicyRunner``.

The vendored ``MjlabOnPolicyRunner`` strips optional actor/critic keys then calls
``OnPolicyRunner.__init__`` without an ``eval_env`` argument. This subclass keeps
the same stripping behavior and ONNX/save/load extensions from mjlab, but
initializes ``OnPolicyRunner`` with ``eval_env`` (required for separate eval
rollouts).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.runners import OnPolicyRunner

if TYPE_CHECKING:
  from rsl_rl.env import VecEnv


class MjlabOnPolicyRunnerWithEval(MjlabOnPolicyRunner):
  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    eval_env: VecEnv | None = None,
  ) -> None:
    for key in ("actor", "critic"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
    OnPolicyRunner.__init__(self, env, train_cfg, log_dir, device, eval_env=eval_env)
