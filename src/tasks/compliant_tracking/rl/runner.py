import os
from typing import cast

import wandb
from rsl_rl.env.vec_env import VecEnv

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.tracking.mdp import MotionCommand


class CompliantMotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  """Tracking runner for the compliant task.

  Unlike ``MotionTrackingOnPolicyRunner`` this does not bundle the motion
  reference into an ONNX model: the compliant command concatenates the whole
  augmented dataset, which is deployed separately as a nominal motion NPZ.
  Only the plain ``policy.onnx`` used by the deploy stack is exported.
  """

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    self.export_policy_to_onnx(policy_path, "policy.onnx")
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    motion_term = cast(
      MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
    )
    metadata.update(
      {
        "anchor_body_name": motion_term.cfg.anchor_body_name,
        "body_names": list(motion_term.cfg.body_names),
      }
    )
    attach_metadata_to_onnx(os.path.join(policy_path, "policy.onnx"), metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + "policy.onnx", base_path=os.path.dirname(policy_path))
      if self.registry_name is not None:
        wandb.run.use_artifact(self.registry_name)  # type: ignore
        self.registry_name = None
