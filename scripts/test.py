"""Script to test trained policies and log episode statistics.

mjlab analog of the SoftMimic ``scripts/rsl_rl/test.py``: rolls out a saved
checkpoint (or zero actions) headlessly, tracks per-episode returns/lengths,
and optionally records a video.

Usage:
  python scripts/test.py Unitree-G1-23Dof-Compliant-Tracking-No-State-Estimation \
    --motion_file=src/assets/compliant_motions/g1_23dof/stand \
    --checkpoint_file=logs/rsl_rl/g1_23dof_compliant_tracking/<run>/model_30000.pt \
    --num_envs 64 --duration 30
"""

import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TestConfig:
  checkpoint_file: str | None = None
  """Checkpoint to load. If None, zero actions are used."""
  motion_file: str | None = None
  """Motion NPZ file or directory (required for tracking tasks)."""
  num_envs: int = 64
  duration: float = 30.0
  """Test duration in seconds (ignored when num_episodes is set)."""
  num_episodes: int | None = None
  """Stop after this many completed episodes instead of a fixed duration."""
  use_zero_action: bool = False
  """Force zero actions even if a checkpoint is given."""
  video: bool = False
  video_length: int = 500
  device: str | None = None


class EvaluationStats:
  """Tracks per-episode returns and lengths."""

  def __init__(self):
    self.episode_returns: list[float] = []
    self.episode_lengths: list[int] = []

  def add_episode(self, episode_return: float, episode_length: int):
    self.episode_returns.append(episode_return)
    self.episode_lengths.append(episode_length)

  @property
  def episode_count(self) -> int:
    return len(self.episode_returns)

  def summary(self) -> dict:
    if not self.episode_returns:
      return {"episode_count": 0}
    returns = np.array(self.episode_returns)
    lengths = np.array(self.episode_lengths)
    return {
      "mean_return": float(returns.mean()),
      "std_return": float(returns.std()),
      "min_return": float(returns.min()),
      "max_return": float(returns.max()),
      "mean_length": float(lengths.mean()),
      "std_length": float(lengths.std()),
      "episode_count": len(returns),
    }


def run_test(task_id: str, cfg: TestConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)

  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.observations["actor"].enable_corruption = False

  # Handle tracking tasks that require a motion file.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )
  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("Tracking tasks require --motion-file path/to/motion(.npz|dir)")
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    env_cfg.commands["motion"].motion_file = str(motion_path)

  log_dir = (
    Path("logs")
    / "test_runs"
    / task_id
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  )
  log_dir.mkdir(parents=True, exist_ok=True)
  print(f"[INFO] Logging results to: {log_dir}")

  render_mode = "rgb_array" if cfg.video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  if cfg.video:
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording video.")
  env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # --- Load policy (or fall back to zero actions) ---
  policy = None
  if not cfg.use_zero_action and cfg.checkpoint_file is not None:
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env_wrapped, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    print("[INFO] Policy loaded successfully.")
  else:
    print("[INFO] No policy loaded. Using zero actions.")

  # --- Rollout ---
  if cfg.num_episodes is not None:
    max_steps = float("inf")
    max_episodes = cfg.num_episodes
    print(f"[INFO] Running test for {cfg.num_episodes} episodes")
  else:
    max_steps = int(cfg.duration / env_wrapped.unwrapped.step_dt)
    max_episodes = float("inf")
    print(f"[INFO] Running test for {cfg.duration}s ({max_steps} steps)")

  stats = EvaluationStats()
  num_envs = env_wrapped.num_envs
  action_dim = env_wrapped.unwrapped.action_space.shape[1]
  current_returns = torch.zeros(num_envs, device=device)
  current_lengths = torch.zeros(num_envs, device=device, dtype=torch.int)

  obs, _ = env_wrapped.reset()
  timestep = 0
  with torch.inference_mode():
    while timestep < max_steps and stats.episode_count < max_episodes:
      if policy is not None:
        actions = policy(obs)
      else:
        actions = torch.zeros(num_envs, action_dim, device=device)

      obs, rewards, dones, extras = env_wrapped.step(actions)

      current_returns += rewards
      current_lengths += 1
      done_mask = dones.bool()
      if done_mask.any():
        for idx in torch.where(done_mask)[0]:
          stats.add_episode(
            current_returns[idx].item(), int(current_lengths[idx].item())
          )
          current_returns[idx] = 0.0
          current_lengths[idx] = 0

      timestep += 1
      if timestep % 500 == 0:
        print(f"[INFO] Step {timestep}, episodes completed: {stats.episode_count}")

  summary = stats.summary()
  print(f"\n[INFO] Test completed. Steps: {timestep}")
  if summary["episode_count"] > 0:
    print(
      f"  Mean Return: {summary['mean_return']:.3f} ± {summary['std_return']:.3f} "
      f"(min {summary['min_return']:.3f}, max {summary['max_return']:.3f})"
    )
    print(f"  Mean Length: {summary['mean_length']:.1f} ± {summary['std_length']:.1f}")
    print(f"  Episodes: {summary['episode_count']}")
  else:
    print("  No episodes completed (increase --duration or --num-episodes).")

  env_wrapped.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TestConfig,
    args=remaining_args,
    default=TestConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  run_test(chosen_task, args)


if __name__ == "__main__":
  main()
