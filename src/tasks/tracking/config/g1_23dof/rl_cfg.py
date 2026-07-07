"""RL configuration for Unitree G1_23Dof tracking task."""

from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class VisitationCriticCfg:
  """Config for the state-action visitation critic (SB3 VCPPO port)."""

  enabled: bool = False
  # Conditional sampling method used to propose alternative actions:
  # sdedit (default), gode, inpainting, cfg.
  sample_method: str = "sdedit"
  # Base blend coefficient. The runner multiplies this by a trapezoidal envelope
  # (linear ramp up over warmup_iters, hold, optional linear decay).
  alpha: float = 0.5
  warmup_iters: int = 250
  # Iteration at which the alpha envelope hits zero. None = no decay (hold).
  stop_iter: int | None = None
  # Iteration at which the decay leg begins. None = stop_iter // 2 when stop_iter is set.
  decay_start_iter: int | None = None
  # Flow-matching model hyperparameters.
  model_train_every: int = 1
  model_train_steps: int = 80
  model_batch_size: int = 256
  model_lr: float = 1e-3
  model_lambda_steps: int = 50
  model_net: tuple[int, ...] = (128, 128, 128)
  # Visitation buffer + MCQ admission filter.
  buffer_size: int = 100_000
  q_top_fraction: float = 0.25
  q_filter_k: int = 16
  gamma_mcq: float = 0.99
  # Candidate sampling hyperparameters.
  num_samples: int = 50
  policy_trust_std: float = 3.0
  tau: float = 0.9
  sigma: float = 0.2
  guidance_scale: float = 1.0
  cfg_dropout: float = 0.1
  # Optional side Q-critic for candidate ranking. "off" disables it entirely.
  q_mode: str = "off"
  q_net: tuple[int, ...] = (256, 256)
  q_lr: float = 3e-4
  q_batch_size: int = 256
  q_train_steps: int = 300
  q_replay_size: int = 1_000_000
  q_tau: float = 0.005
  seed: int = 0


@dataclass
class RslRlPpoAlgorithmCfgWithVc(RslRlPpoAlgorithmCfg):
  visitation_critic_cfg: VisitationCriticCfg = field(default_factory=VisitationCriticCfg)


@dataclass
class EvalCfg:
  enabled: bool = False
  eval_every_n_iters: int = 100
  eval_num_episodes: int = 1000
  eval_num_envs: int = 1000


@dataclass
class RslRlOnPolicyRunnerCfgWithEval(RslRlOnPolicyRunnerCfg):
  eval: EvalCfg = field(default_factory=EvalCfg)


def unitree_g1_23dof_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1_23Dof tracking task."""
  return RslRlOnPolicyRunnerCfgWithEval(
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
    },
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfgWithVc(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_23dof_tracking",
    wandb_project="mjlab_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30001,
  )
