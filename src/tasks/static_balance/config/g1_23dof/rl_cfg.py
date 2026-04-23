"""RL configuration for Unitree G1-23DOF velocity task."""

from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class VisitationCriticCfg:
  enabled: bool = False
  conditioning_type: str = "discrete"
  label_mode: str = "l2_ball"
  l2_radius: float = 4.0 # criteria for good/bad end state
  num_classes: int = 2
  train_every_n_iters: int = 500
  num_warmup_iterations: int = 3000
  num_train_steps: int = 5000
  learning_rate: float = 3e-4
  batch_size: int = 1024
  max_trajectories: int = 10000
  guidance_scale: float = 2.5
  num_euler_steps: int = 100
  cfg_dropout_prob: float = 0.2
  reset_condition_label: int = 0
  # Probabilistic bin sampling for reward_bins mode. Must have length 4
  # and sum to 1.0. Order: (fail-low, fail-high, succeed-low, succeed-high).
  reset_bin_probs: tuple[float, ...] = (0.25, 0.50, 0.20, 0.05)
  # CFM vector-field network architecture.
  hidden_dims: tuple[int, ...] = (1024, 1024, 1024)
  class_dim: int = 8
  # Deterministic trajectory collection (runs right before each VC training step).
  num_collect_trajectories: int = 10000
  disable_push_during_collection: bool = False
  max_num_trains: int = 1  # -1 = unlimited; M > 0 = train at most M times total


@dataclass
class RslRlPpoAlgorithmCfgWithVc(RslRlPpoAlgorithmCfg):
  visitation_critic_cfg: VisitationCriticCfg = field(default_factory=VisitationCriticCfg)


@dataclass
class EvalCfg:
  """Configuration for periodic evaluation on a separate medium-perturbation env."""

  enabled: bool = True
  """Whether to run evaluation during training."""
  eval_every_n_iters: int = 100
  """Run evaluation every N PPO iterations."""
  eval_num_episodes: int = 1000
  """Number of episodes to average over per evaluation."""
  eval_num_envs: int = 1000
  """Number of parallel envs in the eval simulator."""


@dataclass
class RslRlOnPolicyRunnerCfgWithEval(RslRlOnPolicyRunnerCfg):
  """Runner cfg extended with an evaluation block."""

  eval: EvalCfg = field(default_factory=EvalCfg)


def unitree_g1_23dof_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1-23DOF velocity task."""
  return RslRlOnPolicyRunnerCfgWithEval(
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
      "relative_state": ("relative_state",),
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
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_23dof_static_balance",
    wandb_project="mjlab_static_balance",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
