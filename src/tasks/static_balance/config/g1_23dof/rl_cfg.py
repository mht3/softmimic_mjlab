"""RL configuration for Unitree G1-23DOF velocity task."""

from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


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
    algorithm=RslRlPpoAlgorithmCfg(
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
