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
  guidance_scale: float = 2.0
  num_euler_steps: int = 100
  cfg_dropout_prob: float = 0.2
  reset_condition_label: int = 0
  # CFM vector-field network architecture.
  hidden_dims: tuple[int, ...] = (512, 512, 512)
  class_dim: int = 8
  # Deterministic trajectory collection (runs right before each VC training step).
  num_collect_trajectories: int = 10000
  disable_push_during_collection: bool = False


@dataclass
class RslRlPpoAlgorithmCfgWithVc(RslRlPpoAlgorithmCfg):
  visitation_critic_cfg: VisitationCriticCfg = field(default_factory=VisitationCriticCfg)


def unitree_g1_23dof_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1-23DOF velocity task."""
  return RslRlOnPolicyRunnerCfg(
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
