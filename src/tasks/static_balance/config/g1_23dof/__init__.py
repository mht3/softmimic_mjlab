from mjlab.tasks.registry import register_mjlab_task
from src.tasks.static_balance.rl import BalanceOnPolicyRunner

from .env_cfgs import (
  unitree_g1_23dof_flat_env_cfg,
  unitree_g1_23dof_balance_push_curriculum_env_cfg,
)
from .rl_cfg import unitree_g1_23dof_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Balance-Flat",
  env_cfg=unitree_g1_23dof_flat_env_cfg(),
  play_env_cfg=unitree_g1_23dof_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_ppo_runner_cfg(),
  runner_cls=BalanceOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Balance-Flat-Push-Curriculum",
  env_cfg=unitree_g1_23dof_balance_push_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_23dof_balance_push_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_ppo_runner_cfg(),
  runner_cls=BalanceOnPolicyRunner,
)
