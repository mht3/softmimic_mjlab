from mjlab.tasks.registry import register_mjlab_task
from src.tasks.compliant_tracking.rl import CompliantMotionTrackingOnPolicyRunner

from .env_cfgs import unitree_g1_23dof_flat_compliant_tracking_env_cfg
from .rl_cfg import unitree_g1_23dof_compliant_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Compliant-Tracking",
  env_cfg=unitree_g1_23dof_flat_compliant_tracking_env_cfg(),
  play_env_cfg=unitree_g1_23dof_flat_compliant_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_compliant_tracking_ppo_runner_cfg(),
  runner_cls=CompliantMotionTrackingOnPolicyRunner,
)

# Standing reference-conditioned actor, including all six nominal channels.
register_mjlab_task(
  task_id="Unitree-G1-23Dof-Compliant-Tracking-Stand",
  env_cfg=unitree_g1_23dof_flat_compliant_tracking_env_cfg(stand_deployable=True),
  play_env_cfg=unitree_g1_23dof_flat_compliant_tracking_env_cfg(
    stand_deployable=True, play=True
  ),
  rl_cfg=unitree_g1_23dof_compliant_tracking_ppo_runner_cfg(),
  runner_cls=CompliantMotionTrackingOnPolicyRunner,
)

from .env_cfgs import unitree_g1_23dof_stable_stand_env_cfg
from .rl_cfg import unitree_g1_23dof_stable_stand_ppo_runner_cfg
register_mjlab_task(
  task_id="Unitree-G1-23Dof-Compliant-Tracking-Stand-Stable",
  env_cfg=unitree_g1_23dof_stable_stand_env_cfg(),
  play_env_cfg=unitree_g1_23dof_stable_stand_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_stable_stand_ppo_runner_cfg(),
  runner_cls=CompliantMotionTrackingOnPolicyRunner,
)
