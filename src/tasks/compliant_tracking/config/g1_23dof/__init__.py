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

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Compliant-Tracking-No-State-Estimation",
  env_cfg=unitree_g1_23dof_flat_compliant_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_23dof_flat_compliant_tracking_env_cfg(
    has_state_estimation=False, play=True
  ),
  rl_cfg=unitree_g1_23dof_compliant_tracking_ppo_runner_cfg(),
  runner_cls=CompliantMotionTrackingOnPolicyRunner,
)

# Velocity-conditioned variant: the actor additionally observes the reference
# root velocity (heading frame), so a walking policy trained on this task can be
# steered with the GUI velocity joystick in play.py. The observation dimension
# differs from the tasks above, so their checkpoints are not interchangeable.
register_mjlab_task(
  task_id="Unitree-G1-23Dof-Compliant-Tracking-Velocity",
  env_cfg=unitree_g1_23dof_flat_compliant_tracking_env_cfg(
    has_state_estimation=False, velocity_conditioning=True
  ),
  play_env_cfg=unitree_g1_23dof_flat_compliant_tracking_env_cfg(
    has_state_estimation=False, velocity_conditioning=True, play=True
  ),
  rl_cfg=unitree_g1_23dof_compliant_tracking_ppo_runner_cfg(),
  runner_cls=CompliantMotionTrackingOnPolicyRunner,
)
