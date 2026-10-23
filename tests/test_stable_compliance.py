from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from src.tasks.compliant_tracking.config.g1_23dof.env_cfgs import unitree_g1_23dof_stable_stand_env_cfg
from src.tasks.compliant_tracking.config.g1_23dof.rl_cfg import unitree_g1_23dof_stable_stand_ppo_runner_cfg
from src.tasks.compliant_tracking.mdp.commands import resolve_motion_files
from src.tasks.compliant_tracking.rl.health_metrics import ModeHealthMetrics
from src.tasks.compliant_tracking.rl.action_diagnostics import ActionDiagnostics
from src.tasks.compliant_tracking.mdp.terminations import extreme_robot_state


def test_rebalance_and_bounded_control(tmp_path):
    for mode,count in [('forcefield',40),('collision-emulator',40),('zero-wrench',5)]:
        folder=tmp_path/mode;folder.mkdir()
        for i in range(count):(folder/f'{i}.npz').touch()
    cfg=unitree_g1_23dof_stable_stand_env_cfg();cmd=cfg.commands['motion']
    files=resolve_motion_files(str(tmp_path),cmd.mode_weights)
    assert len(files)==80 and sum(p.parent.name=='forcefield' for p in files)==40
    assert not any(p.parent.name=='collision-emulator' for p in files)
    assert cmd.max_force==140 and cmd.max_torque==10
    assert unitree_g1_23dof_stable_stand_ppo_runner_cfg().clip_actions==5
    assert cfg.observations['actor'].terms['actions'].clip==(-5,5)
    assert cfg.sim.nan_guard.enabled


def test_recent_completion_is_not_cumulative():
    m=ModeHealthMetrics(['forcefield'],'cpu');one=torch.ones(1);zero=torch.zeros(1)
    m.update(one,one,one,zero);m.flush()  # initial episode censored
    m.update(one,one,one,zero);assert m.flush()['Compliance/forcefield/timeout_fraction_this_update']==1
    m.update(one,one,zero,zero);v=m.flush()
    assert v['Compliance/forcefield/timeout_fraction_this_update']==0
    assert v['Compliance/forcefield/timeout_fraction_since_resume']==.5


def test_diagnostics_capture_action_before_reset_and_never_mutate(tmp_path):
    data=NS(joint_pos=torch.zeros(1,23),joint_vel=torch.ones(1,23))
    command=NS(robot=NS(data=data),time_steps=torch.tensor([71]),motion_ids=torch.tensor([4]),forcefield_forces_w=torch.zeros(1,3))
    d=ActionDiagnostics(tmp_path,5);action=torch.full((1,23),1.e15);before=action.clone();d.observe(action,command)
    torch.testing.assert_close(action,before)
    assert '71' in (tmp_path/'action_outliers.jsonl').read_text()
    assert d.flush()['Stability/action_bound_exceedance_fraction']==1
    with pytest.raises(RuntimeError,match='Non-finite'):d.observe(action*float('nan'),command)
    env=NS(scene={'robot':NS(data=NS(joint_vel=torch.full((1,23),101.),root_link_ang_vel_b=torch.zeros(1,3),root_link_lin_vel_w=torch.zeros(1,3)))})
    assert extreme_robot_state(env).item()
