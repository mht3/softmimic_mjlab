"""Do not lose an event's rollback point before its release is feasible."""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from compliant_motion_augmentation.config import SimulationConfig
from compliant_motion_augmentation.runner import SimulationRunner
import compliant_motion_augmentation.runner as module


def test_completed_event_keeps_checkpoint_and_defers_next_ramp(monkeypatch):
    cfg = SimulationConfig(mode='generate-data',model_path='',motion_path='',seed=42,
                           com_cost=.1,com_cost_z_factor=0.,force_mode='forcefield')
    runner = SimulationRunner(cfg, 0)
    root = Path(__file__).resolve().parents[1]
    model = mujoco.MjModel.from_xml_path(str(root/'softmimic_deploy/src/assets/g1/g1_23dof.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    configuration = SimpleNamespace(q=data.qpos.copy())
    configuration.update = lambda q: setattr(configuration,'q',q.copy())
    runner.ik_solver = SimpleNamespace(model=model,data=data,configuration=configuration)
    runner.reusable_ref_data = mujoco.MjData(model)
    event = dict(link_name='right_wrist_roll_rubber_hand',end_time=2.)
    runner.current_event=event
    runner.event_start_data_idx=17
    runner.event_start_step=17
    runner.qpos_before_event=data.qpos.copy()
    monkeypatch.setattr(module,'perform_single_ik_update',lambda *args: (None,np.zeros(3),None,None,None,None,None))
    runner._complete_event_if_finished(2.,data.qpos.copy())
    assert runner.current_event is None
    assert runner.releasing_event is event
    assert runner.event_start_data_idx==runner.event_start_step==17
    assert runner.qpos_before_event is not None
    upcoming=dict(start_time=2.1,hold_start_time=2.3,hold_end_time=2.5,end_time=2.7,
                  torque_start_time=2.2,amplitude=10.,ramp_duration=.2)
    runner.event_queue=[upcoming]
    runner._maybe_start_next_event(3.)
    assert runner.current_event is None and runner.event_queue==[upcoming]
    runner.releasing_event=None
    runner._maybe_start_next_event(3.)
    assert runner.current_event is upcoming
    assert upcoming['start_time']==3.
    assert np.isclose(upcoming['end_time'],3.6)
    assert np.isclose(upcoming['torque_start_time'],3.1)
    assert upcoming['amplitude']==10. and upcoming['ramp_duration']==.2
