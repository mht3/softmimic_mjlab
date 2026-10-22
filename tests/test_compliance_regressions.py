"""Behavioral regressions for the compliance port (no GPU required)."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tasks.compliant_tracking.mdp.commands import CompliantMotionCommand, CompliantMotionLoader
from src.tasks.compliant_tracking.mdp.actions import ForcefieldWrenchAction
from src.utils.deploy_export import _to_plain_value


def command_stub():
    c = CompliantMotionCommand.__new__(CompliantMotionCommand)
    c.cfg = NS(force_computation_mode='forcefield', max_force=140., max_torque=10.)
    c._env = NS(num_envs=3, device='cpu', scene=NS(env_origins=torch.zeros(3, 3)))
    c.time_steps = torch.arange(3)
    ident = torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)
    c.motion = NS(ff_stiffness=torch.tensor([100., 0., 0.]),
                  ff_rotational_stiffness=torch.tensor([0., 1., 0.]),
                  ff_origin=torch.zeros(3, 3), ff_setpoint_rot=ident.clone(),
                  ff_normal=torch.zeros(3, 3), force_body_index=torch.tensor([0, 0, -1]),
                  root_pos_w=torch.zeros(3, 3), root_quat_w=ident.clone(),
                  force_vector=torch.zeros(3, 3), torque_vector=torch.tensor([[0.,0.,0.],[0.,0.,1.],[0.,0.,0.]]))
    c.robot = NS(data=NS(root_link_pos_w=torch.zeros(3, 3), root_link_quat_w=ident.clone(),
                        body_link_pos_w=torch.ones(3, 1, 3)*0.1,
                        body_link_quat_w=ident[:,None,:].clone()))
    c._last_ff_stiffness=torch.zeros(3)
    c._last_force_body=torch.full((3,),-1)
    c._ff_anchor_pos=torch.zeros(3,3)
    c._ff_anchor_ref_pos=torch.zeros(3,3)
    c._ff_anchor_rot=ident.clone()
    return c


def test_zero_target_spring_and_pure_torque_are_active():
    c=command_stub()
    c._update_compliance_buffers()
    assert c.active_force_mask.tolist() == [True,True,False]
    torch.testing.assert_close(c.forcefield_forces_w[0],torch.full((3,),-10.))
    c.motion.ff_setpoint_rot[1]=torch.tensor([0.70710678,0,0,0.70710678])
    c._update_compliance_buffers()
    assert c.forcefield_torques_w[1,2] > 1.5


def test_feedforward_torque_only_is_active():
    c=command_stub();c.cfg.force_computation_mode='feedforward'
    c._update_compliance_buffers()
    assert c.active_force_mask.tolist()==[False,True,False]


def test_switching_force_body_reanchors():
    c=command_stub();c._update_compliance_buffers()
    c.motion.force_body_index[0]=1
    c.robot.data.root_link_pos_w[0,0]=2.
    c._update_compliance_buffers()
    assert c._ff_anchor_pos[0,0]==2.


def test_disabling_dataset_force_clears_persistent_wrench_once():
    writes=[]
    a=ForcefieldWrenchAction.__new__(ForcefieldWrenchAction)
    a.cfg=NS(command_name='motion');a._env=NS(command_manager=NS(get_term=lambda _:NS(dataset_forces_enabled=False)))
    a._entity=NS(write_external_wrench_to_sim=lambda f,t:writes.append((f.clone(),t.clone())))
    a._forces=torch.ones(1,2,3);a._torques=torch.ones_like(a._forces);a._was_enabled=True
    a.apply_actions();a.apply_actions()
    assert len(writes)==1 and writes[0][0].count_nonzero()==0 and writes[0][1].count_nonzero()==0


def test_deploy_preserves_controller_precision():
    values={'dt':0.0025,'offset':[0.1234567],'scale':0.00432}
    assert _to_plain_value(values)==values


def test_nominal_actor_is_separate_from_adapted_target(tmp_path):
    root=Path(__file__).resolve().parents[1]
    source=root/'src/assets/compliant_motions/g1_23dof/stand/forcefield/stand_augmented_mink_001.npz'
    with np.load(source) as data:
        arrays={k:v[:2] if v.ndim and len(v)>2 else v for k,v in data.items()}
    for key in ('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w','foot_contacts'):
        arrays['nominal_'+key]=arrays[key].copy()
    arrays['nominal_joint_pos'][:]=0.123
    dest=tmp_path/'clip.npz';np.savez(dest,**arrays)
    loader=CompliantMotionLoader(str(dest),torch.tensor([0]))
    c=CompliantMotionCommand.__new__(CompliantMotionCommand)
    c.motion=loader;c.time_steps=torch.tensor([0]);c.cfg=NS(require_nominal_reference=True)
    torch.testing.assert_close(c.command[0,:23],torch.full((23,),.123))
    assert not torch.allclose(c.joint_pos,c.command[:,:23])


def test_residual_decay_and_ik_share_one_velocity_budget():
    import numpy as np
    import mujoco
    from types import SimpleNamespace
    from compliant_motion_augmentation.ik_update import bound_frame_joint_step
    from compliant_motion_augmentation.constants import MAX_IK_JOINT_VEL
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model)
    class Configuration:
        q=np.array([.314159265])  # decay pi + IK 2pi, integrated at 30 Hz
        def update(self,q):self.q=q;data.qpos[:]=q
    solver=SimpleNamespace(configuration=Configuration(),model=model,data=data,actuated_qpos_indices=[0])
    bound_frame_joint_step(solver,np.zeros(1),1/30)
    np.testing.assert_allclose(solver.configuration.q,[MAX_IK_JOINT_VEL/30])


def test_rejection_rewinds_to_actual_start_frame_without_duplicate():
    import numpy as np
    from types import SimpleNamespace
    from compliant_motion_augmentation.runner import SimulationRunner
    from compliant_motion_augmentation.config import SimulationConfig
    cfg=SimulationConfig(mode='generate-data',model_path='',motion_path='',seed=0,
                         com_cost=.1,com_cost_z_factor=0.,force_mode='forcefield')
    runner=SimulationRunner(cfg,0)
    configuration=SimpleNamespace(q=np.r_[np.zeros(3),1.,np.zeros(26)])
    configuration.update=lambda q:setattr(configuration,'q',q.copy())
    runner.ik_solver=SimpleNamespace(configuration=configuration,model=SimpleNamespace(nq=30),
        actuated_qpos_indices=list(range(7,30)),get_reference_motion=lambda t:(configuration.q.copy(),None,None))
    row=np.r_[np.zeros(6),1.,np.zeros(23)]
    runner.all_adapted_qpos=[row.copy() for _ in range(31)]
    runner.all_reference_qpos=[np.zeros(32) for _ in range(31)]
    runner.all_force_data=[np.zeros(9) for _ in range(31)]
    runner.all_collision_metadata=[np.zeros(12) for _ in range(31)]
    runner.event_queue=[{'start_time':1.01,'amplitude':.5,'torque_amplitude':0.,'ramp_duration':.2}]
    runner._maybe_start_next_event(31/30)
    assert runner._handle_infeasible_event(1.2)==31
    assert len(runner.all_adapted_qpos)==31
