"""CPU checks of released SoftMimic observation/reward/termination semantics."""
from pathlib import Path
from types import SimpleNamespace as NS
import math
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tasks.compliant_tracking.config.g1_23dof.env_cfgs import unitree_g1_23dof_flat_compliant_tracking_env_cfg
from src.tasks.compliant_tracking.mdp import observations as obs
from src.tasks.compliant_tracking.mdp import rewards, terminations


def fixture():
    n, frames, bodies = 2, 101, 14
    ident = torch.tensor([1., 0., 0., 0.])
    motion = NS()
    for prefix in ('', 'nominal_'):
        setattr(motion, prefix+'joint_pos', torch.zeros(frames, 23))
        setattr(motion, prefix+'body_pos_w', torch.zeros(frames, bodies, 3))
        setattr(motion, prefix+'body_quat_w', ident.repeat(frames, bodies, 1))
        setattr(motion, prefix+'body_lin_vel_w', torch.zeros(frames, bodies, 3))
        setattr(motion, prefix+'body_ang_vel_w', torch.zeros(frames, bodies, 3))
        setattr(motion, prefix+'foot_contacts', torch.ones(frames, 2))
    motion.force_body_index=torch.where(torch.arange(frames)%2==1,12,13)
    motion.force_body_pos_w=torch.zeros(frames,3)
    motion.force_body_quat_w=ident.repeat(frames,1)
    motion.force_vector=torch.zeros(frames,3)
    motion.torque_vector=torch.zeros(frames,3)
    data = NS(joint_pos=torch.zeros(n,23), joint_vel=torch.zeros(n,23),
              default_joint_pos=torch.zeros(n,23), default_joint_vel=torch.zeros(n,23),
              root_link_pos_w=torch.zeros(n,3), root_link_quat_w=ident.repeat(n,1),
              root_link_ang_vel_b=torch.zeros(n,3), root_link_lin_vel_b=torch.zeros(n,3),
              projected_gravity_b=torch.tensor([0.,0.,-1.]).repeat(n,1),
              body_link_pos_w=torch.zeros(n,bodies,3),
              body_link_quat_w=ident.repeat(n,bodies,1),
              body_link_lin_vel_w=torch.zeros(n,bodies,3))
    names=['pelvis']+[f'body{i}' for i in range(1,12)]+['left_wrist_roll_rubber_hand','right_wrist_roll_rubber_hand']
    robot=NS(data=data,body_names=names)
    command=NS(motion=motion, robot=robot, cfg=NS(body_names=tuple(names)),
               time_steps=torch.tensor([0,1]), foot_body_indexes=[3,6],
               force_body_indexes=torch.tensor([12,13]), active_force_mask=torch.ones(n,dtype=torch.bool),
               forcefield_forces_w=torch.zeros(n,3), target_forces_w=torch.zeros(n,3),
               forcefield_torques_w=torch.zeros(n,3), target_torques_w=torch.zeros(n,3),
               desired_stiffness=torch.ones(n)*40,desired_rotational_stiffness=torch.ones(n),
               forcefield_stiffness=torch.ones(n)*100,forcefield_rotational_stiffness=torch.ones(n),
               body_pos_relative_w=torch.zeros(n,bodies,3),robot_body_pos_w=data.body_link_pos_w,
               body_quat_relative_w=ident.repeat(n,bodies,1),robot_body_quat_w=data.body_link_quat_w,
               body_quat_w=ident.repeat(n,bodies,1),
               force_body_pos_relative_w=torch.zeros(n,3),force_body_quat_relative_w=ident.repeat(n,1),
               _ff_anchor_rot=ident.repeat(n,1),_ff_anchor_pos=torch.zeros(n,3),
               _ff_anchor_ref_pos=torch.zeros(n,3),joint_pos=torch.zeros(n,23),
               robot_body_lin_vel_w=data.body_link_lin_vel_w,
               robot_body_ang_vel_w=torch.zeros(n,bodies,3),
               desired_foot_contacts=torch.ones(n,2))
    command.future_time_steps=lambda steps: (command.time_steps[:,None]+torch.tensor(steps)[None]).clamp(max=frames-1)
    sensor=NS(data=NS(force=torch.zeros(n,2,3),force_history=torch.zeros(n,2,3,3)))
    class Scene(dict):
        env_origins=torch.zeros(n,3)
    env=NS(num_envs=n,device='cpu',scene=Scene(robot=robot,feet_ground_contact=sensor),
           command_manager=NS(get_term=lambda name:command),action_manager=NS(action=torch.zeros(n,23)))
    return env,command


def test_actor_and_critic_contract_and_reference_separation():
    env,c=fixture()
    cfg=unitree_g1_23dof_flat_compliant_tracking_env_cfg(stand_deployable=True)
    actor=cfg.observations['actor'].terms
    assert list(actor)==['base_ang_vel','projected_gravity','joint_pos','joint_vel','actions',
                         'reference_dof_pos','reference_xy_vel','reference_yaw_vel',
                         'reference_root_height','reference_gravity_vector','reference_foot_contact',
                         'desired_stiffness_log','desired_rotational_stiffness_log']
    dims={name:sum(term.func(env,**term.params).shape[1]*max(term.history_length,1)
                   for term in group.terms.values()) for name,group in cfg.observations.items()}
    assert dims=={'actor':327,'critic':1951}
    assert all(term.history_length==3 for term in actor.values())
    assert cfg.scene.sensors[0].history_length==3
    c.motion.nominal_joint_pos[:]=.7;c.motion.joint_pos[:]=-.2
    torch.testing.assert_close(obs.reference_dof_pos(env,'motion'),torch.full((2,23),.7))
    torch.testing.assert_close(obs.reference_dof_pos(env,'motion',nominal=False),torch.full((2,23),-.2))
    assert set(cfg.terminations)=={'time_out','bad_keypoints'}
    assert cfg.commands['motion'].anchor_body_name=='pelvis'
    assert cfg.rewards['joint_position_tracking'].weight==2.


def test_reference_velocity_is_local_and_future_is_separate():
    env,c=fixture()
    yaw90=torch.tensor([math.sqrt(.5),0.,0.,math.sqrt(.5)])
    c.motion.nominal_body_quat_w[:,0]=yaw90
    c.motion.nominal_body_lin_vel_w[:,0,0]=1.
    c.motion.nominal_body_ang_vel_w[:,0,2]=.4
    c.motion.nominal_body_pos_w[:,0,2]=.76
    c.motion.nominal_joint_pos[5]=1.
    torch.testing.assert_close(obs.reference_xy_vel(env,'motion'),torch.tensor([[0.,-1.],[0.,-1.]]),atol=1e-6,rtol=0)
    torch.testing.assert_close(obs.reference_ang_vel(env,'motion'),torch.full((2,1),.4))
    torch.testing.assert_close(obs.reference_root_height(env,'motion'),torch.full((2,1),.76))
    horizon=obs.reference_dof_pos(env,'motion',future_steps=(0,5))
    assert horizon.shape==(2,46)
    assert horizon[0,:23].sum()==0 and horizon[0,23:].sum()==23


def test_force_observations_keep_body_identity_and_local_frame():
    env,c=fixture()
    yaw90=torch.tensor([math.sqrt(.5),0.,0.,math.sqrt(.5)])
    c.robot.data.body_link_quat_w[:,12:]=yaw90
    c.forcefield_forces_w[:,1]=50.
    values=obs.forcefield_force_applied(env,'motion')
    torch.testing.assert_close(values,torch.tensor([[1.,0.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.]]),atol=1e-6,rtol=0)
    c.force_body_indexes[0]=-1
    assert obs.forcefield_force_applied(env,'motion')[0].abs().sum()==0


def test_contacts_and_sliding_use_history_maximum_not_found_flag():
    env,c=fixture();sensor=env.scene['feet_ground_contact'].data
    sensor.force_history[:,0,2,2]=10.;sensor.force_history[:,1,0,2]=.5
    torch.testing.assert_close(obs.foot_contact(env,'feet_ground_contact'),torch.tensor([[1.,0.],[1.,0.]]))
    assert obs.foot_contact_forces(env,'feet_ground_contact').shape==(2,2)
    c.robot.data.body_link_lin_vel_w[:,3,0]=2.
    torch.testing.assert_close(rewards.feet_slide_proportional(env,'feet_ground_contact','motion'),torch.full((2,),20.))
    torch.testing.assert_close(rewards.foot_contact_schedule_mismatch(env,'motion','feet_ground_contact'),torch.ones(2))


def test_joint_kernel_and_full_distance_termination():
    env,c=fixture();c.robot.data.joint_pos[:,0]=.5
    torch.testing.assert_close(rewards.joint_deviation_from_command_exp(env,'motion'),torch.full((2,),math.exp(-.5)))
    c.robot_body_pos_w[0,5,0]=.51;c.robot_body_pos_w[1,5,0]=.49
    assert terminations.bad_motion_body_pos(env,'motion',.5).tolist()==[True,False]


def test_force_link_reward_and_critic_error_use_fixed_target():
    env,c=fixture()
    c.motion.force_body_pos_w[:,0]=1.
    c.robot.data.body_link_pos_w[:,12:,0]=.9
    # Changing the current pelvis does not recenter the fixed onset target.
    c.robot.data.root_link_pos_w[:,0]=.4
    torch.testing.assert_close(rewards.force_link_position_tracking_exp(env,'motion'),torch.full((2,),math.exp(-1)),atol=1e-6,rtol=0)
    torch.testing.assert_close(obs.all_adapted_keypoints_pos_error_local(env,'motion'),torch.tensor([[.1,0,0],[.1,0,0]]),atol=1e-6,rtol=0)
    c.active_force_mask[1]=False
    assert rewards.force_link_position_tracking_exp(env,'motion')[1]==1.
    assert obs.all_adapted_keypoints_pos_error_local(env,'motion')[1].abs().sum()==0


def test_reward_uses_next_frame_and_termination_uses_current_live_alignment():
    env,c=fixture()
    c.time_steps[:]=0
    # Robot moves globally after its last observation; local tracking should
    # use the live pelvis, not the previous command's cached target.
    c.robot.data.root_link_pos_w[:,0]=2.
    c.robot_body_pos_w[:,:,0]=2.
    c.motion.body_pos_w[1,5,0]=.6
    assert terminations.bad_motion_body_pos(env,'motion',.5).tolist()==[False,False]
    assert obs.keypoint_error_local_shifted(env,'motion').abs().max()==0.
    expected=math.exp(-(.6**2 / 14)/(.3**2))
    torch.testing.assert_close(rewards.motion_relative_body_position_error_exp(env,'motion',.3),torch.full((2,),expected))
    # Nonconstant future joint and contact commands must affect rewards now,
    # while current actor reference remains the current original pose.
    c.motion.joint_pos[1,0]=.5
    c.motion.foot_contacts[1]=0.
    env.scene['feet_ground_contact'].data.force_history[:,:,:,2]=10.
    torch.testing.assert_close(rewards.joint_deviation_from_command_exp(env,'motion'),torch.full((2,),math.exp(-.5)))
    torch.testing.assert_close(rewards.foot_contact_schedule_mismatch(env,'motion','feet_ground_contact'),torch.full((2,),2.))
    assert obs.reference_dof_pos(env,'motion').abs().max()==0.


def test_future_force_link_and_wrench_keep_current_onset_transform():
    env,c=fixture();c.time_steps[:]=0
    c.motion.force_body_index[1]=13  # Future reward body differs fromcurrent.
    c.motion.force_body_pos_w[1,0]=1.
    c.motion.force_vector[1,0]=20.
    c.motion.torque_vector[1,2]=2.
    c._ff_anchor_pos[:,0]=3.
    c.robot.data.body_link_pos_w[:,13,0]=4.
    c.robot.data.body_link_pos_w[:,12,0]=0.
    torch.testing.assert_close(rewards.force_link_position_tracking_exp(env,'motion'),torch.ones(2))
    torch.testing.assert_close(rewards.force_command_tracking(env,'motion'),torch.full((2,),math.exp(-1)))
    torch.testing.assert_close(rewards.torque_command_tracking(env,'motion'),torch.full((2,),math.exp(-1)))
    torch.testing.assert_close(obs.forcefield_force_desired(env,'motion')[0,0],torch.tensor(.4))
    assert obs.all_adapted_keypoints_pos_error_local(env,'motion').abs().sum()==0.


def test_future_root_velocity_and_gravity_reward():
    env,c=fixture();c.time_steps[:]=0
    c.motion.body_lin_vel_w[1,0,0]=.5
    c.motion.body_ang_vel_w[1,0,2]=2.
    torch.testing.assert_close(rewards.motion_base_lin_vel_local_error_exp(env,'motion'),torch.full((2,),math.exp(-1)))
    torch.testing.assert_close(rewards.motion_base_ang_vel_local_error_exp(env,'motion'),torch.full((2,),math.exp(-1)))
    # 90deg pitch rotates gravity away; currentreference remains upright.
    c.motion.body_quat_w[1,0]=torch.tensor([math.sqrt(.5),0.,math.sqrt(.5),0.])
    torch.testing.assert_close(rewards.motion_base_gravity_error_exp(env,'motion'),torch.full((2,),math.exp(-1)))
