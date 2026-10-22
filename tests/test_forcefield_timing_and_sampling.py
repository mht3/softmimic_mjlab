"""CPU behavioral checks of released force timing, anchoring and clip sampling."""
from types import SimpleNamespace as NS
import math
import torch
import pytest
from src.tasks.compliant_tracking.mdp.actions import ForcefieldWrenchAction
from src.tasks.compliant_tracking.mdp.commands import CompliantMotionCommand, resolve_motion_files


def test_force_is_cached_at_policy_rate_and_rotates_with_body():
    command=NS(dataset_forces_enabled=True,active_force_mask=torch.tensor([True]),
               force_body_indexes=torch.tensor([0]),forcefield_forces_w=torch.tensor([[10.,0.,0.]]),
               forcefield_torques_w=torch.zeros(1,3))
    quat=torch.tensor([[[1.,0.,0.,0.]]]);writes=[]
    a=ForcefieldWrenchAction.__new__(ForcefieldWrenchAction)
    a.cfg=NS(command_name='motion');a._env=NS(command_manager=NS(get_term=lambda _:command))
    a._entity=NS(data=NS(body_link_quat_w=quat),write_external_wrench_to_sim=lambda f,t:writes.append(f.clone()))
    a._forces=torch.zeros(1,1,3);a._torques=torch.zeros_like(a._forces)
    a._local_forces=torch.zeros_like(a._forces);a._local_torques=torch.zeros_like(a._forces);a._was_enabled=False
    a.cache_wrench();a.apply_actions()
    command.forcefield_forces_w[:]=torch.tensor([[100.,0.,0.]])
    quat[:]=torch.tensor([math.sqrt(.5),0.,0.,math.sqrt(.5)])
    a.process_actions(torch.zeros(1,0));a.apply_actions()
    torch.testing.assert_close(writes[0][0,0],torch.tensor([10.,0.,0.]))
    torch.testing.assert_close(writes[1][0,0],torch.tensor([0.,10.,0.]),atol=1e-5,rtol=0)
    a.cache_wrench();a.apply_actions()
    torch.testing.assert_close(writes[-1][0,0],torch.tensor([100.,0.,0.]),atol=1e-4,rtol=0)


def test_force_target_keeps_onset_xyz_when_current_pelvis_moves():
    c=CompliantMotionCommand.__new__(CompliantMotionCommand)
    c._env=NS(scene=NS(env_origins=torch.tensor([[10.,0.,0.]])))
    c.time_steps=torch.tensor([0])
    c.motion=NS(force_body_pos_w=torch.tensor([[1.,0.,.5]]),force_body_quat_w=torch.tensor([[1.,0.,0.,0.]]))
    c._ff_anchor_pos=torch.tensor([[12.,2.,1.]])
    c._ff_anchor_ref_pos=torch.tensor([[10.,0.,.7]])
    c._ff_anchor_rot=torch.tensor([[math.sqrt(.5),0.,0.,math.sqrt(.5)]])
    c._update_force_body_targets()
    torch.testing.assert_close(c.force_body_pos_relative_w,torch.tensor([[12.,3.,.8]]),atol=1e-6,rtol=0)


def test_initial_phase_is_uniform_with_fixed_file_and_failure_continues():
    c=CompliantMotionCommand.__new__(CompliantMotionCommand)
    c._env=NS(device='cpu');c.cfg=NS(sampling_mode='uniform')
    c.motion_ids=torch.tensor([0,1,0,1]);c._phase_initialized=torch.zeros(4,dtype=torch.bool)
    c.motion=NS(num_motions=2,motion_starts=torch.tensor([0,10]),motion_ends=torch.tensor([10,110]))
    c.motion.motion_end_of=lambda t:torch.where(t<10,10,110)
    c.time_steps=torch.tensor([0,10,0,10]);c._clear_force_state=lambda ids:None
    writes=[];c._write_reference_state=lambda ids:writes.append(c.time_steps[ids].clone())
    ids=torch.arange(4);c._resample_command(ids)
    first=c.time_steps.clone()
    assert (first[[0,2]]<10).all() and (first[[1,3]]>=10).all()
    c._resample_command(ids)
    torch.testing.assert_close(c.time_steps,first)


def test_hybrid_equal_mode_weights_and_incomplete_mode_rejected(tmp_path):
    for mode,count in [('forcefield',40),('collision-emulator',40),('zero-wrench',5)]:
        folder=tmp_path/mode;folder.mkdir()
        for i in range(count):(folder/f'{i}.npz').touch()
    files=resolve_motion_files(str(tmp_path))
    assert [sum(f.parent.name==mode for f in files) for mode in ('forcefield','collision-emulator','zero-wrench')]==[40,40,40]
    for p in (tmp_path/'zero-wrench').iterdir():p.unlink()
    (tmp_path/'zero-wrench').rmdir()
    with pytest.raises(ValueError,match='missing modes'):resolve_motion_files(str(tmp_path))
