"""Standing export must reproduce nominal actor inputs without adapted leakage."""
from pathlib import Path
from types import SimpleNamespace as NS
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "algorithms/rsl_rl")]
from src.utils.deploy_export import _constant_nominal_reference, export_deploy_cfg
from src.tasks.compliant_tracking.config.g1_23dof.env_cfgs import unitree_g1_23dof_flat_compliant_tracking_env_cfg


def command_stub():
    n = 6
    quat = torch.tensor([2**-.5, 0., 0., 2**-.5]).repeat(n, 1, 1)
    motion = NS(has_nominal_reference=True,
        nominal_joint_pos=torch.arange(23, dtype=torch.float32).repeat(n, 1)*.01,
        nominal_body_pos_w=torch.tensor([0., 0., .74]).repeat(n, 1, 1),
        nominal_body_quat_w=quat,
        nominal_body_lin_vel_w=torch.tensor([1., 2., 3.]).repeat(n, 1, 1),
        nominal_body_ang_vel_w=torch.tensor([4., 5., 6.]).repeat(n, 1, 1),
        nominal_foot_contacts=torch.tensor([.25, .75]).repeat(n, 1))
    command = NS(motion=motion, cfg=NS(body_names=("pelvis",)),
        desired_stiffness=torch.full((2,), 200.), desired_rotational_stiffness=torch.ones(2))
    command.future_time_steps = lambda steps: torch.tensor([[0], [3]]) + torch.tensor(steps)
    return command


def test_nominal_channels_use_pelvis_local_velocity_and_raw_joint_positions():
    values = _constant_nominal_reference(command_stub())
    np.testing.assert_allclose(values["reference_xy_vel"], [2., -1.], atol=1e-6)
    np.testing.assert_allclose(values["reference_yaw_vel"], [6.])
    np.testing.assert_allclose(values["reference_gravity_vector"], [0., 0., -1.])
    np.testing.assert_allclose(values["reference_dof_pos"], np.arange(23)*.01, atol=1e-7)
    np.testing.assert_allclose(values["reference_foot_contact"], [.25, .75])


def test_export_rejects_changing_reference_and_accepts_float32_derivative_noise():
    command = command_stub()
    command.motion.nominal_body_ang_vel_w[3, 0, 2] += 6e-6
    _constant_nominal_reference(command)
    command.motion.nominal_joint_pos[4, 8] += .01
    with pytest.raises(ValueError, match="dynamic nominal replay is unimplemented"):
        _constant_nominal_reference(command)


@pytest.mark.parametrize("moving", [False, True])
def test_actual_standing_actor_exports_13_terms_327_values_and_all_constants(tmp_path, moving):
    cfg = unitree_g1_23dof_flat_compliant_tracking_env_cfg(stand_deployable=not moving)
    actor = cfg.observations["actor"].terms
    data = NS(default_joint_pos=torch.full((2, 23), -.2), joint_pos=torch.zeros(2, 23),
        default_joint_vel=torch.zeros(2, 23), joint_vel=torch.zeros(2, 23),
        root_link_ang_vel_b=torch.tensor([[1., 2., 3.]]).repeat(2, 1),
        projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(2, 1))
    command = command_stub()
    command.cfg.deployment_reference_mode = cfg.commands["motion"].deployment_reference_mode
    if moving: command.motion.nominal_joint_pos[3,0] += .1
    env = NS(cfg=cfg, num_envs=2, step_dt=.02, scene={"robot": NS(data=data)},
        command_manager=NS(get_term=lambda _: command),
        action_manager=NS(_terms={}, action=torch.zeros(2, 23)),
        observation_manager=NS(active_terms={"actor": list(actor)}, _group_obs_term_cfgs={"actor": list(actor.values())}))
    export_deploy_cfg(env, tmp_path)
    exported = yaml.safe_load((tmp_path / "params/deploy.yaml").read_text())
    expected_names = ["base_ang_vel", "projected_gravity", "joint_pos_rel", "joint_vel_rel", "last_action",
        "reference_dof_pos", "reference_xy_vel", "reference_yaw_vel", "reference_root_height",
        "reference_gravity_vector", "reference_foot_contact", "desired_stiffness_log", "desired_rotational_stiffness_log"]
    assert list(exported["observations"]) == expected_names
    assert exported["observation_dim"] == 327
    assert all(v["history_length"] == 3 for v in exported["observations"].values())
    assert exported["observations"]["base_ang_vel"]["scale"] == [.25]*3
    assert exported["observations"]["joint_vel_rel"]["scale"] == [.05]*23
    # The command stub mirrors the reference mode of the selected task.
    if not moving:
        assert exported["reference_mode"] == "constant_nominal"
        for name, value in _constant_nominal_reference(command).items():
            assert exported["observations"][name]["params"] == {"value": value, "reference_mode": "constant_nominal"}
    else:
        assert exported["reference_mode"] == "nominal_motion"
        for name in expected_names[5:11]:
            assert exported["observations"][name]["params"] == {"reference_mode": "nominal_motion"}
    # Unknown actor terms must fail export, never silently disappear at runtime.
    env.observation_manager.active_terms["actor"] = ["unsupported_input"]
    env.observation_manager._group_obs_term_cfgs["actor"] = [next(iter(actor.values()))]
    with pytest.raises(ValueError, match="no supported C\\+\\+ deployment"):
        export_deploy_cfg(env, tmp_path)


def test_cpp_reference_validation_and_termwise_history(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "reference.cpp"
    source.write_text(r'''
#include <algorithm>
#include <cassert>
#include <yaml-cpp/yaml.h>
#include "isaaclab/envs/mdp/observations/constant_reference.h"
#include "isaaclab/manager/manager_term_cfg.h"
int main() {
    auto p = YAML::Load("reference_mode: constant_nominal\nvalue: [0.25, 0.75]\n");
    const auto v = isaaclab::mdp::constant_nominal_reference(p, "feet", 2);
    assert(v[0] == .25f && v[1] == .75f);
    bool failed=false;
    try { isaaclab::mdp::constant_nominal_reference(p, "joints", 23); } catch(const std::runtime_error&) { failed=true; }
    assert(failed);
    p.remove("reference_mode"); failed=false;
    try { isaaclab::mdp::constant_nominal_reference(p, "feet", 2); } catch(const std::runtime_error&) { failed=true; }
    assert(failed);
    isaaclab::ObservationTermCfg t; t.history_length=3; t.scale={.25f, .25f};
    t.reset({1.f, 2.f}); t.add({3.f, 4.f}); t.add({5.f, 6.f});
    assert(t.get() == std::vector<float>({.25f,.5f,.75f,1.f,1.25f,1.5f}));
}
''')
    binary = tmp_path / "reference"
    subprocess.run([compiler, "-std=c++17", str(source), "-I", str(ROOT / "deploy/include"),
                    "-lyaml-cpp", "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True)
