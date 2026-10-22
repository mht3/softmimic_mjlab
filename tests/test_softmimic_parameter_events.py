"""Exercise batched parameter writes and reset cadence without a GPU."""
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "algorithms/rsl_rl")]
from mjlab.actuator import BuiltinPositionActuator
from mjlab.managers.event_manager import EventManager, RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg
spec = importlib.util.spec_from_file_location("parity_events_test", ROOT / "src/tasks/compliant_tracking/mdp/parity_events.py")
events_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(events_module)
additive_actuator_damping = events_module.additive_actuator_damping
additive_joint_limits = events_module.additive_joint_limits
body_mass_with_inertia = events_module.body_mass_with_inertia
make_softmimic_parameter_events = events_module.make_softmimic_parameter_events
MASS_SCALE_BODIES = events_module.MASS_SCALE_BODIES


def fake_env(n=3):
    defaults = {
        "actuator_biasprm": torch.tensor([[0., -40., -5.], [0., -5., -1.], [0., -20., -2.]]),
        "jnt_range": torch.tensor([[-1., 1.], [-2., 2.], [-3., 3.]]),
        "body_mass": torch.tensor([0., 4., 2., 3.]),
        "body_inertia": torch.tensor([[0., 0., 0.], [1., 2., 3.], [.1, .2, .3], [.4, .5, .6]]),
    }
    model = NS(**{k: v.unsqueeze(0).repeat(n, *([1]*v.ndim)) for k, v in defaults.items()})
    actuator = BuiltinPositionActuator.__new__(BuiltinPositionActuator)
    actuator._global_ctrl_ids = torch.tensor([0, 2])
    asset = NS(actuators=[actuator], num_joints=3,
        indexing=NS(joint_ids=torch.arange(3, dtype=torch.int32), body_ids=torch.tensor([1, 2, 3], dtype=torch.int32)),
        cfg=NS(articulation=NS(soft_joint_pos_limit_factor=.9)),
        data=NS(joint_pos_limits=defaults["jnt_range"].repeat(n, 1, 1),
                soft_joint_pos_limits=.9*defaults["jnt_range"].repeat(n, 1, 1),
                default_joint_pos=torch.zeros(n, 3)))
    return NS(num_envs=n, device="cpu", scene={"robot": asset},
        sim=NS(model=model, get_default_field=lambda k: defaults[k]))


def test_kd_resets_from_default_leaves_other_gains_and_envs_unchanged():
    env = fake_env()
    original = env.sim.model.actuator_biasprm.clone()
    additive_actuator_damping(env, torch.tensor([1]), damping_range=(1., 1.))
    expected = original.clone(); expected[1, [0, 2], 2] -= 1.
    torch.testing.assert_close(env.sim.model.actuator_biasprm, expected)
    additive_actuator_damping(env, torch.tensor([1]), damping_range=(1., 1.))
    torch.testing.assert_close(env.sim.model.actuator_biasprm, expected)
    assert additive_actuator_damping.model_fields == ("actuator_biasprm",)


def test_actual_event_manager_enforces_720_step_reset_gate():
    env = fake_env()
    term = make_softmimic_parameter_events()["joint_damping"]
    term.params["damping_range"] = (1., 1.)
    manager = EventManager.__new__(EventManager)
    manager._env = env
    manager._mode_term_cfgs = {"reset": [term]}
    manager._reset_term_last_triggered_step_id = [torch.zeros(3, dtype=torch.int32)]
    manager._reset_term_last_triggered_once = [torch.zeros(3, dtype=torch.bool)]
    manager.apply("reset", env_ids=torch.tensor([1]), global_env_step_count=0)
    assert env.sim.model.actuator_biasprm[1, 0, 2] == -6.
    term.params["damping_range"] = (2., 2.)
    manager.apply("reset", env_ids=torch.tensor([1]), global_env_step_count=719)
    assert env.sim.model.actuator_biasprm[1, 0, 2] == -6.
    manager.apply("reset", env_ids=torch.tensor([1]), global_env_step_count=720)
    assert env.sim.model.actuator_biasprm[1, 0, 2] == -7.


def test_joint_limits_independent_offsets_and_matching_buffers():
    torch.manual_seed(123)
    env = fake_env(n=128)
    cfg = SceneEntityCfg("robot", joint_ids=[0, 2])
    original = env.sim.model.jnt_range.clone()
    additive_joint_limits(env, torch.arange(1, 128), asset_cfg=cfg)
    delta = env.sim.model.jnt_range - original
    assert (delta.abs() <= .020001).all()
    assert not torch.equal(delta[1:, 0, 0], delta[1:, 0, 1])
    assert delta[0].count_nonzero() == 0 and delta[:, 1].count_nonzero() == 0
    torch.testing.assert_close(env.scene["robot"].data.joint_pos_limits, env.sim.model.jnt_range)
    limits = env.sim.model.jnt_range
    mean = limits.mean(-1); half = .45*(limits[..., 1]-limits[..., 0])
    torch.testing.assert_close(env.scene["robot"].data.soft_joint_pos_limits,
                               torch.stack((mean-half, mean+half), -1))
    # Repeated startup application samples from default limits, never accumulates.
    additive_joint_limits(env, None, limit_range=(.01, .01), asset_cfg=cfg)
    additive_joint_limits(env, None, limit_range=(.01, .01), asset_cfg=cfg)
    torch.testing.assert_close(env.sim.model.jnt_range[:, [0, 2]], original[:, [0, 2]]+.01)


def test_mass_and_inertia_share_exact_ratio_and_preserve_other_bodies():
    env = fake_env()
    before_mass = env.sim.model.body_mass.clone()
    before_inertia = env.sim.model.body_inertia.clone()
    cfg = SceneEntityCfg("robot", body_ids=[0, 2])  # Global bodies 1 and 3.
    body_mass_with_inertia(env, torch.tensor([1]), ranges=(1.2, 1.2), operation="scale", asset_cfg=cfg)
    expected = before_mass.clone(); expected[1, [1, 3]] *= 1.2
    inertia = before_inertia.clone(); inertia[1, [1, 3]] *= 1.2
    torch.testing.assert_close(env.sim.model.body_mass, expected)
    torch.testing.assert_close(env.sim.model.body_inertia, inertia)
    body_mass_with_inertia(env, torch.tensor([1]), ranges=(1., 1.), operation="add", asset_cfg=cfg)
    torch.testing.assert_close(env.sim.model.body_mass[1, [1, 3]], before_mass[1, [1, 3]]+1.)
    torch.testing.assert_close(env.sim.model.body_inertia[1, [1, 3]],
        before_inertia[1, [1, 3]] * ((before_mass[1, [1, 3]]+1.)/before_mass[1, [1, 3]])[:, None])
    assert body_mass_with_inertia.recompute == RecomputeLevel.set_const
    assert {"body_mass", "body_inertia"} <= set(body_mass_with_inertia.model_fields)


def test_event_declarations_match_effective_original_configuration():
    events = make_softmimic_parameter_events()
    assert "encoder_bias" not in events
    assert events["joint_damping"].mode == "reset"
    assert events["joint_damping"].min_step_count_between_reset == 720
    assert events["link_mass_scale"].params["asset_cfg"].body_names == MASS_SCALE_BODIES
    assert len(MASS_SCALE_BODIES) == 7 and "torso_link" not in MASS_SCALE_BODIES
    assert events["link_mass_scale"].params["ranges"] == (.7, 1.3)
    assert events["base_com"].params["ranges"] == {i: (-.02, .02) for i in range(3)}
    assert events["joint_limits"].params["limit_range"] == (-.02, .02)
