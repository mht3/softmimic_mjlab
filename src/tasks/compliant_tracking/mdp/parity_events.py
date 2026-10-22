"""Parameter randomization matching the local Isaac SoftMimic configuration.

Contact friction/restitution are deliberately excluded: PhysX material
coefficients do not have a one-to-one MuJoCo contact parameter mapping.
"""
from __future__ import annotations

import torch

from mjlab.actuator import BuiltinPositionActuator
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg


MASS_SCALE_BODIES = (
    "pelvis", "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link",
    "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link",
)


def _env_ids(env, env_ids):
    return (torch.arange(env.num_envs, device=env.device, dtype=torch.long)
            if env_ids is None else env_ids.to(device=env.device, dtype=torch.long))


@requires_model_fields("actuator_biasprm")
def additive_actuator_damping(
    env, env_ids, damping_range=(0.0, 2.0), asset_cfg=SceneEntityCfg("robot"),
):
    """Sample kd = default kd + U(range); never accumulate across resets.

    The event manager supplies the original 720-step reset gating through
    EventTermCfg.min_step_count_between_reset (see the factory below).
    """
    ids = _env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    actuators = asset.actuators
    if not isinstance(asset_cfg.actuator_ids, slice):
        actuators = [actuators[i] for i in asset_cfg.actuator_ids]
    else:
        actuators = actuators[asset_cfg.actuator_ids]
    default = env.sim.get_default_field("actuator_biasprm")
    for actuator in actuators:
        if not isinstance(actuator, BuiltinPositionActuator):
            raise TypeError("SoftMimic damping randomization requires built-in position actuators")
        ctrl_ids = actuator.global_ctrl_ids
        noise = torch.empty((len(ids), len(ctrl_ids)), device=env.device).uniform_(*damping_range)
        env.sim.model.actuator_biasprm[ids[:, None], ctrl_ids, 2] = (
            default[ctrl_ids, 2] - noise
        )


@requires_model_fields("jnt_range")
def additive_joint_limits(
    env, env_ids, limit_range=(-0.02, 0.02), asset_cfg=SceneEntityCfg("robot"),
):
    """Independently perturb each limit and update hard/soft limit buffers."""
    dr.joint_limits(env, env_ids, ranges={0: limit_range, 1: limit_range},
                    asset_cfg=asset_cfg, operation="add", distribution="uniform")
    ids = _env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    local_ids = torch.arange(asset.num_joints, device=env.device)[asset_cfg.joint_ids]
    global_ids = asset.indexing.joint_ids[local_ids]
    limits = env.sim.model.jnt_range[ids[:, None], global_ids]
    if torch.any(limits[..., 0] > limits[..., 1]):
        raise ValueError("Randomized lower joint limit exceeds upper joint limit")
    asset.data.joint_pos_limits[ids[:, None], local_ids] = limits
    mean = limits.mean(dim=-1)
    half = (limits[..., 1] - limits[..., 0]) * .5 * asset.cfg.articulation.soft_joint_pos_limit_factor
    asset.data.soft_joint_pos_limits[ids[:, None], local_ids] = torch.stack((mean-half, mean+half), dim=-1)
    # IsaacLab's write_joint_position_limit_to_sim also clamps default offsets.
    asset.data.default_joint_pos[ids[:, None], local_ids] = torch.clamp(
        asset.data.default_joint_pos[ids[:, None], local_ids], limits[..., 0], limits[..., 1]
    )


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def body_mass_with_inertia(
    env, env_ids, ranges, operation, asset_cfg=SceneEntityCfg("robot"),
):
    """Uniform mass perturbation with IsaacLab's default inertia recomputation.

    Both additive torso payload and multiplicative link mass use default
    inertia times new_mass/default_mass, preserving inertial-frame orientation.
    """
    if operation not in ("add", "scale"):
        raise ValueError("Expected mass operation 'add' or 'scale'")
    ids = _env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
    default_mass = env.sim.get_default_field("body_mass")[body_ids]
    default_inertia = env.sim.get_default_field("body_inertia")[body_ids]
    sample = torch.empty((len(ids), len(body_ids)), device=env.device).uniform_(*ranges)
    mass = default_mass + sample if operation == "add" else default_mass * sample
    if torch.any(default_mass <= 0) or torch.any(mass <= 0):
        raise ValueError("Mass randomization requires positive original and sampled masses")
    env.sim.model.body_mass[ids[:, None], body_ids] = mass
    env.sim.model.body_inertia[ids[:, None], body_ids] = (
        default_inertia * (mass/default_mass).unsqueeze(-1)
    )


def make_softmimic_parameter_events() -> dict[str, EventTermCfg]:
    """Fresh event declarations; caller retains its force/reset/contact events."""
    return {
        "payload_mass": EventTermCfg(mode="startup", func=body_mass_with_inertia, params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            "ranges": (-2., 2.), "operation": "add",
        }),
        "link_mass_scale": EventTermCfg(mode="startup", func=body_mass_with_inertia, params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MASS_SCALE_BODIES),
            "ranges": (.7, 1.3), "operation": "scale",
        }),
        "base_com": EventTermCfg(mode="startup", func=dr.body_com_offset, params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            "operation": "add", "ranges": {i: (-.02, .02) for i in range(3)},
        }),
        "joint_damping": EventTermCfg(mode="reset", func=additive_actuator_damping,
            min_step_count_between_reset=720, params={"damping_range": (0., 2.)}),
        "joint_armature": EventTermCfg(mode="startup", func=dr.joint_armature, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "add", "ranges": (.01, .1),
        }),
        "joint_friction": EventTermCfg(mode="startup", func=dr.joint_friction, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "add", "ranges": (0., .01),
        }),
        "joint_limits": EventTermCfg(mode="startup", func=additive_joint_limits, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "limit_range": (-.02, .02),
        }),
    }
