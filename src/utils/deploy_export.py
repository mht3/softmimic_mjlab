"""Export deploy.yaml from a trained ManagerBasedRlEnv."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.assets.robots.deploy_entity_cfg import DeployEntityCfg


_REFERENCE_OBSERVATIONS = {
    "reference_dof_pos", "reference_xy_vel", "reference_yaw_vel",
    "reference_root_height", "reference_gravity_vector", "reference_foot_contact",
}
_SUPPORTED_DEPLOY_OBSERVATIONS = _REFERENCE_OBSERVATIONS | {
    "base_ang_vel", "projected_gravity", "joint_pos", "joint_pos_rel",
    "joint_vel_rel", "last_action", "velocity_commands", "gait_phase",
    "motion_command", "motion_anchor_ori_b", "desired_stiffness_log",
    "desired_rotational_stiffness_log",
}


def _constant_nominal_reference(command) -> dict[str, list[float]]:
    """Export actual reference channels only if constant over the entire dataset.

    This is a standing deployment contract, not an approximation of a changing
    trajectory. Dynamic nominal-reference replay needs a separate runtime port.
    """
    motion = command.motion
    if not motion.has_nominal_reference:
        raise ValueError("Deployment requires the complete nominal reference arrays")
    if command.cfg.body_names[0] != "pelvis":
        raise ValueError("Nominal deployment reference expects pelvis as body slot zero")
    quat = motion.nominal_body_quat_w[:, 0]
    gravity = torch.zeros_like(quat[:, :3]); gravity[:, 2] = -1.
    values = {
        "reference_dof_pos": motion.nominal_joint_pos,
        "reference_xy_vel": quat_apply_inverse(quat, motion.nominal_body_lin_vel_w[:, 0])[:, :2],
        "reference_yaw_vel": quat_apply_inverse(quat, motion.nominal_body_ang_vel_w[:, 0])[:, 2:3],
        "reference_root_height": motion.nominal_body_pos_w[:, 0, 2:3],
        "reference_gravity_vector": quat_apply_inverse(quat, gravity),
        "reference_foot_contact": motion.nominal_foot_contacts,
    }
    result = {}
    for name, value in values.items():
        if value.ndim != 2 or len(value) == 0 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid nominal deployment channel {name}")
        if not torch.allclose(value, value[:1].expand_as(value), atol=1.e-5, rtol=0.):
            raise ValueError(
                f"Cannot export changing nominal channel {name}: C++ deployment currently "
                "supports constant standing references only; dynamic nominal replay is unimplemented"
            )
        result[name] = value[0].detach().cpu().tolist()
    return result


class _InlineListDumper(yaml.SafeDumper):
    pass


def _represent_list_inline(dumper: _InlineListDumper, data: list[Any]) -> yaml.Node:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_InlineListDumper.add_representer(list, _represent_list_inline)


def _to_plain_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.item())
        return _to_plain_value(value.detach().cpu().tolist())
    if isinstance(value, tuple):
        return [_to_plain_value(v) for v in value]
    if isinstance(value, list):
        return [_to_plain_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_plain_value(v) for k, v in value.items()}
    return value


def _obs_export_name(train_name: str, params: dict[str, Any]) -> str:
    if train_name == "command":
        cmd_name = params.get("command_name")
        if cmd_name == "twist":
            return "velocity_commands"
        if cmd_name == "motion":
            return "motion_command"
    if train_name == "phase":
        return "gait_phase"
    if train_name == "joint_pos":
        return "joint_pos_rel"
    if train_name == "joint_vel":
        return "joint_vel_rel"
    if train_name == "actions":
        return "last_action"
    return train_name


def _obs_export_params(train_name: str, params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    if train_name == "command" and out.get("command_name") == "twist":
        out["command_name"] = "base_velocity"
    if train_name == "phase":
        out = {"period": out.get("period", 0.6)}
    if train_name in {"joint_pos", "joint_vel", "actions"}:
        out = {}
    return out


def export_deploy_cfg(env: ManagerBasedRlEnv, log_dir: Path) -> None:
    """Export deploy.yaml from a live training environment.

    The robot's EntityCfg must be a DeployEntityCfg carrying hardware constants
    (joint_ids_map, hardware_stiffness, hardware_damping). If it is not, the
    export is skipped with a warning.
    """
    robot_cfg = env.cfg.scene.entities.get("robot")
    if not isinstance(robot_cfg, DeployEntityCfg):
        print(
            "[WARNING] export_deploy_cfg: robot EntityCfg is not a DeployEntityCfg "
            "— deploy.yaml not exported."
        )
        return

    output_path = Path(log_dir) / "params" / "deploy.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    robot = env.scene["robot"]
    default_joint_pos = robot.data.default_joint_pos[0].detach().cpu().tolist()

    cfg: dict[str, Any] = {
        "joint_ids_map": robot_cfg.joint_ids_map,
        "step_dt": float(env.step_dt),
        "stiffness": robot_cfg.hardware_stiffness,
        "damping": robot_cfg.hardware_damping,
        "default_joint_pos": default_joint_pos,
    }

    # Commands
    commands: dict[str, Any] = {}
    if "twist" in env.cfg.commands:
        ranges_cfg = env.cfg.commands["twist"].ranges
        commands["base_velocity"] = {
            "ranges": {
                "lin_vel_x": list(ranges_cfg.lin_vel_x),
                "lin_vel_y": list(ranges_cfg.lin_vel_y),
                "ang_vel_z": list(ranges_cfg.ang_vel_z),
                "heading": None,
            }
        }
    cfg["commands"] = commands

    # Actions
    cfg["actions"] = {}
    for term in env.action_manager._terms.values():
        action_dim = term.action_dim
        if action_dim == 0:
            # Sim-only terms (e.g. the forcefield wrench) are not part of the
            # deployed action interface.
            continue
        scale = term._scale
        offset = term._offset
        scale_list = (
            scale[0].detach().cpu().tolist()
            if isinstance(scale, torch.Tensor)
            else [float(scale)] * action_dim
        )
        offset_list = (
            offset[0].detach().cpu().tolist()
            if isinstance(offset, torch.Tensor)
            else [float(offset)] * action_dim
        )
        cfg["actions"][type(term).__name__] = {
            "clip": _to_plain_value(getattr(term.cfg, "clip", None)),
            "joint_names": list(getattr(term.cfg, "actuator_names", (".*",))),
            "scale": scale_list,
            "offset": offset_list,
            "joint_ids": None,
        }

    # Observations — prefer "actor" group (mjlab), fall back to "policy" (Isaac Lab style)
    obs_group = "actor" if "actor" in env.observation_manager.active_terms else "policy"
    obs_names = env.observation_manager.active_terms[obs_group]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs[obs_group]
    cfg["observations"] = {}
    references = None
    if any(name in _REFERENCE_OBSERVATIONS for name in obs_names):
        references = _constant_nominal_reference(env.command_manager.get_term("motion"))
        cfg["reference_mode"] = "constant_nominal"
    observation_dim = 0
    for train_name, obs_cfg in zip(obs_names, obs_cfgs, strict=True):
        params = dict(obs_cfg.params)
        export_name = _obs_export_name(train_name, params)
        export_params = _obs_export_params(train_name, params)
        if export_name not in _SUPPORTED_DEPLOY_OBSERVATIONS:
            raise ValueError(f"Observation {export_name!r} has no supported C++ deployment implementation")
        if export_name in cfg["observations"]:
            raise ValueError(f"Duplicate exported observation name {export_name!r}")

        obs_sample = obs_cfg.func(env, **params)
        obs_dim = int(obs_sample.shape[1]) if obs_sample.ndim > 1 else int(obs_sample.shape[0])
        if train_name in _REFERENCE_OBSERVATIONS:
            if params.get("command_name") != "motion" or not params.get("nominal", True):
                raise ValueError(f"{train_name} must use the nominal motion reference")
            constant = references[train_name]
            expected = torch.tensor(constant, device=obs_sample.device, dtype=obs_sample.dtype)
            if obs_dim != len(constant) or not torch.allclose(obs_sample, expected.expand_as(obs_sample), atol=1.e-5, rtol=0.):
                raise ValueError(f"Training/deployment reference mismatch for {train_name}")
            export_params = {"value": constant, "reference_mode": "constant_nominal"}

        scale = obs_cfg.scale
        if scale is None:
            scale_list = [1.0] * obs_dim
        else:
            plain = _to_plain_value(scale)
            scale_list = plain if isinstance(plain, list) else [float(plain)] * obs_dim

        history_length = int(obs_cfg.history_length) if obs_cfg.history_length else 1
        observation_dim += obs_dim * history_length
        cfg["observations"][export_name] = {
            "params": _to_plain_value(export_params),
            "clip": _to_plain_value(obs_cfg.clip),
            "scale": _to_plain_value(scale_list),
            "history_length": history_length,
        }
    cfg["observation_dim"] = observation_dim

    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            _to_plain_value(cfg),
            f,
            Dumper=_InlineListDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=False,
            width=120,
        )

    print(f"[INFO] deploy.yaml exported to: {output_path}")
