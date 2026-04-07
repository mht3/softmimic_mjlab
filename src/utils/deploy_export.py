"""Export deploy.yaml from a trained ManagerBasedRlEnv."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from mjlab.envs import ManagerBasedRlEnv

from src.assets.robots.deploy_entity_cfg import DeployEntityCfg


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
    if isinstance(value, float):
        if abs(value) < 1.0:
            return float(f"{value:.2f}")
        return float(f"{value:.1f}")
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
    for train_name, obs_cfg in zip(obs_names, obs_cfgs, strict=True):
        params = dict(obs_cfg.params)
        export_name = _obs_export_name(train_name, params)
        export_params = _obs_export_params(train_name, params)

        obs_sample = obs_cfg.func(env, **params)
        obs_dim = int(obs_sample.shape[1]) if obs_sample.ndim > 1 else int(obs_sample.shape[0])

        scale = obs_cfg.scale
        if scale is None:
            scale_list = [1.0] * obs_dim
        else:
            plain = _to_plain_value(scale)
            scale_list = plain if isinstance(plain, list) else [float(plain)] * obs_dim

        history_length = int(obs_cfg.history_length) if obs_cfg.history_length else 1
        cfg["observations"][export_name] = {
            "params": _to_plain_value(export_params),
            "clip": _to_plain_value(obs_cfg.clip),
            "scale": _to_plain_value(scale_list),
            "history_length": history_length,
        }

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
