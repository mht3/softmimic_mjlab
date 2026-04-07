from dataclasses import dataclass, field

from mjlab.entity import EntityCfg


@dataclass
class DeployEntityCfg(EntityCfg):
    """EntityCfg subclass carrying hardware deploy constants."""

    joint_ids_map: list[int] = field(default_factory=list)
    hardware_stiffness: list[float] = field(default_factory=list)
    hardware_damping: list[float] = field(default_factory=list)
