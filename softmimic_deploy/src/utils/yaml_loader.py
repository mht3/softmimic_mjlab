"""YAML loading helpers with a restricted but compatible constructor set."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


class SoftMimicLoader(yaml.SafeLoader):
    """Safe loader extended to support legacy IsaacLab tags."""


def _construct_python_tuple(loader: SoftMimicLoader, node: yaml.SequenceNode) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node))


def _construct_python_slice(loader: SoftMimicLoader, node: yaml.SequenceNode) -> slice:
    start, stop, step = loader.construct_sequence(node)
    return slice(start, stop, step)


SoftMimicLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)
SoftMimicLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", _construct_python_slice)


def load_yaml(path: str | Path) -> Any:
    """Load a YAML file using the hardened loader."""
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=SoftMimicLoader)


def load_yaml_stream(stream: Iterable[str]) -> Any:
    return yaml.load(stream, Loader=SoftMimicLoader)
