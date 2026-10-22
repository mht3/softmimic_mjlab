"""Generation entry points must not require caller-supplied import paths."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_generator_children_use_checkout_and_preserve_environment(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "generation_import_test", REPO_ROOT / "scripts/generate_compliant_23dof.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: calls.append(kw))
    env = {"PYTHONPATH": "/caller/imports", "CUDA_VISIBLE_DEVICES": ""}
    module._run([sys.executable, "child.py"], cwd="/tmp", env=env)
    child = calls[0]["env"]
    assert child["PYTHONPATH"].split(os.pathsep) == [
        str(REPO_ROOT), str(REPO_ROOT / "algorithms/rsl_rl"), "/caller/imports"
    ]
    assert child["CUDA_VISIBLE_DEVICES"] == ""
    assert env["PYTHONPATH"] == "/caller/imports"


@pytest.mark.parametrize("script", ["generate_compliant_23dof.py", "compliant_csv_to_npz.py"])
def test_generation_entrypoint_help_without_pythonpath(script, tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--input-fps" in result.stdout
