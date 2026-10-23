"""Published inputs must work without the author's ignored datasets or logs."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path('deploy/robots/g1_23dof/config/policy/compliant_mimic/stand')


def test_relocated_bundle_and_mismatch_rejection(tmp_path):
    files = [Path('deploy/check_bundle.py'), Path('demos/sonic_web/reference.py'),
             Path('demos/ardy_web/robot_contract.json')]
    files += [BUNDLE / name for name in ['manifest.json','controller_config.yaml',
              'exported/policy.onnx','params/deploy.yaml','params/reference.csv']]
    for name in files:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    command = [sys.executable, str(tmp_path/'deploy/check_bundle.py')]
    good = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    assert 'PASS' in good.stdout
    with (tmp_path/BUNDLE/'params/reference.csv').open('a') as stream:
        stream.write('\n')
    bad = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert bad.returncode != 0 and 'Mismatched bundle file' in bad.stderr
