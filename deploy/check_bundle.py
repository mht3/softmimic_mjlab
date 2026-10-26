"""Offline integrity and interface check; never opens DDS or controls a robot."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import onnx
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'demos/sonic_web'))
from reference import read_reference, validate_reference


def check(bundle, default_config=None):
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    required = ('exported/policy.onnx', 'params/deploy.yaml', 'params/reference.csv', 'controller_config.yaml')
    if not set(required).issubset(manifest['files']):
        raise ValueError('Manifest is missing required bundle hashes')
    for name in required:
        actual = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        if actual != manifest['files'][name]:
            raise ValueError(f'Mismatched bundle file: {name}')
    cfg = yaml.safe_load((bundle / 'params/deploy.yaml').read_text())
    validate_reference(read_reference(bundle / 'params/reference.csv'), cfg)
    if cfg['joint_ids_map'] != [*range(13),15,16,17,18,19,22,23,24,25,26] or cfg['step_dt'] != .02:
        raise ValueError('Incorrect hardware mapping/control rate')
    model = onnx.load(bundle / 'exported/policy.onnx')
    onnx.checker.check_model(model)
    props = {p.key: p.value for p in model.metadata_props}
    if int(props.get('checkpoint_iteration', -1)) != manifest['checkpoint_iteration']:
        raise ValueError('ONNX checkpoint metadata differs from manifest')
    if props.get('checkpoint_sha256') != manifest['checkpoint_sha256']:
        raise ValueError('ONNX checkpoint hash differs from manifest')
    dims = lambda value: [d.dim_value for d in value.type.tensor_type.shape.dim]
    if len(model.graph.input) != 1 or len(model.graph.output) != 1 or dims(model.graph.input[0]) != [1,327] or dims(model.graph.output[0]) != [1,23]:
        raise ValueError('Incorrect ONNX interface')
    fsm = yaml.safe_load((bundle / 'controller_config.yaml').read_text())['FSM']
    if set(fsm['_']) != {'Passive','FixStand','Mimic_Compliant_Stand'}:
        raise ValueError('Expected standing-only FSM')
    pose = np.array(fsm['FixStand']['qs'][-1])[cfg['joint_ids_map']]
    np.testing.assert_allclose(pose, cfg['observations']['reference_dof_pos']['params']['value'], atol=1e-6)
    for key in ('policy_dir', 'reference_csv'):
        if Path(fsm['Mimic_Compliant_Stand'][key]).is_absolute():
            raise ValueError('Published controller paths must be portable')
    state = fsm['Mimic_Compliant_Stand']
    if state.get('limit_angle') != 1.25 or state.get('policy_watchdog_ms') != 100:
        raise ValueError('Unexpected fall cutoff or policy watchdog')
    if not state.get('loop') or cfg.get('reference_mode') != 'constant_nominal':
        raise ValueError('Published standing bundle must use a looping constant reference')
    if default_config is not None:
        default_config = Path(default_config)
        if yaml.safe_load(default_config.read_text()) != {'FSM': fsm}:
            raise ValueError('Default controller configuration differs from the matched bundle')
        project = default_config.parent.parent.resolve()
        if (project/state['policy_dir']).resolve() != bundle.resolve():
            raise ValueError('Default controller policy path does not resolve to this bundle')
        if (project/state['reference_csv']).resolve() != (bundle/'params/reference.csv').resolve():
            raise ValueError('Default controller reference path does not resolve to this CSV')
    print(f'PASS: matched checkpoint {manifest["checkpoint_iteration"]}; 327 observations, 23 actions, 50 Hz; nominal CSV and FixStand agree. No hardware was contacted.')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=ROOT / 'deploy/robots/g1_23dof/config/policy/compliant_mimic/stand')
    p.add_argument('--default-config', type=Path, help='Also verify the active controller config and portable paths')
    a = p.parse_args()
    default = ROOT/'deploy/robots/g1_23dof/config/config.yaml'
    if a.default_config is None and a.bundle.resolve() == (default.parent/'policy/compliant_mimic/stand').resolve():
        a.default_config = default
    check(a.bundle, a.default_config)
