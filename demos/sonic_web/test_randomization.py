"""Check native/WASM physical mutations and the saved training ranges on CPU."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import mujoco
import numpy as np
import yaml
from domain_randomization import apply_draw, MASS_BODIES

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--assets',type=Path,required=True)
p.add_argument('--training-config',type=Path,required=True,help='Saved run params/env.yaml')
a=p.parse_args()
# BaseLoader reads only YAML data, never executes saved Python tags.
c=yaml.load(a.training_config.read_text(),Loader=yaml.BaseLoader)
events=c['events']
for event,key,expected in [('payload_mass','ranges',[-2,2]),('link_mass_scale','ranges',[.7,1.3]),
                         ('joint_damping','damping_range',[0,2]),('joint_armature','ranges',[.01,.1]),
                         ('joint_friction','ranges',[0,.01]),('joint_limits','limit_range',[-.02,.02]),
                         ('foot_friction','ranges',[.5,2])]:
    assert [float(x) for x in events[event]['params'][key]]==expected,event
assert events['foot_friction']['params']['shared_random']=='true'
assert set(events['link_mass_scale']['params']['asset_cfg']['body_names'])==set(MASS_BODIES)
assert events['base_com']['params']['operation']=='add'
for axis,limits in events['base_com']['params']['ranges'].items():
    assert [float(x) for x in limits]==[-.02,.02]
for channel,bound in [('base_ang_vel',.2),('projected_gravity',.01),('joint_pos',.01),('joint_vel',1.5)]:
    noise=c['observations']['actor']['terms'][channel]['noise']
    assert float(noise['n_min'])==-bound and float(noise['n_max'])==bound,channel
with tempfile.TemporaryDirectory() as tmp:
    output=Path(tmp)/'wasm.json'
    subprocess.run(['node',str(Path(__file__).with_suffix('.mjs')),str(a.assets),str(output)],check=True)
    model=mujoco.MjModel.from_xml_path(str((a.assets/'robot/scene.xml').resolve()))
    data=mujoco.MjData(model);apply_draw(model);mujoco.mj_setConst(model,data)
    for key,values in json.loads(output.read_text()).items():
        np.testing.assert_allclose(getattr(model,key).reshape(-1),values,rtol=1e-12,atol=1e-12,err_msg=key)
print('PASS: saved training ranges and native/WASM randomized model parity')
