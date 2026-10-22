"""Create an isolated browser ONNX parity test; never installs the demo policy."""
import argparse,json,shutil
from pathlib import Path
import numpy as np,yaml
from onnx.reference import ReferenceEvaluator
p=argparse.ArgumentParser();p.add_argument('--deploy',type=Path,required=True);p.add_argument('--policy',type=Path,required=True);a=p.parse_args()
here=Path(__file__).resolve().parent; out=here/'diagnostic';out.mkdir(exist_ok=True)
c=yaml.safe_load(a.deploy.read_text());c.update(effort_limits=json.loads((here/'assets/robot/contract.json').read_text())['effort_limits'],stiffness_command=100,rotational_stiffness_command=10)
sess=ReferenceEvaluator(str(a.policy))
x=np.random.default_rng(42).normal(size=(1,327)).astype(np.float32)
y=sess.run(None,{sess.input_names[0]:x})[0]
(out/'fixture.json').write_text(json.dumps(dict(config=c,input=x[0].tolist(),expected=y[0].tolist())))
shutil.copy2(a.policy,out/'policy.onnx')
(out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Diagnostic only — ONNX parity</title><script src="../lib/ort.min.js"></script><h1>Diagnostic only — not a trained controller</h1><pre id="result">Testing…</pre><script type="module">
import {SoftMimicPolicy} from '../js/policy.js';
try {const f=await fetch('fixture.json').then(r=>r.json());const p=new SoftMimicPolicy();await p.load(f.config,'policy.onnx');p.observation=Float32Array.from(f.input);const y=await p.infer();const error=Math.max(...y.map((v,i)=>Math.abs(v-f.expected[i])));if(error>1e-4)throw Error('Max error '+error);document.getElementById('result').textContent='PASS: browser WASM ONNX matches CPU ONNX ReferenceEvaluator; max absolute error '+error;}catch(e){document.getElementById('result').textContent='FAIL: '+e.stack;}
</script>''')
print('Open /diagnostic/ for CPU ONNX parity only. No main policy installed.')
