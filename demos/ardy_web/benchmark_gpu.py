"""One bounded motion-model GPU probe; cached text encoding stays on CPU."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"softmimic_mjlab"))
import torch
from demos.ardy_web.engine import Engine
output=ROOT/'sonic23/ardy_gpu_probe'
if output.exists():raise SystemExit('Probe already exists; refusing another run')
free,total=torch.cuda.mem_get_info()
if free<8*1024**3:raise SystemExit('Insufficient GPU headroom; probe skipped')
# Hard allocator ceiling for this process only; does not change training.
torch.cuda.set_per_process_memory_fraction(.10)
torch.cuda.reset_peak_memory_stats()
engine=Engine(print,device='cuda')
result=engine.generate('A person is standing.',4,42,output,print)
torch.cuda.synchronize()
report=dict(scope='ARDY motion model on GPU; cached CPU LLM2Vec embedding. Text encoder was NOT placed on GPU.',
            device=torch.cuda.get_device_name(),free_before_bytes=free,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            generation_seconds=result['generation_seconds'],retarget_seconds=result['retarget_seconds'],
            motion_model_disk_bytes=sum(p.stat().st_size for p in (ROOT/'sonic23/checkpoints/ARDY-G1-RP-25FPS-Horizon8').rglob('*') if p.is_file()),
            text_dependencies_disk_bytes=sum(m['bytes'] for m in json.loads((ROOT/'sonic23/ardy_text_models.json').read_text())['models']))
(output/'gpu_memory_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
