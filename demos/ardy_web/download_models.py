"""Download pinned ARDY text dependencies; no license acceptance or GPU use."""
import json, os
from pathlib import Path
from datetime import datetime, timezone
from huggingface_hub import snapshot_download
ROOT = Path(__file__).resolve().parents[3]
cache = ROOT/'sonic23/checkpoints/huggingface'
os.environ['HF_HUB_CACHE'] = str(cache)
repos = ['meta-llama/Meta-Llama-3-8B-Instruct',
         'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp',
         'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised']
manifest = ROOT/'sonic23/ardy_text_models.json'
DEFAULT_REVISIONS = {'meta-llama/Meta-Llama-3-8B-Instruct': '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2', 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp': '31474e395ada192e8ed1586db6be79fb3b70c9c0', 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised': 'baa8ebf04a1c2500e61288e7dad65e8ae42601a7'}
previous = json.loads(manifest.read_text()) if manifest.exists() else {}
pinned = {m['repo']:m['revision'] for m in previous.get('models',[])}
report = {'created_at':previous.get('created_at',datetime.now(timezone.utc).isoformat()), 'models':[]}
for repo in repos:
    revision = pinned.get(repo) or DEFAULT_REVISIONS[repo]
    print(f'Downloading {repo} @ {revision}', flush=True)
    path = snapshot_download(repo, revision=revision, cache_dir=str(cache),
        allow_patterns=['*.json','*.safetensors','*.model','*.txt','LICENSE*','README*'], max_workers=2)
    # Also record the pinned snapshot as the offline main cache entry used by Transformers.
    ref = cache/('models--'+repo.replace('/','--'))/'refs/main'
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(revision)
    files = [p for p in Path(path).rglob('*') if p.is_file()]
    report['models'].append(dict(repo=repo, revision=revision, path=path,
                                bytes=sum(p.stat().st_size for p in files)))
    (ROOT/'sonic23/ardy_text_models.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)

# The generation server is offline, so fetch its motion weights during setup too.
snapshot_download('nvidia/ARDY-G1-RP-25FPS-Horizon8',
    revision='fb815e8021d700af523ac32d9851b7e489fbb60b',
    local_dir=str(ROOT/'sonic23/checkpoints/ARDY-G1-RP-25FPS-Horizon8'),max_workers=2)
