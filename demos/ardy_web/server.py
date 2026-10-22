"""Loopback-only ARDY CPU generation queue and 23DOF motion preview."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import queue
import sys
import threading
import traceback
from urllib.parse import urlparse,unquote
import uuid
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"softmimic_mjlab"))
from demos.ardy_web.engine import Engine,presets
from demos.ardy_web.streaming import validate_request,generate_stream,Cancelled
WEB=Path(__file__).resolve().parent
SHARED=WEB.parent/'sonic_web'
DATA=ROOT/'sonic23/ardy_motions'
DATA.mkdir(exist_ok=True)
STATE={'phase':'Loading models','ready':False,'jobs':{}}
QUEUE=queue.Queue(maxsize=4)
LOCK=threading.Lock()
CONSTRAINT_ADAPTER=None

def phase(message):
    with LOCK: STATE['phase']=message
    print(message,flush=True)

def worker():
    global CONSTRAINT_ADAPTER
    try:
        engine=Engine(phase)
        from demos.ardy_web.constraints import ConstraintAdapter
        CONSTRAINT_ADAPTER=ConstraintAdapter(engine)
        with LOCK: STATE['ready']=True
    except Exception as error:
        phase('Model loading failed: '+str(error)); traceback.print_exc(); return
    while True:
        job=QUEUE.get()
        identifier=job['id']
        def progress(message):
            with LOCK: STATE['jobs'][identifier]['status']=message
            print(identifier,message,flush=True)
        def publish(preview):
            with LOCK: STATE['jobs'][identifier]['preview']=preview
        def cancelled():
            with LOCK: return STATE['jobs'][identifier].get('cancel_requested',False)
        try:
            result=generate_stream(engine,job,DATA/identifier,progress,publish,cancelled)
            with LOCK: STATE['jobs'][identifier].update(status='complete',result=result)
        except Cancelled as error:
            with LOCK: STATE['jobs'][identifier].update(status='cancelled',error=str(error))
        except Exception as error:
            with LOCK: STATE['jobs'][identifier].update(status='failed',error=str(error))
            (DATA/identifier).mkdir(exist_ok=True)
            (DATA/identifier/'failure.json').write_text(json.dumps(dict(job=job,error=str(error)),indent=2))
            traceback.print_exc()
        finally: QUEUE.task_done()

class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy','same-origin')
        self.send_header('Cross-Origin-Embedder-Policy','require-corp')
        self.send_header('Cache-Control','no-store')
        super().end_headers()
    def reply(self,value,status=200):
        body=json.dumps(value).encode(); self.send_response(status)
        self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        path=urlparse(self.path).path
        if path=='/builtin/stand/motion.json':
            import numpy as np
            from demos.sonic_web.reference import read_reference
            motion=read_reference(ROOT/'softmimic_mjlab/datasets/streamer/stand.csv')
            raw=np.c_[motion['root_pos'],motion['root_quat'],motion['joint_pos']]
            if not np.allclose(raw,raw[:1],atol=1e-6):return self.reply({'error':'Built-in standing reference is not constant'},400)
            return self.reply(dict(qpos=raw.tolist(),fps=motion['fps'],joint_names=motion['joint_names'],quaternion_order='wxyz'))
        if path=='/api/status':
            with LOCK: return self.reply(STATE)
        if path=='/api/presets': return self.reply(presets())
        if path=='/api/clips':
            clips=[json.loads(p.read_text()) for p in sorted(DATA.glob('*/metadata.json'),reverse=True)]
            return self.reply(clips)
        return super().do_GET()
    def translate_path(self,path):
        path=unquote(urlparse(path).path)
        if path.startswith('/shared/lib/'): base=SHARED/'lib'; relative=path[len('/shared/lib/'):]
        elif path=='/shared/js/viewer.js': return str(SHARED/'js/viewer.js')
        elif path.startswith('/assets/robot/'): base=SHARED/'assets/robot'; relative=path[len('/assets/robot/'):]
        elif path.startswith('/clips/'): base=DATA; relative=path[len('/clips/'):]
        elif path in ('/','/index.html','/app.js','/style.css','/sequence.js','/timeline-model.mjs','/constraints-ui.js','/constraints-model.mjs'): base=WEB; relative=path.lstrip('/') or 'index.html'
        else: return str(WEB/'not-found')
        target=(base/relative).resolve()
        if not target.is_relative_to(base.resolve()) or target.is_dir(): return str(WEB/'not-found')
        return str(target)
    def do_POST(self):
        path=urlparse(self.path).path
        if path not in ('/api/generate','/api/constraint-pose') and not (path.startswith('/api/jobs/') and path.endswith('/cancel')): return self.reply({'error':'Not found'},404)
        # Only same-origin JSON requests can enqueue inference (no permissive CORS).
        origin=self.headers.get('Origin')
        if origin and origin!=f'http://127.0.0.1:{self.server.server_port}': return self.reply({'error':'Origin rejected'},403)
        if path.endswith('/cancel'):
            identifier=path.split('/')[3]
            with LOCK:
                if identifier not in STATE['jobs']: return self.reply({'error':'Unknown job'},404)
                STATE['jobs'][identifier]['cancel_requested']=True
            return self.reply({'cancel_requested':True})
        try:
            length=int(self.headers.get('Content-Length',0))
            if not 0<length<=120000 or self.headers.get('Content-Type','').split(';')[0]!='application/json': raise ValueError('Expected small JSON request')
            body=json.loads(self.rfile.read(length))
            if not isinstance(body,dict):raise ValueError('Expected a JSON object')
            if path=='/api/constraint-pose':
                if CONSTRAINT_ADAPTER is None:return self.reply({'error':'Model not ready'},503)
                from demos.ardy_web.constraints import validate_constraints
                item=validate_constraints([dict(type='fullbody',time=0,pose=body.get('pose'))],4)[0]
                return self.reply(CONSTRAINT_ADAPTER.anchors(item['pose']))
            request=validate_request(body)
            with LOCK:
                if not STATE['ready']: return self.reply({'error':STATE['phase']},503)
                job=dict(id=uuid.uuid4().hex,**request,status='queued')
                if QUEUE.full(): return self.reply({'error':'Queue is full'},429)
                STATE['jobs'][job['id']]=job.copy(); QUEUE.put_nowait(job)
            return self.reply(job,202)
        except (ValueError,TypeError,json.JSONDecodeError) as error: return self.reply({'error':str(error)},400)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8089); args=parser.parse_args()
    threading.Thread(target=worker,daemon=True).start()
    print(f'ARDY 23DOF motion preview: http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
