"""Loopback-only browser simulation with WASM isolation headers."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import subprocess
import sys

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=str(Path(__file__).resolve().parent),**kwargs)
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy','same-origin')
        self.send_header('Cross-Origin-Embedder-Policy','require-corp')
        self.send_header('Cache-Control','no-cache')
        super().end_headers()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8088)
    p.add_argument('--policy',type=Path,required=True,help='Trained ONNX policy')
    p.add_argument('--deploy',type=Path,required=True,help='Matching deployment YAML')
    p.add_argument('--reference',type=Path,required=True,help='Nominal reference CSV')
    p.add_argument('--nominal-motion',type=Path,help='Prepared nominal NPZ for a dynamic-reference policy')
    p.add_argument('--reference-fps',type=float,default=30.)
    p.add_argument('--stiffness',type=float,default=200.)
    p.add_argument('--rotational-stiffness',type=float,default=1.)
    args=p.parse_args()
    if args.reference.suffix.lower()!='.csv':p.error('--reference must be a CSV')
    # Validate the three supplied files before rebuilding assets or opening a port.
    for path in (args.policy,args.deploy,args.reference):
        if not path.is_file():p.error(f'Missing input: {path}')
    if args.nominal_motion:
        from reference import read_reference,validate_nominal_csv_pair
        try:validate_nominal_csv_pair(read_reference(args.reference,args.reference_fps),read_reference(args.nominal_motion))
        except ValueError as error:p.error(str(error))
    subprocess.run([sys.executable,str(Path(__file__).with_name('prepare.py')),
                    '--policy',str(args.policy.resolve()),'--deploy',str(args.deploy.resolve()),
                    '--reference',str((args.nominal_motion or args.reference).resolve()),
                    '--reference-fps',str(args.reference_fps),'--stiffness',str(args.stiffness),
                    '--rotational-stiffness',str(args.rotational_stiffness)],check=True)
    print(f'Compliant Humanoid: http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
