"""Loopback-only browser simulation with WASM isolation headers."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from functools import partial
import argparse
import errno
import subprocess
import sys
from contact_model import add_contact_arguments

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,assets_dir=None,**kwargs):
        self.assets_dir=assets_dir
        super().__init__(*args,directory=str(Path(__file__).resolve().parent),**kwargs)
    def translate_path(self,path):
        translated=Path(super().translate_path(path))
        if self.assets_dir is not None:
            try:
                relative=translated.relative_to(Path(__file__).resolve().parent/'assets')
            except ValueError:
                pass
            else:
                return str(self.assets_dir/relative)
        return str(translated)
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy','same-origin')
        self.send_header('Cross-Origin-Embedder-Policy','require-corp')
        self.send_header('Cache-Control','no-cache')
        super().end_headers()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8088)
    p.add_argument('--assets-dir',type=Path,default=Path(__file__).resolve().parent/'assets',help='Use a separate directory for each simultaneous policy preview')
    p.add_argument('--policy',type=Path,required=True,help='Trained ONNX policy')
    p.add_argument('--deploy',type=Path,required=True,help='Matching deployment YAML')
    p.add_argument('--reference',type=Path,required=True,help='Nominal reference CSV')
    p.add_argument('--nominal-motion',type=Path,help='Prepared nominal NPZ for a dynamic-reference policy')
    p.add_argument('--reference-fps',type=float,default=30.)
    p.add_argument('--stiffness',type=float,default=200.)
    p.add_argument('--rotational-stiffness',type=float,default=1.)
    add_contact_arguments(p)
    args=p.parse_args()
    if args.reference.suffix.lower()!='.csv':p.error('--reference must be a CSV')
    # Validate the three supplied files before rebuilding assets or opening a port.
    for path in (args.policy,args.deploy,args.reference):
        if not path.is_file():p.error(f'Missing input: {path}')
    if args.nominal_motion:
        from reference import read_reference,validate_nominal_csv_pair
        try:validate_nominal_csv_pair(read_reference(args.reference,args.reference_fps),read_reference(args.nominal_motion))
        except ValueError as error:p.error(str(error))
    # Reserve the port before modifying assets shared with a running viewer.
    try:
        server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,assets_dir=args.assets_dir.resolve()))
    except OSError as error:
        if error.errno==errno.EADDRINUSE:
            p.exit(2,f'Port {args.port} is already in use. No assets were changed.\n'
                   f'If the demo is already running, open http://127.0.0.1:{args.port}.\n'
                   'To restart it, stop the existing demo process first.\n')
        raise
    with server:
        subprocess.run([sys.executable,str(Path(__file__).with_name('prepare.py')),
                        '--assets-dir',str(args.assets_dir.resolve()),
                        '--policy',str(args.policy.resolve()),'--deploy',str(args.deploy.resolve()),
                        '--reference',str((args.nominal_motion or args.reference).resolve()),
                        '--reference-fps',str(args.reference_fps),'--stiffness',str(args.stiffness),
                        '--rotational-stiffness',str(args.rotational_stiffness),
                        '--contact-profile',args.contact_profile,'--body-friction',str(args.body_friction)],check=True)
        print(f'Compliant Humanoid: http://127.0.0.1:{args.port}',flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
