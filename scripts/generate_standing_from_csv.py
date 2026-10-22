"""Prepare a streamer CSV, run direct 23DOF IK, convert and fully audit before installation."""
from pathlib import Path
import argparse,hashlib,json,os,sys,tempfile
from prepare_streamer_reference import prepare,REPO
from generate_compliant_23dof import augment,to_npz,_run
from check_compliant_dataset_ready import check

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--name',default='streamer_stand');p.add_argument('--num-files',nargs=3,type=int,default=[40,40,5]);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if not a.name.replace('_','').replace('-','').isalnum() or min(a.num_files)<1:p.error('Use a simple name and positive mode counts')
    root=REPO/'src/assets/compliant_motions/g1_23dof';destination=root/a.name
    if destination.exists():p.error(f'{destination} already exists; choose a new name. Existing datasets are preserved.')
    root.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.streamer-',dir=root))
    reference=prepare(a.input,stage/f'{a.name}.csv',standing=True,duration=30)
    modes=['forcefield','collision-emulator','zero-wrench']
    for mode,count in zip(modes,a.num_files):
        folder=augment(a.name,mode,reference,count,a.seed,stage)
        to_npz(folder,30,50,'cpu')
    generated=stage/a.name
    _run([sys.executable,REPO/'scripts/audit_compliant_dataset.py',generated,'--stride','1','--input-fps','30','--output',generated/'audit.json'],cwd=REPO)
    count=check(generated)
    if count!=sum(a.num_files):raise ValueError('Unexpected number of audited files')
    reference_dir=REPO/'datasets/reference23';reference_dir.mkdir(parents=True,exist_ok=True)
    saved=reference_dir/reference.name
    if saved.exists():raise FileExistsError(saved)
    reference.replace(saved);reference.with_suffix('.json').replace(saved.with_suffix('.json'))
    (generated/'generation.json').write_text(json.dumps(dict(motion=a.name,modes=modes,num_files=a.num_files,seed=a.seed,input_fps=30,output_fps=50,reference=str(saved),reference_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),streamer_csv=str(a.input.resolve()),streamer_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),audited_files=count),indent=2)+'\n')
    generated.rename(destination);stage.rmdir()
    print(f'PASS: {count} audited pairs installed at {destination}\nNominal reference: {saved}',flush=True)
if __name__=='__main__':main()
