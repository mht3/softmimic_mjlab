"""Compare wrist motion under identical CPU standing/noise conditions."""
import argparse,json
from pathlib import Path
import onnx
from evaluate_browser_actuation import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundles',type=Path,nargs='+',required=True)
    p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 0<a.seconds<=300:p.error('--seconds must be in (0,300]')
    rows=[]
    for bundle in a.bundles:
        model=onnx.load(bundle/'exported/policy.onnx')
        metadata={item.key:item.value for item in model.metadata_props}
        for noise,seed in [(False,a.seeds[0])]+[(True,s) for s in a.seeds]:
            row=run(bundle,a.seconds,'implicit',sensor_noise=noise,seed=seed)
            row['iteration']=int(metadata['checkpoint_iteration'])
            rows.append(row);print(json.dumps(row),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
