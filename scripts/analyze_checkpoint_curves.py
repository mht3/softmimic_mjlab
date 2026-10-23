"""Plot downloaded W&B curves and summarize trailing checkpoint neighborhoods."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    history=json.loads(a.history.read_text());rows=sorted(history['rows'],key=lambda r:r['_step'])
    metrics=[('Train/mean_reward','Training reward',1),
             ('Compliance/forcefield/body_tracking_error_m','Forcefield body error (cm)',100),
             ('Compliance/zero-wrench/body_tracking_error_m','Unforced body error (cm)',100),
             ('Episode_Reward/action_rate_l2','Action-rate penalty magnitude',-1),
             ('Episode_Reward/joint_vel_l2','Joint-velocity penalty magnitude',-1),
             ('Train/mean_episode_length','Episode length (steps)',1)]
    checkpoints=list(range(15000,20000,500))+[19999]
    summaries=[]
    for i in checkpoints:
        window=[r for r in rows if i-249<=r['_step']<=i]
        values={key:float(np.mean([r[key] for r in window if key in r])) for key,_,_ in metrics}
        for mode in ('forcefield','zero-wrench'):
            count=f'Compliance/{mode}/completed_episodes_this_update'
            fraction=f'Compliance/{mode}/timeout_fraction_this_update'
            usable=[r for r in window if count in r and fraction in r and r[count]>0]
            total=sum(r[count] for r in usable)
            values[f'{mode}_completed_episode_timeout_fraction']=sum(r[count]*r[fraction] for r in usable)/total if total else None
        summaries.append(dict(iteration=i,window=[i-249,i],samples=len(window),metrics=values))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'wandb_checkpoint_summary.json').write_text(json.dumps(dict(run_url=history['url'],window_updates=250,checkpoints=summaries),indent=2)+'\n')
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,(key,title,scale) in zip(axs.flat,metrics):
        points=[r for r in rows if key in r]
        x=np.array([r['_step'] for r in points]);y=np.array([r[key]*scale for r in points])
        ax.plot(x,y,color='#6da2c6',alpha=.22,lw=.6)
        smoothed=np.convolve(y,np.ones(250)/250,mode='valid')
        ax.plot(x[249:],smoothed,color='#075c96',lw=1.8)
        ax.scatter(checkpoints,[s['metrics'][key]*scale for s in summaries],color='#d46a15',s=24,zorder=3)
        ax.set(xlim=(15000,20000),title=title,xlabel='Training iteration')
        ax.grid(alpha=.2)
    fig.suptitle('Completed Stable run: W&B training curves\nRaw values (faint), 250-update trailing means, saved checkpoints (orange)',fontsize=14)
    fig.savefig(a.output_dir/'wandb_checkpoint_curves.png',dpi=150)
    print(a.output_dir/'wandb_checkpoint_curves.png')


if __name__=='__main__':main()
