"""Headless held-out-dataset evaluation. Early terminations include tracking loss.

Use a dataset generated with a different seed from training. This evaluates
reactive dataset forcefields, not a hardware safety certificate or fixed-force
stiffness sweep. The interactive play viewer supplies independent load tests.
"""
import argparse,json
from dataclasses import asdict
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
import src.tasks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--motion-file',type=Path,required=True)
    p.add_argument('--task',default='Unitree-G1-23Dof-Compliant-Tracking-Stand')
    p.add_argument('--num-envs',type=int,default=64);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--seed',type=int,default=123);p.add_argument('--output',type=Path,default=Path('validation.json'))
    args=p.parse_args()
    if args.steps<1 or args.num_envs<1:p.error('steps and environments must be positive')
    torch.set_num_threads(4)
    cfg=load_env_cfg(args.task);cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
    cfg.commands['motion'].motion_file=str(args.motion_file.resolve())
    cfg.commands['motion'].sampling_mode='uniform'
    cfg.observations['actor'].enable_corruption=False
    acfg=load_rl_cfg(args.task);acfg.logger='tensorboard'
    env=ManagerBasedRlEnv(cfg,device='cuda:0');wrapped=RslRlVecEnvWrapper(env,clip_actions=acfg.clip_actions)
    runner=load_runner_cls(args.task)(wrapped,asdict(acfg),device='cuda:0')
    runner.load(str(args.checkpoint),load_cfg={'actor':True},strict=True,map_location='cuda:0')
    policy=runner.get_inference_policy(device='cuda:0');obs=wrapped.get_observations()
    command=env.command_manager.get_term('motion');totals={};early=timeout=0;active=0;peak=0.;finite=True
    try:
        with torch.inference_mode():
            for _ in range(args.steps):
                obs,reward,done,info=wrapped.step(policy(obs))
                finite=finite and bool(torch.isfinite(reward).all()) and all(bool(torch.isfinite(v).all()) for v in obs.values())
                if not finite:raise RuntimeError('non-finite rollout')
                timeouts=info['time_outs'];timeout+=int(timeouts.sum());early+=int((done.bool()&~timeouts).sum())
                active+=int(command.active_force_mask.sum())
                peak=max(peak,float(command.forcefield_forces_w.norm(dim=-1).max()))
                for k,v in command.metrics.items():
                    if k.startswith('error_'):totals[k]=totals.get(k,0.)+float(v.mean())
        result={'checkpoint':str(args.checkpoint),'dataset':str(args.motion_file),'seed':args.seed,
            'environment_seconds':args.steps*args.num_envs*env.step_dt,'finite':finite,
            'early_terminations':early,'time_outs':timeout,'early_terminations_per_robot_minute':early/(args.steps*args.num_envs*env.step_dt/60),
            'active_force_fraction':active/(args.steps*args.num_envs),'peak_reactive_force_N':peak,
            'mean_errors':{k:v/args.steps for k,v in totals.items()},
            'note':'Early terminations include tracking error, not only falls. Reset randomization remains enabled.'}
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:env.close()

if __name__=='__main__':main()
