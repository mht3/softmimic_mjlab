"""CPU prompt streaming using ARDY's upstream autoregressive_step API."""
import json
import math
from pathlib import Path
import time

STILL = 'A person is standing still.'


def validate_request(body):
    if not isinstance(body, dict):
        raise ValueError('Expected a JSON object')
    seed = body.get('seed', 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError('Invalid seed')
    segments = body.get('segments', [{'prompt': body.get('prompt', ''), 'seconds': body.get('seconds', 4)}])
    if not isinstance(segments, list) or not 1 <= len(segments) <= 30:
        raise ValueError('Use 1–30 prompt blocks')
    clean = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError('Invalid prompt block')
        prompt, seconds = segment.get('prompt'), segment.get('seconds')
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 500:
            raise ValueError('Prompt must contain 1–500 characters')
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 2 <= seconds <= 30:
            raise ValueError('Each duration must be 2–30 seconds')
        clean.append(dict(prompt=prompt.strip(), seconds=round(seconds * 25) / 25))
    total = sum(s['seconds'] for s in clean)
    if total > 30:
        raise ValueError('The complete sequence, including a standing finish, must be at most 30 seconds')
    from demos.ardy_web.constraints import validate_constraints
    constraints=validate_constraints(body.get('constraints',[]),total)
    return dict(segments=clean, seconds=total, seed=seed, constraints=constraints, prompt=' → '.join(s['prompt'] for s in clean))


class Cancelled(Exception):
    pass


def generate_stream(engine, job, output, progress, publish, cancelled):
    import numpy as np
    import torch
    from demos.ardy_web.export import export_motion
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fps = float(engine.model.motion_rep.fps)
    horizon = engine.model.gen_horizon_len
    token = engine.model.num_frames_per_token
    # Matches interactive_demo/common.py DEFAULT_HISTORY_CROP_LENGTH (frames).
    history_length = 4 // token * token
    if history_length == 0:
        raise ValueError('Model token size exceeds upstream history default')
    plan = job['segments']
    total = int(round(job['seconds'] * fps))
    from demos.ardy_web.constraints import ConstraintAdapter,window_conditions,constraint_fit
    constraint_items=job.get('constraints',[])
    adapter=ConstraintAdapter(engine) if constraint_items else None
    mask,observed=adapter.conditions(constraint_items,total) if adapter else (None,None)
    boundaries = np.cumsum([round(s['seconds'] * fps) for s in plan])
    embeddings = {}
    start = time.perf_counter()
    for s in plan:
        if cancelled():
            raise Cancelled('Cancelled before generation')
        if s['prompt'] not in embeddings:
            progress('Encoding: ' + s['prompt'])
            embeddings[s['prompt']] = engine.encode_prompt(s['prompt'], progress).to(engine.device)
    encoding_s = time.perf_counter() - start
    torch.manual_seed(job['seed'])
    history = None
    source = []
    events = []
    generated = 0
    next_preview = round(4 * fps)
    generation_s = 0.
    preview_s = 0.
    revision = 0
    last_segment = None
    while generated < total:
        if cancelled():
            raise Cancelled('Cancelled; previously audited previews remain available')
        index = min(int(np.searchsorted(boundaries, generated, side='right')), len(plan)-1)
        if index != last_segment:
            events.append(dict(frame=generated, seconds=generated/fps, prompt=plan[index]['prompt']))
            last_segment = index
        progress(f'Generating {generated/fps:.2f} / {total/fps:.2f} s · {plan[index]["prompt"]}')
        begin = time.perf_counter()
        with torch.inference_mode():
            h = 0 if history is None else history.shape[1]
            window,step_mask,step_observed=window_conditions(mask,observed,generated,h,horizon,token,fps)
            samples = engine.model.autoregressive_step(
                num_frames=window,
                num_denoising_steps=int(engine.model.diffusion.num_base_steps),
                motion_mask=step_mask, observed_motion=step_observed, cfg_weight=(2., 2.),
                text_feat=embeddings[plan[index]['prompt']],
                text_pad_mask=torch.ones((1, 1), dtype=torch.bool, device=engine.device),
                init_history_sequence=history,
                init_first_heading_angle=torch.zeros(1, device=engine.device) if history is None else None)
            new = samples[:, h:h+horizon]
            history = torch.cat(([history] if history is not None else []) + [new], dim=1)[:, -history_length:].clone()
            decoded = engine.model.motion_rep.inverse(new, is_normalized=True)
            q = np.asarray(engine.converter.dict_to_qpos(decoded, root_quat_w_first=True))[0]
        if q.shape != (horizon, 36) or not np.isfinite(q).all():
            raise ValueError('Invalid streamed ARDY output')
        source.append(q[:min(horizon,total-generated)])
        generated += len(source[-1])
        generation_s += time.perf_counter() - begin
        if generated < next_preview and generated != total:
            continue
        revision += 1
        q29 = np.concatenate(source)
        final = generated == total
        folder = output if final else output/'stream'/f'{revision:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/'source_29dof.npz', qpos=q29, fps=fps,
                            quaternion_order='wxyz', joint_names=engine.retargeter.source_names)
        schedule = dict(requested=plan, constraints=constraint_items, coordinate_system='MuJoCo Z-up, metres/radians', actual_prompt_events=events, history_frames=history_length,
                        horizon_frames=horizon, fps=fps, final=final,
                        transition_method='ARDY autoregressive history conditioning; no pose crossfade')
        (folder/'schedule.json').write_text(json.dumps(schedule, indent=2)+'\n')
        progress(f'Auditing {generated/fps:.2f} s continuous prefix including every transition')
        begin = time.perf_counter()
        metadata = export_motion(folder, q29, fps, job['prompt'], generated/fps, job['seed'], engine.retargeter,
                                 timings=dict(encoder_seconds=encoding_s, generation_seconds=generation_s,
                                              previous_preview_seconds=preview_s), device=engine.device)
        if constraint_items:
            with np.load(folder/'motion_23dof.npz') as saved:
                fit=constraint_fit(adapter,constraint_items,saved['qpos'],float(saved['fps']))
            metadata['audit']['constraint_errors']=fit
            metadata['audit']['constraint_review_required']=any(r['needs_review'] for r in fit)
            (folder/'audit.json').write_text(json.dumps(metadata['audit'],indent=2)+'\n')
            partial=folder/'metadata.partial';partial.write_text(json.dumps(metadata,indent=2)+'\n');partial.replace(folder/'metadata.json')
        preview_s += time.perf_counter()-begin
        relative = folder.relative_to(output.parent).as_posix()
        publish(dict(revision=revision, final=final, seconds=generated/fps,
                     base_url='/clips/'+relative, clip=metadata))
        next_preview = generated + round(4*fps)
    return metadata
