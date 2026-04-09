"""Collect a dataset of trajectories from a fixed trained policy.

All trajectories are stored in a single compressed NPZ file (dataset.npz)
using a ragged-array layout: all timesteps are concatenated along axis 0
and a traj_lengths array records how many steps belong to each trajectory.

  Layout:
    obs_policy   : (T_total, D_actor)
    obs_critic   : (T_total, D_critic)
    actions      : (T_total, N_joints)
    state_qpos   : (T_total, 7+N)
    state_qvel   : (T_total, 6+N)
    ref_qpos     : (T_total, 7+N)
    ref_qvel     : (T_total, 6+N)
    traj_lengths : (N_trajs,)  int32 — steps per trajectory

  Reconstruct trajectory i:
    offsets = np.concatenate([[0], np.cumsum(data["traj_lengths"])])
    obs = data["obs_policy"][offsets[i] : offsets[i+1]]

qpos layout : root_pos_env_local(3) + root_quat_wxyz(4) + joint_pos(N)
qvel layout : root_lin_vel_body(3)  + root_ang_vel_body(3) + joint_vel(N)
ref_qvel    : world frame for tracking tasks; zeros for balance tasks.

Perturbation levels (no interval pushes — initial conditions only):

  | Level  | init_joint | init_lin_vel | init_z_vel | init_ang_vel | init_pos_z | init_rp |
  |--------|------------|--------------|------------|--------------|------------|---------|
  | none   | 0          | 0            | 0          | 0            | 0          | 0       |
  | small  | 0.1        | 0.5          | 0.2        | 0.78         | 0.01       | 0.1     |
  | medium | 0.2        | 1.25         | 0.2        | 0.78         | 0.01       | 0.3     |
  | hard   | 0.4        | 2.0          | 0.2        | 0.78         | 0.01       | 0.4     |

Usage:
  python scripts/collect_dataset.py Unitree-G1-23Dof-Balance-Flat \\
      --policy logs/rsl_rl/g1_23dof_static_balance/.../model_10000.pt \\
      --level small --num-trajectories 1000 --num-envs 64

  python scripts/collect_dataset.py Unitree-G1-23Dof-Tracking-No-State-Estimation \\
      --policy logs/.../model_10000.pt \\
      --motion-file src/assets/motions/g1_23dof/walk.npz \\
      --level small --num-trajectories 1000
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul
from mjlab.utils.torch import configure_torch_backends
import mjlab.tasks  # noqa: F401
import src.tasks    # noqa: F401

_FIELD_NAMES = (
    "obs_policy",
    "obs_critic",
    "actions",
    "state_qpos",
    "state_qvel",
    "ref_qpos",
    "ref_qvel",
)

# ---------------------------------------------------------------------------
# Perturbation level definitions — mirrors PERTURB_LEVELS in evaluate.py.
# No interval pushes; initial conditions only.
# ---------------------------------------------------------------------------
_LIN, _Z, _ANG = 0.5, 0.2, 0.78

PERTURB_LEVELS: dict[str, dict] = {
    "none": dict(
        init_joint_range=0.0, init_lin_vel_range=0.0, init_z_vel_range=0.0,
        init_ang_vel_range=0.0, init_pos_z_range=0.0, init_rp_range=0.0,
    ),
    "small": dict(
        init_joint_range=0.1, init_lin_vel_range=_LIN, init_z_vel_range=_Z,
        init_ang_vel_range=_ANG, init_pos_z_range=0.01, init_rp_range=0.1,
    ),
    "medium": dict(
        init_joint_range=0.2, init_lin_vel_range=1.25, init_z_vel_range=_Z,
        init_ang_vel_range=_ANG, init_pos_z_range=0.01, init_rp_range=0.3,
    ),
    "hard": dict(
        init_joint_range=0.4, init_lin_vel_range=2.0, init_z_vel_range=_Z,
        init_ang_vel_range=_ANG, init_pos_z_range=0.01, init_rp_range=0.4,
    ),
}


def _empty_buffer() -> dict[str, list]:
    return {k: [] for k in _FIELD_NAMES}


def _flush_dataset(
    combined: dict[str, list],
    traj_lengths: list,
    output_path: Path,
) -> None:
    arrays = {k: np.concatenate(combined[k], axis=0).astype(np.float32) for k in _FIELD_NAMES}
    arrays["traj_lengths"] = np.array(traj_lengths, dtype=np.int32)
    np.savez_compressed(output_path, **arrays)


def _update_bal_ref_qpos(
    bal_ref_qpos: torch.Tensor,
    robot,
    env_origins: torch.Tensor,
    default_joint_pos: torch.Tensor,
    idx,  # int or slice — indexes into the num_envs dimension
) -> None:
    """Store post-reset equilibrium pose as the balance reference for env(s) `idx`."""
    bal_ref_qpos[idx, :3] = (robot.data.root_link_pos_w[idx] - env_origins[idx]).clone()
    bal_ref_qpos[idx, 3:7] = robot.data.root_link_quat_w[idx].clone()
    bal_ref_qpos[idx, 7:] = default_joint_pos[idx].clone()


def _apply_init_noise_balance(
    robot,
    env_id: int,
    device: torch.device,
    init_joint_range: float,
    init_lin_vel_range: float,
    init_z_vel_range: float,
    init_ang_vel_range: float,
    init_pos_z_range: float,
    init_rp_range: float,
) -> None:
    """Apply uniform init noise to joint positions and base velocity for one env.

    After a balance reset the robot is at rest, so we add noise around that state.
    Velocities are read in body frame, noised, then transformed to world frame
    before writing (write_root_state_to_sim expects world-frame velocities).

    Note: the obs returned by env.step() for the done env was computed from the
    clean reset state (before this noise), so obs[0] of the new trajectory will
    be slightly inconsistent with state_qpos[0]. This affects at most one sample
    per trajectory and is negligible in practice.
    """
    env_ids = torch.tensor([env_id], device=device)

    if init_joint_range > 0.0:
        n_j = robot.data.joint_pos.shape[1]
        noise_j = torch.empty(1, n_j, device=device).uniform_(
            -init_joint_range, init_joint_range
        )
        noisy_jpos = robot.data.joint_pos[env_ids] + noise_j
        if hasattr(robot.data, "soft_joint_pos_limits"):
            limits = robot.data.soft_joint_pos_limits[env_ids]  # (1, N, 2)
            noisy_jpos = torch.clamp(noisy_jpos, limits[:, :, 0], limits[:, :, 1])
        robot.write_joint_state_to_sim(
            noisy_jpos, robot.data.joint_vel[env_ids], env_ids=env_ids
        )

    apply_pose_or_vel = (
        init_lin_vel_range > 0.0 or init_z_vel_range > 0.0 or init_ang_vel_range > 0.0
        or init_pos_z_range > 0.0 or init_rp_range > 0.0
    )
    if apply_pose_or_vel:
        pos = robot.data.root_link_pos_w[env_ids].clone()
        quat = robot.data.root_link_quat_w[env_ids].clone()

        if init_pos_z_range > 0.0:
            pos[:, 2:3] += torch.empty(1, 1, device=device).uniform_(
                -init_pos_z_range, init_pos_z_range
            )
        if init_rp_range > 0.0:
            roll  = torch.empty(1, device=device).uniform_(-init_rp_range, init_rp_range)
            pitch = torch.empty(1, device=device).uniform_(-init_rp_range, init_rp_range)
            delta = quat_from_euler_xyz(roll, pitch, torch.zeros(1, device=device))
            quat  = quat_mul(delta, quat)

        # Read body-frame velocities (approximately zero after reset).
        lin_vel_b = robot.data.root_link_lin_vel_b[env_ids].clone()
        ang_vel_b = robot.data.root_link_ang_vel_b[env_ids].clone()

        if init_lin_vel_range > 0.0:
            lin_vel_b[:, :2] += torch.empty(1, 2, device=device).uniform_(
                -init_lin_vel_range, init_lin_vel_range
            )
        if init_z_vel_range > 0.0:
            lin_vel_b[:, 2:3] += torch.empty(1, 1, device=device).uniform_(
                -init_z_vel_range, init_z_vel_range
            )
        if init_ang_vel_range > 0.0:
            ang_vel_b += torch.empty(1, 3, device=device).uniform_(
                -init_ang_vel_range, init_ang_vel_range
            )

        # Transform to world frame for write_root_state_to_sim.
        lin_vel_w = quat_apply(quat, lin_vel_b)
        ang_vel_w = quat_apply(quat, ang_vel_b)

        root_state = torch.cat([pos, quat, lin_vel_w, ang_vel_w], dim=-1)
        robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    robot.clear_state(env_ids=env_ids)


@torch.no_grad()
def collect_dataset(
    env: RslRlVecEnvWrapper,
    policy,
    num_trajectories: int,
    output_dir: Path,
    is_tracking: bool,
    device: torch.device,
    init_joint_range: float,
    init_lin_vel_range: float,
    init_z_vel_range: float,
    init_ang_vel_range: float,
    init_pos_z_range: float,
    init_rp_range: float,
    name: str = "dataset",
) -> int:
    robot = env.unwrapped.scene["robot"]
    num_envs = env.num_envs
    env_origins = env.unwrapped.scene.env_origins.clone()  # [num_envs, 3]
    default_joint_pos = robot.data.default_joint_pos.clone()  # [num_envs, N]
    n_joints = default_joint_pos.shape[1]

    motion_cmd: MotionCommand | None = None
    if is_tracking:
        motion_cmd = env.unwrapped.command_manager.get_term("motion")

    has_init_noise = (
        init_joint_range > 0.0
        or init_lin_vel_range > 0.0
        or init_z_vel_range > 0.0
        or init_ang_vel_range > 0.0
        or init_pos_z_range > 0.0
        or init_rp_range > 0.0
    )

    # Per-env balance reference: [pos_env_local(3), quat(4), default_joint_pos(N)]
    # Constant throughout each episode; updated after each reset.
    bal_ref_qpos = torch.zeros(num_envs, 7 + n_joints, device=device)
    bal_ref_qvel = torch.zeros(num_envs, 6 + n_joints, device=device)

    env_buffers: list[dict[str, list]] = [_empty_buffer() for _ in range(num_envs)]
    saved_count = 0

    # Accumulator for combined NPZ.
    combined: dict[str, list] = {k: [] for k in _FIELD_NAMES}
    traj_lengths: list[int] = []

    obs, _ = env.reset()

    if not is_tracking:
        _update_bal_ref_qpos(bal_ref_qpos, robot, env_origins, default_joint_pos, slice(None))
        # Apply init noise to the first batch so all trajectories (not just those
        # after the first done) start from perturbed initial conditions.
        if has_init_noise:
            for i in range(num_envs):
                _apply_init_noise_balance(
                    robot, i, device,
                    init_joint_range, init_lin_vel_range,
                    init_z_vel_range, init_ang_vel_range,
                    init_pos_z_range, init_rp_range,
                )
            robot.update(env.unwrapped.step_dt)

    pbar = tqdm(total=num_trajectories, desc="Trajectories", unit="traj")

    while saved_count < num_trajectories:
        # --- Snapshot state and reference BEFORE step so that (obs, state, ref, action)
        # all correspond to the same simulation timestep t. ---

        pos_w = robot.data.root_link_pos_w          # [num_envs, 3]
        quat_w = robot.data.root_link_quat_w         # [num_envs, 4]
        lin_vel_b = robot.data.root_link_lin_vel_b   # [num_envs, 3] body frame
        ang_vel_b = robot.data.root_link_ang_vel_b   # [num_envs, 3] body frame
        joint_pos = robot.data.joint_pos             # [num_envs, N]
        joint_vel = robot.data.joint_vel             # [num_envs, N]

        state_qpos = torch.cat(
            [pos_w - env_origins, quat_w, joint_pos], dim=-1
        )  # [num_envs, 7+N]
        state_qvel = torch.cat(
            [lin_vel_b, ang_vel_b, joint_vel], dim=-1
        )  # [num_envs, 6+N]

        if is_tracking:
            ref_qpos = torch.cat(
                [
                    motion_cmd.anchor_pos_w - env_origins,
                    motion_cmd.anchor_quat_w,
                    motion_cmd.joint_pos,
                ],
                dim=-1,
            )  # [num_envs, 7+N]
            ref_qvel = torch.cat(
                [
                    motion_cmd.anchor_lin_vel_w,
                    motion_cmd.anchor_ang_vel_w,
                    motion_cmd.joint_vel,
                ],
                dim=-1,
            )  # [num_envs, 6+N]
        else:
            ref_qpos = bal_ref_qpos  # [num_envs, 7+N]
            ref_qvel = bal_ref_qvel  # zeros

        obs_policy_np = obs["actor"].cpu().numpy()
        obs_critic_np = obs["critic"].cpu().numpy()
        state_qpos_np = state_qpos.cpu().numpy()
        state_qvel_np = state_qvel.cpu().numpy()
        ref_qpos_np = ref_qpos.cpu().numpy()
        ref_qvel_np = ref_qvel.cpu().numpy()

        actions = policy(obs)
        actions_np = actions.cpu().numpy()

        obs, _, dones, _ = env.step(actions)

        # Append step data to per-env buffers.
        for i in range(num_envs):
            buf = env_buffers[i]
            buf["obs_policy"].append(obs_policy_np[i])
            buf["obs_critic"].append(obs_critic_np[i])
            buf["actions"].append(actions_np[i])
            buf["state_qpos"].append(state_qpos_np[i])
            buf["state_qvel"].append(state_qvel_np[i])
            buf["ref_qpos"].append(ref_qpos_np[i])
            buf["ref_qvel"].append(ref_qvel_np[i])

        done_mask = dones.bool()
        if not done_mask.any():
            continue

        done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.dim() == 0:
            done_ids = done_ids.unsqueeze(0)

        for i in done_ids.tolist():
            buf = env_buffers[i]
            if len(buf["actions"]) >= 2 and saved_count < num_trajectories:
                T = len(buf["actions"])
                for k in _FIELD_NAMES:
                    combined[k].append(np.array(buf[k], dtype=np.float32))
                traj_lengths.append(T)
                saved_count += 1
                pbar.update(1)

            env_buffers[i] = _empty_buffer()

            if not is_tracking:
                # After env.step() the done env has been reset; robot.data reflects
                # the clean reset state. Capture reference, then optionally apply noise.
                _update_bal_ref_qpos(
                    bal_ref_qpos, robot, env_origins, default_joint_pos, i
                )
                if has_init_noise:
                    _apply_init_noise_balance(
                        robot, i, device,
                        init_joint_range, init_lin_vel_range,
                        init_z_vel_range, init_ang_vel_range,
                        init_pos_z_range, init_rp_range,
                    )
                    # write_root_state_to_sim only updates the physics buffer; robot.data
                    # still caches the pre-noise values until the next robot.update() call.
                    # Refresh now so the next loop iteration records the noisy state at t=0.
                    robot.update(env.unwrapped.step_dt)

    pbar.close()
    _flush_dataset(combined, traj_lengths, output_dir / f"{name}.npz")
    return saved_count


def main():

    all_tasks = list_tasks()

    parser = argparse.ArgumentParser(
        description="Collect a trajectory dataset from a fixed trained policy.",
    )
    parser.add_argument("task", choices=all_tasks, help="Registered task ID.")
    parser.add_argument("--policy", required=True, metavar="PATH",
                        help="Path to checkpoint (.pt).")
    parser.add_argument(
        "--init-perturb-mode", default="small", choices=list(PERTURB_LEVELS),
        dest="init_perturb_mode",
        help="Perturbation level for initial conditions (no interval pushes). "
             "Default: small.",
    )
    parser.add_argument("--num-trajectories", type=int, default=1000,
                        help="Total complete trajectories to save (default: 1000).")
    parser.add_argument("--num-envs", type=int, default=64,
                        help="Number of parallel environments (default: 64).")
    parser.add_argument("--motion-file", type=str, default=None,
                        help="NPZ motion file (required for tracking tasks).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Default: <checkpoint_dir>/dataset/")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-obs-noise", action="store_true",
                        help="Disable actor observation corruption.")
    parser.add_argument("--name", type=str, default=None,
                        help="Base name for output files. Produces <name>.npz and "
                             "metadata_<name>.json. Default: dataset_<level>.")
    args = parser.parse_args()

    if args.name is None:
        args.name = f"dataset_{args.init_perturb_mode}"

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))

    ckpt_path = Path(args.policy).expanduser().resolve()
    if not ckpt_path.exists():
        parser.error(f"Checkpoint not found: {ckpt_path}")

    spec = PERTURB_LEVELS[args.init_perturb_mode]

    # --- Environment ---
    env_cfg = load_env_cfg(args.task, play=False)
    agent_cfg = load_rl_cfg(args.task)

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.curriculum = {}
    if args.disable_obs_noise:
        env_cfg.observations["actor"].enable_corruption = False

    # Always disable interval pushes for dataset collection.
    env_cfg.events.pop("push_robot", None)

    is_tracking = bool(
        env_cfg.commands
        and "motion" in env_cfg.commands
        and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
    )

    # Zero built-in balance reset noise — _apply_init_noise_balance is the sole
    # source of init perturbation (matching the approach in evaluate.py).
    if not is_tracking:
        if hasattr(env_cfg, "events") and "reset_base" in (env_cfg.events or {}):
            rb = env_cfg.events["reset_base"]
            rb.params["pose_range"] = {k: (0.0, 0.0) for k in rb.params.get("pose_range", {})}
            rb.params["velocity_range"] = {}
        if hasattr(env_cfg, "events") and "reset_robot_joints" in (env_cfg.events or {}):
            rj = env_cfg.events["reset_robot_joints"]
            rj.params["position_range"] = (0.0, 0.0)
            rj.params["velocity_range"] = (0.0, 0.0)
        if hasattr(env_cfg, "events") and "reset_rp_noise" in (env_cfg.events or {}):
            env_cfg.events["reset_rp_noise"].params["rp_range"] = 0.0

    if is_tracking:
        if not args.motion_file:
            parser.error("Tracking tasks require --motion-file path/to/motion.npz")
        motion_path = Path(args.motion_file).expanduser().resolve()
        if not motion_path.exists():
            parser.error(f"Motion file not found: {motion_path}")
        env_cfg.commands["motion"].motion_file = str(motion_path)
        env_cfg.commands["motion"].sampling_mode = "start"

        # Zero built-in noise; level spec is the sole source of randomization.
        env_cfg.commands["motion"].pose_range = {}
        env_cfg.commands["motion"].velocity_range = {}
        env_cfg.commands["motion"].joint_position_range = (0.0, 0.0)

        lv = spec["init_lin_vel_range"]
        zv = spec["init_z_vel_range"]
        av = spec["init_ang_vel_range"]
        rp = spec["init_rp_range"]
        pz = spec["init_pos_z_range"]
        jr = spec["init_joint_range"]
        if jr > 0.0:
            env_cfg.commands["motion"].joint_position_range = (-jr, jr)
        if lv > 0.0 or zv > 0.0 or av > 0.0:
            env_cfg.commands["motion"].velocity_range = {
                "x": (-lv, lv), "y": (-lv, lv), "z": (-zv, zv),
                "roll": (-av, av), "pitch": (-av, av), "yaw": (-av, av),
            }
        if pz > 0.0 or rp > 0.0:
            env_cfg.commands["motion"].pose_range = {
                "x": (-0.05, 0.05), "y": (-0.05, 0.05),
                "z": (-pz, pz),
                "roll": (-rp, rp), "pitch": (-rp, rp),
                "yaw": (-rp * 2, rp * 2),
            }

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- Policy ---
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env_wrapped, asdict(agent_cfg), device=device)
    runner.load(str(ckpt_path), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    # --- Output directory ---
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ckpt_path.parent / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Task:         {args.task}")
    print(f"[INFO] Checkpoint:   {ckpt_path}")
    print(f"[INFO] Level:        {args.init_perturb_mode}  {spec}")
    print(f"[INFO] Output:       {output_dir}")
    print(f"[INFO] Trajectories: {args.num_trajectories}")
    print(f"[INFO] Envs:         {args.num_envs}")
    print(f"[INFO] Tracking:     {is_tracking}")

    saved = collect_dataset(
        env_wrapped, policy,
        args.num_trajectories, output_dir,
        is_tracking, torch.device(device),
        spec["init_joint_range"], spec["init_lin_vel_range"],
        spec["init_z_vel_range"], spec["init_ang_vel_range"],
        spec["init_pos_z_range"], spec["init_rp_range"],
        name=args.name,
    )

    metadata = {
        "task": args.task,
        "checkpoint": str(ckpt_path),
        "num_trajectories_requested": args.num_trajectories,
        "num_trajectories_saved": saved,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "level": args.init_perturb_mode,
        "disable_obs_noise": args.disable_obs_noise,
        "is_tracking": is_tracking,
        "motion_file": (
            str(Path(args.motion_file).expanduser().resolve())
            if args.motion_file else None
        ),
        "step_dt": env_wrapped.unwrapped.step_dt,
        "episode_length_s": env_wrapped.unwrapped.max_episode_length_s,
        **{f"spec_{k}": v for k, v in spec.items()},
        "npz_fields": list(_FIELD_NAMES),
        "qpos_layout": "root_pos_env_local(3) + root_quat_wxyz(4) + joint_pos(N)",
        "qvel_layout": "root_lin_vel_body(3) + root_ang_vel_body(3) + joint_vel(N)",
        "ref_qvel_frame": "world frame (tracking) / zeros (balance)",
        "state_qvel_frame": "body frame — apply quat_apply(root_quat, lin_vel_b) to get world frame",
    }
    with open(output_dir / f"metadata_{args.name}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    env_wrapped.close()
    print(f"\n[INFO] Done. Saved {saved} trajectories -> {output_dir / args.name}.npz")


if __name__ == "__main__":
    main()
