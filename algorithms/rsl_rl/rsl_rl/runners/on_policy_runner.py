# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import time
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger


def _episode_reward_extra_snapshot(extras: dict) -> dict:
    """Extract mjlab ``Episode_Reward/*`` keys (same merge as ``Logger.process_env_step``).

    Values are ``episodic_sum / max_episode_length_s`` per term from
    ``mjlab.managers.reward_manager.RewardManager.reset``.
    """
    merged: dict = {}
    if "episode" in extras:
        merged.update(extras["episode"])
    if "log" in extras:
        merged.update(extras["log"])
    return {k: v for k, v in merged.items() if k.startswith("Episode_Reward/")}


class OnPolicyRunner:
    """On-policy runner for reinforcement learning algorithms."""

    alg: PPO
    """The actor-critic algorithm."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        eval_env: VecEnv | None = None,
    ) -> None:
        """Construct the runner, algorithm, and logging stack."""
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.eval_env = eval_env
        self._eval_cfg = train_cfg.get("eval", {}) or {}

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from the environment for algorithm construction
        obs = self.env.get_observations()

        # Create the algorithm
        alg_class: type[PPO] = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)

        # Visitation critic (state-action density model, alpha-blends actions during rollout).
        self.visitation_critic = self.alg.visitation_critic

        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations."""
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample policy action; stores transition.actions and log_prob
                    actions = self.alg.act(obs)
                    # --- Visitation critic alpha-blend (no-op when disabled) ---
                    if self.visitation_critic is not None:
                        alpha = self.visitation_critic.alpha(it)
                        self.visitation_critic.last_alpha = alpha
                        if alpha > 0.0 and self.visitation_critic.is_ready():
                            a_mean, a_std = self.alg.transition.distribution_params[:2]
                            a_vc = self.visitation_critic.vc_action(obs, actions, a_mean, a_std)
                            blended = (1.0 - alpha) * actions + alpha * a_vc
                            self.visitation_critic.record_blend_stats(blended, a_mean, a_std)
                            self.alg.blend_action_into_transition(blended)
                            actions = blended
                    # --- end VC ---
                    # Step the environment
                    next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        check_nan(next_obs, rewards, dones)
                    # Move to device
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    # Feed VC with raw rewards before PPO mutates them for the time-out bootstrap.
                    if self.visitation_critic is not None:
                        self.visitation_critic.record_step(obs, actions, rewards, next_obs, dones)
                    # Process the step (mutates rewards in place for time_outs)
                    self.alg.process_env_step(next_obs, rewards, dones, extras)
                    # Extract intrinsic rewards if RND is used (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                    obs = next_obs

                stop = time.time()
                collect_time = stop - start
                start = stop

            with torch.inference_mode():
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update(iteration=it)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            # Periodic evaluation on a separate medium-perturbation env.
            if self._should_evaluate(it):
                self._evaluate(it)

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

    # --- Evaluation on separate medium-perturbation env ---

    def _should_evaluate(self, it: int) -> bool:
        """True iff eval is configured and this iteration is an eval step."""
        if self.eval_env is None:
            return False
        if not self._eval_cfg.get("enabled", False):
            return False
        every = int(self._eval_cfg.get("eval_every_n_iters", 100))
        if every <= 0:
            return False
        return it % every == 0

    @torch.inference_mode()
    def _evaluate(self, it: int) -> None:
        """Roll out the deterministic policy on ``self.eval_env`` until
        ``eval_num_episodes`` complete, then log mean_reward and mean_episode_length.
        """
        num_episodes = int(self._eval_cfg.get("eval_num_episodes", 1000))
        env = self.eval_env
        actor = self.alg.actor
        actor_was_training = actor.training
        actor.eval()
        is_recurrent = getattr(actor, "is_recurrent", False)

        num_envs = env.num_envs
        cur_reward = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        cur_length = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        completed_rewards: list[float] = []
        completed_lengths: list[float] = []
        eval_ep_reward_extras: list[dict] = []

        obs, _extras_reset = env.reset()
        obs = obs.to(self.device)
        if is_recurrent:
            actor.reset()  # clear hidden state across all envs at episode start
        # Cap steps to avoid infinite loops if envs never terminate.
        max_steps = int(env.max_episode_length) * (num_episodes // num_envs + 2)
        steps = 0
        while len(completed_rewards) < num_episodes and steps < max_steps:
            actions = actor(obs, stochastic_output=False)
            obs, rewards, dones, extras = env.step(actions.to(env.device))
            snap = _episode_reward_extra_snapshot(extras)
            if snap:
                eval_ep_reward_extras.append(snap)
            obs = obs.to(self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)
            cur_reward += rewards
            cur_length += 1
            if dones.any():
                done_mask = dones > 0
                done_ids = done_mask.nonzero(as_tuple=True)[0]
                for i in done_ids.tolist():
                    completed_rewards.append(cur_reward[i].item())
                    completed_lengths.append(cur_length[i].item())
                    if len(completed_rewards) >= num_episodes:
                        break
                cur_reward[done_mask] = 0.0
                cur_length[done_mask] = 0.0
                if is_recurrent:
                    actor.reset(dones)  # zero hidden state for finished envs
            steps += 1

        if actor_was_training:
            actor.train()

        if not completed_rewards:
            return
        trimmed_rewards = completed_rewards[:num_episodes]
        trimmed_lengths = completed_lengths[:num_episodes]
        mean_reward = sum(trimmed_rewards) / len(trimmed_rewards)
        mean_length = sum(trimmed_lengths) / len(trimmed_lengths)
        if self.logger.writer is not None:
            self.logger.writer.add_scalar("Evaluate/mean_reward", mean_reward, it)
            self.logger.writer.add_scalar("Evaluate/mean_episode_length", mean_length, it)
            self.logger.writer.add_scalar("Evaluate/num_episodes", len(trimmed_rewards), it)
            self._log_eval_episode_reward_extras(eval_ep_reward_extras, it)

    def _log_eval_episode_reward_extras(self, ep_extras: list[dict], it: int) -> None:
        """Log ``Episode_Reward/*`` from eval the same way as train (mean over rollout snapshots).

        mjlab defines each ``Episode_Reward/term`` as episodic sum (already × dt when
        ``scale_rewards_by_dt``) divided by ``max_episode_length_s``. Training averages
        all snapshots in ``ep_extras`` for the iteration; here we average over one
        eval sweep's snapshots that carried reward-term keys.
        """
        if not ep_extras or self.logger.writer is None:
            return
        all_keys: set[str] = set()
        for d in ep_extras:
            all_keys.update(d.keys())
        for key in sorted(all_keys):
            infotensor = torch.tensor([], device=self.device)
            for ep_info in ep_extras:
                if key not in ep_info:
                    continue
                v = ep_info[key]
                if not isinstance(v, torch.Tensor):
                    v = torch.as_tensor(v, dtype=torch.float, device=self.device).reshape(1)
                else:
                    v = v.to(self.device)
                    if v.ndim == 0:
                        v = v.unsqueeze(0)
                infotensor = torch.cat((infotensor, v))
            if infotensor.numel() == 0:
                continue
            value = torch.mean(infotensor).item()
            self.logger.writer.add_scalar(f"Evaluate/{key}", value, it)

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state to a given path and upload them if external logging is used."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state from a given path.

        Args:
            path (str): Path to load the model from.
            load_cfg (dict | None): Optional dictionary that defines what models and states to load. If None, all
                models and states are loaded.
            strict (bool): Whether state_dict loading should be strict.
            map_location (str | None): Device mapping for loading the model.
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """Return the policy on the requested device for inference."""
        self.alg.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        return self.alg.get_policy().to(device)  # type: ignore

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the model to a Torch JIT file."""
        jit_model = self.alg.get_policy().as_jit()
        jit_model.to("cpu")

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        traced_model = torch.jit.script(jit_model)
        traced_model.save(save_path)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the model into an ONNX file."""
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),  # type: ignore
            save_path,
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,  # type: ignore
            output_names=onnx_model.output_names,  # type: ignore
        )

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository path whose git status should be logged."""
        self.logger.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
