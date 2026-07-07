from typing import Any, Dict, List, Optional, Tuple
import os
import random
import time

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from config import SimulationConfig
from constants import (
    FORCEABLE_LINKS,
    MAX_ROBOT_ROTATIONAL_STIFFNESS,
    MAX_ROBOT_STIFFNESS,
    MIN_ROBOT_ROTATIONAL_STIFFNESS,
    MIN_ROBOT_STIFFNESS,
    TELEPORT_THRESHOLD,
)
from force_profile import generate_random_force_profile
from ik_solver import G1_Mink_IK_Solver
from ik_update import perform_single_ik_update
from physics import is_ik_solution_feasible
from visualization import add_visual_overlays

try:
    import imageio
except ImportError:  # pragma: no cover - handled gracefully at runtime
    imageio = None


class SimulationRunner:
    """Encapsulates the state and control flow for a single simulation/generation run."""

    SIMULATION_FREQUENCY = 30.0
    LOGGING_FREQUENCY = 30.0

    def __init__(self, config: SimulationConfig, file_index: int):
        self.config = config
        self.file_index = file_index
        self.is_interactive = config.mode == "interactive"
        self.timestep = 1.0 / self.SIMULATION_FREQUENCY
        self.logging_interval_steps = max(
            1, round(self.SIMULATION_FREQUENCY / self.LOGGING_FREQUENCY)
        )
        self.ik_solver: Optional[G1_Mink_IK_Solver] = None
        self.motion_duration = 0.0
        self.force_profile: List[Dict[str, Any]] = []
        self.event_queue: List[Dict[str, Any]] = []
        self.current_event: Optional[Dict[str, Any]] = None
        self.release_info: Optional[Dict[str, Any]] = None
        self.reusable_ref_data: Optional[mujoco.MjData] = None
        self.prev_qpos_ref: Optional[np.ndarray] = None
        self.event_start_data_idx = -1
        self.qpos_before_event: Optional[np.ndarray] = None
        self.rewind_indicator_until = 0.0
        self.viewer = None
        self.camera_tracking = True
        self.renderer = None
        self.cam = None
        self.frames: List[np.ndarray] = []
        self.all_adapted_qpos: List[np.ndarray] = []
        self.all_reference_qpos: List[np.ndarray] = []
        self.all_force_data: List[np.ndarray] = []
        self.all_collision_metadata: List[np.ndarray] = []
        self.stiffness_state = {
            "stiffness": 140.0,
            "rot_stiffness": 1.0,
            "next_update_time": -1.0,
        }
        self.num_steps = 0

    def run(self):
        self._announce_start()
        self._seed_random_generators()
        self._initialize_solver()
        self._prepare_force_profile()
        self._prepare_timing()
        self._initialize_zero_wrench_state()
        self._setup_rendering()
        try:
            self._simulate_loop()
        finally:
            self._teardown_rendering()
        if not self.is_interactive:
            self._save_outputs()

    def _announce_start(self):
        if self.is_interactive:
            print(
                f"\n--- Starting Interactive Simulation (Force Mode: {self.config.force_mode}) ---"
            )
        else:
            os.makedirs(self.config.output_dir, exist_ok=True)
            print(
                f"\n[File {self.file_index + 1}/{self.config.num_files}] "
                f"Generating profile (seed={self.config.seed}, mode={self.config.force_mode})..."
            )

    def _seed_random_generators(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _initialize_solver(self):
        self.ik_solver = G1_Mink_IK_Solver(
            self.config.model_path,
            self.config.motion_path,
            self.config.repeat_frame_time,
            self.config.com_cost,
            self.config.com_cost_z_factor,
            self.config.upper_joint_cost,
            self.config.torso_orientation_cost,
        )
        if self.ik_solver.motion_lib:
            self.motion_duration = self.ik_solver.motion_lib.get_max_times(torch.tensor(0)).item()
        else:
            self.motion_duration = 20.0
        self.reusable_ref_data = mujoco.MjData(self.ik_solver.model)

    def _prepare_force_profile(self):
        self.force_profile = generate_random_force_profile(
            self.motion_duration,
            FORCEABLE_LINKS,
            self.config.force_mode,
            self.ik_solver,
            self.config,
        )
        print(
            f"[File {self.file_index + 1}/{self.config.num_files}] "
            f"Generated profile with {len(self.force_profile)} candidate events."
        )
        self.event_queue = list(self.force_profile)

    def _prepare_timing(self):
        self.num_steps = int(np.ceil(self.motion_duration / self.timestep))

    def _initialize_zero_wrench_state(self):
        if self.config.force_mode == "zero-wrench":
            self._update_stiffness_state(0.0)

    def _update_stiffness_state(self, current_time: float):
        self.stiffness_state["stiffness"] = np.exp(
            random.uniform(np.log(MIN_ROBOT_STIFFNESS), np.log(MAX_ROBOT_STIFFNESS))
        )
        self.stiffness_state["rot_stiffness"] = np.exp(
            random.uniform(
                np.log(MIN_ROBOT_ROTATIONAL_STIFFNESS), np.log(MAX_ROBOT_ROTATIONAL_STIFFNESS)
            )
        )
        hold_duration = random.uniform(2.0, 5.0)
        self.stiffness_state["next_update_time"] = current_time + hold_duration

    def _setup_rendering(self):
        if self.is_interactive:
            self.viewer = mujoco.viewer.launch_passive(self.ik_solver.model, self.ik_solver.data)
            self.viewer.cam.azimuth, self.viewer.cam.elevation, self.viewer.cam.distance = 90, -15, 4.0
            self.viewer.cam.lookat[:] = self.ik_solver.data.body("torso_link").xpos

            def key_callback(_keycode):
                self.camera_tracking = not self.camera_tracking

            self.viewer.key_callback = key_callback

        if self.config.record_video:
            if imageio is None:
                print("Warning: 'imageio' not found. Cannot record video.")
                self.config.record_video = False
            else:
                self.renderer = mujoco.Renderer(self.ik_solver.model, height=480, width=640)
                self.cam = mujoco.MjvCamera()
                self.cam.azimuth, self.cam.elevation, self.cam.distance = 90, -15, 4.0
                print(f"Recording video to {self.config.output_filename}")

    def _simulate_loop(self):
        step = 0
        while step < self.num_steps:
            if self.is_interactive and not self.viewer.is_running():
                break

            loop_start = time.time()
            current_time = step * self.timestep

            if (
                self.config.force_mode == "zero-wrench"
                and current_time >= self.stiffness_state["next_update_time"]
            ):
                self._update_stiffness_state(current_time)

            qpos_ref, _, contacts_ref = self.ik_solver.get_reference_motion(current_time)
            self._handle_possible_teleport(qpos_ref, current_time)

            self._complete_event_if_finished(current_time, qpos_ref)
            self._maybe_start_next_event(current_time)

            vis_data = perform_single_ik_update(
                self.ik_solver,
                self.reusable_ref_data,
                self.current_event,
                self.release_info,
                current_time,
                self.timestep,
                self.config,
                qpos_ref,
            )

            if self.current_event:
                self.reusable_ref_data.qpos[:] = qpos_ref
                mujoco.mj_forward(self.reusable_ref_data.model, self.reusable_ref_data)
                is_feasible, _ = is_ik_solution_feasible(
                    self.ik_solver.data,
                    self.reusable_ref_data,
                    self.current_event["link_name"],
                    vis_data[1],
                    vis_data[2],
                    self.current_event["stiffness"],
                    self.current_event.get("rotational_stiffness", 1.0),
                    self.ik_solver.total_mass,
                )
                if not is_feasible:
                    step = self._handle_infeasible_event(current_time)
                    continue

            self._finalize_release_if_needed()

            if not self.is_interactive:
                self._record_step(step, qpos_ref, contacts_ref, vis_data, current_time)

            self._update_viewer(vis_data, current_time, loop_start)
            self._capture_video_frame(vis_data, current_time)

            step += 1

    def _handle_possible_teleport(self, qpos_ref: np.ndarray, current_time: float):
        if self.prev_qpos_ref is None:
            self.prev_qpos_ref = qpos_ref.copy()
            return

        dist_sq = np.sum((qpos_ref[:2] - self.prev_qpos_ref[:2]) ** 2)
        if dist_sq > TELEPORT_THRESHOLD**2:
            teleport_vector = qpos_ref[:3] - self.prev_qpos_ref[:3]
            print(
                f"Teleport detected at t={current_time:.2f}s (dist={np.sqrt(dist_sq):.2f}m). Adjusting IK state accordingly."
            )
            current_q = self.ik_solver.configuration.q
            current_q[0:3] += teleport_vector
            self.ik_solver.configuration.update(q=current_q)

            if self.qpos_before_event is not None:
                self.qpos_before_event[0:3] += teleport_vector

            if self.current_event:
                if "forcefield_setpoint_pos" in self.current_event:
                    self.current_event["forcefield_setpoint_pos"] += teleport_vector
                if "collision_plane_origin" in self.current_event:
                    self.current_event["collision_plane_origin"] += teleport_vector

            if self.release_info:
                self.release_info["start_pos"] += teleport_vector

        self.prev_qpos_ref = qpos_ref.copy()

    def _complete_event_if_finished(self, current_time: float, qpos_ref: np.ndarray):
        if not self.current_event or current_time < self.current_event["end_time"]:
            return

        (
            _task_body_name,
            last_force_ext,
            _,
            _,
            _,
            _,
            _,
        ) = perform_single_ik_update(
            self.ik_solver,
            self.reusable_ref_data,
            self.current_event,
            None,
            current_time,
            self.timestep,
            self.config,
            qpos_ref,
        )
        link_name = self.current_event["link_name"]
        self.release_info = {
            "link_name": link_name,
            "start_time": current_time,
            "start_pos": self.ik_solver.data.body(link_name).xpos.copy(),
            "start_rot": Rotation.from_matrix(
                self.ik_solver.data.body(link_name).xmat.reshape(3, 3)
            ),
            "start_force": last_force_ext.copy(),
        }
        self.current_event = None
        self.event_start_data_idx = -1
        self.qpos_before_event = None

    def _maybe_start_next_event(self, current_time: float):
        if self.current_event or not self.event_queue:
            return
        if current_time < self.event_queue[0]["start_time"]:
            return

        self.current_event = self.event_queue.pop(0)
        if self.is_interactive:
            self.qpos_before_event = self.ik_solver.configuration.q.copy()
        else:
            self.event_start_data_idx = len(self.all_adapted_qpos)

    def _handle_infeasible_event(self, current_time: float) -> int:
        event_to_rewind = self.current_event
        if self.config.force_mode in ["triangle", "forcefield", "collision-emulator"]:
            scale_factor = 0.8
            if self.config.force_mode in ["forcefield", "triangle"]:
                new_amplitude = event_to_rewind.get("amplitude", 0.0) * scale_factor
                new_torque_amplitude = event_to_rewind.get("torque_amplitude", 0.0) * scale_factor
                old_ramp_duration = event_to_rewind["ramp_duration"]
                new_ramp_duration = old_ramp_duration * scale_factor
                min_ramp_duration = 0.1
                if (
                    (new_amplitude < 1.0 and new_torque_amplitude < 1.0)
                    or new_ramp_duration < min_ramp_duration
                ):
                    self.current_event = None
                else:
                    start_time = event_to_rewind["start_time"]
                    hold_duration = (
                        event_to_rewind["hold_end_time"] - event_to_rewind["hold_start_time"]
                    )
                    event_to_rewind["amplitude"] = new_amplitude
                    event_to_rewind["torque_amplitude"] = new_torque_amplitude
                    event_to_rewind["ramp_duration"] = new_ramp_duration
                    event_to_rewind["hold_start_time"] = start_time + new_ramp_duration
                    event_to_rewind["hold_end_time"] = event_to_rewind["hold_start_time"] + hold_duration
                    event_to_rewind["end_time"] = event_to_rewind["hold_end_time"] + new_ramp_duration
            elif self.config.force_mode == "collision-emulator":
                new_end_time = current_time
                start_time = event_to_rewind["start_time"]
                min_valid_duration = 0.2
                if (new_end_time - start_time) < min_valid_duration:
                    self.current_event = None
                else:
                    event_to_rewind["end_time"] = new_end_time
        else:
            self.current_event = None

        rewind_to_time = event_to_rewind["start_time"]
        self.release_info = None

        if self.is_interactive:
            self.rewind_indicator_until = current_time + 1.0
            if self.qpos_before_event is not None:
                self.ik_solver.configuration.update(q=self.qpos_before_event)
            self.qpos_before_event = None
        else:
            self.all_adapted_qpos = self.all_adapted_qpos[: self.event_start_data_idx]
            self.all_reference_qpos = self.all_reference_qpos[: self.event_start_data_idx]
            self.all_force_data = self.all_force_data[: self.event_start_data_idx]
            self.all_collision_metadata = self.all_collision_metadata[: self.event_start_data_idx]
            if self.all_adapted_qpos:
                last_valid_qpos = self.all_adapted_qpos[-1]
            else:
                last_valid_qpos = self.ik_solver.get_reference_motion(rewind_to_time)[0]
            self.ik_solver.configuration.update(q=last_valid_qpos)

        return int(rewind_to_time / self.timestep)

    def _finalize_release_if_needed(self):
        if self.release_info and self.release_info.get("finished", False):
            self.release_info = None

    def _record_step(
        self,
        step: int,
        qpos_ref: np.ndarray,
        contacts_ref: np.ndarray,
        vis_data: Tuple,
        current_time: float,
    ):
        if step % self.logging_interval_steps != 0:
            return

        self.all_reference_qpos.append(
            np.concatenate(
                [
                    qpos_ref[0:3],
                    qpos_ref[[4, 5, 6, 3]],
                    qpos_ref[self.ik_solver.actuated_qpos_indices],
                    contacts_ref,
                ]
            )
        )
        q = self.ik_solver.configuration.q.copy()
        self.all_adapted_qpos.append(
            np.concatenate(
                [
                    q[0:3],
                    q[[4, 5, 6, 3]],
                    q[self.ik_solver.actuated_qpos_indices],
                ]
            )
        )

        target_event = (
            self.current_event if self.current_event else (self.event_queue[0] if self.event_queue else None)
        )
        if target_event:
            link_id = mujoco.mj_name2id(
                self.ik_solver.model, mujoco.mjtObj.mjOBJ_BODY, target_event["link_name"]
            )
            stiffness = target_event["stiffness"]
            rot_stiffness = target_event.get("rotational_stiffness", 1.0)
            force_info = np.array([link_id, *vis_data[1], *vis_data[2], stiffness, rot_stiffness])
        else:
            if self.config.force_mode == "zero-wrench":
                stiffness = self.stiffness_state["stiffness"]
                rot_stiffness = self.stiffness_state["rot_stiffness"]
            else:
                stiffness = 140.0
                rot_stiffness = 1.0
            force_info = np.array([-1, 0, 0, 0, 0, 0, 0, stiffness, rot_stiffness])

        self.all_force_data.append(force_info)

        ff_stiffness, rot_ff_stiffness = 0.0, 0.0
        ff_setpoint_pos = np.zeros(3)
        ff_setpoint_rot_quat = np.array([0.0, 0.0, 0.0, 1.0])
        plane_normal = np.zeros(3)

        if self.current_event and self.config.force_mode in [
            "forcefield",
            "collision-emulator",
            "collision-emulator-1d",
        ]:
            ff_stiffness = self.current_event.get("forcefield_stiffness", 0.0)
            rot_ff_stiffness = self.current_event.get("rotational_forcefield_stiffness", 0.0)
            if (
                self.config.force_mode == "collision-emulator-1d"
                and "collision_plane_normal" in self.current_event
            ):
                ff_setpoint_pos = (
                    self.current_event["collision_plane_origin"]
                    + self.current_event.get("plane_velocity_vec", np.zeros(3))
                    * max(0, current_time - self.current_event["initial_spawn_time"])
                )
                plane_normal = self.current_event["collision_plane_normal"]
            elif "forcefield_setpoint_pos" in self.current_event:
                ff_setpoint_pos = self.current_event["forcefield_setpoint_pos"]
            if "forcefield_setpoint_rot" in self.current_event:
                ff_setpoint_rot_quat = self.current_event["forcefield_setpoint_rot"].as_quat()

        self.all_collision_metadata.append(
            np.concatenate(
                [
                    [ff_stiffness, rot_ff_stiffness],
                    ff_setpoint_pos,
                    ff_setpoint_rot_quat,
                    plane_normal,
                ]
            )
        )

    def _update_viewer(self, vis_data: Tuple, current_time: float, loop_start: float):
        if not (self.is_interactive and self.viewer.is_running()):
            return

        if self.camera_tracking:
            self.viewer.cam.lookat[:] = self.ik_solver.data.body("torso_link").xpos
        self.viewer.user_scn.ngeom = 0
        add_visual_overlays(
            self.viewer.user_scn,
            self.ik_solver.data,
            *vis_data,
            self.current_event,
            current_time,
            self.rewind_indicator_until,
        )
        self.viewer.sync()
        time.sleep(max(0, self.timestep - (time.time() - loop_start)))

    def _capture_video_frame(self, vis_data: Tuple, current_time: float):
        if not (self.config.record_video and self.renderer):
            return

        active_cam = (
            self.viewer.cam if (self.is_interactive and self.viewer.is_running()) else self.cam
        )
        if not (self.is_interactive and self.viewer.is_running()):
            active_cam.lookat[:] = self.ik_solver.data.body("torso_link").xpos
        self.renderer.update_scene(self.ik_solver.data, camera=active_cam)
        add_visual_overlays(
            self.renderer.scene,
            self.ik_solver.data,
            *vis_data,
            self.current_event,
            current_time,
            self.rewind_indicator_until,
        )
        self.frames.append(self.renderer.render())

    def _teardown_rendering(self):
        if self.viewer:
            self.viewer.close()
        if self.config.record_video and self.frames:
            print(
                f"\nSaving video with {len(self.frames)} frames to '{self.config.output_filename}'..."
            )
            try:
                imageio.mimsave(
                    self.config.output_filename,
                    self.frames,
                    fps=int(1.0 / self.timestep),
                    quality=7,
                )
                print("Video saved successfully.")
            except Exception as exc:
                print(f"Error saving video: {exc}")

    def _save_outputs(self):
        num_frames = len(self.all_adapted_qpos)
        if not (
            len(self.all_reference_qpos) == num_frames
            and len(self.all_force_data) == num_frames
            and len(self.all_collision_metadata) == num_frames
        ):
            print(
                "Error: Data length mismatch! Skipping file save. lengths: "
                f"{len(self.all_adapted_qpos)}, {len(self.all_reference_qpos)}, "
                f"{len(self.all_force_data)}, {len(self.all_collision_metadata)}"
            )
            return

        print(
            f"\n[File {self.file_index + 1}/{self.config.num_files}] "
            f"Saving CSV with {num_frames} rows..."
        )
        ref_df = pd.DataFrame(
            np.array(self.all_reference_qpos)[:, : self.ik_solver.model.nq + 2]
        )
        adapted_df = pd.DataFrame(np.array(self.all_adapted_qpos))
        force_df = pd.DataFrame(np.array(self.all_force_data))
        collision_df = pd.DataFrame(np.array(self.all_collision_metadata))

        out_df = pd.concat([ref_df, adapted_df, force_df, collision_df], axis=1)
        basename = os.path.splitext(os.path.basename(self.config.motion_path))[0]
        out_path = os.path.join(
            self.config.output_dir,
            f"{basename}_augmented_mink_{self.file_index + 1:03d}.csv",
        )
        out_df.to_csv(out_path, index=False, header=False)
        print(f"Saved to '{out_path}'")


def run_simulation_or_generation(config: SimulationConfig, file_index: int = 0):
    SimulationRunner(config, file_index).run()
