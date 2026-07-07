from typing import Any, Dict, List
import random

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from config import SimulationConfig
from constants import (
    DOWNWARD_ONLY_FORCEABLE_LINKS,
    MAX_FORCEFIELD_ROTATIONAL_STIFFNESS,
    MAX_FORCEFIELD_STIFFNESS,
    MAX_ROBOT_ROTATIONAL_STIFFNESS,
    MAX_ROBOT_STIFFNESS,
    MIN_FORCEFIELD_ROTATIONAL_STIFFNESS,
    MIN_FORCEFIELD_STIFFNESS,
    MIN_ROBOT_ROTATIONAL_STIFFNESS,
    MIN_ROBOT_STIFFNESS,
)
from ik_solver import G1_Mink_IK_Solver


def generate_random_force_profile(total_duration: float, possible_links: List[str], force_mode: str, ik_solver: G1_Mink_IK_Solver, config: SimulationConfig) -> List[Dict[str, Any]]:
    
    if config.force_mode == 'zero-wrench':
        print("Zero-wrench mode enabled: No force events will be generated.")
        return []
    
    profile = []
    temp_data = mj.MjData(ik_solver.model)

    v_norm = 3.0
    if force_mode in ['collision-emulator', 'collision-emulator-1d', 'forcefield']:
        print("Pre-computing reference motion velocity profile...")
        all_velocities = []
        for t in np.arange(0.0, total_duration, 0.02 * 5):
            q_t, _, _ = ik_solver.get_reference_motion(t); q_t_minus_dt, _, _ = ik_solver.get_reference_motion(t - 0.02)
            vel_ref = np.zeros(ik_solver.model.nv); mj.mj_differentiatePos(ik_solver.model, vel_ref, 0.02, q_t_minus_dt, q_t)
            temp_data.qpos[:], temp_data.qvel[:] = q_t, vel_ref; mj.mj_forward(ik_solver.model, temp_data)
            for link_name in possible_links:
                link_id = mj.mj_name2id(ik_solver.model, mj.mjtObj.mjOBJ_BODY, link_name)
                link_vel_vec = np.zeros(6); mj.mj_objectVelocity(ik_solver.model, temp_data, mj.mjtObj.mjOBJ_BODY, link_id, link_vel_vec, 0)
                all_velocities.append(np.linalg.norm(link_vel_vec[:3]))
        if all_velocities: v_norm = max(np.percentile(all_velocities, 90), 0.5)
        print(f"Velocity normalization factor for sampling set to: {v_norm:.2f} m/s")

    if force_mode == 'collision-emulator-1d':
        print("Pre-computing all possible collision events (without feasibility pre-check)...")
        candidate_events = []
        MIN_INTERACTION_TIME, SPAWN_BUFFER = 0.2, 0.01
        for t_spawn in tqdm(np.arange(0.0, total_duration, 0.1)):
            link_name = random.choice(possible_links)
            q_t, _, _ = ik_solver.get_reference_motion(t_spawn); q_t_minus_dt, _, _ = ik_solver.get_reference_motion(t_spawn - 0.02)
            vel_ref = np.zeros(ik_solver.model.nv); mj.mj_differentiatePos(ik_solver.model, vel_ref, 0.02, q_t_minus_dt, q_t)
            temp_data.qpos[:], temp_data.qvel[:] = q_t, vel_ref; mj.mj_forward(ik_solver.model, temp_data)
            link_id = mj.mj_name2id(ik_solver.model, mj.mjtObj.mjOBJ_BODY, link_name)
            link_vel_vec = np.zeros(6); mj.mj_objectVelocity(ik_solver.model, temp_data, mj.mjtObj.mjOBJ_BODY, link_id, link_vel_vec, 0)
            link_vel, link_vel_mag = link_vel_vec[:3], np.linalg.norm(link_vel_vec[:3])
            if random.random() > np.clip(link_vel_mag / v_norm, 0.0, 1.0): continue
            p_ref_start = temp_data.body(link_name).xpos.copy()
            plane_normal = link_vel / link_vel_mag if link_vel_mag > 1e-6 else np.zeros(3)
            if link_vel_mag > 1e-4 and np.dot(plane_normal, link_vel) > 0: plane_normal = -plane_normal
            plane_origin = p_ref_start - plane_normal * SPAWN_BUFFER
            plane_velocity_vec = plane_normal * 0.01
            penetration_times = []

            for t_future in np.arange(t_spawn, min(t_spawn + 1.5, total_duration), 0.02):
                q_future, _, _ = ik_solver.get_reference_motion(t_future); temp_data.qpos[:] = q_future; mj.mj_forward(ik_solver.model, temp_data)
                penetration = -np.dot(temp_data.body(link_name).xpos.copy() - plane_origin, plane_normal)
                if penetration > 0: penetration_times.append(t_future)
            
            if penetration_times and (penetration_times[-1] - penetration_times[0]) >= MIN_INTERACTION_TIME:
                event_params = {'stiffness': np.exp(random.uniform(np.log(MIN_ROBOT_STIFFNESS), np.log(MAX_ROBOT_STIFFNESS))), 'forcefield_stiffness': np.exp(random.uniform(np.log(MIN_FORCEFIELD_STIFFNESS), np.log(MAX_FORCEFIELD_STIFFNESS)))}
                
                torque_duration = random.uniform(0.5, penetration_times[-1] - penetration_times[0])
                torque_start_offset = random.uniform(0, (penetration_times[-1] - penetration_times[0]) - torque_duration)
                event_params['torque_start_time'] = penetration_times[0] + torque_start_offset
                event_params['torque_end_time'] = event_params['torque_start_time'] + torque_duration
                torque_hold_duration = random.uniform(0.15, 0.5) * torque_duration
                event_params['torque_ramp_duration'] = max(0.1, (torque_duration - torque_hold_duration) / 2.0)
                event_params['torque_hold_start_time'] = event_params['torque_start_time'] + event_params['torque_ramp_duration']
                event_params['torque_hold_end_time'] = event_params['torque_hold_start_time'] + torque_hold_duration
                
                torque_range, rot_stiff_range, rot_disp_range = (0.0, 10.0), (0.3, 30.0), (0.0, 2.0)
                while True:
                    rot_stiff = np.exp(random.uniform(np.log(rot_stiff_range[0]), np.log(rot_stiff_range[1]))); lower_rd, upper_rd = max(rot_disp_range[0], torque_range[0]/rot_stiff), min(rot_disp_range[1], torque_range[1]/rot_stiff)
                    if lower_rd < upper_rd: break
                event_params['rotational_stiffness'] = rot_stiff
                event_params['rotational_forcefield_stiffness'] = np.exp(random.uniform(np.log(MIN_FORCEFIELD_ROTATIONAL_STIFFNESS), np.log(MAX_FORCEFIELD_ROTATIONAL_STIFFNESS)))
                torque_amplitude = rot_stiff * random.uniform(lower_rd, upper_rd)
                torque_axis = np.random.randn(3); torque_axis /= np.linalg.norm(torque_axis)
                event_params['torque_amplitude'] = torque_amplitude
                event_params['torque_axis'] = torque_axis
                if np.linalg.norm(event_params['torque_axis']) < 1e-6: continue
                
                event_params['collision_plane_origin'] = plane_origin
                event_params['collision_plane_normal'] = plane_normal
                event_params['plane_velocity_vec'] = plane_velocity_vec
                event_params['initial_spawn_time'] = t_spawn
                
                hold_start_time = penetration_times[0]
                hold_end_time = penetration_times[-1]
                ramp_duration = max(0.1, (hold_end_time - hold_start_time) * 0.3)
                final_event = {
                    'start_time': hold_start_time - ramp_duration,
                    'hold_start_time': hold_start_time,
                    'hold_end_time': hold_end_time,
                    'end_time': hold_end_time + ramp_duration,
                    'ramp_duration': ramp_duration,
                    'link_name': link_name,
                    **event_params
                }
                candidate_events.append(final_event)
        
        candidate_events.sort(key=lambda x: x['start_time'])
        last_event_end_time = -np.inf
        for cand in candidate_events:
            if cand['start_time'] >= last_event_end_time:
                profile.append(cand); last_event_end_time = cand['end_time']
    else:
        current_time = 0.0
        while current_time < total_duration:
            wait_duration = random.uniform(0.5, 1.5); start_time = current_time + wait_duration
            if start_time > total_duration: break
            link_name = random.choice(possible_links)
            event_duration = random.uniform(2.0, 4.0)
            event_params = {}
            
            q_t_start, _, _ = ik_solver.get_reference_motion(start_time)
            temp_data.qpos[:] = q_t_start; mj.mj_forward(ik_solver.model, temp_data)
            p_ref = temp_data.body(link_name).xpos.copy()
            r_ref = Rotation.from_matrix(temp_data.body(link_name).xmat.reshape(3,3))

            if force_mode == 'collision-emulator':
                 q_t_minus_dt, _, _ = ik_solver.get_reference_motion(start_time - 0.02)
                 if start_time - 0.02 < 0: current_time = start_time + 0.5; continue
                 vel_ref = np.zeros(ik_solver.model.nv); mj.mj_differentiatePos(ik_solver.model, vel_ref, 0.02, q_t_minus_dt, q_t_start)
                 temp_data.qvel[:] = vel_ref; mj.mj_forward(ik_solver.model, temp_data)
                 link_id = mj.mj_name2id(ik_solver.model, mj.mjtObj.mjOBJ_BODY, link_name)
                 link_vel_vec = np.zeros(6); mj.mj_objectVelocity(ik_solver.model, temp_data, mj.mjtObj.mjOBJ_BODY, link_id, link_vel_vec, 0)
                 if random.random() > np.clip(np.linalg.norm(link_vel_vec[:3]) / v_norm, 0.0, 1.0): current_time = start_time + 0.5; continue
                 
                 # The setpoint is the link's pose at the start of the event.
                 event_params['forcefield_setpoint_pos'] = p_ref
                 event_params['forcefield_setpoint_rot'] = r_ref
                 
                 # Sample stiffness properties
                 stiffness = np.exp(random.uniform(np.log(MIN_ROBOT_STIFFNESS), np.log(MAX_ROBOT_STIFFNESS)))
                 rot_stiffness = np.exp(random.uniform(np.log(MIN_ROBOT_ROTATIONAL_STIFFNESS), np.log(MAX_ROBOT_ROTATIONAL_STIFFNESS)))
                 event_params.update({
                     'stiffness': stiffness,
                     'forcefield_stiffness': np.exp(random.uniform(np.log(MIN_FORCEFIELD_STIFFNESS), np.log(MAX_FORCEFIELD_STIFFNESS))),
                     'rotational_stiffness': rot_stiffness,
                     'rotational_forcefield_stiffness': np.exp(random.uniform(np.log(MIN_FORCEFIELD_ROTATIONAL_STIFFNESS), np.log(MAX_FORCEFIELD_ROTATIONAL_STIFFNESS)))
                 })

                 ramp_duration = random.uniform(0.2, 1.0)

            elif force_mode in ['forcefield', 'triangle']:
                # Both modes sample a target force/torque profile first.
                force_range, stiffness_range, displacement_range = (0.0, 140.0), (MIN_ROBOT_STIFFNESS, MAX_ROBOT_STIFFNESS), (0.0, 0.7)
                stiffness = np.exp(random.uniform(np.log(stiffness_range[0]), np.log(stiffness_range[1])))
                lower_d, upper_d = max(displacement_range[0], force_range[0] / stiffness), min(displacement_range[1], force_range[1] / stiffness)
                if lower_d >= upper_d: current_time = start_time + 0.5; continue
                
                amplitude = stiffness * random.uniform(lower_d, upper_d)
                force_axis = np.random.randn(3); force_axis /= np.linalg.norm(force_axis)
                if link_name in DOWNWARD_ONLY_FORCEABLE_LINKS: force_axis[2] = -abs(force_axis[2])

                torque_range, rot_stiff_range, rot_disp_range = (0.0, 10.0), (MIN_ROBOT_ROTATIONAL_STIFFNESS, MAX_ROBOT_ROTATIONAL_STIFFNESS), (0.0, 2.0)
                rot_stiff = np.exp(random.uniform(np.log(rot_stiff_range[0]), np.log(rot_stiff_range[1])))
                lower_rd, upper_rd = max(rot_disp_range[0], torque_range[0]/rot_stiff), min(rot_disp_range[1], torque_range[1]/rot_stiff)
                if lower_rd >= upper_rd: current_time = start_time + 0.5; continue
                torque_amplitude = rot_stiff * random.uniform(lower_rd, upper_rd)
                torque_axis = np.random.randn(3); torque_axis /= np.linalg.norm(torque_axis)

                event_params['stiffness'] = stiffness
                event_params['rotational_stiffness'] = rot_stiff
                
                # Store the peak force/torque parameters directly.
                event_params['amplitude'] = amplitude
                event_params['force_axis'] = force_axis
                event_params['torque_amplitude'] = torque_amplitude
                event_params['torque_axis'] = torque_axis
                
                if force_mode == 'forcefield':
                    # Forcefield additionally needs the environment stiffness properties.
                    event_params['forcefield_stiffness'] = np.exp(random.uniform(np.log(MIN_FORCEFIELD_STIFFNESS), np.log(MAX_FORCEFIELD_STIFFNESS)))
                    event_params['rotational_forcefield_stiffness'] = np.exp(random.uniform(np.log(MIN_FORCEFIELD_ROTATIONAL_STIFFNESS), np.log(MAX_FORCEFIELD_ROTATIONAL_STIFFNESS)))

                # hold_duration = random.uniform(0.15, 0.5) * event_duration
                # ramp_duration = max(0.1, (event_duration - hold_duration) / 2.0)
                
                # 1. Sample target velocities from desired distributions.
                #    Example: Uniform distribution from 0.0 to 2.0 m/s for linear motion.
                target_linear_velocity = random.uniform(0.1, 1.0)  # Min 0.1 to avoid division by zero
                target_angular_velocity = random.uniform(0.2, 4.0) # rad/s

                # 2. Calculate the required ramp duration for both linear and angular components.
                #    Formula: Ramp Duration = Peak Force / (Stiffness * Target Velocity)
                required_ramp_lin = 0.0
                if target_linear_velocity > 1e-6 and stiffness > 1e-6 and amplitude > 1e-6:
                    required_ramp_lin = amplitude / (stiffness * target_linear_velocity)

                # required_ramp_rot = 0.0
                # if target_angular_velocity > 1e-6 and rot_stiff > 1e-6 and torque_amplitude > 1e-6:
                #     required_ramp_rot = torque_amplitude / (rot_stiff * target_angular_velocity)
                
                # 3. Use the LONGER of the two required durations to ensure neither velocity exceeds its target.
                #    Also, clamp the duration to a reasonable range for stability and realism.
                ramp_duration = required_ramp_lin
                ramp_duration = np.clip(ramp_duration, 0.1, 2.0) # Clamp to [0.2s, 2.0s]

            # 4. Reconstruct the other timings based on the new ramp_duration.
            #    Let's make the hold duration proportional to the ramp duration.
            hold_duration = random.uniform(0.5, 1.0)# * ramp_duration
            
            hold_start_time = start_time + ramp_duration
            hold_end_time = hold_start_time + hold_duration
            end_time = hold_end_time + ramp_duration
            if end_time > total_duration: break

            final_event = {
                'start_time': start_time, 'end_time': end_time, 
                'hold_start_time': hold_start_time, 'hold_end_time': hold_end_time, 
                'ramp_duration': ramp_duration, 'link_name': link_name, **event_params
            }
            profile.append(final_event)
            current_time = end_time
    return sorted(profile, key=lambda x: x['start_time'])
