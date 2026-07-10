import argparse
import multiprocessing
import os
import sys
from dataclasses import replace as dataclasses_replace

import pandas as pd

try:
    import mink
except ImportError:  # pragma: no cover - handled at runtime
    mink = None

try:
    from softmimic_deploy.src.motion_lib.motion_lib_from_multi_csv import (
        ProceduralMotionLibFromDemo,
    )
except ImportError:  # pragma: no cover - handled at runtime
    ProceduralMotionLibFromDemo = None

if __package__ is None or __package__ == "":
    # Running as a standalone script (e.g. `python mink_generator_ff.py`).
    # Import via the real package name so sibling modules' relative imports
    # (e.g. `from .constants import ...`) still resolve.
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from compliant_motion_augmentation.config import SimulationConfig
    from compliant_motion_augmentation.ik_solver import G1_Mink_IK_Solver
    from compliant_motion_augmentation.runner import run_simulation_or_generation
else:
    from .config import SimulationConfig
    from .ik_solver import G1_Mink_IK_Solver
    from .runner import run_simulation_or_generation

__all__ = ["SimulationConfig", "G1_Mink_IK_Solver", "run_simulation_or_generation"]


def main():
    if any(lib is None for lib in [mink, ProceduralMotionLibFromDemo, pd]):
        print("Aborting due to missing dependencies.")
        return

    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Solve IK on the 23-DOF robot we deploy. The reference CSV must be a 23-DOF
    # motion (see scripts/csv_29dof_to_23dof.py); pass --model_path with the
    # 29-DOF XML to run the legacy 29-DOF pipeline against a 29-DOF CSV.
    default_model_path = os.path.abspath(
        os.path.join(script_dir, "../softmimic_deploy/src/assets/g1/g1_23dof.xml")
    )

    parser = argparse.ArgumentParser(
        description="Run G1 IK solver using 'mink' with multiple force models and dynamic rejection sampling."
    )
    parser.add_argument(
        "mode", choices=["interactive", "generate-data"], help="Execution mode."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=default_model_path,
        help="Path to the MuJoCo XML model file.",
    )
    parser.add_argument(
        "--motion_path",
        type=str,
        required=True,
        help="Path to the reference motion CSV file.",
    )
    parser.add_argument(
        "--force_mode",
        type=str,
        choices=["triangle", "forcefield", "collision-emulator", "collision-emulator-1d", "zero-wrench"],
        default="triangle",
        help="The model used to generate external forces.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for force profile generation."
    )
    parser.add_argument(
        "--com_cost",
        type=float,
        default=0.1,
        help="Cost for the CoM task in the XY plane.",
    )
    parser.add_argument(
        "--com_cost_z_factor",
        type=float,
        default=0.00001,
        help="Multiplier for CoM cost in Z.",
    )
    parser.add_argument(
        "--upper_joint_cost",
        type=float,
        default=0.0,
        help="Cost for the upper body joints.",
    )
    parser.add_argument(
        "--torso_orientation_cost",
        type=float,
        default=0.0,
        help="Cost for the torso orientation.",
    )
    parser.add_argument(
        "--repeat_frame_time",
        type=float,
        default=None,
        help="If set, freezes the reference motion at this time (in seconds).",
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=10,
        help="Number of augmented data files to generate in 'generate-data' mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./augmented_data_mink",
        help="Directory to save generated data files.",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Record the simulation to a video file.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="ik_simulation.mp4",
        help="Name of the output video file.",
    )

    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        return

    config = SimulationConfig(**vars(args))

    if config.mode == "interactive":
        run_simulation_or_generation(config)
    elif config.mode == "generate-data":
        if config.record_video:
            print(
                "Warning: Video recording is enabled in 'generate-data' mode. This will record the first generated file only."
            )
            run_simulation_or_generation(config, 0)
            tasks = [
                (dataclasses_replace(config, record_video=False, seed=config.seed + i), i)
                for i in range(1, config.num_files)
            ]
        else:
            tasks = [
                (dataclasses_replace(config, seed=config.seed + i), i)
                for i in range(config.num_files)
            ]

        if tasks:
            num_workers = min(10, len(tasks))
            print(
                f"\n--- Starting parallel data generation for {len(tasks)} files using up to {num_workers} workers ---"
            )
            with multiprocessing.Pool(processes=num_workers) as pool:
                pool.starmap(run_simulation_or_generation, tasks)
        print("\n--- Data Generation Complete ---")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
