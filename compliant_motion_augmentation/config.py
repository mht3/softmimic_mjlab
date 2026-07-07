from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """Configuration settings for the simulation or data generation process."""

    mode: str
    model_path: str
    motion_path: str
    seed: int
    com_cost: float
    com_cost_z_factor: float
    force_mode: str
    upper_joint_cost: float = 0.0
    torso_orientation_cost: float = 0.0
    repeat_frame_time: Optional[float] = None
    record_video: bool = False
    output_filename: str = "ik_simulation.mp4"
    num_files: int = 10
    output_dir: str = "./augmented_data"
