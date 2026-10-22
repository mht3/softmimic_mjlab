"""Render the G1 23-DoF robot over the offline dataset, viser-ghost style.

Uses the same ghost convention as the mjlab viser viewer (mjlab
``viewer/viser_visualizer.py::add_ghost_mesh`` driven by
``MotionCommandCfg.VizCfg.ghost_color``): the robot's own visual meshes,
tinted light green and translucent. Rendering goes through MuJoCo's offscreen
renderer so the geometry is exactly what the simulator/viewer draws.

Two outputs:
  * ``ghost_reference_*.png`` — a single reference stand pose.
  * ``ghost_overlay_*.png``   — many dataset poses drawn at once (one giant
    green blob showing the training-data distribution).

The overlay replicates the robot body N times inside one MjSpec, so every
sampled pose is a real robot render in the same scene.

Usage:
  python scripts/render_offline_data_ghost.py
  python scripts/render_offline_data_ghost.py --num-poses 400 --motions stand
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "src" / "assets" / "compliant_motions" / "g1_23dof"
G1_23DOF_XML = (
  REPO_ROOT / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "g1_23dof.xml"
)

# mjlab viser ghost color: MotionCommandCfg.VizCfg.ghost_color.
GHOST_COLOR = (0.5, 0.7, 0.5, 0.5)
# Visual geoms are group 2, collision geoms group 3 (see the G1 MJCF).
VISUAL_GEOM_GROUP = 2

# (name, azimuth, elevation) — distance/lookat are fit to the pose set.
VIEWS = (
  ("front", 90.0, -10.0),
  ("side", 0.0, -10.0),
  ("iso", 135.0, -20.0),
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
  parser.add_argument(
    "--motions", nargs="*", default=None, help="Motion names (e.g. stand walk)."
  )
  parser.add_argument(
    "--modes", nargs="*", default=None, help="Force modes (e.g. forcefield)."
  )
  parser.add_argument(
    "--num-poses",
    type=int,
    default=2000,
    help="Number of dataset poses drawn as full ghost robots in the overlay.",
  )
  parser.add_argument(
    "--overlay-alpha",
    type=float,
    default=None,
    help="Per-robot alpha in the overlay (default: auto from --num-poses).",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=25,
    help="Robots posed per render pass; frames are composited (memory bound).",
  )
  parser.add_argument(
    "--pelvis-relative",
    action="store_true",
    default=True,
    help="Align every pose at the pelvis (default) so poses overlap.",
  )
  parser.add_argument(
    "--world-frame",
    dest="pelvis_relative",
    action="store_false",
    help="Keep world root positions instead of aligning at the pelvis.",
  )
  parser.add_argument("--width", type=int, default=1600)
  parser.add_argument("--height", type=int, default=1600)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--output-dir", type=Path, default=None)
  return parser.parse_args()


def discover_files(
  data_root: Path, motions: list[str] | None, modes: list[str] | None
) -> list[Path]:
  files = []
  for f in sorted(data_root.rglob("*.npz")):
    rel = f.relative_to(data_root)
    motion = rel.parts[0] if len(rel.parts) > 1 else ""
    mode = rel.parts[1] if len(rel.parts) > 2 else ""
    if motions and motion not in motions:
      continue
    if modes and mode not in modes:
      continue
    files.append(f)
  if not files:
    raise FileNotFoundError(f"No NPZ files found under {data_root}")
  return files


def reference_file(files: list[Path]) -> Path:
  """Prefer a zero-wrench stand file (adapted motion == nominal reference)."""
  for f in files:
    if "zero-wrench" in f.parts and "stand" in f.parts:
      return f
  return files[0]


def sample_poses(
  files: list[Path], num_poses: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Sample dataset frames as (root_pos, root_quat_wxyz, joint_pos)."""
  rng = np.random.default_rng(seed)
  per_file = max(1, round(num_poses / len(files)))
  roots, quats, joints = [], [], []
  for f in files:
    with np.load(f) as data:
      n = data["joint_pos"].shape[0]
      idx = rng.choice(n, size=min(per_file, n), replace=False)
      idx.sort()
      roots.append(data["body_pos_w"][idx, 0])
      quats.append(data["body_quat_w"][idx, 0])
      joints.append(data["joint_pos"][idx])
  root = np.concatenate(roots).astype(np.float64)
  quat = np.concatenate(quats).astype(np.float64)
  joint = np.concatenate(joints).astype(np.float64)
  if len(root) > num_poses:  # trim evenly across files
    keep = rng.choice(len(root), size=num_poses, replace=False)
    keep.sort()
    root, quat, joint = root[keep], quat[keep], joint[keep]
  return root, quat, joint


def build_ghost_spec(num_copies: int, alpha: float) -> mujoco.MjSpec:
  """MjSpec holding ``num_copies`` ghost-tinted robots on a blank scene.

  Every copy is a complete robot with its own free joint, so each dataset
  frame renders as a full ghost robot rather than a set of joint points.
  Copies duplicate the mesh assets, so keep this to a few dozen and use
  ``accumulate_overlay`` for larger pose counts.
  """
  spec = mujoco.MjSpec.from_file(str(G1_23DOF_XML))
  for geom in spec.geoms:
    if geom.group == VISUAL_GEOM_GROUP:
      geom.rgba = [*GHOST_COLOR[:3], alpha]
      geom.material = ""  # drop textures/materials so the tint is uniform
    else:
      geom.group = 4  # collision geoms: hidden group
  if num_copies == 1:
    return spec

  merged = mujoco.MjSpec()
  merged.copy_during_attach = True
  for i in range(num_copies):
    frame = merged.worldbody.add_frame()
    frame.attach_body(spec.worldbody.first_body(), f"g{i}_", "")
  return merged


def set_pose(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  copy_idx: int,
  root_pos: np.ndarray,
  root_quat: np.ndarray,
  joint_pos: np.ndarray,
) -> None:
  """Write one robot copy's free joint + 23 hinge joints into qpos."""
  stride = 7 + joint_pos.shape[0]
  adr = copy_idx * stride
  data.qpos[adr : adr + 3] = root_pos
  data.qpos[adr + 3 : adr + 7] = root_quat  # wxyz, as stored in the NPZ
  data.qpos[adr + 7 : adr + stride] = joint_pos


def style_scene(spec: mujoco.MjSpec) -> None:
  """Light viser-like canvas: pale gradient skybox, soft headlight."""
  spec.visual.headlight.ambient = [0.55, 0.55, 0.55]
  spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
  spec.visual.headlight.specular = [0.1, 0.1, 0.1]
  spec.visual.rgba.haze = [1.0, 1.0, 1.0, 0.0]

  skybox = None
  for texture in spec.textures:
    if texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
      skybox = texture
      break
  if skybox is None:
    skybox = spec.add_texture()
    skybox.name = "ghost_skybox"
    skybox.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    skybox.width, skybox.height = 512, 512
  skybox.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
  skybox.rgb1 = [0.97, 0.97, 0.98]
  skybox.rgb2 = [0.85, 0.86, 0.89]


def make_renderer(model: mujoco.MjModel, width: int, height: int):
  model.vis.global_.offwidth = width
  model.vis.global_.offheight = height
  renderer = mujoco.Renderer(model, height=height, width=width, max_geom=200_000)
  # Visual geoms only; no collision geoms, no world frame decorations.
  renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
  renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
  return renderer


def fit_camera(data: mujoco.MjData, model: mujoco.MjModel) -> tuple[np.ndarray, float]:
  """Lookat + distance that frame all rendered bodies."""
  pos = data.xpos[1:]  # skip the world body
  lo, hi = pos.min(axis=0), pos.max(axis=0)
  lookat = (lo + hi) / 2
  extent = float(np.linalg.norm(hi - lo))
  return lookat, max(extent * 1.25, 1.5)


def visual_only_options() -> mujoco.MjvOption:
  options = mujoco.MjvOption()
  options.geomgroup[:] = 0
  options.geomgroup[VISUAL_GEOM_GROUP] = 1  # visual meshes only
  return options


def make_camera(lookat: np.ndarray, distance: float) -> mujoco.MjvCamera:
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = lookat
  camera.distance = distance
  return camera


def render_views(
  data: mujoco.MjData,
  renderer,
  out_dir: Path,
  prefix: str,
  lookat: np.ndarray,
  distance: float,
) -> None:
  options = visual_only_options()
  camera = make_camera(lookat, distance)
  for name, azimuth, elevation in VIEWS:
    camera.azimuth = azimuth
    camera.elevation = elevation
    renderer.update_scene(data, camera=camera, scene_option=options)
    path = out_dir / f"{prefix}_{name}.png"
    Image.fromarray(renderer.render()).save(path)
    print(f"Saved {path}")


def accumulate_overlay(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  renderer,
  poses: tuple[np.ndarray, np.ndarray, np.ndarray],
  batch_size: int,
  lookat: np.ndarray,
  distance: float,
) -> dict[str, np.ndarray]:
  """Render every pose as a full ghost robot and composite the frames.

  Copying the robot thousands of times inside one model exhausts memory (the
  mesh assets are duplicated per copy), so a small batch of robots is posed
  and rendered repeatedly, and the frames are combined with a darken blend.
  Overlapping ghosts therefore accumulate exactly as they would in one scene.
  """
  root, quat, joint = poses
  num = len(root)
  options = visual_only_options()
  camera = make_camera(lookat, distance)
  accumulators = {
    name: np.full((renderer.height, renderer.width, 3), 255, dtype=np.uint8)
    for name, _, _ in VIEWS
  }

  for start in range(0, num, batch_size):
    chunk = slice(start, min(start + batch_size, num))
    n_chunk = len(root[chunk])
    for i in range(batch_size):
      # Unused copies are parked far below the camera frustum.
      if i < n_chunk:
        set_pose(model, data, i, root[chunk][i], quat[chunk][i], joint[chunk][i])
      else:
        set_pose(model, data, i, np.array([0.0, 0.0, -50.0]), quat[0], joint[0])
    # Kinematics only: mj_forward would run collision detection between the
    # interpenetrating ghost copies and overflow the constraint stack.
    mujoco.mj_kinematics(model, data)
    for name, azimuth, elevation in VIEWS:
      camera.azimuth = azimuth
      camera.elevation = elevation
      renderer.update_scene(data, camera=camera, scene_option=options)
      frame = renderer.render()
      np.minimum(accumulators[name], frame, out=accumulators[name])
    done = min(start + batch_size, num)
    print(f"  rendered {done}/{num} ghost poses", end="\r", flush=True)
  print()
  return accumulators


def main() -> None:
  args = parse_args()
  files = discover_files(args.data_root, args.motions, args.modes)
  print(f"Found {len(files)} NPZ files")

  out_dir = args.output_dir
  if out_dir is None:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = REPO_ROOT / "logs" / "offline_data_viz" / f"ghost_{stamp}"
  out_dir.mkdir(parents=True, exist_ok=True)

  # -- Reference stand pose --------------------------------------------------
  ref = reference_file(files)
  with np.load(ref) as data_npz:
    ref_root = data_npz["body_pos_w"][0, 0].astype(np.float64)
    ref_quat = data_npz["body_quat_w"][0, 0].astype(np.float64)
    ref_joint = data_npz["joint_pos"][0].astype(np.float64)
  print(f"Reference pose from {ref.relative_to(args.data_root)}")

  spec = build_ghost_spec(1, GHOST_COLOR[3])
  style_scene(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  set_pose(model, data, 0, np.array([0.0, 0.0, ref_root[2]]), ref_quat, ref_joint)
  mujoco.mj_forward(model, data)
  renderer = make_renderer(model, args.width, args.height)
  lookat, distance = fit_camera(data, model)
  render_views(data, renderer, out_dir, "ghost_reference", lookat, distance)
  # Kept alive (not closed, not dropped): mujoco's EGL context is
  # process-global, so freeing this renderer — including via garbage
  # collection — tears the context down for the overlay renderer below.
  reference_renderer = renderer

  # -- Overlay of many dataset poses ----------------------------------------
  root, quat, joint = sample_poses(files, args.num_poses, args.seed)
  num = len(root)
  print(f"Overlay: {num} poses from the dataset")

  if args.pelvis_relative:
    # Align every pose at the pelvis so the cloud of poses overlaps.
    root = np.tile([0.0, 0.0, float(np.mean(root[:, 2]))], (num, 1))

  # Many overlapping ghosts saturate to opaque, so thin each one out; the
  # blob's density then reads as how often the data visits that pose.
  alpha = args.overlay_alpha
  if alpha is None:
    alpha = float(np.clip(30.0 / num, 0.04, 0.5))
  batch = min(args.batch_size, num)
  print(f"Overlay: ghost alpha {alpha:.3f}, {batch} robots per render pass")

  spec = build_ghost_spec(batch, alpha)
  style_scene(spec)
  model = spec.compile()
  data = mujoco.MjData(model)

  # Fit the camera to the whole pose set: sweep every pose through copy 0 and
  # take the extent of all body positions, so nothing is cropped.
  lo = np.full(3, np.inf)
  hi = np.full(3, -np.inf)
  bodies_per_copy = (model.nbody - 1) // batch
  for i in range(num):
    set_pose(model, data, 0, root[i], quat[i], joint[i])
    mujoco.mj_kinematics(model, data)
    pos = data.xpos[1 : 1 + bodies_per_copy]
    lo = np.minimum(lo, pos.min(axis=0))
    hi = np.maximum(hi, pos.max(axis=0))
  lookat = (lo + hi) / 2
  distance = max(float(np.linalg.norm(hi - lo)) * 1.25, 1.5)

  renderer = make_renderer(model, args.width, args.height)
  frames = accumulate_overlay(
    model, data, renderer, (root, quat, joint), batch, lookat, distance
  )
  for name, image in frames.items():
    path = out_dir / f"ghost_overlay_{name}.png"
    Image.fromarray(image).save(path)
    print(f"Saved {path}")

  del reference_renderer
  print(f"Done. Images in {out_dir}")


if __name__ == "__main__":
  main()
