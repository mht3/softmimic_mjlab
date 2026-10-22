"""Visualize the offline compliant-motion training data distribution.

Loads every augmented-motion NPZ under the data root, transforms all body
positions into the pelvis (root) frame, and overlays every frame as one 3D
point cloud — the spatial distribution the tracking policy is trained on.
A reference stand pose is drawn as a light-green skeleton (the viser ghost
color) and the cloud's convex hull as a dotted gray wireframe. Several camera
angles plus a per-mode facet figure are saved under ``logs/``.

Usage:
  python scripts/visualize_offline_data.py
  python scripts/visualize_offline_data.py --motions stand --max-points 200000
  python scripts/visualize_offline_data.py --color-by uniform --no-hull
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "src" / "assets" / "compliant_motions" / "g1_23dof"
G1_23DOF_XML = (
  REPO_ROOT / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "g1_23dof.xml"
)

# Force-event modes in fixed order with fixed categorical colors (validated
# palette; a mode keeps its color regardless of which modes are loaded).
MODE_ORDER = ("forcefield", "collision-emulator", "zero-wrench")
MODE_COLORS = {
  "forcefield": "#2a78d6",
  "collision-emulator": "#008300",
  "zero-wrench": "#e87ba4",
}
# Light green of the viser ghost robot (tracking/mdp/commands.py VizCfg).
GHOST_GREEN = (0.5, 0.7, 0.5)
HULL_GRAY = "#888888"

# Camera angles: (name, elevation, azimuth). Pelvis frame: x forward, z up.
VIEWS = (
  ("front", 10.0, 0.0),
  ("side", 10.0, 90.0),
  ("iso", 25.0, 45.0),
  ("top", 88.0, -90.0),
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--data-root",
    type=Path,
    default=DEFAULT_DATA_ROOT,
    help="Directory scanned recursively for motion NPZ files.",
  )
  parser.add_argument(
    "--motions",
    nargs="*",
    default=None,
    help="Motion names to include (subdirectory names, e.g. stand walk). "
    "Default: all.",
  )
  parser.add_argument(
    "--modes",
    nargs="*",
    default=None,
    choices=MODE_ORDER,
    help="Force-event modes to include. Default: all.",
  )
  parser.add_argument(
    "--max-points",
    type=int,
    default=600_000,
    help="Target total number of scatter points; sets the frame stride.",
  )
  parser.add_argument(
    "--frame-stride",
    type=int,
    default=None,
    help="Explicit frame subsampling stride (overrides --max-points).",
  )
  parser.add_argument(
    "--color-by",
    choices=("mode", "uniform"),
    default="mode",
    help="Color the cloud per force mode or uniformly in ghost green.",
  )
  parser.add_argument("--no-hull", action="store_true", help="Skip the convex hull.")
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Output directory (default: logs/offline_data_viz/<timestamp>).",
  )
  parser.add_argument("--dpi", type=int, default=200)
  return parser.parse_args()


def discover_files(
  data_root: Path, motions: list[str] | None, modes: list[str] | None
) -> list[tuple[Path, str, str]]:
  """All NPZ files under the root as (path, motion, mode) tuples."""
  entries: list[tuple[Path, str, str]] = []
  for f in sorted(data_root.rglob("*.npz")):
    rel = f.relative_to(data_root)
    motion = rel.parts[0] if len(rel.parts) > 1 else "unknown"
    mode = rel.parts[1] if len(rel.parts) > 2 else "unknown"
    if motions and motion not in motions:
      continue
    if modes and mode not in modes:
      continue
    entries.append((f, motion, mode))
  if not entries:
    raise FileNotFoundError(f"No NPZ files found under {data_root}")
  return entries


def load_pelvis_relative(path: Path, stride: int) -> np.ndarray:
  """Body positions of every stride-th frame in the pelvis frame, (F, B, 3)."""
  with np.load(path) as data:
    body_pos = data["body_pos_w"][::stride].astype(np.float64)  # (F, B, 3)
    root_quat = data["body_quat_w"][::stride, 0].astype(np.float64)  # wxyz
  root_pos = body_pos[:, 0]
  rot = Rotation.from_quat(root_quat[:, [1, 2, 3, 0]])  # xyzw
  rel = body_pos - root_pos[:, None, :]
  # World -> pelvis frame: p_rel = R^T (p - p_root).
  mats = rot.inv().as_matrix()  # (F, 3, 3)
  return np.einsum("fij,fbj->fbi", mats, rel)


def skeleton_edges(num_bodies: int) -> list[tuple[int, int]] | None:
  """Kinematic-tree edges between NPZ body indices, from the MuJoCo model."""
  try:
    import mujoco
  except ImportError:
    return None
  model = mujoco.MjModel.from_xml_path(str(G1_23DOF_XML))
  if model.nbody - 1 != num_bodies:  # body 0 is the world
    print(
      f"[WARN] model has {model.nbody - 1} bodies but NPZ has {num_bodies}; "
      "skipping skeleton overlay"
    )
    return None
  edges = []
  for b in range(2, model.nbody):  # skip world and the root itself
    parent = model.body_parentid[b]
    if parent >= 1:
      edges.append((parent - 1, b - 1))
  return edges


def load_reference_body_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
  """World-frame body positions and quaternions (wxyz) of frame 0."""
  with np.load(path) as data:
    return (
      data["body_pos_w"][0].astype(np.float64),
      data["body_quat_w"][0].astype(np.float64),
    )


def robot_mesh_triangles(
  body_pos_w: np.ndarray, body_quat_w: np.ndarray
) -> np.ndarray | None:
  """Triangles of the robot's visual meshes at the reference pose, (N, 3, 3).

  Each visual geom's mesh is reduced to its convex hull (the full G1 visual
  meshes have ~250k vertices — far too heavy for matplotlib, while per-link
  hulls keep the humanoid silhouette at a few thousand faces). Triangles are
  returned in the pelvis frame, like the point cloud.
  """
  try:
    import mujoco
  except ImportError:
    return None
  model = mujoco.MjModel.from_xml_path(str(G1_23DOF_XML))
  if model.nbody - 1 != body_pos_w.shape[0]:
    return None

  root_rot = Rotation.from_quat(body_quat_w[0][[1, 2, 3, 0]])
  root_pos = body_pos_w[0]

  triangles: list[np.ndarray] = []
  for g in range(model.ngeom):
    body = model.geom_bodyid[g]
    is_visual_mesh = (
      model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
      and model.geom_contype[g] == 0
      and model.geom_conaffinity[g] == 0
      and body >= 1
    )
    if not is_visual_mesh:
      continue
    mesh_id = model.geom_dataid[g]
    adr, num = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    verts = model.mesh_vert[adr : adr + num].astype(np.float64)
    if len(verts) > 500:  # cap hull complexity; visually identical
      rng = np.random.default_rng(0)
      verts = verts[rng.choice(len(verts), 500, replace=False)]
    hull = ConvexHull(verts)
    # geom frame -> body frame -> world -> pelvis frame.
    geom_rot = Rotation.from_quat(model.geom_quat[g][[1, 2, 3, 0]])
    verts = geom_rot.apply(verts) + model.geom_pos[g]
    body_rot = Rotation.from_quat(body_quat_w[body - 1][[1, 2, 3, 0]])
    verts = body_rot.apply(verts) + body_pos_w[body - 1]
    verts = root_rot.inv().apply(verts - root_pos)
    triangles.append(verts[hull.simplices])
  if not triangles:
    return None
  return np.concatenate(triangles, axis=0)


def style_axes(ax, limits: np.ndarray) -> None:
  """Equal-aspect, recessive panes/grid, labeled axes."""
  centers = limits.mean(axis=1)
  half = (limits[:, 1] - limits[:, 0]).max() / 2
  ax.set_xlim(centers[0] - half, centers[0] + half)
  ax.set_ylim(centers[1] - half, centers[1] + half)
  ax.set_zlim(centers[2] - half, centers[2] + half)
  ax.set_box_aspect((1, 1, 1))
  ax.set_xlabel("x [m]", color="#555555", labelpad=8)
  ax.set_ylabel("y [m]", color="#555555", labelpad=8)
  ax.set_zlabel("z [m]", color="#555555", labelpad=8)
  ax.tick_params(colors="#777777", labelsize=8)
  for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    axis.pane.set_edgecolor("#dddddd")
    axis._axinfo["grid"].update(color="#e6e6e6", linewidth=0.5)


def draw_cloud(ax, clouds: dict[str, np.ndarray], color_by: str) -> None:
  for mode in MODE_ORDER:
    if mode not in clouds:
      continue
    pts = clouds[mode].reshape(-1, 3)
    color = MODE_COLORS[mode] if color_by == "mode" else GHOST_GREEN
    alpha = float(np.clip(40_000 / max(len(pts), 1), 0.02, 0.5))
    ax.scatter(
      pts[:, 0], pts[:, 1], pts[:, 2],
      s=1.0, c=color, alpha=alpha, linewidths=0, rasterized=True,
      depthshade=False,
    )


def draw_skeleton(ax, pose: np.ndarray, edges: list[tuple[int, int]]) -> None:
  segments = [(pose[a], pose[b]) for a, b in edges]
  ax.add_collection3d(
    Line3DCollection(segments, colors=[GHOST_GREEN], linewidths=2.5, alpha=0.95)
  )
  ax.scatter(
    pose[:, 0], pose[:, 1], pose[:, 2],
    s=14, c=[GHOST_GREEN], edgecolors="#3c5a3c", linewidths=0.5, depthshade=False,
  )


def draw_robot(ax, triangles: np.ndarray) -> None:
  """Translucent ghost-green robot mesh at the reference pose."""
  ax.add_collection3d(
    Poly3DCollection(
      triangles,
      facecolors=[(*GHOST_GREEN, 0.5)],
      edgecolors=[(*GHOST_GREEN, 0.5)],
      linewidths=0.0,
      shade=True,
    )
  )


def draw_hull(ax, hull: ConvexHull, points: np.ndarray) -> None:
  segments = []
  seen = set()
  for simplex in hull.simplices:
    for i in range(3):
      edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
      if edge not in seen:
        seen.add(edge)
        segments.append((points[edge[0]], points[edge[1]]))
  ax.add_collection3d(
    Line3DCollection(
      segments, colors=[HULL_GRAY], linewidths=0.5, linestyles=":", alpha=0.45
    )
  )


def make_legend(ax, clouds: dict[str, np.ndarray], color_by: str,
                ref_style: str | None, has_hull: bool) -> None:
  handles = []
  if color_by == "mode":
    for mode in MODE_ORDER:
      if mode in clouds:
        n = clouds[mode].reshape(-1, 3).shape[0]
        handles.append(
          Line2D([], [], marker="o", linestyle="", markersize=6,
                 color=MODE_COLORS[mode], label=f"{mode} ({n:,} pts)")
        )
  else:
    n = sum(c.reshape(-1, 3).shape[0] for c in clouds.values())
    handles.append(
      Line2D([], [], marker="o", linestyle="", markersize=6,
             color=GHOST_GREEN, label=f"all frames ({n:,} pts)")
    )
  if ref_style == "mesh":
    handles.append(
      Patch(facecolor=(*GHOST_GREEN, 0.55), edgecolor="#3c5a3c",
            label="reference stand pose")
    )
  elif ref_style == "skeleton":
    handles.append(
      Line2D([], [], color=GHOST_GREEN, linewidth=2.5, label="reference stand pose")
    )
  if has_hull:
    handles.append(
      Line2D([], [], color=HULL_GRAY, linestyle=":", label="convex hull")
    )
  ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)


def main() -> None:
  args = parse_args()

  entries = discover_files(args.data_root, args.motions, args.modes)
  motions = sorted({m for _, m, _ in entries})
  print(f"Found {len(entries)} NPZ files (motions: {', '.join(motions)})")

  # Frame stride so the total cloud lands near --max-points.
  if args.frame_stride is not None:
    stride = args.frame_stride
  else:
    total_frames = 0
    for f, _, _ in entries:
      with np.load(f) as data:
        total_frames += data["foot_contacts"].shape[0]
    with np.load(entries[0][0]) as data:
      num_bodies = data["body_pos_w"].shape[1]
    stride = max(1, round(total_frames * num_bodies / args.max_points))
  print(f"Frame stride: {stride}")

  clouds: dict[str, list[np.ndarray]] = {}
  reference_file: Path | None = None
  for f, motion, mode in entries:
    rel = load_pelvis_relative(f, stride)
    clouds.setdefault(mode, []).append(rel)
    # Reference pose: frame 0 of the first zero-wrench stand file.
    if reference_file is None and mode == "zero-wrench" and motion == "stand":
      reference_file = f
  merged = {mode: np.concatenate(parts, axis=0) for mode, parts in clouds.items()}
  if reference_file is None:  # no zero-wrench stand file present
    reference_file = entries[0][0]
  reference_pose = load_pelvis_relative(reference_file, 1)[0]

  all_points = np.concatenate([c.reshape(-1, 3) for c in merged.values()], axis=0)
  num_bodies = reference_pose.shape[0]
  print(f"Cloud: {all_points.shape[0]:,} points ({num_bodies} bodies/frame)")

  # Reference robot rendering: visual-mesh ghost, skeleton as fallback.
  robot_tris = robot_mesh_triangles(*load_reference_body_poses(reference_file))
  edges = skeleton_edges(num_bodies) if robot_tris is None else None
  ref_style = "mesh" if robot_tris is not None else (
    "skeleton" if edges is not None else None
  )
  if robot_tris is not None:
    print(f"Reference robot mesh: {len(robot_tris):,} triangles")

  hull = None
  hull_points = all_points
  if not args.no_hull:
    # Full point set: qhull is fast even for millions of points, and the hull
    # extremes are rare frames a subsample would miss.
    hull = ConvexHull(hull_points)
    print(f"Convex hull: volume {hull.volume:.3f} m^3, {len(hull.simplices)} facets")

  limits = np.stack(
    [all_points.min(axis=0) - 0.05, all_points.max(axis=0) + 0.05], axis=1
  )

  out_dir = args.output_dir
  if out_dir is None:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = REPO_ROOT / "logs" / "offline_data_viz" / stamp
  out_dir.mkdir(parents=True, exist_ok=True)

  n_files = len(entries)
  subtitle = (
    f"{n_files} files | {all_points.shape[0]:,} points | pelvis frame | "
    f"stride {stride}"
  )

  # One figure per camera angle.
  for name, elev, azim in VIEWS:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")
    draw_cloud(ax, merged, args.color_by)
    if robot_tris is not None:
      draw_robot(ax, robot_tris)
    elif edges is not None:
      draw_skeleton(ax, reference_pose, edges)
    if hull is not None:
      draw_hull(ax, hull, hull_points)
    style_axes(ax, limits)
    ax.view_init(elev=elev, azim=azim)
    make_legend(ax, merged, args.color_by, ref_style, hull is not None)
    ax.set_title(
      f"Offline compliant-motion data — {name} view", fontsize=12, color="#222222"
    )
    ax.text2D(
      0.5, -0.06, subtitle, transform=ax.transAxes, ha="center",
      fontsize=8, color="#777777",
    )
    path = out_dir / f"cloud_{name}.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

  # Facet figure: one subplot per mode (direct labeling of each mode's cloud).
  present_modes = [m for m in MODE_ORDER if m in merged]
  if len(present_modes) > 1:
    fig = plt.figure(figsize=(5.5 * len(present_modes), 6))
    for i, mode in enumerate(present_modes):
      ax = fig.add_subplot(1, len(present_modes), i + 1, projection="3d")
      draw_cloud(ax, {mode: merged[mode]}, args.color_by)
      if robot_tris is not None:
        draw_robot(ax, robot_tris)
      elif edges is not None:
        draw_skeleton(ax, reference_pose, edges)
      style_axes(ax, limits)
      ax.view_init(elev=25, azim=45)
      n = merged[mode].reshape(-1, 3).shape[0]
      ax.set_title(f"{mode}\n{n:,} pts", fontsize=11, color="#222222")
    fig.suptitle(
      "Offline compliant-motion data by force mode", fontsize=13, color="#222222"
    )
    path = out_dir / "cloud_modes_facet.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

  print(f"Done. Images in {out_dir}")


if __name__ == "__main__":
  main()
