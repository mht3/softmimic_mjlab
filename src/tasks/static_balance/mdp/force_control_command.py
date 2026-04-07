"""Interactive force/torque control command for evaluation/play mode.

Click "Select Body" to enter pick mode (camera navigation pauses). Click a
robot body in the 3D viewport to select it — pick mode then exits and camera
navigation is restored. A transform gizmo (translation axes only) appears at
the body. Drag an axis to apply force proportional to the displacement; an
arrow (cylinder shaft + cone head) is drawn from the body centre outward in
the force direction. Force clears on mouse release. Use "Clear selection" to
hide the gizmo and stop targeting a body (viser allows only one scene pointer
callback, so background clicks cannot deselect without breaking orbit).
Does not contribute to observations — add to cfg.commands in play mode only.

Coordinate note: viser's scene is offset by -tracked_body_pos so the camera
target stays centred. All viser positions (gizmo, arrow, ray) live in this
offset frame; body positions from entity data are in MuJoCo world frame and
must be converted via _get_body_pos_viser().
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _vec_to_wxyz(direction: np.ndarray) -> np.ndarray:
  """Quaternion rotating +Y axis to the given direction vector."""
  direction = direction / (np.linalg.norm(direction) + 1e-8)
  y = np.array([0.0, 1.0, 0.0])
  cross = np.cross(y, direction)
  cross_norm = np.linalg.norm(cross)
  dot = float(np.dot(y, direction))
  if cross_norm < 1e-6:
    return np.array([1.0, 0.0, 0.0, 0.0]) if dot > 0 else np.array([0.0, 1.0, 0.0, 0.0])
  angle = math.atan2(cross_norm, dot)
  axis = cross / cross_norm
  s = math.sin(angle / 2)
  return np.array([math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])


def _quat_mul_wxyz(q: np.ndarray, r: np.ndarray) -> np.ndarray:
  """Hamilton product q * r (w,x,y,z). Applies r then q to a vector."""
  w1, x1, y1, z1 = q
  w2, x2, y2, z2 = r
  return np.array(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dtype=np.float64,
  )


# viser CylinderMesh rotates geometry by π/2 about X so height lies along local +Z;
# our cone mesh uses +Y. This quaternion maps cylinder +Z → +Y before _vec_to_wxyz.
_half_neg_x = -0.25 * math.pi
_WXYZ_VISER_CYLINDER_TO_CONE = np.array(
  [math.cos(_half_neg_x), math.sin(_half_neg_x), 0.0, 0.0], dtype=np.float64
)


def _make_cone_mesh(n: int = 16) -> tuple[np.ndarray, np.ndarray]:
  """Unit cone: tip at (0, 1, 0), base radius 1 at y=0, +Y is forward."""
  angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
  apex = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
  ring = np.stack(
    [np.cos(angles), np.zeros(n), np.sin(angles)], axis=1
  ).astype(np.float32)
  center = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
  vertices = np.vstack([apex, ring, center])  # (n+2, 3)

  # Side triangles: apex(0) → ring[i] → ring[(i+1)%n]
  side = [[0, i + 1, (i + 1) % n + 1] for i in range(n)]
  # Base cap: center(n+1) → ring[(i+1)%n] → ring[i]
  base = [[n + 1, (i + 1) % n + 1, i + 1] for i in range(n)]

  faces = np.array(side + base, dtype=np.uint32)
  return vertices, faces


_CONE_VERTICES, _CONE_FACES = _make_cone_mesh()


class ForceControlCommand(CommandTerm):
  cfg: ForceControlCommandCfg

  def __init__(self, cfg: ForceControlCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._server: "viser.ViserServer | None" = None
    self._gizmo: "viser.TransformControlsHandle | None" = None
    self._arrow_shaft: "viser.CylinderHandle | None" = None
    self._arrow_head: "viser.MeshHandle | None" = None
    self._body_label: "viser.GuiTextHandle | None" = None
    self._force_label: "viser.GuiTextHandle | None" = None
    self._select_btn: "viser.GuiButtonHandle | None" = None
    self._clear_btn: "viser.GuiButtonHandle | None" = None
    self._selected_body_idx: int | None = None
    self._selected_body_name: str = ""
    self._is_dragging: bool = False
    self._pending_force: np.ndarray = np.zeros(3)
    self._follow_counter: int = 0

  @property
  def command(self) -> torch.Tensor:
    """Zero tensor — GUI-only, never observed by the policy."""
    return torch.zeros(self.num_envs, 1, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pass

  def _update_command(self) -> None:
    if self._gizmo is None:
      return
    if self._is_dragging and self._selected_body_idx is not None:
      self._apply_wrench(self._pending_force)
      self._update_arrow()
    else:
      self._clear_wrench()
      self._hide_arrow()
      self._follow_counter += 1
      if self._follow_counter >= self.cfg.gizmo_follow_interval and self._selected_body_name:
        self._follow_counter = 0
        self._snap_gizmo_to_body()

  def _update_metrics(self) -> None:
    pass

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    self._server = server

    with server.gui.add_folder("Force Control"):
      self._select_btn = server.gui.add_button("Select Body")
      self._clear_btn = server.gui.add_button(
        "Clear selection",
        disabled=True,
        hint="Hide the XYZ gizmo and stop applying force to the selected body.",
      )
      self._body_label = server.gui.add_text(
        "Body", initial_value="(none)", disabled=True
      )
      self._force_label = server.gui.add_text(
        "Force (N)", initial_value="0.0", disabled=True
      )

    # Gizmo: translation axes only (no rotation rings, no plane sliders).
    self._gizmo = server.scene.add_transform_controls(
      name="/force_control_gizmo",
      scale=self.cfg.gizmo_scale,
      disable_rotations=True,
      disable_sliders=True,
      depth_test=False,
      opacity=0.9,
      visible=False,
    )

    # Arrow shaft (cylinder) and head (cone mesh), both hidden initially.
    self._arrow_shaft = server.scene.add_cylinder(
      name="/force_control_arrow/shaft",
      radius=self.cfg.arrow_shaft_radius,
      height=0.01,
      color=(255, 140, 0),
      opacity=0.9,
      visible=False,
    )
    self._arrow_head = server.scene.add_mesh_simple(
      name="/force_control_arrow/head",
      vertices=_CONE_VERTICES,
      faces=_CONE_FACES,
      color=(255, 140, 0),
      opacity=0.9,
      side="double",
      visible=False,
    )

    @self._clear_btn.on_click
    def _(_) -> None:
      self._clear_body_selection()

    @self._select_btn.on_click
    def _(_) -> None:
      self._select_btn.disabled = True

      @server.scene.on_pointer_event(event_type="click")
      def _on_body_click(ev: "viser.ScenePointerEvent") -> None:
        if ev.ray_origin is not None and ev.ray_direction is not None:
          idx, body_name = self._find_clicked_body(ev.ray_origin, ev.ray_direction)
          if body_name is not None:
            self._selected_body_idx = idx
            self._selected_body_name = body_name
            if self._body_label is not None:
              self._body_label.value = body_name
            self._snap_gizmo_to_body()
            self._gizmo.visible = True
            if self._clear_btn is not None:
              self._clear_btn.disabled = False
        server.scene.remove_pointer_callback()

      @server.scene.on_pointer_callback_removed
      def _() -> None:
        self._select_btn.disabled = False

    @self._gizmo.on_drag_start
    def _(_: "viser.TransformControlsEvent") -> None:
      self._is_dragging = True

    @self._gizmo.on_update
    def _(_: "viser.TransformControlsEvent") -> None:
      if not self._selected_body_name:
        return
      body_pos_viser = self._get_body_pos_viser(self._selected_body_name)
      displacement = self._gizmo.position - body_pos_viser
      raw_mag = float(np.linalg.norm(displacement))
      if raw_mag < 1e-6:
        self._pending_force = np.zeros(3)
        return
      direction = displacement / raw_mag
      force_mag = min(self.cfg.force_scale * raw_mag, self.cfg.force_max)
      self._pending_force = direction * force_mag

    @self._gizmo.on_drag_end
    def _(_: "viser.TransformControlsEvent") -> None:
      self._is_dragging = False
      self._pending_force = np.zeros(3)
      self._clear_wrench()
      self._hide_arrow()
      if self._force_label is not None:
        self._force_label.value = "0.0"
      self._snap_gizmo_to_body()

  # -------------------------------------------------------------------------
  # Coordinate helpers
  # -------------------------------------------------------------------------

  def _get_tracked_pos(self) -> np.ndarray:
    robot = self._env.scene[self.cfg.entity_name]
    try:
      tracked_name = self._env.cfg.viewer.body_name
      if tracked_name:
        tracked_ids, _ = robot.find_bodies(tracked_name)
        return robot.data.body_link_pos_w[0, tracked_ids[0]].cpu().numpy()
    except Exception:
      pass
    return np.zeros(3)

  def _get_body_pos_viser(self, body_name: str) -> np.ndarray:
    robot = self._env.scene[self.cfg.entity_name]
    body_ids, _ = robot.find_bodies(body_name)
    mj_pos = robot.data.body_link_pos_w[0, body_ids[0]].cpu().numpy()
    return mj_pos - self._get_tracked_pos()

  # -------------------------------------------------------------------------
  # Helpers
  # -------------------------------------------------------------------------

  def _clear_body_selection(self) -> None:
    """Hide gizmo, clear sim wrench, reset labels. Pointer mode unchanged."""
    self._is_dragging = False
    self._pending_force = np.zeros(3)
    self._selected_body_idx = None
    self._selected_body_name = ""
    self._clear_wrench()
    self._hide_arrow()
    if self._body_label is not None:
      self._body_label.value = "(none)"
    if self._force_label is not None:
      self._force_label.value = "0.0"
    if self._gizmo is not None:
      self._gizmo.visible = False
    if self._clear_btn is not None:
      self._clear_btn.disabled = True

  def _snap_gizmo_to_body(self) -> None:
    if self._gizmo is None or not self._selected_body_name:
      return
    self._gizmo.position = self._get_body_pos_viser(self._selected_body_name)
    self._gizmo.wxyz = np.array([1.0, 0.0, 0.0, 0.0])

  def _find_clicked_body(
    self,
    ray_origin: tuple[float, float, float],
    ray_direction: tuple[float, float, float],
  ) -> tuple[int | None, str | None]:
    robot = self._env.scene[self.cfg.entity_name]
    mj_pos = robot.data.body_link_pos_w[0].cpu()
    tracked = torch.tensor(self._get_tracked_pos(), dtype=torch.float32)
    body_pos = mj_pos - tracked.unsqueeze(0)  # viser frame

    ray_o = torch.tensor(ray_origin, dtype=torch.float32)
    ray_d = torch.tensor(ray_direction, dtype=torch.float32)
    ray_d = ray_d / ray_d.norm()

    to_body = body_pos - ray_o.unsqueeze(0)
    t = (to_body * ray_d.unsqueeze(0)).sum(dim=1)
    closest = ray_o.unsqueeze(0) + t.unsqueeze(1) * ray_d.unsqueeze(0)
    dist = (closest - body_pos).norm(dim=1)

    mask = t > 0
    if not mask.any():
      return None, None
    dist[~mask] = float("inf")
    idx = int(dist.argmin().item())
    return idx, robot.body_names[idx]

  def _apply_wrench(self, force_vec: np.ndarray) -> None:
    if self._selected_body_idx is None:
      return
    robot = self._env.scene[self.cfg.entity_name]
    force_mag = float(np.linalg.norm(force_vec))

    n = robot.num_bodies
    forces = torch.zeros(1, n, 3, device=self.device)
    torques = torch.zeros(1, n, 3, device=self.device)
    forces[0, self._selected_body_idx] = torch.tensor(
      force_vec, device=self.device, dtype=torch.float32
    )
    env_ids = torch.tensor([0], device=self.device)
    robot.write_external_wrench_to_sim(forces, torques, env_ids=env_ids)

    if self._force_label is not None:
      self._force_label.value = f"{force_mag:.1f}"

  def _clear_wrench(self) -> None:
    robot = self._env.scene[self.cfg.entity_name]
    n = robot.num_bodies
    zeros = torch.zeros(1, n, 3, device=self.device)
    env_ids = torch.tensor([0], device=self.device)
    robot.write_external_wrench_to_sim(zeros, zeros, env_ids=env_ids)

  def _update_arrow(self) -> None:
    if self._arrow_shaft is None or self._arrow_head is None:
      return
    force_mag = float(np.linalg.norm(self._pending_force))
    if force_mag < 1e-4:
      self._hide_arrow()
      return

    direction = self._pending_force / force_mag
    wxyz = _vec_to_wxyz(direction)

    # Visual length scales linearly with force, capped at max_arrow_length.
    vis_len = self.cfg.max_arrow_length * (force_mag / self.cfg.force_max)

    head_h = self.cfg.arrow_head_height
    head_r = self.cfg.arrow_head_radius
    shaft_h = max(0.0, vis_len - head_h)

    body_pos = self._get_body_pos_viser(self._selected_body_name)

    # Shaft: centred at body_pos + direction * shaft_h/2
    wxyz_shaft = _quat_mul_wxyz(wxyz, _WXYZ_VISER_CYLINDER_TO_CONE)
    self._arrow_shaft.height = shaft_h if shaft_h > 1e-4 else 1e-4
    self._arrow_shaft.position = body_pos + direction * (shaft_h / 2)
    self._arrow_shaft.wxyz = wxyz_shaft
    self._arrow_shaft.visible = shaft_h > 1e-4

    # Head: cone base at end of shaft, tip pointing outward.
    # _CONE_VERTICES has tip at (0,1,0) and base at y=0, so:
    # - position at shaft tip (body_pos + direction * shaft_h)
    # - scale (head_r, head_h, head_r) in (x, y, z)
    self._arrow_head.position = body_pos + direction * shaft_h
    self._arrow_head.wxyz = wxyz
    self._arrow_head.scale = (head_r, head_h, head_r)
    self._arrow_head.visible = True

  def _hide_arrow(self) -> None:
    if self._arrow_shaft is not None:
      self._arrow_shaft.visible = False
    if self._arrow_head is not None:
      self._arrow_head.visible = False

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    pass


@dataclass(kw_only=True)
class ForceControlCommandCfg(CommandTermCfg):
  """Configuration for the interactive force/torque control command."""

  entity_name: str
  force_scale: float = 150.0       # N per metre of gizmo displacement
  force_max: float = 100.0         # N — force is clamped to this
  max_arrow_length: float = 0.5    # visual arrow length at force_max
  arrow_shaft_radius: float = 0.012
  arrow_head_height: float = 0.08
  arrow_head_radius: float = 0.04
  gizmo_scale: float = 0.35
  gizmo_follow_interval: int = 10

  def build(self, env: ManagerBasedRlEnv) -> ForceControlCommand:
    return ForceControlCommand(self, env)
