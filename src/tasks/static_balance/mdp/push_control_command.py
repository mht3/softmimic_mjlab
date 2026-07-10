"""Interactive push control command for evaluation/play mode.

Adds sliders and a push button to the Viser viewer so you can manually
trigger perturbations during evaluation. Does not contribute to observations
— add to cfg.commands in play mode only.

Checkbox unchecked (default — Auto): random pushes every ``push_interval_s``
seconds. Sliders update to show the applied body-frame velocity as feedback.

Checkbox checked (Manual): sliders control the exact push velocity (x/y are
in the robot's body frame). Push button fires once.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_ALL_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


class PushControlCommand(CommandTerm):
  cfg: PushControlCommandCfg

  def __init__(self, cfg: PushControlCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._sliders: dict[str, "viser.GuiSliderHandle"] = {}
    self._enabled: "viser.GuiCheckboxHandle | None" = None
    self._push_btn: "viser.GuiButtonHandle | None" = None
    self._reset_btn: "viser.GuiButtonHandle | None" = None
    self._auto_push_countdown: int = -1  # -1 = GUI not yet created
    # When False, the command is inert (no auto/manual pushes) and its controls
    # are greyed out. Used to hand perturbation control to the dataset forces
    # while augmented motions play.
    self._controls_active: bool = True

  @property
  def command(self) -> torch.Tensor:
    """Zero tensor — this command is GUI-only and never observed by the policy."""
    return torch.zeros(self.num_envs, 1, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pass

  def _update_command(self) -> None:
    # No-op until GUI is created, when handed off (augmented motions), or when
    # in manual mode (pushes only fire from the button).
    if self._enabled is None or not self._controls_active or self._enabled.value:
      return
    self._auto_push_countdown -= 1
    if self._auto_push_countdown <= 0:
      maxes = {
        "x":     self.cfg.default_max_lin,
        "y":     self.cfg.default_max_lin,
        "z":     self.cfg.default_max_z,
        "roll":  self.cfg.default_max_ang,
        "pitch": self.cfg.default_max_ang,
        "yaw":   self.cfg.default_max_ang,
      }
      sampled = {ax: random.uniform(-maxes[ax], maxes[ax]) for ax in _ALL_AXES}
      self._apply_and_update_sliders(sampled)
      self._reset_countdown()

  def _update_metrics(self) -> None:
    pass

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    from viser import Icon

    slider_maxes = {
      "x":     self.cfg.slider_max_lin,
      "y":     self.cfg.slider_max_lin,
      "z":     self.cfg.slider_max_z,
      "roll":  self.cfg.slider_max_ang,
      "pitch": self.cfg.slider_max_ang,
      "yaw":   self.cfg.slider_max_ang,
    }

    with server.gui.add_folder("Push Control"):
      self._enabled = server.gui.add_checkbox(
        "Manual", initial_value=self.cfg.manual_by_default
      )

      for axis in _ALL_AXES:
        m = slider_maxes[axis]
        self._sliders[axis] = server.gui.add_slider(
          axis,
          min=-m,
          max=m,
          step=0.05,
          initial_value=0.0,
        )

      push_btn = server.gui.add_button("Push", icon=Icon.ARROW_RIGHT)
      push_btn.disabled = not self.cfg.manual_by_default  # only active in manual mode
      self._push_btn = push_btn

      reset_btn = server.gui.add_button("Reset", icon=Icon.REFRESH)
      self._reset_btn = reset_btn

      @self._enabled.on_update
      def _(ev: "viser.GuiUpdateEvent[viser.GuiCheckboxHandle]") -> None:
        # Manual toggle only enables the push button while controls are active.
        push_btn.disabled = not ev.target.value or not self._controls_active

      @push_btn.on_click
      def _(_) -> None:
        sampled = {ax: self._sliders[ax].value for ax in _ALL_AXES}
        self._apply_and_update_sliders(sampled)

      @reset_btn.on_click
      def _(_) -> None:
        for ax in _ALL_AXES:
          self._sliders[ax].value = 0.0

    # Start auto-push countdown after GUI is fully set up.
    self._reset_countdown()

  def set_controls_active(self, active: bool) -> None:
    """Enable/disable the push controls (and auto/manual pushes).

    Used to hand perturbation control to the dataset forcefield while augmented
    motions play. Greys out the folder's controls so the state is obvious.
    """
    self._controls_active = active
    for handle in (self._enabled, self._reset_btn, *self._sliders.values()):
      if handle is not None:
        handle.disabled = not active
    if self._push_btn is not None:
      # Push button also requires manual mode.
      manual = self._enabled is not None and self._enabled.value
      self._push_btn.disabled = not active or not manual

  def _reset_countdown(self) -> None:
    interval_s = random.uniform(*self.cfg.push_interval_s)
    self._auto_push_countdown = max(1, int(interval_s / self._env.step_dt))

  def _apply_and_update_sliders(self, body_frame: dict[str, float]) -> None:
    """Apply push given body-frame velocities, then update sliders as feedback."""
    robot = self._env.scene[self.cfg.entity_name]
    env_ids = torch.arange(self._env.num_envs, device=self.device)
    heading = robot.data.heading_w  # [num_envs]

    vx_b = torch.full((self._env.num_envs,), body_frame["x"], device=self.device)
    vy_b = torch.full((self._env.num_envs,), body_frame["y"], device=self.device)

    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    vx_w = cos_h * vx_b - sin_h * vy_b
    vy_w = sin_h * vx_b + cos_h * vy_b

    vel_w = robot.data.root_link_vel_w[env_ids].clone()
    vel_w[:, 0] += vx_w
    vel_w[:, 1] += vy_w
    vel_w[:, 2] += body_frame["z"]
    vel_w[:, 3] += body_frame["roll"]
    vel_w[:, 4] += body_frame["pitch"]
    vel_w[:, 5] += body_frame["yaw"]
    robot.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)

    # Update sliders with the applied body-frame values.
    slider_maxes = {
      "x":     self.cfg.slider_max_lin,
      "y":     self.cfg.slider_max_lin,
      "z":     self.cfg.slider_max_z,
      "roll":  self.cfg.slider_max_ang,
      "pitch": self.cfg.slider_max_ang,
      "yaw":   self.cfg.slider_max_ang,
    }
    for ax, val in body_frame.items():
      if ax in self._sliders:
        m = slider_maxes[ax]
        self._sliders[ax].value = max(-m, min(m, val))

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    pass


@dataclass(kw_only=True)
class PushControlCommandCfg(CommandTermCfg):
  """Configuration for the interactive push control command."""

  entity_name: str
  default_max_lin: float = 1.0
  default_max_z: float = 0.4
  default_max_ang: float = 0.78
  slider_max_lin: float = 2.0
  slider_max_z: float = 0.5
  slider_max_ang: float = 1.5
  push_interval_s: tuple[float, float] = field(default_factory=lambda: (2.0, 5.0))
  manual_by_default: bool = False
  """Start in manual mode (no automatic pushes, sliders at zero)."""

  def build(self, env: ManagerBasedRlEnv) -> PushControlCommand:
    return PushControlCommand(self, env)
