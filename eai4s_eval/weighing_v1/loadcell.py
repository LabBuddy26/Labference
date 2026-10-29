"""PhysX joint-wrench readback for the feedback-weighing scene.

Only the scale's internal support joint is measured. No configured object mass
is ever used to produce a reading. Importing this module does not start Isaac.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any


SENSOR_SOURCE = "physx_loadcell_joint"
SEGMENTS = {
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd",
    "4": "fgbc", "5": "afgcd", "6": "afgecd", "7": "abc",
    "8": "abcdefg", "9": "abfgcd", " ": "", "-": "g",
    "O": "abcdef", "L": "fed",
}


class LoadCellError(RuntimeError):
    """The physical sensor cannot supply a trustworthy reading."""


@dataclass(frozen=True)
class LoadCellReading:
    step: int
    valid: bool
    error: str | None
    sensor_source: str
    raw_force_n: float | None
    raw_grams: float | None
    filtered_grams: float | None
    displayed_grams: float | None
    display_text: str
    stable: bool
    zero_force_n: float | None
    overloaded: bool = False

    def as_dict(self) -> dict[str, Any]:
        return vars(self).copy()


def format_grams(value: float) -> str:
    """Four digits and one decimal point, with explicit overrange failure."""
    if not math.isfinite(value) or value < -0.05 or value > 999.95:
        raise LoadCellError("measured mass outside display range [0, 999.9] g")
    return f"{max(0.0, min(999.9, round(value, 1))):.1f}"


def segment_state(text: str) -> dict[str, bool]:
    if text == "----":
        digits, dot_after = "----", None
    elif text == "OL":
        digits, dot_after = "  OL", None
    else:
        whole, sep, decimal = text.partition(".")
        if sep != "." or len(decimal) != 1 or not whole.isdigit() or len(whole) > 3:
            raise ValueError(f"invalid scale display text: {text!r}")
        digits, dot_after = whole.rjust(3) + decimal, 2
    state = {}
    for index, digit in enumerate(digits):
        active = SEGMENTS[digit]
        for segment in "abcdefg":
            state[f"d{index}_{segment}"] = segment in active
        state[f"d{index}_dp"] = index == dot_after
    return state


def low_pass(previous: float, current: float, dt: float, tau_s: float = 0.25) -> float:
    if not all(math.isfinite(v) for v in (previous, current, dt, tau_s)) or dt <= 0 or tau_s <= 0:
        raise ValueError("low-pass arguments must be finite and dt/tau positive")
    alpha = 1.0 - math.exp(-dt / tau_s)
    return previous + alpha * (current - previous)


class PhysxLoadCell:
    """Read the tray's incoming joint wrench after each World.step().

    Lifecycle: build_scene -> World.reset -> initialize -> calibrate(empty pan)
    -> SceneHandles.place_initial_loads -> update(step, dt). Invalid updates
    return valid=False and display_text='----'; no mass-sum fallback exists.
    """

    def __init__(self, stage: Any, joint_path: str, *, tray_path: str | None = None,
                 gravity_m_s2: float = 9.81, filter_tau_s: float = 0.25,
                 display_quantum_g: float = 0.1, stable_window_s: float = 1.0,
                 stable_range_g: float = 0.5) -> None:
        if not math.isfinite(gravity_m_s2) or gravity_m_s2 <= 0:
            raise ValueError("gravity_m_s2 must be positive")
        if not math.isfinite(filter_tau_s) or filter_tau_s <= 0:
            raise ValueError("filter_tau_s must be positive")
        if not math.isfinite(display_quantum_g) or display_quantum_g <= 0:
            raise ValueError("display_quantum_g must be positive")
        self.stage = stage
        self.joint_path = joint_path
        self.tray_path = tray_path
        self.gravity_m_s2 = gravity_m_s2
        self.filter_tau_s = filter_tau_s
        self.display_quantum_g = display_quantum_g
        self.stable_window_s = stable_window_s
        self.stable_range_g = stable_range_g
        self.articulation = None
        self.link_index = None
        self.zero_force_n = None
        self.filtered_grams = None
        self.history: deque[tuple[float, float]] = deque()
        self.elapsed_s = 0.0

    def initialize(self) -> None:
        from pxr import UsdGeom, UsdPhysics, Usd
        from isaacsim.core.prims import SingleArticulation

        joint = UsdPhysics.FixedJoint.Get(self.stage, self.joint_path)
        if not joint.GetPrim().IsValid():
            raise LoadCellError(f"missing load-cell joint: {self.joint_path}")
        targets = list(map(str, joint.GetBody1Rel().GetTargets()))
        if self.tray_path is None:
            if len(targets) != 1:
                raise LoadCellError("load-cell joint must have one tray body target")
            self.tray_path = targets[0]
        if targets != [self.tray_path]:
            raise LoadCellError(f"load-cell joint is missing or does not target {self.tray_path}")
        tray = self.stage.GetPrimAtPath(self.tray_path)
        if not tray.IsValid() or not tray.HasAPI(UsdPhysics.RigidBodyAPI):
            raise LoadCellError(f"tray is not a rigid body: {self.tray_path}")
        transform = UsdGeom.Xformable(tray).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        up = transform.TransformDir((0, 0, 1))
        up_norm = math.sqrt(sum(float(v) ** 2 for v in up))
        if up_norm <= 0 or abs(float(up[0]) / up_norm) > 1e-3 or \
                abs(float(up[1]) / up_norm) > 1e-3 or float(up[2]) / up_norm < 0.999:
            raise LoadCellError("load-cell tray local +Z must align with world gravity axis")
        scale_root = str(joint.GetPrim().GetParent().GetPath())
        self.articulation = SingleArticulation(prim_path=scale_root, name="weighing_loadcell")
        self.articulation.initialize()
        view = self.articulation._articulation_view
        self.link_index = view.get_link_index(tray.GetName())
        if self.link_index is None or int(self.link_index) < 1:
            raise LoadCellError("PhysX did not expose the tray incoming joint")
        self.link_index = int(self.link_index)
        self._read_force_n()

    def _read_force_n(self) -> float:
        if self.articulation is None or self.link_index is None:
            raise LoadCellError("load cell is not initialized")
        import numpy as np

        values = self.articulation.get_measured_joint_forces(joint_indices=np.array([self.link_index]))
        if values is None:
            raise LoadCellError("PhysX returned no joint wrench")
        values = np.asarray(values, dtype=float)
        if values.shape != (1, 6) or not np.isfinite(values).all():
            raise LoadCellError(f"invalid PhysX joint wrench shape/values: {values.shape}")
        return float(values[0, 2])

    def calibrate(self, world: Any, steps: int = 120) -> float:
        """Step an empty pan and measure its support-force bias.

        Call exactly once before SceneHandles.place_initial_loads().
        """
        if self.zero_force_n is not None:
            raise LoadCellError("load cell was already calibrated")
        if steps < 60:
            raise ValueError("empty-pan calibration needs at least 60 physics steps")
        readings = []
        for index in range(steps):
            world.step(render=False)
            force = self._read_force_n()
            if index >= steps // 2:
                readings.append(force)
        if not readings or not all(math.isfinite(x) for x in readings):
            raise LoadCellError("empty-pan force readback is unavailable")
        spread_g = (max(readings) - min(readings)) * 1000.0 / self.gravity_m_s2
        baseline = sum(readings) / len(readings)
        if baseline <= 0.05 or spread_g > 0.5:
            raise LoadCellError(f"empty-pan force invalid: baseline={baseline:.4f} N, spread={spread_g:.3f} g")
        self.zero_force_n = baseline
        self.filtered_grams = 0.0
        self.history.clear()
        self.elapsed_s = 0.0
        return baseline

    def update(self, step: int, dt: float) -> LoadCellReading:
        if self.zero_force_n is None:
            return self._invalid(step, "not_calibrated")
        try:
            if not math.isfinite(dt) or dt <= 0:
                raise LoadCellError("invalid physics timestep")
            raw_force_n = self._read_force_n()
            raw_grams = (raw_force_n - self.zero_force_n) * 1000.0 / self.gravity_m_s2
            if not math.isfinite(raw_grams):
                raise LoadCellError("joint force implies non-finite load")
            if raw_grams < -2.0 or raw_grams > 999.95:
                self.elapsed_s += dt
                self.history.clear()
                held = float(self.filtered_grams) if self.filtered_grams is not None else 0.0
                return LoadCellReading(step, True, None, SENSOR_SOURCE, raw_force_n, raw_grams,
                                       held, 999.9, "OL", False, self.zero_force_n, True)
            self.filtered_grams = low_pass(float(self.filtered_grams), raw_grams, dt, self.filter_tau_s)
            if not math.isfinite(self.filtered_grams):
                raise LoadCellError("filtered load is non-finite")
            quantized = round(self.filtered_grams / self.display_quantum_g) * self.display_quantum_g
            displayed = float(f"{quantized:.1f}")
            display_text = format_grams(displayed)
            self.elapsed_s += dt
            self.history.append((self.elapsed_s, self.filtered_grams))
            cutoff = self.elapsed_s - self.stable_window_s
            while self.history and self.history[0][0] < cutoff:
                self.history.popleft()
            values = [value for _, value in self.history]
            stable = bool(self.history and self.history[0][0] <= cutoff + dt and
                          max(values) - min(values) <= self.stable_range_g)
            return LoadCellReading(step, True, None, SENSOR_SOURCE, raw_force_n, raw_grams,
                                   self.filtered_grams, displayed, display_text, stable, self.zero_force_n)
        except Exception as exc:
            return self._invalid(step, f"{type(exc).__name__}: {exc}")

    def _invalid(self, step: int, error: str) -> LoadCellReading:
        self.history.clear()
        self.filtered_grams = None
        return LoadCellReading(step, False, error, SENSOR_SOURCE, None, None, None,
                               None, "----", False, self.zero_force_n)
