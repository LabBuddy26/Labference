from __future__ import annotations

import numpy as np

from .scaleweights_task import ScaleWeightsTask


class WeighingFeedbackTask(ScaleWeightsTask):
    """Main-compatible weighing state with a live stable mass readout."""

    MASSES_G = (10.0, 20.0, 50.0)

    def reset(self):
        super().reset()
        self._mass_history = []

    def step(self):
        state = super().step()
        if state is None:
            return None
        positions = (
            np.asarray(state["object_position"], dtype=float),
            np.asarray(state["weight_01_position"], dtype=float),
            np.asarray(state["weight_02_position"], dtype=float),
        )
        slots = (
            np.asarray(state["target_position"], dtype=float),
            np.asarray(state["target_01_position"], dtype=float),
            np.asarray(state["target_02_position"], dtype=float),
        )
        tolerance = float(self.cfg.scale_weights_success.place_xy_tolerance_m)
        placed = [
            bool(np.linalg.norm(position[:2] - slot[:2]) <= tolerance)
            for position, slot in zip(positions, slots)
        ]
        mass = float(sum(value for value, present in zip(self.MASSES_G, placed) if present))
        self._mass_history.append(mass)
        stable_frames = int(self.cfg.weighing_feedback.get("stable_frames", 30))
        if len(self._mass_history) > stable_frames:
            self._mass_history = self._mass_history[-stable_frames:]
        state.update({
            "display_mass_g": mass,
            "target_mass_g": float(self.cfg.weighing_feedback.target_mass_g),
            "load_cell_stable": (
                len(self._mass_history) >= stable_frames
                and max(self._mass_history) - min(self._mass_history)
                <= float(self.cfg.weighing_feedback.stable_tolerance_g)
            ),
            "sensor_source": str(self.cfg.weighing_feedback.sensor_source),
        })
        return state

