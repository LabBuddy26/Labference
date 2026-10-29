from __future__ import annotations

import numpy as np

from .rackplace_task import RackPlaceTask


class RackRecallTask(RackPlaceTask):
    """R10 rack scene with an episode-stable remembered slot and return pose."""

    def reset(self):
        super().reset()
        self._return_position = None

    def step(self):
        state = super().step()
        if state is None:
            return None
        if self._return_position is None:
            self._return_position = np.asarray(state["object_position"], dtype=float).copy()
        state["return_position"] = self._return_position.copy()
        state["remembered_hole_index"] = int(self.hole_index)
        return state

