"""Translate gripper position commands into stable binary action labels."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class GripperActionTracker:
    """Track the most recently commanded binary gripper target.

    The simulator emits sparse joint-position commands: arm-motion commands
    usually leave the gripper joints as ``None``.  This tracker updates only
    when a controller explicitly supplies a gripper target and otherwise
    retains the preceding target state.

    Labels use the dataset convention ``0=closed/grasp`` and
    ``1=open/release``.
    """

    CLOSED = 0
    OPEN = 1

    def __init__(
        self,
        joint_indices: Sequence[int],
        open_target_threshold: float = 0.035,
    ) -> None:
        self._joint_indices = tuple(joint_indices)
        self._open_target_threshold = float(open_target_threshold)
        self.reset()

    def reset(self) -> None:
        """Reset an episode to the safe default open target."""
        self._state = self.OPEN

    @property
    def state(self) -> int:
        return self._state

    def update_from_action(self, action) -> int:
        """Apply an ``ArticulationAction`` and return the current label.

        The value is intentionally derived from the controller command, not
        from recorded robot observations or the shifted ``actions`` dataset.
        Explicit targets at or above ``open_target_threshold`` are opening
        commands; object-specific grasp widths fall below the threshold and
        are closing commands.
        """
        joint_positions = getattr(action, "joint_positions", None)
        if joint_positions is None:
            return self._state

        joint_indices = getattr(action, "joint_indices", None)
        try:
            if joint_indices is not None:
                targets = [
                    float(target)
                    for index, target in zip(joint_indices, joint_positions)
                    if int(index) in self._joint_indices and target is not None
                ]
            else:
                targets = [
                    float(joint_positions[index])
                    for index in self._joint_indices
                    if index < len(joint_positions) and joint_positions[index] is not None
                ]
        except (IndexError, TypeError, ValueError):
            return self._state

        if not targets:
            return self._state

        target = float(np.mean(targets))
        if np.isfinite(target):
            self._state = self.OPEN if target >= self._open_target_threshold else self.CLOSED
        return self._state
