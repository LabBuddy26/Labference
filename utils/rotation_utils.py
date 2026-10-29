"""Quaternion conversions shared by scripted robot controllers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def isaac_quaternion_from_euler_deg(euler_deg) -> np.ndarray:
    """Return an Isaac-compatible scalar-first ``[w, x, y, z]`` quaternion."""
    quat_xyzw = Rotation.from_euler("xyz", np.radians(euler_deg)).as_quat()
    return np.roll(quat_xyzw, 1)


__all__ = ["isaac_quaternion_from_euler_deg"]
