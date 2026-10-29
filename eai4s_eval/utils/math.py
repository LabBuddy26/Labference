from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def normalize_quat_wxyz(quat: Iterable[float]) -> np.ndarray:
    q = np.asarray(list(quat), dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_multiply_wxyz(a: Iterable[float], b: Iterable[float]) -> np.ndarray:
    aw, ax, ay, az = normalize_quat_wxyz(a)
    bw, bx, by, bz = normalize_quat_wxyz(b)
    return normalize_quat_wxyz(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def rotvec_to_quat_wxyz(rotvec: Iterable[float]) -> np.ndarray:
    rv = np.asarray(list(rotvec), dtype=np.float64)
    angle = float(np.linalg.norm(rv))
    if angle <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rv / angle
    half = angle * 0.5
    return normalize_quat_wxyz([math.cos(half), *(math.sin(half) * axis)])


def rms_third_difference(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] < 4:
        return None
    third = np.diff(arr, n=3, axis=0)
    return float(np.sqrt(np.mean(np.square(third))))


def path_length(points: np.ndarray) -> float:
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())
