from __future__ import annotations

from statistics import median
from typing import Any

import numpy as np

from eai4s_eval.utils.math import path_length, rms_third_difference


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


class TaskMetricManager:
    @staticmethod
    def grasp_attempts(trajectory: list[dict[str, Any]]) -> int:
        attempts = 0
        prev_open: bool | None = None
        for entry in trajectory:
            raw = entry.get("vla_action", {}).get("gripper_raw")
            if raw is None:
                continue
            is_open = float(raw) > 0.5
            if prev_open is not None and prev_open and not is_open:
                attempts += 1
            prev_open = is_open
        return attempts

    @staticmethod
    def trajectory_metrics(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
        if not trajectory:
            return {
                "path_length_m": 0.0,
                "eef_jerk_rms": None,
                "joint_jerk_rms": None,
            }
        ee_pos = np.asarray([step["actual_state"]["ee_pos"] for step in trajectory], dtype=np.float64)
        joint_pos = np.asarray(
            [step.get("actual_state", {}).get("joint_pos", []) for step in trajectory],
            dtype=object,
        )
        joint_jerk = None
        if len(joint_pos) and isinstance(joint_pos[0], (list, tuple)) and len(joint_pos[0]) > 0:
            joint_jerk = rms_third_difference(np.asarray(joint_pos.tolist(), dtype=np.float64))
        return {
            "path_length_m": round(path_length(ee_pos), 6),
            "eef_jerk_rms": rms_third_difference(ee_pos),
            "joint_jerk_rms": joint_jerk,
        }

    @staticmethod
    def latency_metrics(latencies_ms: list[float]) -> dict[str, Any]:
        return {
            "inference_latency_p50_ms": percentile(latencies_ms, 50),
            "inference_latency_p95_ms": percentile(latencies_ms, 95),
            "inference_latency_median_ms": float(median(latencies_ms)) if latencies_ms else None,
        }
