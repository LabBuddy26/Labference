from __future__ import annotations

from typing import Any


class ExternalInferenceEngine:
    """Deferred engine used while the benchmark policy bridge is being bound.

    It intentionally performs no network initialization. Evaluation creates
    the legacy task controller first, then replaces this object with
    ``PolicyTrajectoryInferenceEngine`` before the first episode reset/step.
    """

    def __init__(self, cfg: Any, trajectory_controller: Any) -> None:
        self.cfg = cfg
        self.trajectory_controller = trajectory_controller

    def reset(self) -> None:
        reset = getattr(self.trajectory_controller, "reset", None)
        if callable(reset):
            reset()

    def step_inference(self, _state: Any) -> Any:
        raise RuntimeError(
            "external inference placeholder was stepped before the benchmark "
            "policy trajectory engine was bound"
        )

    def close(self) -> None:
        return None


__all__ = ["ExternalInferenceEngine"]
