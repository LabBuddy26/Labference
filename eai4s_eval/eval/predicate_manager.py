from __future__ import annotations

from typing import Any

from eai4s_eval.eval.success import PredicateCall, SuccessEvaluation, SuccessSpec, TaskSuccessEvaluator

PredicateResult = SuccessEvaluation


class LegacyPredicateManager:
    """Compatibility wrapper around the robust semantic success evaluator."""

    def __init__(
        self,
        *,
        success_hold_steps: int = 10,
        height_threshold: float = 0.05,
        distance_threshold: float = 0.15,
        grip_threshold: float = 0.04,
        linear_velocity_threshold: float = 0.05,
        angular_velocity_threshold: float = 0.50,
        require_gripper_closed: bool = True,
        require_stable: bool = False,
        workspace_radius: float = 2.0,
    ):
        goal_all = [
            PredicateCall(
                "grasped" if require_gripper_closed else "object_near_hand",
                args={"grip_threshold": grip_threshold, "distance_threshold_m": distance_threshold},
                source="legacy_pick_lift_manager",
            ),
            PredicateCall(
                "lifted",
                args={"height_threshold_m": height_threshold},
                source="legacy_pick_lift_manager",
            ),
        ]
        stability_all = [
            PredicateCall(
                "object_velocity_below",
                args={
                    "linear_velocity_threshold_mps": linear_velocity_threshold,
                    "angular_velocity_threshold_radps": angular_velocity_threshold,
                },
                source="legacy_pick_lift_manager",
            )
        ]
        safety_hard_fail = [
            PredicateCall(
                "object_out_of_workspace",
                args={"workspace_radius_m": workspace_radius},
                source="legacy_pick_lift_manager",
            )
        ]
        self.spec = SuccessSpec(
            hold_steps=success_hold_steps,
            goal_all=goal_all,
            stability_required=require_stable,
            stability_all=stability_all,
            safety_hard_fail=safety_hard_fail,
            source="legacy_pick_lift_manager",
        )
        self.evaluator = TaskSuccessEvaluator(self.spec)

    @property
    def hold_counter(self) -> int:
        return self.evaluator.hold_counter

    @property
    def last_result(self) -> PredicateResult | None:
        return self.evaluator.last_result

    def reset(self) -> None:
        self.evaluator.reset()

    def update(self, snapshot: dict[str, Any]) -> PredicateResult:
        return self.evaluator.update(snapshot)
