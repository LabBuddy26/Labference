from __future__ import annotations

from typing import Any

from eai4s_eval.eval.success.predicates import MissingPredicateState, default_predicate_registry
from eai4s_eval.eval.success.registry import PredicateRegistry
from eai4s_eval.eval.success.schemas import PredicateCall, PredicateValue, SuccessEvaluation, SuccessSpec
from eai4s_eval.eval.success.snapshot import EvaluationSnapshot
from eai4s_eval.eval.success.trackers import SequenceTracker, ViolationTracker


class TaskSuccessEvaluator:
    def __init__(self, spec: SuccessSpec, registry: PredicateRegistry | None = None):
        self.spec = spec
        self.registry = registry or default_predicate_registry()
        self.hold_counter = 0
        self.last_result: SuccessEvaluation | None = None
        self.sequence_tracker = SequenceTracker(spec.sequence_events)
        self.violation_tracker = ViolationTracker()

    def reset(self) -> None:
        self.hold_counter = 0
        self.last_result = None
        self.sequence_tracker.reset()
        self.violation_tracker.reset()

    def update(self, snapshot: dict[str, Any] | EvaluationSnapshot) -> SuccessEvaluation:
        eval_snapshot = EvaluationSnapshot.from_mapping(snapshot)
        predicate_values: dict[str, PredicateValue] = {}
        goal_values = self._evaluate_group(eval_snapshot, [*self.spec.goal_all, *self.spec.goal_any], predicate_values)
        terminal_values = self._evaluate_group(eval_snapshot, self.spec.terminal_all, predicate_values)
        stability_values = self._evaluate_group(eval_snapshot, self.spec.stability_all, predicate_values)
        safety_values = self._evaluate_group(eval_snapshot, self.spec.safety_hard_fail, predicate_values)
        sequence_pass = self.sequence_tracker.update(eval_snapshot)

        goal_all_pass = all(self._passed(call, goal_values) for call in self.spec.goal_all)
        goal_any_pass = True if not self.spec.goal_any else any(self._passed(call, goal_values) for call in self.spec.goal_any)
        goal_pass = goal_all_pass and goal_any_pass
        terminal_pass = True if not self.spec.terminal_required else all(
            self._passed(call, terminal_values) for call in self.spec.terminal_all
        )
        stability_pass = True if not self.spec.stability_required else all(
            self._passed(call, stability_values) for call in self.spec.stability_all
        )
        current_hard_failure = any(self._passed(call, safety_values) for call in self.spec.safety_hard_fail)
        current_failure_type = self._first_passed_name(self.spec.safety_hard_fail, safety_values) if current_hard_failure else None
        latched_violations = self.violation_tracker.update(current_failure_type)
        hard_failure = bool(latched_violations)
        safety_valid = not hard_failure
        failure_type = latched_violations[0] if latched_violations else None

        sequence_valid_for_success = sequence_pass if self.spec.sequence_required else True
        current_success = goal_pass and sequence_valid_for_success and terminal_pass and stability_pass and safety_valid
        self.hold_counter = self.hold_counter + 1 if current_success else 0
        task_success = self.hold_counter >= self.spec.hold_steps
        completed_stage, stage_completion = self._stage_progress(predicate_values, goal_pass, terminal_pass, stability_pass, task_success)
        quality_metrics = self._quality_metrics(eval_snapshot)

        result = SuccessEvaluation(
            success_logic=self.spec.logic,
            task_success=task_success,
            goal_pass=goal_pass,
            sequence_pass=sequence_pass,
            sequence_required=self.spec.sequence_required,
            sequence_stage=self.sequence_tracker.stage,
            terminal_pass=terminal_pass,
            stability_pass=stability_pass,
            safety_valid=safety_valid,
            hard_failure=hard_failure,
            failure_type=failure_type,
            latched_violations=latched_violations,
            success_hold_counter=self.hold_counter,
            hold_steps=self.spec.hold_steps,
            hold_time_sec=self.spec.hold_time_sec,
            predicate_values=predicate_values,
            goal_predicates={call.key: self._passed(call, goal_values) for call in [*self.spec.goal_all, *self.spec.goal_any]},
            terminal_predicates={call.key: self._passed(call, terminal_values) for call in self.spec.terminal_all},
            stability_predicates={call.key: self._passed(call, stability_values) for call in self.spec.stability_all},
            safety_predicates={call.key: self._passed(call, safety_values) for call in self.spec.safety_hard_fail},
            quality_metrics=quality_metrics,
            completed_stage=completed_stage,
            stage_completion=stage_completion,
            grasped=self._legacy_bool(predicate_values, "grasped"),
            object_lifted=self._legacy_bool(predicate_values, "lifted") or self._legacy_bool(predicate_values, "object_lifted"),
            gripper_closed=self._legacy_detail_bool(predicate_values, "grasped", "gripper_closed")
            or self._legacy_bool(predicate_values, "gripper_closed"),
            object_near_hand=self._legacy_detail_bool(predicate_values, "grasped", "object_near_hand")
            or self._legacy_bool(predicate_values, "object_near_hand"),
            stable=self._legacy_bool(predicate_values, "stable")
            or self._legacy_bool(predicate_values, "object_velocity_below")
            or self._legacy_bool(predicate_values, "stable_pose"),
            hand_object_distance=self._legacy_float(predicate_values, "grasped", "hand_object_distance_m")
            or self._legacy_value_float(predicate_values, "object_near_hand"),
            object_height_change=self._legacy_float(predicate_values, "lifted", "height_change_m")
            or self._legacy_float(predicate_values, "object_lifted", "height_change_m"),
        )
        self.last_result = result
        return result

    def _quality_metrics(self, snapshot: EvaluationSnapshot) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for name in self.spec.quality_metrics:
            value = snapshot.get(name, None)
            if value is None:
                value = snapshot.get(f"task_state.{name}", None)
            if value is not None:
                metrics[name] = value
        return metrics

    def _evaluate_group(
        self,
        snapshot: EvaluationSnapshot,
        calls: list[PredicateCall],
        predicate_values: dict[str, PredicateValue],
    ) -> dict[str, PredicateValue]:
        group_values: dict[str, PredicateValue] = {}
        for call in calls:
            key = self._unique_key(call, predicate_values)
            try:
                value = self.registry.evaluate(call.predicate, snapshot, call.args)
            except MissingPredicateState:
                raise
            value.name = call.predicate
            predicate_values[key] = value
            group_values[key] = value
        return group_values

    @staticmethod
    def _unique_key(call: PredicateCall, existing: dict[str, PredicateValue]) -> str:
        base = call.key
        if base not in existing:
            return base
        index = 2
        while f"{base}_{index}" in existing:
            index += 1
        return f"{base}_{index}"

    @staticmethod
    def _passed(call: PredicateCall, values: dict[str, PredicateValue]) -> bool:
        if call.key in values:
            return values[call.key].passed
        return any(value.name == call.predicate and value.passed for value in values.values())

    @staticmethod
    def _first_passed_name(calls: list[PredicateCall], values: dict[str, PredicateValue]) -> str | None:
        for call in calls:
            if TaskSuccessEvaluator._passed(call, values):
                return call.key
        return None

    @staticmethod
    def _legacy_bool(values: dict[str, PredicateValue], name: str) -> bool:
        value = values.get(name)
        if value is not None:
            return bool(value.passed)
        return any(item.name == name and item.passed for item in values.values())

    @staticmethod
    def _legacy_detail_bool(values: dict[str, PredicateValue], name: str, key: str) -> bool:
        value = values.get(name)
        if value is not None and key in value.details:
            return bool(value.details[key])
        for item in values.values():
            if item.name == name and key in item.details:
                return bool(item.details[key])
        return False

    @staticmethod
    def _legacy_float(values: dict[str, PredicateValue], name: str, key: str) -> float:
        value = values.get(name)
        if value is not None and key in value.details:
            return round(float(value.details[key]), 4)
        for item in values.values():
            if item.name == name and key in item.details:
                return round(float(item.details[key]), 4)
        return 0.0

    @staticmethod
    def _legacy_value_float(values: dict[str, PredicateValue], name: str) -> float:
        value = values.get(name)
        if value is not None and isinstance(value.value, (int, float)):
            return round(float(value.value), 4)
        return 0.0

    def _stage_progress(
        self,
        values: dict[str, PredicateValue],
        goal_pass: bool,
        terminal_pass: bool,
        stability_pass: bool,
        task_success: bool,
    ) -> tuple[str, float]:
        if task_success:
            return "stabilize", 1.0
        if self._legacy_bool(values, "lifted") or self._legacy_bool(values, "object_lifted"):
            return "lift", 0.80
        if self._legacy_bool(values, "grasped") or self._legacy_bool(values, "object_near_hand"):
            return "grasp", 0.50
        if goal_pass and terminal_pass and stability_pass:
            return "hold", min(0.95, self.hold_counter / max(1, self.spec.hold_steps))

        predicate_count = len(self.spec.goal_all) + len(self.spec.goal_any)
        if predicate_count:
            passed = sum(1 for value in values.values() if value.passed and value.name not in {"object_out_of_workspace"})
            return "partial", round(min(0.75, passed / max(1, predicate_count) * 0.75), 4)
        return "start", 0.0
