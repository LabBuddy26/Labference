from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from eai4s_eval.eval.success.schemas import PredicateCall, SuccessSpec


@dataclass
class SuccessCriteria:
    """Legacy pick/lift criteria kept for compatibility with external imports."""

    success_hold_steps: int = 10
    height_threshold: float = 0.05
    distance_threshold: float = 0.15
    grip_threshold: float = 0.04
    linear_velocity_threshold: float = 0.05
    angular_velocity_threshold: float = 0.50
    require_gripper_closed: bool = True
    require_stable: bool = False
    source: str = "task_spec"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _predicate_name(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("predicate") or item.get("name") or "")
    return ""


def _predicate_args(item: Any) -> dict[str, Any]:
    if isinstance(item, dict) and isinstance(item.get("args"), dict):
        return item["args"]
    return {}


def _predicate_calls(items: Any, *, source: str) -> list[PredicateCall]:
    if not items:
        return []
    if not isinstance(items, list):
        raise ValueError(f"Expected a list of predicates in {source}, got {type(items).__name__}")
    return [PredicateCall.from_raw(item, source=source) for item in items]


def _sequence_events(items: Any) -> list[str]:
    if not items:
        return []
    if not isinstance(items, list):
        raise ValueError(f"Expected a list of sequence events, got {type(items).__name__}")
    events: list[str] = []
    for item in items:
        if isinstance(item, str):
            events.append(item)
        elif isinstance(item, dict):
            name = item.get("event") or item.get("name") or item.get("type") or item.get("id")
            if not name:
                raise ValueError(f"Sequence event is missing a name: {item!r}")
            events.append(str(name))
        else:
            raise ValueError(f"Unsupported sequence event item: {item!r}")
    return events


def _quality_metric_names(items: Any) -> list[str]:
    if not items:
        return []
    if not isinstance(items, list):
        raise ValueError(f"Expected a list of quality metrics, got {type(items).__name__}")
    metrics: list[str] = []
    for item in items:
        if isinstance(item, str):
            metrics.append(item)
        elif isinstance(item, dict):
            name = item.get("metric") or item.get("name") or item.get("key") or item.get("id")
            if not name:
                raise ValueError(f"Quality metric is missing a name: {item!r}")
            metrics.append(str(name))
        else:
            raise ValueError(f"Unsupported quality metric item: {item!r}")
    return metrics


def _string_list(items: Any) -> list[str]:
    if not items:
        return []
    if isinstance(items, str):
        return [items]
    if not isinstance(items, list):
        raise ValueError(f"Expected a string list, got {type(items).__name__}")
    return [str(item) for item in items]


def task_metadata_from_task_spec(task_spec: dict[str, Any]) -> dict[str, Any]:
    task_family = str(task_spec.get("task_family") or "basic")
    leaderboard_group = str(task_spec.get("leaderboard_group") or task_family)
    failure_modes = task_spec.get("failure_modes") or []
    if not isinstance(failure_modes, list):
        raise ValueError("TaskSpec failure_modes must be a list when provided")
    return {
        "task_family": task_family,
        "task_level": str(task_spec.get("task_level") or "base"),
        "task_tags": _string_list(task_spec.get("task_tags")),
        "leaderboard_group": leaderboard_group,
        "failure_modes": failure_modes,
    }


def _hold_config(task_spec: dict[str, Any], success: dict[str, Any]) -> tuple[int, float | None]:
    if "hold_steps" in success:
        return int(success["hold_steps"]), float(success["hold_time_sec"]) if "hold_time_sec" in success else None
    hold_time_sec = success.get("hold_time_sec")
    if hold_time_sec is not None:
        predicate_hz = float(
            task_spec.get("predicate_hz", task_spec.get("control_hz", 10))
        )
        return max(1, int(math.ceil(float(hold_time_sec) * predicate_hz))), float(hold_time_sec)
    return int(task_spec.get("success_hold_steps", 10)), None


def success_spec_from_task_spec(task_spec: dict[str, Any]) -> SuccessSpec:
    success = task_spec.get("success") or {}
    if not isinstance(success, dict):
        raise ValueError("TaskSpec success section must be a mapping")

    if success.get("logic") == "robust_semantic_v0":
        return _robust_success_spec_from_task_spec(task_spec, success)
    return _legacy_success_spec_from_task_spec(task_spec, success)


def _robust_success_spec_from_task_spec(task_spec: dict[str, Any], success: dict[str, Any]) -> SuccessSpec:
    goal = success.get("goal") or {}
    sequence = success.get("sequence") or {}
    terminal = success.get("terminal") or {}
    stability = success.get("stability") or {}
    safety = success.get("safety") or {}
    if not isinstance(goal, dict) or not isinstance(sequence, dict) or not isinstance(terminal, dict) or not isinstance(stability, dict) or not isinstance(safety, dict):
        raise ValueError("TaskSpec success goal/sequence/terminal/stability/safety sections must be mappings")

    hold_steps, hold_time_sec = _hold_config(task_spec, success)
    hard_fail = _predicate_calls(safety.get("hard_fail"), source="success.safety.hard_fail")
    legacy_failure = task_spec.get("failure") or {}
    if isinstance(legacy_failure, dict):
        hard_fail.extend(
            call
            for call in _predicate_calls(legacy_failure.get("any"), source="failure.any")
            if call.predicate != "timeout"
        )

    return SuccessSpec(
        logic="robust_semantic_v0",
        hold_steps=hold_steps,
        hold_time_sec=hold_time_sec,
        goal_all=_predicate_calls(goal.get("all"), source="success.goal.all"),
        goal_any=_predicate_calls(goal.get("any"), source="success.goal.any"),
        sequence_required=bool(sequence.get("required", False)),
        sequence_events=_sequence_events(sequence.get("events") or sequence.get("order") or sequence.get("steps")),
        terminal_required=bool(terminal.get("required", False)),
        terminal_all=_predicate_calls(terminal.get("all"), source="success.terminal.all"),
        stability_required=bool(stability.get("required", False)),
        stability_all=_predicate_calls(stability.get("all"), source="success.stability.all"),
        safety_hard_fail=_dedupe_calls(hard_fail),
        quality_metrics=_quality_metric_names(task_spec.get("quality_metrics") or success.get("quality_metrics")),
        source="task_spec",
    )


def _legacy_success_spec_from_task_spec(task_spec: dict[str, Any], success: dict[str, Any]) -> SuccessSpec:
    hold_steps = int(task_spec.get("success_hold_steps", 10))
    goal_all: list[PredicateCall] = []
    stability_all: list[PredicateCall] = []
    stability_required = False

    for item in success.get("all") or []:
        name = _predicate_name(item)
        args = _predicate_args(item)
        if name == "sustained":
            hold_steps = int(args.get("success_hold_steps", hold_steps))
        elif name == "stable":
            stability_required = True
            stability_all.append(PredicateCall.from_raw(item, source="success.all"))
        elif name:
            goal_all.append(PredicateCall.from_raw(item, source="success.all"))

    for item in success.get("optional_strict") or []:
        name = _predicate_name(item)
        if name in {"stable", "object_velocity_below", "stable_pose"}:
            stability_all.append(PredicateCall.from_raw(item, source="success.optional_strict"))

    failure = task_spec.get("failure") or {}
    hard_fail = []
    if isinstance(failure, dict):
        hard_fail = [
            call
            for call in _predicate_calls(failure.get("any"), source="failure.any")
            if call.predicate != "timeout"
        ]

    return SuccessSpec(
        logic="robust_semantic_v0",
        hold_steps=hold_steps,
        goal_all=goal_all,
        terminal_required=False,
        terminal_all=[],
        stability_required=stability_required,
        stability_all=_dedupe_calls(stability_all),
        safety_hard_fail=_dedupe_calls(hard_fail),
        source="legacy_task_spec",
    )


def _dedupe_calls(calls: list[PredicateCall]) -> list[PredicateCall]:
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    deduped: list[PredicateCall] = []
    for call in calls:
        signature = (call.predicate, tuple(sorted((key, repr(value)) for key, value in call.args.items())))
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(call)
    return deduped


def apply_success_cli_overrides_to_spec(spec: SuccessSpec, args: Any) -> SuccessSpec:
    changed = False
    if getattr(args, "success_hold_steps", None) is not None:
        spec.hold_steps = int(args.success_hold_steps)
        changed = True

    height = getattr(args, "success_height_threshold", None)
    distance = getattr(args, "success_distance_threshold", None)
    grip = getattr(args, "success_grip_threshold", None)
    lin_stable = getattr(args, "success_require_stable", None)
    require_gripper_closed = getattr(args, "success_require_gripper_closed", None)

    if height is not None:
        _update_predicate_args(spec, {"lifted", "object_lifted"}, height_threshold_m=float(height))
        changed = True
    if distance is not None:
        _update_predicate_args(spec, {"grasped", "object_near_hand", "near"}, distance_threshold_m=float(distance))
        changed = True
    if grip is not None:
        _update_predicate_args(spec, {"grasped", "gripper_closed"}, grip_threshold=float(grip))
        changed = True
    if require_gripper_closed is False:
        spec.goal_all = [
            PredicateCall("object_near_hand", args=dict(call.args), source="cli_override")
            if call.predicate == "grasped"
            else call
            for call in spec.goal_all
        ]
        changed = True
    if lin_stable is not None:
        spec.stability_required = bool(lin_stable)
        if spec.stability_required and not spec.stability_all:
            spec.stability_all.append(PredicateCall("object_velocity_below", source="cli_override"))
        changed = True

    if changed:
        spec.source = f"{spec.source}_with_cli_overrides"
    return spec


def _update_predicate_args(spec: SuccessSpec, names: set[str], **updates: Any) -> None:
    for attr in ("goal_all", "goal_any", "terminal_all", "stability_all", "safety_hard_fail"):
        calls = getattr(spec, attr)
        setattr(
            spec,
            attr,
            [call.with_args(**updates) if call.predicate in names else call for call in calls],
        )


def success_criteria_from_task_spec(task_spec: dict[str, Any]) -> SuccessCriteria:
    """Return legacy pick/lift criteria derived from the robust success spec."""

    spec = success_spec_from_task_spec(task_spec)
    criteria = SuccessCriteria(success_hold_steps=spec.hold_steps, source=spec.source)
    criteria.require_gripper_closed = any(call.predicate == "grasped" for call in spec.goal_all)
    criteria.require_stable = spec.stability_required

    for call in [*spec.goal_all, *spec.stability_all]:
        args = call.args
        if call.predicate == "grasped":
            criteria.grip_threshold = float(args.get("grip_threshold", criteria.grip_threshold))
            criteria.distance_threshold = float(args.get("distance_threshold_m", criteria.distance_threshold))
        elif call.predicate in {"object_near_hand", "near"}:
            criteria.distance_threshold = float(args.get("distance_threshold_m", criteria.distance_threshold))
        elif call.predicate in {"lifted", "object_lifted"}:
            criteria.height_threshold = float(args.get("height_threshold_m", criteria.height_threshold))
        elif call.predicate in {"stable", "object_velocity_below", "stable_pose"}:
            criteria.linear_velocity_threshold = float(
                args.get("linear_velocity_threshold_mps", criteria.linear_velocity_threshold)
            )
            criteria.angular_velocity_threshold = float(
                args.get("angular_velocity_threshold_radps", criteria.angular_velocity_threshold)
            )
    return criteria


def apply_success_cli_overrides(criteria: SuccessCriteria, args: Any) -> SuccessCriteria:
    overrides = {
        "success_hold_steps": "success_hold_steps",
        "success_height_threshold": "height_threshold",
        "success_distance_threshold": "distance_threshold",
        "success_grip_threshold": "grip_threshold",
        "success_require_gripper_closed": "require_gripper_closed",
        "success_require_stable": "require_stable",
    }
    changed = False
    for arg_name, field_name in overrides.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(criteria, field_name, value)
            changed = True
    if changed:
        criteria.source = "task_spec_with_cli_overrides"
    return criteria
