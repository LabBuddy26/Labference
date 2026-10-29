from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PredicateCall:
    predicate: str
    args: dict[str, Any] = field(default_factory=dict)
    source: str = "task_spec"

    @classmethod
    def from_raw(cls, item: Any, *, source: str = "task_spec") -> "PredicateCall":
        if isinstance(item, str):
            return cls(predicate=item, source=source)
        if isinstance(item, dict):
            name = item.get("predicate") or item.get("name")
            if not name:
                raise ValueError(f"Predicate item is missing a predicate name: {item!r}")
            args = item.get("args") or {}
            if not isinstance(args, dict):
                raise ValueError(f"Predicate args must be a mapping for {name!r}: {args!r}")
            return cls(predicate=str(name), args=dict(args), source=source)
        raise ValueError(f"Unsupported predicate item: {item!r}")

    @property
    def key(self) -> str:
        return str(self.args.get("id") or self.predicate)

    def with_args(self, **updates: Any) -> "PredicateCall":
        args = dict(self.args)
        args.update({key: value for key, value in updates.items() if value is not None})
        return PredicateCall(predicate=self.predicate, args=args, source=self.source)

    def to_dict(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "args": dict(self.args), "source": self.source}


@dataclass
class SuccessSpec:
    logic: str = "robust_semantic_v0"
    hold_steps: int = 10
    hold_time_sec: float | None = None
    goal_all: list[PredicateCall] = field(default_factory=list)
    goal_any: list[PredicateCall] = field(default_factory=list)
    sequence_required: bool = False
    sequence_events: list[str] = field(default_factory=list)
    terminal_required: bool = False
    terminal_all: list[PredicateCall] = field(default_factory=list)
    stability_required: bool = False
    stability_all: list[PredicateCall] = field(default_factory=list)
    safety_hard_fail: list[PredicateCall] = field(default_factory=list)
    quality_metrics: list[str] = field(default_factory=list)
    source: str = "task_spec"

    @property
    def success_hold_steps(self) -> int:
        return self.hold_steps

    def all_calls(self) -> list[PredicateCall]:
        return [
            *self.goal_all,
            *self.goal_any,
            *self.terminal_all,
            *self.stability_all,
            *self.safety_hard_fail,
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "logic": self.logic,
            "hold_steps": self.hold_steps,
            "hold_time_sec": self.hold_time_sec,
            "goal": {
                "all": [call.to_dict() for call in self.goal_all],
                "any": [call.to_dict() for call in self.goal_any],
            },
            "sequence": {
                "required": self.sequence_required,
                "events": list(self.sequence_events),
            },
            "terminal": {
                "required": self.terminal_required,
                "all": [call.to_dict() for call in self.terminal_all],
            },
            "stability": {
                "required": self.stability_required,
                "all": [call.to_dict() for call in self.stability_all],
            },
            "safety": {
                "hard_fail": [call.to_dict() for call in self.safety_hard_fail],
            },
            "quality_metrics": list(self.quality_metrics),
            "source": self.source,
        }


@dataclass
class PredicateValue:
    name: str
    passed: bool
    value: Any = None
    details: dict[str, Any] = field(default_factory=dict)
    missing: bool = False
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuccessEvaluation:
    success_logic: str
    task_success: bool
    goal_pass: bool
    sequence_pass: bool
    sequence_required: bool
    sequence_stage: str
    terminal_pass: bool
    stability_pass: bool
    safety_valid: bool
    hard_failure: bool
    failure_type: str | None
    latched_violations: list[str]
    success_hold_counter: int
    hold_steps: int
    hold_time_sec: float | None
    predicate_values: dict[str, PredicateValue]
    goal_predicates: dict[str, bool]
    terminal_predicates: dict[str, bool]
    stability_predicates: dict[str, bool]
    safety_predicates: dict[str, bool]
    quality_metrics: dict[str, Any]
    completed_stage: str
    stage_completion: float
    grasped: bool = False
    object_lifted: bool = False
    gripper_closed: bool = False
    object_near_hand: bool = False
    stable: bool = False
    hand_object_distance: float = 0.0
    object_height_change: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
