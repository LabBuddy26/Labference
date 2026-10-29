from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eai4s_eval.eval.success.snapshot import EvaluationSnapshot


def _event_name(event: Any) -> str | None:
    if isinstance(event, str):
        return event
    if isinstance(event, dict):
        value = event.get("event") or event.get("name") or event.get("type") or event.get("id")
        return str(value) if value else None
    return None


def _snapshot_events(snapshot: EvaluationSnapshot) -> list[str]:
    raw_events = snapshot.get("events", None)
    if raw_events is None:
        raw_events = snapshot.get("task_state.events", [])
    if raw_events is None:
        return []
    if isinstance(raw_events, (str, dict)):
        raw_events = [raw_events]
    events: list[str] = []
    for item in raw_events:
        name = _event_name(item)
        if name:
            events.append(name)
    return events


@dataclass
class SequenceTracker:
    expected_events: list[str] = field(default_factory=list)
    current_index: int = 0
    violated: bool = False
    violation_event: str | None = None

    def reset(self) -> None:
        self.current_index = 0
        self.violated = False
        self.violation_event = None

    @property
    def configured(self) -> bool:
        return bool(self.expected_events)

    @property
    def complete(self) -> bool:
        return not self.violated and self.current_index >= len(self.expected_events)

    @property
    def stage(self) -> str:
        if not self.expected_events:
            return "not_configured"
        if self.violated:
            return f"invalid:{self.violation_event or 'sequence'}"
        if self.complete:
            return "completed"
        return self.expected_events[self.current_index]

    def update(self, snapshot: EvaluationSnapshot) -> bool:
        if not self.expected_events or self.violated:
            return self.complete if self.expected_events else True
        for event in _snapshot_events(snapshot):
            if self.current_index < len(self.expected_events) and event == self.expected_events[self.current_index]:
                self.current_index += 1
                continue
            if event in self.expected_events[self.current_index + 1 :]:
                self.violated = True
                self.violation_event = event
                break
        return self.complete


@dataclass
class ViolationTracker:
    violations: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.violations.clear()

    def update(self, violation: str | None) -> list[str]:
        if violation and violation not in self.violations:
            self.violations.append(violation)
        return list(self.violations)
