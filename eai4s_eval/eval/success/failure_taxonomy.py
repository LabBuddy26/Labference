from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


FAILURE_CATEGORIES = {
    "none",
    "timeout",
    "collision",
    "drop",
    "wrong_place",
    "wrong_object",
    "safety_violation",
    "sequence_error",
    "instability",
    "policy_error",
    "unknown_failure",
}


_CODE_CATEGORY = {
    "timeout": "timeout",
    "forbidden_collision": "collision",
    "contact_force_exceeded": "collision",
    "self_collision": "collision",
    "object_out_of_workspace": "drop",
    "object_dropped": "drop",
    "object_tipped_over": "drop",
    "wrong_target_reached": "wrong_place",
    "wrong_container": "wrong_place",
    "wrong_hole_inserted": "wrong_place",
    "wrong_object_moved": "wrong_object",
    "wrong_object_selected": "wrong_object",
    "spill_below": "safety_violation",
    "severe_spill": "safety_violation",
    "no_cross_contamination": "safety_violation",
    "cross_contamination": "safety_violation",
    "overflow": "safety_violation",
    "unsafe_device_operation": "safety_violation",
    "policy_output_invalid": "policy_error",
    "policy_request_failed": "policy_error",
}


_TERMINATION_CATEGORY = {
    "timeout": "timeout",
    "policy_output_invalid": "policy_error",
    "policy_request_failed": "policy_error",
}


@dataclass(frozen=True)
class FailureClassification:
    failure_category: str
    failure_code: str
    failure_source: str
    failure_severity: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def classify_failure(
    *,
    success: bool,
    termination_reason: str | None,
    failure_type: str | None,
    success_result: Any | None = None,
) -> FailureClassification:
    if success:
        return FailureClassification("none", "none", "none", "none")

    code = str(failure_type or "unknown_failure")
    source = "success_predicate" if failure_type else "runner"
    severity = "hard" if failure_type else "soft"

    if failure_type and code in _CODE_CATEGORY:
        return FailureClassification(_CODE_CATEGORY[code], code, source, severity)

    if success_result is not None:
        if getattr(success_result, "sequence_required", False) and not getattr(success_result, "sequence_pass", True):
            return FailureClassification("sequence_error", "sequence_incomplete_or_invalid", "sequence", "hard")
        if getattr(success_result, "goal_pass", False) and getattr(success_result, "terminal_pass", False) and not getattr(success_result, "stability_pass", True):
            return FailureClassification("instability", "stability_not_held", "stability", "soft")

    if termination_reason in _TERMINATION_CATEGORY:
        return FailureClassification(_TERMINATION_CATEGORY[str(termination_reason)], str(termination_reason), "runner", "hard")

    if str(termination_reason or "") == "timeout":
        return FailureClassification("timeout", "timeout", "runner", "soft")

    return FailureClassification("unknown_failure", code, source, "soft")
