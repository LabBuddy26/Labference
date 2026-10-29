from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from eai4s_eval.eval.success.schemas import PredicateCall, PredicateValue, SuccessEvaluation, SuccessSpec
from eai4s_eval.utils.io import ensure_dir


PROGRESS_SCHEMA_VERSION = "eai4s.eval_progress.v1"


def _safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _remaining(actual: Any, operator: str, threshold: Any, tolerance: Any = None) -> float | None:
    actual_f = _finite_float(actual)
    if actual_f is None:
        return None
    if operator == "range" and isinstance(threshold, (list, tuple)) and len(threshold) == 2:
        low = _finite_float(threshold[0])
        high = _finite_float(threshold[1])
        if low is None or high is None:
            return None
        return max(low - actual_f, actual_f - high, 0.0)
    threshold_f = _finite_float(threshold)
    if threshold_f is None:
        return None
    if operator in {"<=", "<"}:
        return max(0.0, actual_f - threshold_f)
    if operator in {">=", ">"}:
        return max(0.0, threshold_f - actual_f)
    if operator == "within":
        tolerance_f = _finite_float(tolerance)
        if tolerance_f is None:
            return None
        return max(0.0, abs(actual_f - threshold_f) - tolerance_f)
    return None


def _condition(
    name: str,
    actual: Any,
    operator: str,
    threshold: Any,
    unit: str = "",
    *,
    tolerance: Any = None,
) -> dict[str, Any]:
    remaining = _remaining(actual, operator, threshold, tolerance)
    condition = {
        "name": name,
        "actual": _safe(actual),
        "operator": operator,
        "threshold": _safe(threshold),
        "unit": unit,
        "remaining": remaining,
    }
    if tolerance is not None:
        condition["tolerance"] = _safe(tolerance)
    condition["satisfied"] = bool(remaining == 0.0) if remaining is not None else None
    return condition


def _predicate_conditions(value: PredicateValue, *, safety: bool = False) -> list[dict[str, Any]]:
    """Normalize predicate-specific details into comparable requirements."""

    name = value.name
    details = value.details or {}
    provided = details.get("criteria")
    if isinstance(provided, list):
        normalized: list[dict[str, Any]] = []
        for item in provided:
            if not isinstance(item, dict):
                continue
            condition = dict(item)
            condition.setdefault("unit", "")
            condition["remaining"] = _remaining(
                condition.get("actual"),
                str(condition.get("operator", "==")),
                condition.get("threshold"),
                condition.get("tolerance"),
            )
            if condition["remaining"] is not None:
                condition["satisfied"] = condition["remaining"] == 0.0
            else:
                condition.setdefault("satisfied", None)
            normalized.append(_safe(condition))
        if normalized:
            return normalized

    if name == "grasped":
        return [
            _condition("gripper_mean", details.get("gripper_mean"), "<", details.get("grip_threshold"), "m"),
            _condition(
                "hand_object_distance",
                details.get("hand_object_distance_m"),
                "<",
                details.get("distance_threshold_m"),
                "m",
            ),
        ]
    if name == "gripper_closed":
        return [_condition("gripper_mean", value.value, "<", details.get("threshold"), "m")]
    if name in {"near", "object_near_hand"}:
        return [_condition("distance", details.get("distance_m", value.value), "<=", details.get("threshold_m"), "m")]
    if name in {"lifted", "object_lifted"}:
        return [_condition("height_change", details.get("height_change_m", value.value), ">=", details.get("threshold_m"), "m")]
    if name in {"object_velocity_below", "stable", "stable_pose"}:
        return [
            _condition("linear_speed", details.get("linear_velocity_mps"), "<", details.get("linear_threshold_mps"), "m/s"),
            _condition("angular_speed", details.get("angular_velocity_radps"), "<", details.get("angular_threshold_radps"), "rad/s"),
        ]
    if name == "released":
        return [_condition("gripper_mean", value.value, ">=", details.get("open_threshold"), "m")]
    if name == "tcp_far":
        return [_condition("tcp_object_distance", details.get("distance_m", value.value), ">", details.get("threshold_m"), "m")]
    if name in {"inserted", "insert_depth_reached"}:
        return [_condition("insert_depth", details.get("insert_depth_m", value.value), ">=", details.get("min_depth_m"), "m")]
    if name == "aligned":
        return [_condition("alignment_cosine", details.get("alignment_cosine", value.value), ">=", details.get("min_cosine"), "")]
    if name in {"volume_transferred", "pipette_aspirated", "pipette_dispensed"}:
        return [
            _condition(
                "volume",
                details.get("volume_ml", value.value),
                "within" if details.get("exact_target", False) else ">=",
                details.get("target_ml"),
                "mL",
                tolerance=details.get("tolerance_ml"),
            )
        ]
    if name == "spill_below":
        return [_condition("spill", details.get("spill_ml", value.value), "<=", details.get("max_spill_ml"), "mL")]
    if name == "tip_in_liquid" and "tip_depth_m" in details:
        return [_condition("tip_depth", details.get("tip_depth_m", value.value), ">=", details.get("min_tip_depth_m"), "m")]
    if name == "no_cross_contamination" and "contamination_events" in details:
        return [_condition("contamination_events", details.get("contamination_events", value.value), "<=", 0, "event")]
    if name in {"container_upright", "object_tipped_over"}:
        operator = ">=" if not safety else ">="
        return [_condition("upright_cosine", details.get("upright_cosine", value.value), operator, details.get("min_cosine"), "")]
    if name == "knob_turned":
        return [
            _condition(
                "angle",
                details.get("angle_rad", value.value),
                "within" if details.get("exact_target", False) else ">=",
                details.get("target_rad"),
                "rad",
                tolerance=details.get("tolerance_rad"),
            )
        ]
    if name == "joint_in_range":
        return [_condition("joint_value", value.value, "range", [details.get("min"), details.get("max")], "rad")]
    if name == "device_state":
        return [
            {
                "name": "device_state",
                "actual": _safe(value.value),
                "operator": "is",
                "threshold": _safe(details.get("expected")),
                "unit": "",
                "remaining": None,
                "remaining_reason": "waiting_for_expected_state",
                "satisfied": bool(value.passed),
            }
        ]
    if name == "object_out_of_workspace":
        return [_condition("workspace_distance_xy", details.get("distance_xy_m", value.value), "<=", details.get("radius_m"), "m")]
    if name == "object_dropped":
        return [_condition("object_height", details.get("object_height_m", value.value), ">=", details.get("min_height_m"), "m")]
    if name == "contact_force_exceeded":
        return [_condition("contact_force", details.get("max_contact_force_n", value.value), "<=", details.get("threshold_n"), "N")]
    if name == "wrong_object_moved":
        return [_condition("distractor_motion", details.get("max_motion_m", value.value), "<=", details.get("threshold_m"), "m")]
    if name == "timeout":
        return [_condition("step", value.value, "<", details.get("max_steps"), "step")]

    # Boolean and categorical predicates cannot expose a meaningful metric
    # distance. Keep the raw source and make that limitation explicit.
    return [
        {
            "name": str(details.get("key") or name),
            "actual": _safe(value.value),
            "operator": "is",
            "threshold": False if safety else True,
            "unit": "",
            "remaining": None,
            "remaining_reason": "waiting_for_false" if safety else "waiting_for_true",
            "satisfied": (not value.passed) if safety else bool(value.passed),
        }
    ]


def _group_for_key(result: SuccessEvaluation, key: str) -> str:
    if key in result.goal_predicates:
        return "goal"
    if key in result.terminal_predicates:
        return "terminal"
    if key in result.stability_predicates:
        return "stability"
    if key in result.safety_predicates:
        return "safety"
    return "diagnostic"


def criterion_rows(result: SuccessEvaluation) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in result.predicate_values.items():
        group = _group_for_key(result, key)
        safety = group == "safety"
        rows.append(
            {
                "key": key,
                "predicate": value.name,
                "group": group,
                "required": group != "diagnostic",
                "passed": (not value.passed) if safety else bool(value.passed),
                "triggered": bool(value.passed) if safety else False,
                "value": _safe(value.value),
                "conditions": _predicate_conditions(value, safety=safety),
                "details": _safe(value.details),
                "missing": bool(value.missing),
                "message": value.message,
            }
        )
    return rows


def active_stage(result: SuccessEvaluation, criteria: list[dict[str, Any]]) -> tuple[str, str]:
    if result.hard_failure:
        key = result.failure_type or "safety_violation"
        return f"safety:{key}", str(key).replace("_", " ")
    if result.sequence_required and not result.sequence_pass:
        return f"sequence:{result.sequence_stage}", str(result.sequence_stage).replace("_", " ")
    group_pass = {
        "goal": result.goal_pass,
        "terminal": result.terminal_pass,
        "stability": result.stability_pass,
    }
    for group in ("goal", "terminal", "stability"):
        if group_pass[group]:
            continue
        for row in criteria:
            if row["group"] == group and not row["passed"]:
                return f"{group}:{row['key']}", str(row["key"]).replace("_", " ")
    if result.task_success:
        return "success", "success"
    return "hold", "hold success conditions"


def build_progress_payload(
    result: SuccessEvaluation,
    *,
    task_id: str,
    episode_id: str,
    episode_index: int,
    split: str,
    seed: int,
    action_step: int,
    sim_step: int,
    policy_calls: int,
    attempt: int = 1,
) -> dict[str, Any]:
    criteria = criterion_rows(result)
    stage_id, stage_label = active_stage(result, criteria)
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "timestamp": datetime.now().astimezone().isoformat(),
        "task_id": task_id,
        "episode_id": episode_id,
        "episode_index": int(episode_index),
        "split": split,
        "seed": int(seed),
        "attempt": int(attempt),
        "action_step": int(action_step),
        "sim_step": int(sim_step),
        "policy_calls": int(policy_calls),
        "stage_id": stage_id,
        "stage_label": stage_label,
        "task_success": bool(result.task_success),
        "goal_pass": bool(result.goal_pass),
        "sequence_pass": bool(result.sequence_pass),
        "sequence_required": bool(result.sequence_required),
        "sequence_stage": result.sequence_stage,
        "terminal_pass": bool(result.terminal_pass),
        "stability_pass": bool(result.stability_pass),
        "safety_valid": bool(result.safety_valid),
        "failure_type": result.failure_type,
        "hold": {
            "current_steps": int(result.success_hold_counter),
            "required_steps": int(result.hold_steps),
            "hold_time_sec": result.hold_time_sec,
            "remaining_steps": max(0, int(result.hold_steps) - int(result.success_hold_counter)),
        },
        "criteria": criteria,
    }


def _call_summary(call: PredicateCall) -> str:
    effective_args = dict(call.args)
    defaults: dict[str, dict[str, Any]] = {
        "grasped": {"grip_threshold": 0.035, "distance_threshold_m": 0.15},
        "gripper_closed": {"closed_threshold": 0.035},
        "object_near_hand": {"distance_threshold_m": 0.15},
        "near": {"distance_threshold_m": 0.05, "axes": "xyz"},
        "lifted": {"height_threshold_m": 0.05},
        "object_lifted": {"height_threshold_m": 0.05},
        "object_velocity_below": {
            "linear_velocity_threshold_mps": 0.05,
            "angular_velocity_threshold_radps": 0.50,
        },
        "stable": {
            "linear_velocity_threshold_mps": 0.05,
            "angular_velocity_threshold_radps": 0.50,
        },
        "stable_pose": {
            "linear_velocity_threshold_mps": 0.05,
            "angular_velocity_threshold_radps": 0.50,
        },
        "released": {"open_threshold": 0.035},
        "tcp_far": {"distance_threshold_m": 0.08},
        "inserted": {"min_depth_m": 0.02},
        "insert_depth_reached": {"min_depth_m": 0.02},
        "aligned": {"min_cosine": 0.95},
        "object_out_of_workspace": {"workspace_radius_m": 2.0},
        "object_dropped": {"min_height_m": 0.02},
        "contact_force_exceeded": {"max_force_n": 50.0},
        "object_tipped_over": {"min_upright_cosine": 0.70},
        "container_upright": {"min_cosine": 0.90},
        "spill_below": {"max_spill_ml": 0.1},
        "wrong_object_moved": {"distance_threshold_m": 0.02},
    }
    for key, value in defaults.get(call.predicate, {}).items():
        effective_args.setdefault(key, value)
    args = ", ".join(f"{key}={value}" for key, value in effective_args.items())
    return f"{call.key}({call.predicate}{', ' + args if args else ''})"


class EvaluationProgressReporter:
    """Rate-limited human and JSONL diagnostics for official TaskSpec success."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        run_id: str,
        task_id: str,
        spec: SuccessSpec,
        predicate_hz: float,
        interval_sec: float = 1.0,
        output_format: str = "both",
    ) -> None:
        if interval_sec <= 0:
            raise ValueError("progress interval must be greater than zero")
        if output_format not in {"both", "stdout", "jsonl"}:
            raise ValueError("progress output format must be one of: both, stdout, jsonl")
        self.run_id = run_id
        self.task_id = task_id
        self.spec = spec
        self.interval_steps = max(1, int(round(float(predicate_hz) * interval_sec)))
        self.stdout_enabled = output_format in {"both", "stdout"}
        self.jsonl_enabled = output_format in {"both", "jsonl"}
        self.path = ensure_dir(Path(run_dir) / "logs") / "progress_events.jsonl"
        self._stream = None
        self._jsonl_error_reported = False
        self._episode: dict[str, Any] = {}
        self._last_emit_step = -self.interval_steps
        self._last_stage: str | None = None
        self._last_predicates: tuple[tuple[str, bool, bool], ...] | None = None
        self._last_hold = 0
        if self.jsonl_enabled:
            try:
                self.path.write_text("", encoding="utf-8")
            except OSError as exc:
                print(f"[EvalProgress] JSONL logging disabled after initialization failure: {exc}", flush=True)
                self._jsonl_error_reported = True
                self.jsonl_enabled = False

    def _write(self, payload: dict[str, Any]) -> None:
        payload = {"run_id": self.run_id, **payload}
        if self.jsonl_enabled:
            try:
                if self._stream is None:
                    self._stream = self.path.open("a", encoding="utf-8", buffering=1)
                self._stream.write(json.dumps(_safe(payload), ensure_ascii=False) + "\n")
                self._stream.flush()
            except OSError as exc:
                if not self._jsonl_error_reported:
                    print(f"[EvalProgress] JSONL logging disabled after write failure: {exc}", flush=True)
                    self._jsonl_error_reported = True
                self.jsonl_enabled = False

    def start_episode(
        self,
        *,
        episode_id: str,
        episode_index: int,
        split: str,
        seed: int,
        attempt: int = 1,
    ) -> None:
        self._episode = {
            "task_id": self.task_id,
            "episode_id": episode_id,
            "episode_index": int(episode_index),
            "split": split,
            "seed": int(seed),
            "attempt": int(attempt),
        }
        self._last_emit_step = -self.interval_steps
        self._last_stage = None
        self._last_predicates = None
        self._last_hold = 0
        groups = {
            "goal_all": [_call_summary(call) for call in self.spec.goal_all],
            "goal_any": [_call_summary(call) for call in self.spec.goal_any],
            "sequence": list(self.spec.sequence_events),
            "terminal": [_call_summary(call) for call in self.spec.terminal_all],
            "stability": [_call_summary(call) for call in self.spec.stability_all],
            "safety": [_call_summary(call) for call in self.spec.safety_hard_fail],
        }
        payload = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "timestamp": datetime.now().astimezone().isoformat(),
            "event_type": "criteria",
            **self._episode,
            "success_logic": self.spec.logic,
            "hold": {
                "required_steps": self.spec.hold_steps,
                "hold_time_sec": self.spec.hold_time_sec,
            },
            "groups": groups,
        }
        self._write(payload)
        if self.stdout_enabled:
            print(
                f"[EvalCriteria] task={self.task_id} episode={episode_id} "
                f"goal_all={groups['goal_all']} goal_any={groups['goal_any']} "
                f"sequence={groups['sequence']} terminal={groups['terminal']} "
                f"stability={groups['stability']} safety={groups['safety']} "
                f"hold={self.spec.hold_steps} steps/{self.spec.hold_time_sec}s",
                flush=True,
            )

    @staticmethod
    def _predicate_signature(payload: dict[str, Any]) -> tuple[tuple[str, bool, bool], ...]:
        return tuple(
            (str(row["key"]), bool(row["passed"]), bool(row["triggered"]))
            for row in payload["criteria"]
        )

    def observe(
        self,
        result: SuccessEvaluation,
        *,
        action_step: int,
        sim_step: int,
        policy_calls: int,
        force: bool = False,
        event_type: str | None = None,
        termination_reason: str | None = None,
    ) -> dict[str, Any]:
        payload = build_progress_payload(
            result,
            **self._episode,
            action_step=action_step,
            sim_step=sim_step,
            policy_calls=policy_calls,
        )
        signature = self._predicate_signature(payload)
        stage_changed = payload["stage_id"] != self._last_stage
        predicate_changed = self._last_predicates is not None and signature != self._last_predicates
        hold = int(payload["hold"]["current_steps"])
        hold_event = (hold == 1 and self._last_hold == 0) or (hold == 0 and self._last_hold > 0) or (
            hold >= int(payload["hold"]["required_steps"]) and self._last_hold < hold
        )
        periodic = sim_step - self._last_emit_step >= self.interval_steps
        should_emit = force or periodic or stage_changed or predicate_changed or hold_event
        if not should_emit:
            self._last_stage = payload["stage_id"]
            self._last_predicates = signature
            self._last_hold = hold
            return payload

        if event_type is None:
            if payload["task_success"]:
                event_type = "success"
            elif not payload["safety_valid"]:
                event_type = "safety_violation"
            elif stage_changed:
                event_type = "stage_change"
            elif predicate_changed:
                event_type = "predicate_change"
            elif hold_event:
                event_type = "hold_change"
            else:
                event_type = "periodic"
        payload["event_type"] = event_type
        if termination_reason is not None:
            payload["termination_reason"] = termination_reason
        self._write(payload)
        if self.stdout_enabled:
            self._print_progress(payload)
        self._last_emit_step = sim_step
        self._last_stage = payload["stage_id"]
        self._last_predicates = signature
        self._last_hold = hold
        return payload

    def runtime_failure(
        self,
        *,
        termination_reason: str,
        action_step: int,
        sim_step: int,
        policy_calls: int,
    ) -> None:
        payload = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "timestamp": datetime.now().astimezone().isoformat(),
            "event_type": "failure",
            **self._episode,
            "action_step": int(action_step),
            "sim_step": int(sim_step),
            "policy_calls": int(policy_calls),
            "stage_id": "runtime_failure",
            "stage_label": "runtime failure before a valid predicate snapshot",
            "termination_reason": termination_reason,
            "criteria": [],
        }
        self._write(payload)
        if self.stdout_enabled:
            print(
                f"[EvalProgress] event=failure task={self.task_id} "
                f"episode={self._episode.get('episode_id')} action_step={action_step} "
                f"sim_step={sim_step} stage=runtime_failure "
                f"reason={termination_reason}",
                flush=True,
            )

    def _print_progress(self, payload: dict[str, Any]) -> None:
        group_text = (
            f"goal={'PASS' if payload['goal_pass'] else 'FAIL'} "
            f"sequence={'PASS' if payload['sequence_pass'] else 'FAIL'} "
            f"terminal={'PASS' if payload['terminal_pass'] else 'FAIL'} "
            f"stability={'PASS' if payload['stability_pass'] else 'FAIL'} "
            f"safety={'PASS' if payload['safety_valid'] else 'FAIL'}"
        )
        print(
            f"[EvalProgress] event={payload['event_type']} task={payload['task_id']} "
            f"episode={payload['episode_id']} action_step={payload['action_step']} "
            f"sim_step={payload['sim_step']} stage={payload['stage_id']} {group_text} "
            f"hold={payload['hold']['current_steps']}/{payload['hold']['required_steps']}",
            flush=True,
        )
        active = [
            row
            for row in payload["criteria"]
            if not row["passed"]
            and (
                (row["group"] == "goal" and not payload["goal_pass"])
                or (row["group"] == "terminal" and not payload["terminal_pass"])
                or (row["group"] == "stability" and not payload["stability_pass"])
                or (row["group"] == "safety" and not payload["safety_valid"])
            )
        ]
        if payload["event_type"] in {"failure", "safety_violation"}:
            rows = active
        else:
            rows = [row for row in active if payload["stage_id"].endswith(str(row["key"]))] or active[:1]
        for row in rows:
            condition_text: list[str] = []
            for condition in row["conditions"]:
                remaining = condition.get("remaining")
                if remaining is None:
                    condition_text.append(
                        f"{condition['name']}={condition.get('actual')} "
                        f"expected={condition.get('threshold')} "
                        f"remaining={condition.get('remaining_reason', 'non_numeric')}"
                    )
                else:
                    unit = condition.get("unit", "")
                    condition_text.append(
                        f"{condition['name']}={condition.get('actual')}{unit} "
                        f"{condition.get('operator')} {condition.get('threshold')}{unit} "
                        f"remaining={remaining:.6g}{unit}"
                    )
            print(
                f"  [{row['group']}] {row['key']}={'PASS' if row['passed'] else 'FAIL'}; "
                + "; ".join(condition_text),
                flush=True,
            )

    def close(self) -> None:
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
            self._stream = None


__all__ = [
    "EvaluationProgressReporter",
    "PROGRESS_SCHEMA_VERSION",
    "active_stage",
    "build_progress_payload",
    "criterion_rows",
]
