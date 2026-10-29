"""Policy input boundaries, accessible checkpoints, and replay scoring."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Callable

from .core import (Contract, Observation, SLOTS, TUBES, State, advance_state,
                   evaluate_insertion, insertion_matches_commit, legal_actions)


ORACLES = ("none", "semantic", "decision", "geometry", "skill")
SCHEMA_VERSION = "labference/centrifuge-diagnostic-v1"


@dataclass(frozen=True)
class PolicyCapabilities:
    reports: frozenset[str]
    oracle_transports: frozenset[str]
    action_space: str = "symbolic"
    resettable: bool = True

    def __post_init__(self) -> None:
        if not self.reports <= {"semantic", "decision", "geometry", "execution"}:
            raise ValueError("unknown policy report")
        if not self.oracle_transports <= {"structured", "text"}:
            raise ValueError("unknown Oracle transport")
        if self.action_space not in {"symbolic", "array"}:
            raise ValueError("unknown action space")


@dataclass(frozen=True)
class VerifiedSkillExecutor:
    validation_id: str
    validated_operations: frozenset[str]
    execute: Callable[[dict[str, str]], dict[str, Any]]
    verify_subgoal: Callable[[dict[str, str]], bool]
    verified: bool = False

    def __post_init__(self) -> None:
        if (not self.verified or not self.validation_id.strip() or
                not self.validated_operations <= {"pick", "insert"} or
                not self.validated_operations or not callable(self.execute) or
                not callable(self.verify_subgoal)):
            raise ValueError("Skill executor needs physical validation and an independent verifier")


def evaluate_policy_checkpoint(
    contract: Contract,
    state: State,
    observation: dict[str, Any],
    visible_occupancy: dict[str, str],
    candidate_frames: dict[str, dict[str, Any]],
    policy: Callable[[dict[str, Any]], dict[str, Any]],
    capabilities: PolicyCapabilities,
    *,
    instance_id: str,
    profile: str = "none",
    transport: str = "structured",
    selected_action: dict[str, str] | None = None,
    reference_geometry: dict[str, Any] | None = None,
    physical_subgoal_confirmed: bool | None = None,
    reset_policy: Callable[[], None] | None = None,
    skill_executor: VerifiedSkillExecutor | None = None,
) -> dict[str, Any]:
    """Score one fixed checkpoint without guessing unreported model internals."""
    if profile != "none" and transport not in capabilities.oracle_transports:
        return {"instance_id": instance_id, "status": "unsupported", "reason": "oracle_transport"}
    if not capabilities.resettable or reset_policy is None:
        return {"instance_id": instance_id, "status": "unsupported", "reason": "reset"}
    if profile in {"geometry", "skill"} and selected_action is None:
        return {"instance_id": instance_id, "status": "unsupported", "reason": "selected_action_required"}
    if profile == "skill" and (not isinstance(skill_executor, VerifiedSkillExecutor) or
                               selected_action is None or
                               selected_action.get("type") not in skill_executor.validated_operations):
        return {"instance_id": instance_id, "status": "unsupported", "reason": "reference_executor_required"}
    context = oracle_context(profile, contract, state, visible_occupancy, candidate_frames, selected_action)
    try:
        reset_policy()
    except Exception as exc:
        return {"instance_id": instance_id, "status": "service_error",
                "error_type": type(exc).__name__, "message": str(exc), "phase": "reset"}
    request = (dict(observation) if profile == "none" else
               deliver_input(observation, context, transport, set(capabilities.oracle_transports)))
    check_policy_request(request)
    try:
        response = policy(request)
    except Exception as exc:
        return {"instance_id": instance_id, "status": "service_error",
                "error_type": type(exc).__name__, "message": str(exc)}
    if not isinstance(response, dict):
        response = {}
    reference_executor_success = None
    if profile == "skill":
        try:
            result = skill_executor.execute(dict(selected_action))
            if (not isinstance(result, dict) or
                    normalize_action(result.get("executed_action")) != normalize_action(selected_action)):
                return {"instance_id": instance_id, "status": "invalid_executor_result",
                        "reason": "executed_operation_differs_from_selected"}
            reference_executor_success = skill_executor.verify_subgoal(dict(selected_action))
            if type(reference_executor_success) is not bool:
                raise ValueError("physical subgoal verifier must return bool")
        except Exception as exc:
            return {"instance_id": instance_id, "status": "service_error",
                    "error_type": type(exc).__name__, "message": str(exc), "phase": "reference_executor"}
        physical_subgoal_confirmed = reference_executor_success
    record = {
        "instance_id": instance_id,
        "supported_reports": sorted(capabilities.reports),
        "expected_semantic": dict(sorted(visible_occupancy.items())),
        "semantic_report": response.get("semantic_report"),
        "legal_actions": legal_actions(contract, state),
        "committed_action": response.get("committed_action"),
        "reference_geometry": reference_geometry,
        "geometry_report": response.get("geometry_report"),
        "physical_subgoal_confirmed": physical_subgoal_confirmed,
    }
    return {"instance_id": instance_id, "status": "scored", "profile": profile,
            "reference_executor_success": reference_executor_success,
            "request": request, "response": response, "scores": score_checkpoint(record)}


def policy_observation(images: dict[str, Any], joints: list[float], rule: str,
                       visible_slots: list[str], camera_calibration: dict[str, Any] | None = None,
                       authorized_history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Construct an allowlisted input; caller never passes the evaluator state."""
    return {
        "images": images,
        "joint_positions": joints,
        "task_rule": rule,
        "visible_slot_ids": list(visible_slots),
        "camera_calibration": camera_calibration or {},
        "authorized_history": authorized_history or [],
    }


def oracle_context(
    profile: str,
    contract: Contract,
    state: State,
    visible_occupancy: dict[str, str],
    candidate_frames: dict[str, dict[str, Any]],
    selected_action: dict[str, str] | None = None,
) -> dict[str, Any]:
    if profile not in ORACLES:
        raise ValueError(f"unknown oracle profile: {profile}")
    if profile == "none":
        return {}
    payload: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "profile": profile}
    if profile == "semantic":
        payload["visible_occupancy"] = dict(sorted(visible_occupancy.items()))
        payload["visible_tube_identities"] = list(sorted(set(visible_occupancy.values())))
    elif profile == "decision":
        legal = legal_actions(contract, state)
        if not legal:
            raise ValueError("checkpoint has no legal operation")
        payload["advice"] = legal[0]
    elif profile == "geometry":
        payload["candidate_frames"] = {s: candidate_frames[s] for s in SLOTS if s in candidate_frames}
    elif profile == "skill":
        payload["service"] = "reference_executor"
        if selected_action is not None:
            payload["requested_operation"] = dict(selected_action)
    return payload


def deliver_input(observation: dict[str, Any], context: dict[str, Any], transport: str, supports: set[str]) -> dict[str, Any]:
    """Fail rather than silently changing an adapter's declared Oracle transport."""
    if transport not in {"structured", "text"} or transport not in supports:
        raise ValueError(f"unsupported Oracle transport: {transport}")
    result = dict(observation)
    if "oracle_context" in result or "oracle_prompt" in result:
        raise ValueError("Oracle already delivered")
    if not context:
        return result
    if transport == "structured":
        result["oracle_context"] = context
    else:
        result["oracle_prompt"] = "<ORACLE>" + json.dumps(context, sort_keys=True, separators=(",", ":")) + "</ORACLE>"
    return result


def check_policy_request(request: dict[str, Any], allow_fixed_subgoal: bool = False) -> None:
    if not isinstance(request, dict):
        raise ValueError("policy request must be an object")
    allowed = {"images", "joint_positions", "task_rule", "visible_slot_ids",
               "camera_calibration", "authorized_history", "oracle_context", "oracle_prompt"}
    if allow_fixed_subgoal:
        allowed |= {"selected_operation", "interaction_pose"}
    unknown = set(request) - allowed
    if unknown:
        raise ValueError(f"unexpected policy input fields: {sorted(unknown)}")
    banned = {"evaluator_state", "success", "correct_slot", "target_world_pose", "legal_actions", "answer"}
    if "oracle_context" in request and "oracle_prompt" in request:
        raise ValueError("Oracle delivered twice")
    if allow_fixed_subgoal:
        selected = normalize_action(request.get("selected_operation"))
        pose = request.get("interaction_pose")
        if selected is None or selected.get("type") != "insert" or not isinstance(pose, dict):
            raise ValueError("fixed-subgoal input requires a selected insertion and pose")
        validated_geometry_report(dict(pose, slot=selected["slot"]), selected["slot"])
    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in banned:
                    raise ValueError(f"evaluator field in policy input: {key}")
                inspect(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                inspect(nested)
        elif isinstance(value, str) and re.search(
                r"\b(?:evaluator_state|correct_slot|target_world_pose|legal_actions|"
                r"first_actual_slot|success_label|ground_truth)\b['\"]?\s*[:=]", value, re.IGNORECASE):
            raise ValueError("evaluator label in policy text")
    inspect({k: v for k, v in request.items() if k not in {"oracle_context", "oracle_prompt"}})
    if "oracle_prompt" in request:
        prompt = request["oracle_prompt"]
        if not isinstance(prompt, str) or not prompt.startswith("<ORACLE>") or not prompt.endswith("</ORACLE>"):
            raise ValueError("invalid Oracle text transport")
        try:
            context = json.loads(prompt[len("<ORACLE>"):-len("</ORACLE>")])
        except json.JSONDecodeError as exc:
            raise ValueError("invalid Oracle JSON text") from exc
    else:
        context = request.get("oracle_context")
    if context is not None:
        if not isinstance(context, dict) or context.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("invalid Oracle context schema")
        profile = context.get("profile")
        fields = {
            "semantic": {"visible_occupancy", "visible_tube_identities"},
            "decision": {"advice"},
            "geometry": {"candidate_frames"},
            "skill": {"service", "requested_operation"},
        }
        if profile not in fields or set(context) - {"schema_version", "profile"} - fields[profile]:
            raise ValueError("Oracle context has unexpected fields")
        if profile == "semantic":
            occupancy = context.get("visible_occupancy")
            identities = context.get("visible_tube_identities")
            if (not isinstance(occupancy, dict) or
                    any(slot not in SLOTS or tube not in TUBES for slot, tube in occupancy.items()) or
                    identities != sorted(set(occupancy.values()))):
                raise ValueError("invalid semantic Oracle facts")
        elif profile == "decision" and normalize_action(context.get("advice")) is None:
            raise ValueError("invalid decision Oracle advice")
        elif profile == "geometry":
            frames = context.get("candidate_frames")
            if not isinstance(frames, dict) or not frames or any(slot not in SLOTS for slot in frames):
                raise ValueError("invalid geometry Oracle candidates")
            for slot, candidate in frames.items():
                if not isinstance(candidate, dict) or candidate.get("slot", slot) != slot:
                    raise ValueError("invalid geometry Oracle frame")
                validated_geometry_report(dict(candidate, slot=slot), slot)
        elif profile == "skill" and (context.get("service") != "reference_executor" or
                                     ("requested_operation" in context and
                                      normalize_action(context["requested_operation"]) is None)):
            raise ValueError("invalid skill Oracle request")
        inspect(context)


def normalize_action(action: dict[str, Any] | None) -> dict[str, str] | None:
    if action is None:
        return None
    if not isinstance(action, dict) or action.get("type") not in {"pick", "insert", "wait", "stop"}:
        return None
    kind = action["type"]
    required = {"pick": {"type", "tube"}, "insert": {"type", "tube", "slot"},
                "wait": {"type"}, "stop": {"type"}}[kind]
    if set(action) != required:
        return None
    return {k: str(action[k]) for k in required}


def validated_geometry_report(report: Any, committed_slot: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not isinstance(report, dict) or report.get("slot") != committed_slot:
        raise ValueError("geometry slot differs from committed slot")
    if report.get("frame", "world") != "world" or report.get("units", "m") != "m":
        raise ValueError("geometry report must use world-frame meters")
    try:
        mouth = tuple(float(value) for value in report["mouth_world_m"])
        axis = tuple(float(value) for value in report["axis_world"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("geometry report needs mouth_world_m and axis_world") from exc
    if (len(mouth) != 3 or len(axis) != 3 or
            not all(math.isfinite(value) for value in (*mouth, *axis)) or
            not .5 <= math.hypot(*axis) < math.inf):
        raise ValueError("geometry report has invalid world coordinates")
    return mouth, axis


def score_checkpoint(record: dict[str, Any]) -> dict[str, Any]:
    """All statuses are scoped to the outputs the declared policy exposes."""
    supported = set(record.get("supported_reports", []))
    result: dict[str, Any] = {}
    if "semantic" in supported:
        if record.get("semantic_reference_available") is False:
            result["semantic"] = "unavailable"
        else:
            expected = record.get("expected_semantic")
            report = record.get("semantic_report")
            result["semantic"] = "pass" if isinstance(report, dict) and isinstance(expected, dict) and report == expected else "fail"
    else:
        result["semantic"] = "unsupported"
    if "decision" in supported:
        action = normalize_action(record.get("committed_action"))
        legal = [normalize_action(x) for x in record.get("legal_actions", [])]
        result["decision"] = "pass" if action is not None and action in legal else "fail"
        result["decision_type"] = "pass" if action and any(a and a["type"] == action["type"] for a in legal) else "fail"
    else:
        result["decision"] = "unsupported"
    if "geometry" in supported:
        report = record.get("geometry_report")
        ref = record.get("reference_geometry")
        if record.get("geometry_applicable") is False:
            result["geometry"] = "not_applicable"
        elif record.get("geometry_reference_available") is False:
            result["geometry"] = "unavailable"
        elif not isinstance(report, dict) or not isinstance(ref, dict) or report.get("slot") != ref.get("slot"):
            result["geometry"] = "fail"
        else:
            try:
                mouth, axis = validated_geometry_report(report, ref["slot"])
                reference_mouth, reference_axis = validated_geometry_report(ref, ref["slot"])
                point_error = math.dist(mouth, reference_mouth)
                axis_norm = math.hypot(*axis)
                reference_norm = math.hypot(*reference_axis)
                dot = sum((x / axis_norm) * (y / reference_norm)
                          for x, y in zip(axis, reference_axis))
                angle = math.degrees(math.acos(max(-1, min(1, dot))))
                if not math.isfinite(point_error) or not math.isfinite(angle):
                    raise ValueError("non-finite geometry error")
                result["geometry"] = "pass" if point_error <= record.get("point_tolerance_m", .01) and angle <= record.get("axis_tolerance_deg", 10) else "fail"
                result["point_error_m"] = point_error
                result["axis_error_deg"] = angle
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                result["geometry"] = "fail"
    else:
        result["geometry"] = "unsupported"
    if "execution" in supported:
        result["execution"] = "pass" if record.get("physical_subgoal_confirmed") is True else "fail"
    else:
        result["execution"] = "unsupported"
    result["instance_id"] = record["instance_id"]
    return result


def summarize_checkpoint_scores(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep unsupported and unavailable interfaces out of accuracy denominators."""
    summary = {}
    for layer in ("semantic", "decision", "geometry", "execution"):
        counts = {status: 0 for status in ("pass", "fail", "unsupported", "not_applicable", "unavailable")}
        for checkpoint in checkpoints:
            status = checkpoint.get(layer, "unsupported")
            if status not in counts:
                raise ValueError(f"invalid {layer} checkpoint status: {status}")
            counts[status] += 1
        denominator = counts["pass"] + counts["fail"]
        summary[layer] = {**counts, "denominator": denominator,
                          "accuracy": counts["pass"] / denominator if denominator else None}
    return summary


def score_trace(contract: Contract, trace: list[dict[str, Any]],
                declared_reports: set[str] | None = None) -> dict[str, Any]:
    """Replay measured occupancy before scoring decisions and terminal stability."""
    decisions = []
    physical_events = []
    checkpoints = []
    decision_errors: dict[str, int] = {}
    terminated_correctly = False
    attempts = {kind: {"completed": 0, "succeeded": 0, "failed": 0,
                       "pending": 0, "pending_observed_success": 0}
                for kind in ("pick", "insert")}
    active_attempt: dict[str, Any] | None = None

    def finish_attempt() -> None:
        nonlocal active_attempt
        if active_attempt is not None:
            counts = attempts[active_attempt["type"]]
            counts["completed"] += 1
            counts["succeeded" if active_attempt["observed_success"] else "failed"] += 1
            active_attempt = None

    def match_physical_event(kind: str, tube: str, slot: str | None = None) -> None:
        if (active_attempt is not None and active_attempt["type"] == kind and
                active_attempt["tube"] == tube and
                (kind != "insert" or active_attempt["slot"] == slot)):
            active_attempt["observed_success"] = True
    occupancy: dict[str, str] = {}
    violations = []
    measured = any(entry.get("type") == "measurement" for entry in trace)
    state = State(contract.difficulty) if measured else None
    frame = next((entry for entry in trace if entry.get("type") == "geometry_frame"), None)
    held_replay = bool(frame) and all(
        isinstance(entry.get("step"), int) and isinstance(entry.get("tubes"), dict) and all(
            isinstance(entry["tubes"].get(tube), dict) and
            {"gripper_contact", "position_world_m"} <= entry["tubes"][tube].keys()
            for tube in TUBES)
        for entry in trace if entry.get("type") == "measurement")
    raw_verified = measured and frame is not None and held_replay
    raw_measurement_errors = 0
    initial_heights: dict[str, float] = {}
    had_failure = False
    inserted_after_failure = False
    for entry in trace:
        event = entry.get("type")
        if event == "measurement" and state is not None:
            evidence = entry.get("tubes")
            if not isinstance(evidence, dict):
                raw_verified = False
                raw_measurement_errors += 1
                evidence = {}
            if frame is not None:
                derived = {}
                for tube in TUBES:
                    raw = evidence.get(tube, {})
                    required = {"position_world_m", "axis_world", "linear_speed_m_s",
                                "released", "supported", "position_delta_m"}
                    if not isinstance(raw, dict) or not required <= raw.keys():
                        raw_verified = False
                        raw_measurement_errors += 1
                        derived[tube] = {"actual_slot": None, "stable_sample": False}
                        continue
                    try:
                        observation = Observation(
                            tube=tube, tip_world=tuple(raw["position_world_m"]),
                            axis_world=tuple(raw["axis_world"]),
                            linear_speed_m_s=float(raw["linear_speed_m_s"]),
                            gripper_open=bool(raw["released"]), supported=bool(raw["supported"]),
                            position_delta_m=raw["position_delta_m"])
                        derived[tube] = evaluate_insertion(
                            contract, observation, frame["slot_mouths_world_m"],
                            frame["insertion_axis_world"])
                    except (KeyError, TypeError, ValueError, OverflowError, IndexError):
                        raw_verified = False
                        raw_measurement_errors += 1
                        derived[tube] = {"actual_slot": None, "stable_sample": False}
                evidence = derived
            if held_replay:
                step = entry["step"]
                for tube in TUBES:
                    raw = entry["tubes"][tube]
                    try:
                        height = float(raw["position_world_m"][2])
                        if not math.isfinite(height):
                            raise ValueError("non-finite tube height")
                    except (TypeError, ValueError, IndexError):
                        raw_verified = False
                        raw_measurement_errors += 1
                        continue
                    if step == 30:
                        initial_heights[tube] = height
                    if step <= 30 or tube not in initial_heights:
                        continue
                    if (state.tube_status[tube] == "table" and
                            height - initial_heights[tube] >= .05 and raw["gripper_contact"]):
                        state.tube_status[tube] = "held"
                        physical_events.append({"event": "lifted", "tube": tube, "step": step})
                        match_physical_event("pick", tube)
                    elif (state.tube_status[tube] == "held" and
                            height <= initial_heights[tube] + .025 and not raw["gripper_contact"]):
                        state.tube_status[tube] = "table"
                        had_failure = True
                        physical_events.append({"event": "grasp_lost", "tube": tube, "step": step})
            for change in advance_state(contract, state, evidence):
                tube = change["tube"]
                slot = change["slot"]
                selected = state.selected_slot.get(tube)
                if change["type"] == "inserted" and had_failure:
                    inserted_after_failure = True
                if change["type"] == "insertion_lost" or (
                        change["type"] == "inserted" and selected is not None and selected != slot):
                    had_failure = True
                physical_events.append({"event": change["type"], "tube": tube,
                                        "slot": slot, "execution_deviation": selected is not None and selected != slot,
                                        "precision_pass": change["type"] == "inserted" and
                                        insertion_matches_commit(contract, tube, selected, slot) and
                                        (tube == "NMR_01" or (state.first_actual_slot in SLOTS and
                                         slot == contract.opposite(state.first_actual_slot)))})
                if change["type"] == "inserted":
                    match_physical_event("insert", tube, slot)
        elif event == "decision":
            finish_attempt()
            chosen = normalize_action(entry.get("action"))
            legal = [normalize_action(x) for x in
                     (legal_actions(contract, state) if state is not None else entry.get("legal_actions", []))]
            decision_ok = chosen is not None and chosen in legal
            decisions.append(decision_ok)
            if chosen is not None and chosen["type"] == "stop" and decision_ok:
                terminated_correctly = True
            if not decision_ok:
                had_failure = True
                if chosen is None:
                    reason = "invalid_action"
                elif chosen["type"] == "stop":
                    reason = "premature_stop"
                elif chosen["type"] == "wait":
                    reason = "redundant_wait"
                elif chosen["type"] == "insert" and chosen["tube"] in TUBES and (
                        state is not None and state.tube_status[chosen["tube"]] == "inserted"):
                    reason = "redundant_action"
                elif chosen["type"] == "insert" and any(
                        action and action["type"] == "insert" and
                        action.get("tube") == chosen.get("tube") for action in legal):
                    reason = "wrong_target"
                elif chosen["type"] == "pick" and chosen["tube"] in TUBES and (
                        state is not None and state.tube_status[chosen["tube"]] != "table"):
                    reason = "redundant_action"
                else:
                    reason = "premature_advance"
                decision_errors[reason] = decision_errors.get(reason, 0) + 1
            response = entry.get("policy_response") or {}
            if not isinstance(response, dict):
                response = {}
            listed_reports = (declared_reports if declared_reports is not None else
                              entry.get("supported_reports", ["decision"]))
            supported = set(listed_reports if isinstance(listed_reports, (set, list, tuple)) else []) - {"execution"}
            supported.add("decision")
            geometry_applicable = chosen is not None and chosen["type"] == "insert"
            geometry_ref = (dict(slot=chosen["slot"],
                                 mouth_world_m=frame["slot_mouths_world_m"][chosen["slot"]],
                                 axis_world=frame["insertion_axis_world"])
                            if geometry_applicable and frame is not None and
                            chosen["slot"] in frame.get("slot_mouths_world_m", {}) and
                            "insertion_axis_world" in frame else None)
            checkpoints.append(score_checkpoint({
                "instance_id": f"decision-{entry.get('step', len(decisions))}",
                "supported_reports": sorted(supported),
                "expected_semantic": dict(state.occupied) if state is not None else None,
                "semantic_reference_available": state is not None,
                "semantic_report": response.get("semantic_report"),
                "legal_actions": [action for action in legal if action is not None],
                "committed_action": chosen,
                "geometry_applicable": geometry_applicable,
                "geometry_reference_available": geometry_ref is not None,
                "reference_geometry": geometry_ref,
                "geometry_report": response.get("geometry_report"),
            }))
            if state is not None and chosen is not None and chosen["type"] == "insert":
                if chosen.get("tube") in TUBES:
                    state.selected_slot[chosen["tube"]] = chosen["slot"]
            if chosen is not None and chosen["type"] in attempts:
                active_attempt = {"type": chosen["type"], "tube": chosen["tube"],
                                  "slot": chosen.get("slot"), "observed_success": False}
        elif event == "physical":
            if state is None:
                if entry.get("event") == "inserted" and had_failure:
                    inserted_after_failure = True
                if entry.get("event") in {"grasp_lost", "insertion_lost"} or entry.get("execution_deviation"):
                    had_failure = True
            elif entry.get("event") == "grasp_lost" and not held_replay:
                had_failure = True
            if state is None or (entry.get("event") not in {"inserted", "insertion_lost"} and
                                 not (held_replay and entry.get("event") in {"lifted", "grasp_lost"})):
                physical_events.append(entry)
            if state is not None:
                if entry.get("event") == "lifted" and not held_replay:
                    state.tube_status[entry["tube"]] = "held"
                elif entry.get("event") == "grasp_lost" and not held_replay:
                    state.tube_status[entry["tube"]] = "table"
            elif entry.get("event") == "inserted":
                occupancy[entry["tube"]] = entry["slot"]
            elif entry.get("event") in {"insertion_lost", "removed"}:
                occupancy.pop(entry.get("tube"), None)
            if state is None:
                if entry.get("event") == "lifted":
                    match_physical_event("pick", entry.get("tube"))
                elif entry.get("event") == "inserted":
                    match_physical_event("insert", entry.get("tube"), entry.get("slot"))
            if entry.get("event") == "grasp_attempt_failed":
                finish_attempt()
        elif event == "violation":
            violations.append(entry)
            had_failure = True
            if entry.get("kind") in {"operation_timeout", "invalid_geometry_report",
                                      "runtime_invalid_joint_target", "invalid_policy_action"}:
                finish_attempt()
        elif event == "snapshot":
            finish_attempt()
    if state is not None:
        occupancy = {tube: slot for slot, tube in state.occupied.items()}
    first = occupancy.get("NMR_01")
    complete = first in SLOTS and occupancy.get("NMR_02") == contract.opposite(first)
    latest = trace[-1] if trace else {}
    terminal_steps = state.terminal_steps if state is not None else latest.get("terminal_steps", 0)
    stable = terminal_steps >= contract.final_stable_steps
    decision_pass = bool(decisions) and all(decisions)
    termination_pass = terminated_correctly if contract.task_version == "centrifuge-opposite/v2" else True
    inserted_order = [x.get("tube") for x in physical_events if x.get("event") == "inserted"]
    out_of_order = (contract.difficulty == "hard" and "NMR_02" in inserted_order and
                    ("NMR_01" not in inserted_order or inserted_order.index("NMR_02") < inserted_order.index("NMR_01")))
    protocol = (not out_of_order and all(decisions) and
                not any(x.get("kind") in {"premature_advance", "premature_stop", "wrong_target",
                                          "redundant_action"} for x in violations))
    precision = complete and all(x.get("precision_pass") is True for x in physical_events if x.get("event") == "inserted")
    lifted_tubes = {x.get("tube") for x in physical_events if x.get("event") == "lifted"}
    required_lifts = {"NMR_02"} if contract.difficulty == "easy" else set(TUBES)
    grasp_lift_pass = required_lifts <= lifted_tubes
    if active_attempt is not None:
        attempts[active_attempt["type"]]["pending"] = 1
        attempts[active_attempt["type"]]["pending_observed_success"] = int(
            active_attempt["observed_success"])
    completed_attempts = sum(counts["completed"] for counts in attempts.values())
    succeeded_attempts = sum(counts["succeeded"] for counts in attempts.values())
    safety = not any(x.get("kind") in {"safety", "runtime_invalid_joint_state", "runtime_invalid_joint_target", "non_finite_physics_state"} for x in violations)
    return {
        "task_version": contract.task_version,
        "contract_hash": contract.hash,
        "completion_success": bool(complete and stable),
        "decision_pass": decision_pass,
        "decision_count": len(decisions),
        "decision_correct_count": sum(decisions),
        "illegal_decision_count": len(decisions) - sum(decisions),
        "decision_error_counts": decision_errors,
        "termination_pass": termination_pass,
        "module_scores": summarize_checkpoint_scores(checkpoints),
        "checkpoint_count": len(checkpoints),
        "operation_attempts": attempts,
        "selected_operation_execution_rate": (succeeded_attempts / completed_attempts
                                              if completed_attempts else None),
        "protocol_pass": protocol,
        "precision_pass": precision,
        "grasp_lift_pass": grasp_lift_pass,
        "safety_pass": safety,
        "valid_success": bool(complete and stable and decision_pass and termination_pass and protocol and precision and safety
                              and (contract.task_version != "centrifuge-opposite/v2" or
                                   (raw_verified and grasp_lift_pass))),
        "recovery_completion": bool(complete and stable and safety and inserted_after_failure),
        "terminal_steps": terminal_steps,
        "replay_source": "measurements" if measured else "legacy_events",
        "physical_replay_verified": bool(raw_verified),
        # A verified rollout can validly fail the task. Do not conflate
        # evaluation validity with insertion/decision/completion success.
        "evaluation_valid": bool(raw_verified and raw_measurement_errors == 0),
        "raw_measurement_error_count": raw_measurement_errors,
        "execution_deviation_count": sum(bool(x.get("execution_deviation")) for x in physical_events),
        "grasp_lost_count": sum(x.get("event") == "grasp_lost" for x in physical_events),
        "grasp_attempt_failed_count": sum(x.get("event") == "grasp_attempt_failed"
                                          for x in physical_events),
        "operation_timeout_count": sum(x.get("kind") == "operation_timeout" for x in violations),
        "violation_count": len(violations),
    }


class OnlineEvaluationMonitor:
    """Publish cumulative evaluation while an episode is still running."""

    def __init__(self, contract: Contract, trace: list[dict[str, Any]],
                 stream: Any, declared_reports: set[str], event_stream: Any | None = None,
                 evaluation_version: str | None = None) -> None:
        self.contract = contract
        self.trace = trace
        self.stream = stream
        self.declared_reports = set(declared_reports)
        self.event_stream = event_stream
        self.evaluation_version = evaluation_version
        self.sample_count = 0

    def snapshot(self, step: int, trigger: str) -> dict[str, Any]:
        if self.event_stream is not None:
            self.event_stream.flush()
        if self.evaluation_version == "labference/centrifuge-online-v2":
            from .online_v2 import score_online_v2
            scores = score_online_v2(self.contract, self.trace, self.declared_reports)
        else:
            scores = score_trace(self.contract, self.trace, self.declared_reports)
        record = {"step": step, "trigger": trigger, "event_count": len(self.trace), "scores": scores}
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()
        self.sample_count += 1
        return scores


def verify_online_scores(contract: Contract, trace: list[dict[str, Any]],
                         snapshots: list[dict[str, Any]],
                         declared_reports: set[str], evaluation_version: str | None = None) -> dict[str, Any]:
    """Recompute saved online prefixes; offline replay is an audit, not the primary result."""
    mismatches = []
    previous = -1
    for index, row in enumerate(snapshots):
        count = row.get("event_count")
        if not isinstance(count, int) or not previous <= count <= len(trace):
            mismatches.append({"sample": index, "reason": "invalid_event_count"})
            continue
        if evaluation_version == "labference/centrifuge-online-v2":
            from .online_v2 import score_online_v2
            expected = score_online_v2(contract, trace[:count], declared_reports)
        else:
            expected = score_trace(contract, trace[:count], declared_reports)
        if row.get("scores") != expected:
            mismatches.append({"sample": index, "reason": "score_mismatch"})
        if count and row.get("step") != trace[count - 1].get("step"):
            mismatches.append({"sample": index, "reason": "step_mismatch"})
        previous = count
    final_present = bool(snapshots) and snapshots[-1].get("event_count") == len(trace)
    return {"sample_count": len(snapshots), "matching": not mismatches and final_present,
            "final_snapshot_present": final_present, "mismatches": mismatches}


def compare_pairs(records: list[dict[str, Any]], condition: str,
                  registered_pairs: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    if condition not in ORACLES[1:]:
        raise ValueError(f"invalid comparison condition: {condition}")
    grouped: dict[str, dict[str, dict[str, Any]]] = {
        pair_id: {} for pair_id in registered_pairs} if registered_pairs is not None else {}
    for item in records:
        pair = item["pair_id"]
        key = item["condition"]
        if key not in {"none", condition}:
            raise ValueError(f"unexpected condition in comparison: {key}")
        if registered_pairs is not None:
            if pair not in registered_pairs:
                raise ValueError(f"unregistered pair: {pair}")
            if item.get("policy_uses_evaluator_state") is not False or not item.get("policy_id"):
                raise ValueError("registered Oracle comparisons require a named non-GT policy")
            for field, expected in registered_pairs[pair].items():
                if item.get(field) != expected:
                    raise ValueError(f"pair {pair} disagrees on {field}")
        if key in grouped.setdefault(pair, {}):
            raise ValueError(f"duplicate pair/condition: {pair}/{key}")
        if item.get("policy_uses_evaluator_state") is True:
            raise ValueError("GT rule-expert runs cannot measure Oracle benefit")
        if item.get("status", "scored") == "scored" and not isinstance(item.get("success"), bool):
            raise ValueError(f"pair {pair}/{key} needs a boolean success result")
        accuracy = item.get("decision_accuracy")
        if accuracy is not None and (not isinstance(accuracy, (int, float)) or
                                     not math.isfinite(accuracy) or not 0 <= accuracy <= 1):
            raise ValueError(f"pair {pair}/{key} has invalid decision accuracy")
        grouped[pair][key] = item
    total = len(grouped)
    paired = [block for block in grouped.values() if "none" in block and condition in block]
    scored = []
    for block in paired:
        base, assisted = block["none"], block[condition]
        for key in ("task_version", "contract_hash", "scene_seed", "decision_seed",
                    "first_actual_slot", "initial_slot", "max_steps_budget", "variant", "policy_id", "policy_seed",
                    "sensor_condition", "operation_seed", "evaluation_version", "policy_track",
                    "policy_revision", "policy_reset_scope"):
            if (key in base) != (key in assisted) or (key in base and base[key] != assisted[key]):
                raise ValueError(f"paired runs disagree on {key}")
        if (base.get("status", "scored") == "scored" and
                assisted.get("status", "scored") == "scored" and
                "success" in base and "success" in assisted):
            scored.append(block)
    gains = [int(block[condition]["success"]) - int(block["none"]["success"]) for block in scored]
    complete = bool(total) and len(scored) == total
    decision_gains = [block[condition]["decision_accuracy"] - block["none"]["decision_accuracy"]
                      for block in scored if isinstance(block[condition].get("decision_accuracy"), (int, float))
                      and isinstance(block["none"].get("decision_accuracy"), (int, float))]
    behavior_gains = [block[condition]["online_legal_next_action_rate"] -
                      block["none"]["online_legal_next_action_rate"] for block in scored
                      if isinstance(block[condition].get("online_legal_next_action_rate"), (int, float))
                      and isinstance(block["none"].get("online_legal_next_action_rate"), (int, float))]
    return {
        "condition": condition,
        "registered_pairs": total,
        "complete_pairs": len(paired),
        "missing_pairs": total - len(paired),
        "scored_pairs": len(scored),
        "missing_conditions": {"none": sum("none" not in block for block in grouped.values()),
                               condition: sum(condition not in block for block in grouped.values())},
        "unsupported_pairs": sum(any(x.get("status") == "unsupported" for x in block.values())
                                 for block in grouped.values()),
        "service_error_pairs": sum(any(x.get("status") == "service_error" for x in block.values())
                                   for block in grouped.values()),
        "runtime_error_pairs": sum(any(x.get("status") == "runtime_error" for x in block.values())
                                   for block in grouped.values()),
        "invalid_evaluation_pairs": sum(any(x.get("status") == "invalid_evaluation" for x in block.values())
                                        for block in grouped.values()),
        "coverage": len(paired) / total if total else 0.0,
        "scored_coverage": len(scored) / total if total else 0.0,
        "paired_absolute_gain": sum(gains) / total if complete else None,
        "paired_rollout_decision_accuracy_gain": (sum(decision_gains) / total
                                                  if complete and len(decision_gains) == total else None),
        "paired_on_policy_behavior_rate_gain": (sum(behavior_gains) / total
                                                 if complete and len(behavior_gains) == total else None),
        "diagnostic_ceiling": condition == "skill",
        "ranking_eligible": False,
        "status": "complete" if complete else "incomplete",
    }


def compare_state_pairs(records: list[dict[str, Any]],
                        plan: list[dict[str, Any]]) -> dict[str, Any]:
    """Score whether a policy changes its next operation with measured state."""
    registered: dict[str, dict[str, dict[str, Any]]] = {}
    for item in plan:
        pair_id, variant = item["pair_id"], item["variant"]
        if variant not in {"a", "b"} or variant in registered.setdefault(pair_id, {}):
            raise ValueError(f"invalid state pair plan: {pair_id}/{variant}")
        registered[pair_id][variant] = item
    if any(set(variants) != {"a", "b"} for variants in registered.values()):
        raise ValueError("each registered state pair needs variants a and b")
    observed: dict[str, dict[str, dict[str, Any]]] = {pair_id: {} for pair_id in registered}
    for item in records:
        pair_id, variant = item["pair_id"], item["variant"]
        if pair_id not in registered or variant not in registered[pair_id]:
            raise ValueError(f"unregistered state pair: {pair_id}/{variant}")
        if variant in observed[pair_id]:
            raise ValueError(f"duplicate state pair response: {pair_id}/{variant}")
        expected = registered[pair_id][variant]
        for field in ("task_version", "contract_hash", "scene_seed", "decision_seed",
                      "first_actual_slot"):
            if item.get(field) != expected[field]:
                raise ValueError(f"state pair {pair_id}/{variant} disagrees on {field}")
        if item.get("policy_uses_evaluator_state") is not False or not item.get("policy_id"):
            raise ValueError("state pairs require a named non-GT policy")
        observed[pair_id][variant] = item
    complete_pairs = 0
    scored_pairs = 0
    both_correct = 0
    changed_responses = 0
    for pair_id, variants in observed.items():
        if set(variants) != {"a", "b"}:
            continue
        complete_pairs += 1
        if any(variants["a"].get(key) != variants["b"].get(key)
               for key in ("policy_id", "policy_seed", "sensor_condition", "policy_track",
                           "evaluation_version", "decision_window_steps", "policy_revision",
                           "policy_reset_scope")):
            raise ValueError(f"state pair {pair_id} mixes policy or sensor conditions")
        if any(variants[key].get("status", "scored") != "scored" for key in ("a", "b")):
            continue
        scored_pairs += 1
        actions = {key: normalize_action(variants[key].get("committed_action")) for key in ("a", "b")}
        expected = {key: [normalize_action(action) for action in registered[pair_id][key]["legal_actions"]]
                    for key in ("a", "b")}
        if actions["a"] is not None and actions["b"] is not None and actions["a"] != actions["b"]:
            changed_responses += 1
        if all(actions[key] is not None and actions[key] in expected[key] for key in ("a", "b")):
            both_correct += 1
    total = len(registered)
    full = bool(total) and scored_pairs == total
    return {
        "registered_pairs": total,
        "complete_pairs": complete_pairs,
        "scored_pairs": scored_pairs,
        "missing_pairs": total - complete_pairs,
        "both_correct_pairs": both_correct,
        "changed_response_pairs": changed_responses,
        "counterfactual_consistency_score": both_correct / total if full else None,
        "decision_switch_rate": changed_responses / total if full else None,
        "status": "complete" if full else "incomplete",
    }


def plan_state_pairs(contract: Contract, seeds: list[int], online: bool = False) -> list[dict[str, Any]]:
    """Pre-register symbolic counterfactual checkpoints; no rollout is implied."""
    records = []
    for seed in seeds:
        slots = (SLOTS[seed % len(SLOTS)], SLOTS[(seed + 1) % len(SLOTS)])
        for variant, slot in zip(("a", "b"), slots):
            state = State(contract.difficulty, first_actual_slot=slot)
            state.tube_status["NMR_01"] = "inserted"
            state.tube_status["NMR_02"] = "held"
            state.occupied[slot] = "NMR_01"
            records.append({
                "pair_id": f"{contract.difficulty}-state-{seed:04d}",
                "oracle_pair_id": f"{contract.difficulty}-state-{seed:04d}-{variant}",
                "variant": variant,
                "scene_seed": seed,
                "decision_seed": seed,
                "state_source": ("verified_preload_and_reference_pick_plan" if online else
                                 "synthetic_checkpoint_plan"),
                "changed_variable": "first_tube_actual_slot",
                "first_actual_slot": slot,
                "legal_actions": legal_actions(contract, state),
                "contract_hash": contract.hash,
                "task_version": contract.task_version,
            })
    return records


def plan_online_rollouts(contract: Contract, seeds: list[int],
                         evaluation_version: str | None = None,
                         policy_track: str | None = None,
                         max_steps_budget: int | None = None) -> list[dict[str, Any]]:
    """Register complete Isaac episodes before baseline and Oracle rollouts."""
    if len(set(seeds)) != len(seeds):
        raise ValueError("rollout seeds must be unique")
    if max_steps_budget is not None and not 0 < max_steps_budget <= contract.max_steps:
        raise ValueError("rollout step budget is outside the task contract")
    import random

    return [{"pair_id": f"{contract.difficulty}-online-{seed:04d}",
             "task_version": contract.task_version, "contract_hash": contract.hash,
             "scene_seed": seed, "decision_seed": seed,
             "max_steps_budget": max_steps_budget or contract.max_steps,
             "initial_slot": (SLOTS[random.Random(f"decision:{seed}").randrange(len(SLOTS))]
                              if evaluation_version == "labference/centrifuge-online-v2" and
                              contract.difficulty == "easy" else
                              SLOTS[seed % len(SLOTS)] if contract.difficulty == "easy" else None),
             **({"evaluation_version": evaluation_version, "policy_track": policy_track}
                if evaluation_version is not None else {})}
            for seed in seeds]
