"""Event-defined online measurements for native and reported centrifuge policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .core import Contract, Observation, SLOTS, TUBES, State, advance_state, evaluate_insertion, legal_actions
from .diagnostics import normalize_action, score_checkpoint, score_trace, summarize_checkpoint_scores


EVALUATION_VERSION = "labference/centrifuge-online-v2"


def state_signature(contract: Contract, state: State) -> tuple[Any, ...]:
    """Ignore elapsed frames until they change a legal operation."""
    return (state.first_actual_slot, tuple(sorted(state.occupied.items())),
            tuple((tube, state.tube_status[tube]) for tube in TUBES),
            state.terminal_steps >= contract.final_stable_steps)


@dataclass
class OpportunityBook:
    contract: Contract
    interface: str
    current_id: str | None = None
    current_signature: tuple[Any, ...] | None = None
    next_number: int = 0
    first_choice_recorded: bool = False
    last_choice: dict[str, str] | None = None

    def update(self, state: State, step: int, *, evidence_step: int,
               trigger: str = "state_changed") -> dict[str, Any] | None:
        signature = state_signature(self.contract, state)
        if signature == self.current_signature:
            return None
        self.current_signature = signature
        self.current_id = f"opportunity-{self.next_number:04d}"
        self.next_number += 1
        self.first_choice_recorded = False
        self.last_choice = None
        return {"type": "opportunity_open", "opportunity_id": self.current_id,
                "step": step, "evidence_step": evidence_step, "trigger": trigger,
                "interface": self.interface, "state_signature": list(signature),
                "legal_actions": legal_actions(self.contract, state)}

    def retry_after_failure(self, state: State, step: int, *, evidence_step: int) -> dict[str, Any]:
        self.current_signature = None
        return self.update(state, step, evidence_step=evidence_step,
                           trigger="observable_operation_failure")

    def record_choice(self, action: dict[str, str], step: int, source: str,
                      evidence_step: int) -> dict[str, Any] | None:
        if self.current_id is None or self.first_choice_recorded:
            if self.current_id is None or self.last_choice == action:
                return None
            self.last_choice = dict(action)
            return {"type": "choice_revision", "opportunity_id": self.current_id,
                    "step": step, "evidence_step": evidence_step, "source": source,
                    "action": action}
        self.first_choice_recorded = True
        self.last_choice = dict(action)
        return {"type": "choice", "opportunity_id": self.current_id, "step": step,
                "evidence_step": evidence_step, "source": source,
                "action": action}


def native_behavior(state: State, evidence: dict[str, dict[str, Any]],
                    contract: Contract, gripper_width_m: float) -> dict[str, str] | None:
    """Identify a physical target from contact or entry into a socket, not hovering."""
    picks = []
    for tube in TUBES:
        sample = evidence[tube]
        if (state.tube_status[tube] == "table" and sample.get("both_fingers_contact")
                and gripper_width_m < 0.035):
            picks.append({"type": "pick", "tube": tube})
    if picks:
        return picks[0] if len(picks) == 1 else None
    insertions = []
    for tube in TUBES:
        sample = evidence[tube]
        slot = sample.get("nearest_slot")
        if (state.tube_status[tube] == "held" and slot in SLOTS
                and sample.get("radial_error_m", math.inf) <= contract.socket_radius_m
                and contract.socket_depth_m * 0.15 <= sample.get("insertion_depth_m", -math.inf)
                and sample.get("axis_error_deg", math.inf) <= 20.0):
            insertions.append({"type": "insert", "tube": tube, "slot": slot})
    return insertions[0] if len(insertions) == 1 else None


def _measured_states(contract: Contract, trace: list[dict[str, Any]]) -> dict[int, State]:
    frame = next((row for row in trace if row.get("type") == "geometry_frame"), None)
    if frame is None:
        return {}
    state = State(contract.difficulty)
    heights: dict[str, float] = {}
    states: dict[int, State] = {}
    for row in trace:
        if row.get("type") != "measurement":
            continue
        step = row.get("step")
        raw_tubes = row.get("tubes", {})
        try:
            measured = {}
            for tube in TUBES:
                raw = raw_tubes[tube]
                measured[tube] = evaluate_insertion(
                    contract, Observation(tube, tuple(raw["position_world_m"]),
                                          tuple(raw["axis_world"]), float(raw["linear_speed_m_s"]),
                                          bool(raw["released"]), bool(raw["supported"]),
                                          raw.get("position_delta_m")),
                    frame["slot_mouths_world_m"], frame["insertion_axis_world"])
                z = float(raw["position_world_m"][2])
                if step == 30:
                    heights[tube] = z
                if step > 30 and tube in heights:
                    if (state.tube_status[tube] == "table" and z - heights[tube] >= .05
                            and raw.get("gripper_contact")):
                        state.tube_status[tube] = "held"
                    elif (state.tube_status[tube] == "held" and z <= heights[tube] + .025
                          and not raw.get("gripper_contact")):
                        state.tube_status[tube] = "table"
            advance_state(contract, state, measured)
            states[step] = State(state.difficulty, dict(state.tube_status), dict(state.occupied),
                                 dict(state.selected_slot), dict(state.stable_counts),
                                 state.first_actual_slot, state.terminal_steps)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            continue
    return states


def score_online_v2(contract: Contract, trace: list[dict[str, Any]],
                    reports: set[str] | None = None) -> dict[str, Any]:
    """Recompute references from measurements; count each opportunity once."""
    legacy = score_trace(contract, trace, reports)
    states = _measured_states(contract, trace)
    opportunities: dict[str, dict[str, Any]] = {}
    choices: dict[str, dict[str, Any]] = {}
    raw_by_step = {row.get("step"): row for row in trace if row.get("type") == "measurement"}
    decisions_by_step = {row.get("step"): row for row in trace if row.get("type") == "decision"}
    invalid_references = 0
    fixed_subgoal = None
    frame = next((item for item in trace if item.get("type") == "geometry_frame"), None)
    for row in trace:
        kind = row.get("type")
        if kind == "fixed_subgoal":
            state = states.get(row.get("evidence_step"))
            action = normalize_action(row.get("action"))
            pose = row.get("interaction_pose", {})
            slot = action.get("slot") if action else None
            expected = (frame or {}).get("slot_mouths_world_m", {}).get(slot)
            if (fixed_subgoal is not None or state is None or action is None or
                    action not in legal_actions(contract, state) or
                    pose.get("mouth_world_m") != (list(expected) if expected is not None else None) or
                    pose.get("axis_world") != list((frame or {}).get("insertion_axis_world", [])) or
                    pose.get("frame") != "world" or pose.get("units") != "m"):
                invalid_references += 1
            else:
                fixed_subgoal = row
        elif kind == "opportunity_open":
            key = row.get("opportunity_id")
            state = states.get(row.get("evidence_step"))
            if not isinstance(key, str) or key in opportunities:
                invalid_references += 1
                continue
            if state is None or row.get("legal_actions") != legal_actions(contract, state):
                invalid_references += 1
                continue
            opportunities[key] = row
        elif kind == "choice":
            key = row.get("opportunity_id")
            if key not in opportunities or key in choices:
                continue
            step = row.get("evidence_step")
            reference_step = row.get("reference_step", step)
            state = states.get(reference_step)
            if (state is None or not isinstance(step, int) or
                    not isinstance(reference_step, int) or
                    reference_step < opportunities[key]["evidence_step"] or
                    reference_step > step):
                invalid_references += 1
                continue
            source = row.get("source")
            if source == "physical_interaction":
                if row.get("step") != step:
                    invalid_references += 1
                    continue
                raw = raw_by_step.get(step, {})
                measured = {}
                frame = next((item for item in trace if item.get("type") == "geometry_frame"), None)
                for tube in TUBES:
                    sample = raw.get("tubes", {}).get(tube)
                    if sample is None or frame is None:
                        break
                    try:
                        derived = evaluate_insertion(
                            contract, Observation(tube, tuple(sample["position_world_m"]),
                                                  tuple(sample["axis_world"]), float(sample["linear_speed_m_s"]),
                                                  bool(sample["released"]), bool(sample["supported"]),
                                                  sample.get("position_delta_m")),
                            frame["slot_mouths_world_m"], frame["insertion_axis_world"])
                    except (KeyError, TypeError, ValueError, IndexError):
                        break
                    measured[tube] = dict(sample, **derived)
                    if "left_finger_contact" in sample and "right_finger_contact" in sample:
                        measured[tube]["both_fingers_contact"] = bool(
                            sample["left_finger_contact"] and sample["right_finger_contact"])
                finger_joints = raw.get("finger_joints_m")
                width = (sum(finger_joints) if isinstance(finger_joints, list) and len(finger_joints) == 2
                         else raw.get("gripper_width_m", math.inf))
                if len(measured) != len(TUBES) or native_behavior(
                        state, measured, contract, width) != row.get("action"):
                    invalid_references += 1
                    continue
            elif source == "explicit_commitment":
                decision = decisions_by_step.get(step)
                if decision is None or normalize_action(decision.get("action")) != normalize_action(row.get("action")):
                    invalid_references += 1
                    continue
            else:
                invalid_references += 1
                continue
            choices[key] = row
    report_checkpoints = []
    reported_ids: set[str] = set()
    for row in trace:
        if row.get("type") != "policy_report":
            continue
        report_id = row.get("opportunity_id")
        if (report_id not in opportunities or report_id in reported_ids or
                row.get("evidence_step", -1) < opportunities[report_id]["evidence_step"]):
            invalid_references += 1
            continue
        reported_ids.add(report_id)
        state = states.get(row.get("evidence_step"))
        if state is None:
            invalid_references += 1
            continue
        response = row.get("report", {})
        chosen = normalize_action(response.get("committed_action"))
        geometry = None
        if chosen and chosen["type"] == "insert" and frame is not None:
            slot = chosen["slot"]
            mouth = frame.get("slot_mouths_world_m", {}).get(slot)
            if mouth is not None:
                geometry = {"slot": slot, "mouth_world_m": mouth,
                            "axis_world": frame["insertion_axis_world"]}
        visible = {slot: tube for slot, tube in state.occupied.items()
                   if slot in row.get("visible_slots", [])}
        report_checkpoints.append(score_checkpoint({
            "instance_id": row.get("opportunity_id", str(row.get("step"))),
            "supported_reports": sorted(set(reports or ()) - {"execution"}),
            "expected_semantic": visible, "semantic_report": response.get("semantic_report"),
            "semantic_reference_available": bool(row.get("visible_slots")),
            "legal_actions": legal_actions(contract, state),
            "committed_action": chosen, "geometry_applicable": bool(chosen and chosen["type"] == "insert"),
            "reference_geometry": geometry, "geometry_report": response.get("geometry_report"),
        }))
    scorable = {key: item for key, item in opportunities.items()
                if item.get("interface") != "native" or any(
                    action.get("type") in {"pick", "insert"} for action in item["legal_actions"])}
    correct = sum(normalize_action(choices[key].get("action")) in [
        normalize_action(x) for x in legal_actions(contract, states[item["evidence_step"]])]
        for key, item in scorable.items() if key in choices)
    total = len(scorable)
    selected = sum(key in choices for key in scorable)
    temporal_illegal = sum(key in choices for key in opportunities if key not in scorable)
    evaluation_valid = bool(legacy["physical_replay_verified"] and invalid_references == 0)
    physical = bool(legacy["completion_success"] and legacy["safety_pass"]
                    and legacy["physical_replay_verified"])
    failure_events = any(row.get("event") in {"grasp_lost", "grasp_attempt_failed", "insertion_lost"}
                         or row.get("type") == "violation" for row in trace)
    interface_scores = (summarize_checkpoint_scores(report_checkpoints) if report_checkpoints else
                        legacy["module_scores"])
    fixed_execution = None
    fixed_delivered = False
    if fixed_subgoal is not None:
        handoff = fixed_subgoal["evidence_step"]
        action = fixed_subgoal["action"]
        fixed_delivered = any(
            row.get("type") == "policy_inference" and row.get("step", -1) >= handoff and
            row.get("policy_request", {}).get("selected_operation") == action and
            row.get("policy_request", {}).get("interaction_pose") == fixed_subgoal["interaction_pose"]
            for row in trace)
        if fixed_delivered:
            fixed_execution = any(
                step > handoff and state.tube_status[action["tube"]] == "inserted" and
                state.occupied.get(action["slot"]) == action["tube"] and
                state.terminal_steps >= contract.final_stable_steps
                for step, state in states.items())
        interface_scores = dict(interface_scores)
        interface_scores["execution"] = {"status": "scored" if fixed_delivered else "unavailable",
                                          "accuracy": float(fixed_execution) if fixed_delivered else None,
                                          "denominator": int(fixed_delivered),
                                          "reference_executor": "policy_native_joint_actions"}
    return {"evaluation_version": EVALUATION_VERSION, "task_version": contract.task_version,
            "contract_hash": contract.hash, "opportunity_count": len(opportunities),
            "scored_action_opportunity_count": total,
            "temporal_opportunity_count": len(opportunities) - total,
            "temporal_illegal_choice_count": temporal_illegal,
            "identified_choice_count": selected, "correct_choice_count": correct,
            "online_legal_next_action_rate": correct / total if total and evaluation_valid else None,
            "choice_identification_coverage": selected / total if total and evaluation_valid else None,
            "invalid_opportunity_reference_count": invalid_references,
            "evaluation_valid": evaluation_valid,
            "physical_completion": legacy["completion_success"],
            "task_completion": physical, "clean_completion": physical and total > 0 and
            evaluation_valid and not failure_events and not temporal_illegal and correct == total,
            "recovered_completion": physical and failure_events,
            "physical_replay_verified": legacy["physical_replay_verified"],
            "diagnostic": {"interface_scores": interface_scores,
                           "fixed_subgoal_execution_success": fixed_execution,
                           "fixed_subgoal_delivered": fixed_delivered,
                           "report_checkpoint_count": len(report_checkpoints),
                           "operation_attempts": legacy["operation_attempts"],
                           "commitment_matches_actual_insertion": legacy["precision_pass"],
                           "safety_pass": legacy["safety_pass"]}}


def online_state_pair_record(path: str, plan: list[dict[str, Any]]) -> dict[str, Any]:
    """Read a physically verified checkpoint continuation for a planned state pair."""
    import json
    from pathlib import Path
    from .cli import contract_from_evaluation, parse_trace
    from .diagnostics import verify_online_scores

    root = Path(path)
    if root.is_file():
        root = root.parent
    manifest = json.loads((root / "run.json").read_text())
    slot = manifest.get("checkpoint_requested_slot")
    matching = [item for item in plan if item["scene_seed"] == manifest.get("scene_seed")
                and item["first_actual_slot"] == slot]
    if len(matching) != 1 or matching[0].get("state_source") != "verified_preload_and_reference_pick_plan":
        raise ValueError("run does not match a registered online state checkpoint")
    expected = matching[0]
    trace = parse_trace(str(root / "events.jsonl")) if (root / "events.jsonl").is_file() else []
    evaluation_valid = False
    if manifest.get("checkpoint_verified"):
        contract = contract_from_evaluation(manifest["evaluation"])
        snapshots = parse_trace(str(root / "online_scores.jsonl"))
        audit = verify_online_scores(contract, trace, snapshots,
                                     set(manifest["evaluation"]["policy_interface"].get("supported_reports", [])),
                                     EVALUATION_VERSION)
        if not audit["matching"]:
            raise ValueError("online checkpoint trace failed measurement or score audit")
        evaluation_valid = score_online_v2(contract, trace)["evaluation_valid"]
    first = next((item for item in trace if item.get("type") == "opportunity_open" and
                  item.get("legal_actions") == expected["legal_actions"]), None)
    choice = next((item for item in trace if item.get("type") == "choice" and first is not None and
                   item.get("opportunity_id") == first["opportunity_id"] and
                   item.get("step", math.inf) - first["step"] <=
                   manifest.get("checkpoint_decision_window_steps", 180)), None)
    window = manifest.get("checkpoint_decision_window_steps", 180)
    window_complete = (first is not None and manifest.get("steps_executed", 0) - first["step"] >= window)
    status = ("invalid_evaluation" if manifest.get("checkpoint_verified") and not evaluation_valid else
              "scored" if manifest.get("checkpoint_verified") and window_complete and
              manifest.get("runtime_status") in
              {"finished", "time_limit", "stopped_after_violation"} else
              "service_error" if manifest.get("runtime_status") == "service_error" else "runtime_error")
    return {"pair_id": expected["pair_id"], "variant": expected["variant"],
            "task_version": manifest["task_version"], "contract_hash": manifest["contract_hash"],
            "scene_seed": manifest["scene_seed"], "decision_seed": manifest["decision_seed"],
            "first_actual_slot": slot, "policy_id": manifest["policy_id"],
            "policy_revision": manifest.get("policy_revision"),
            "policy_reset_scope": manifest.get("policy_reset_scope"),
            "policy_uses_evaluator_state": manifest["policy_uses_evaluator_state"],
            "sensor_condition": manifest["sensor_condition"],
            "policy_track": manifest["policy_track"],
            "evaluation_version": manifest["evaluation_version"],
            "choice_source": choice.get("source") if choice else None,
            "committed_action": choice.get("action") if choice else None,
            "decision_window_steps": window,
            "status": status}
