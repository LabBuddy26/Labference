"""Event-driven online scoring for rack-placement recall."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from .core import (Contract, DISTRACTOR_TUBES, EVALUATION_VERSION, Instance, Measurement, RackPose,
                   SCORING_TRACKS, SCORING_VERSION, SLOTS, State, TARGET_TUBE, TubeObservation,
                   advance_state, evaluate_tube, legal_actions, normalize_action, task_complete)


def state_signature(contract: Contract, state: State) -> tuple[Any, ...]:
    return (state.phase, state.target_status, tuple(sorted(state.occupied.items())),
            state.environment_ready,
            state.recovery_stable_count >= contract.final_stable_steps)


@dataclass
class OpportunityBook:
    contract: Contract
    interface: str
    current_signature: tuple[Any, ...] | None = None
    current_id: str | None = None
    next_number: int = 0
    first_choice_recorded: bool = False

    def update(self, state: State, step: int, evidence_step: int,
               trigger: str = "state_changed") -> dict[str, Any] | None:
        signature = state_signature(self.contract, state)
        if signature == self.current_signature:
            return None
        self.current_signature = signature
        self.current_id = f"opportunity-{self.next_number:04d}"
        self.next_number += 1
        self.first_choice_recorded = False
        return {"type": "opportunity_open", "opportunity_id": self.current_id,
                "step": step, "evidence_step": evidence_step, "trigger": trigger,
                "interface": self.interface, "legal_actions": legal_actions(self.contract, state)}

    def retry(self, state: State, step: int, evidence_step: int) -> dict[str, Any]:
        self.current_signature = None
        return self.update(state, step, evidence_step, "observable_operation_failure")

    def record_choice(self, action: Mapping[str, Any], step: int,
                      evidence_step: int, source: str) -> dict[str, Any] | None:
        normalized = normalize_action(action)
        if normalized is None or self.current_id is None:
            return None
        if self.first_choice_recorded:
            return {"type": "choice_revision", "opportunity_id": self.current_id,
                    "step": step, "evidence_step": evidence_step,
                    "source": source, "action": normalized}
        self.first_choice_recorded = True
        return {"type": "choice", "opportunity_id": self.current_id,
                "step": step, "evidence_step": evidence_step,
                "source": source, "action": normalized}


@dataclass
class BehaviorSelectionTracker:
    """Recognize a deliberate slot approach before finger contact."""

    contract: Contract
    candidate_slot: str | None = None
    candidate_start_step: int | None = None
    candidate_start_z_m: float | None = None
    frozen_slot: str | None = None

    def reset_candidate(self) -> None:
        self.candidate_slot = None
        self.candidate_start_step = None
        self.candidate_start_z_m = None

    def update(self, finger_center_world_m: tuple[float, float, float],
               rack_pose: RackPose, step: int) -> dict[str, Any] | None:
        if self.frozen_slot is not None:
            return None
        x, y, z = finger_center_world_m
        candidates = []
        for slot in SLOTS:
            mouth = self.contract.slot_world_center(slot, rack_pose)
            tube_top_z = (mouth[2] - self.contract.rack_bottom_support_depth_m +
                          self.contract.tube_length_m)
            radial = math.dist((x, y), mouth[:2])
            height = z - tube_top_z
            if (radial <= self.contract.behavior_selection_radius_m and
                    self.contract.behavior_selection_min_height_m <= height <=
                    self.contract.behavior_selection_max_height_m):
                candidates.append(slot)
        if len(candidates) != 1:
            self.reset_candidate()
            return None
        slot = candidates[0]
        if slot != self.candidate_slot:
            self.candidate_slot = slot
            self.candidate_start_step = step
            self.candidate_start_z_m = z
            return None
        dwell_steps = math.ceil(
            self.contract.behavior_selection_dwell_s * self.contract.physics_hz)
        descended = float(self.candidate_start_z_m - z)
        if step - int(self.candidate_start_step) < dwell_steps:
            return None
        if descended < self.contract.behavior_selection_descent_m:
            return None
        self.frozen_slot = slot
        return {"type": "retrieve", "slot": slot}


def native_behavior(state: State, measurement: Measurement,
                    evidence: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    target = evidence[TARGET_TUBE]
    if state.phase == "placement":
        if state.target_status == "table" and target["both_fingers_contact"]:
            return {"type": "pick", "tube": TARGET_TUBE}
        if state.target_status == "held" and target["actual_slot"] is not None:
            return {"type": "insert", "tube": TARGET_TUBE,
                    "slot": target["actual_slot"]}
    if state.phase == "recovery" and target["in_recovery_region"]:
        return {"type": "place_recovery", "tube": TARGET_TUBE}
    return None


def measurement_to_dict(measurement: Measurement) -> dict[str, Any]:
    return {"step": measurement.step, "rack_pose": measurement.rack_pose.public(),
            "environment_phase": measurement.environment_phase,
            "occluder_visible": measurement.occluder_visible,
            "rack_motion_complete": measurement.rack_motion_complete,
            "evidence_delivered": measurement.evidence_delivered,
            "tubes": {name: asdict(value) for name, value in measurement.tubes.items()}}


def measurement_from_dict(value: Mapping[str, Any]) -> Measurement:
    tubes = {name: TubeObservation(**dict(sample))
             for name, sample in value["tubes"].items()}
    pose = RackPose(tuple(value["rack_pose"]["position_m"]),
                    float(value["rack_pose"]["yaw_deg"]))
    return Measurement(int(value["step"]), pose, str(value["environment_phase"]),
                       bool(value["occluder_visible"]), tubes,
                       bool(value["rack_motion_complete"]),
                       bool(value["evidence_delivered"]))


def replay_states(contract: Contract, instance: Instance,
                  trace: list[dict[str, Any]]) -> tuple[dict[int, State], list[dict[str, Any]]]:
    state = State.initial(instance)
    checkpoint_rows = [row for row in trace if row.get("type") == "checkpoint_initialization"]
    validation_rows = [row for row in trace if row.get("type") == "checkpoint_validation"]
    checkpoint_mode = bool(checkpoint_rows or validation_rows)
    checkpoint_slot = checkpoint_rows[0].get("slot") if len(checkpoint_rows) == 1 else None
    checkpoint_valid = bool(
        len(checkpoint_rows) == 1 and len(validation_rows) == 1 and
        checkpoint_slot in SLOTS and checkpoint_slot not in instance.distractor_slots and
        validation_rows[0].get("slot") == checkpoint_slot and
        validation_rows[0].get("consecutive_stable_samples", 0) >= contract.stable_steps and
        validation_rows[0].get("verified") is True
    )
    physical: list[dict[str, Any]] = []
    physical_rows_by_step: dict[int, list[dict[str, Any]]] = {}
    for row in trace:
        if row.get("type") == "physical" and type(row.get("step")) is int:
            physical_rows_by_step.setdefault(row["step"], []).append(row)
    if checkpoint_mode and not checkpoint_valid:
        physical.append({"type": "violation", "scope": "environment_validity",
                         "kind": "checkpoint_validation_missing_or_invalid", "step": -1})
    checkpoint_pending = checkpoint_valid
    if checkpoint_valid:
        state.actual_placement_slot = checkpoint_slot
        state.occupied[checkpoint_slot] = TARGET_TUBE
        state.target_status = "inserted"
        state.phase = "transition"
    states: dict[int, State] = {}
    for row in trace:
        if row.get("type") != "measurement":
            continue
        try:
            measurement = measurement_from_dict(row["measurement"])
            if checkpoint_pending:
                sample = evaluate_checkpoint_sample(contract, measurement, checkpoint_slot)
                if not sample:
                    physical.append({"type": "violation", "scope": "environment_validity",
                                     "kind": "checkpoint_state_mismatch",
                                     "step": measurement.step})
                else:
                    physical.append({"type": "environment",
                                     "event": "checkpoint_state_verified",
                                     "slot": checkpoint_slot,
                                     "step": measurement.step})
                checkpoint_pending = False
            # Reference collection runs emit an explicit handoff once the
            # expert controller has completed the physical sub-operation.
            # Replay that measured handoff before evaluating the same-step
            # observation so protocol opportunities use the true state.
            for event in physical_rows_by_step.get(measurement.step, ()):
                kind = event.get("event")
                if kind in {"placement_confirmed", "collection_placement_confirmed",
                            "target_retrieved", "recovery_confirmed"}:
                    physical.append(dict(event))
                if kind in {"placement_confirmed", "collection_placement_confirmed"}:
                    slot = event.get("actual_slot") or event.get("slot")
                    if slot in SLOTS:
                        state.actual_placement_slot = slot
                        state.occupied[slot] = TARGET_TUBE
                        state.target_status = "inserted"
                        state.phase = "transition"
                elif kind == "target_retrieved":
                    state.target_status = "held"
                    state.phase = "recovery"
                elif kind == "recovery_confirmed":
                    state.target_status = "table"
                    state.target_status = "recovered"
                    state.phase = "complete"
                    state.recovery_stable_count = contract.final_stable_steps
            events = advance_state(contract, instance, state, measurement)
        except (KeyError, TypeError, ValueError) as exc:
            physical.append({"type": "violation", "scope": "environment_validity",
                             "kind": "malformed_measurement",
                             "step": row.get("step"),
                             "error": type(exc).__name__})
            continue
        for event in events:
            physical.append({**event, "step": measurement.step})
        states[measurement.step] = State(
            state.difficulty, state.phase, state.target_status,
            state.actual_placement_slot, dict(state.occupied),
            state.insertion_stable_count, state.recovery_stable_count,
            state.environment_ready, state.failure_seen, list(state.violations),
            state.grasp_stable_count, state.released_invalid_count,
            state.occlusion_observed, state.retrieval_stable_count,
            dict(state.distractor_invalid_counts))
    if checkpoint_pending:
        physical.append({"type": "violation", "scope": "environment_validity",
                         "kind": "checkpoint_measurement_missing", "step": -1})
    return states, physical


def evaluate_checkpoint_sample(contract: Contract, measurement: Measurement,
                               slot: str) -> bool:
    sample = evaluate_tube(contract, measurement.tubes[TARGET_TUBE], measurement.rack_pose)
    return bool(sample["actual_slot"] == slot and sample["stable_sample"])


def score_trace(contract: Contract, instance: Instance,
                trace: list[dict[str, Any]]) -> dict[str, Any]:
    if (instance.difficulty != contract.difficulty or
            instance.condition_variant != contract.condition_variant):
        raise ValueError("instance and contract condition mismatch")
    states, physical = replay_states(contract, instance, trace)
    final = states[max(states)] if states else State.initial(instance)
    capabilities = [row for row in trace if row.get("type") == "policy_capabilities"]
    scoring_track = (capabilities[0].get("scoring_track")
                     if len(capabilities) == 1 else None)
    history_protocol = (capabilities[0].get("history_protocol")
                        if len(capabilities) == 1 else None)
    episode_memory = (capabilities[0].get("episode_memory")
                      if len(capabilities) == 1 else None)
    ranking_eligible = bool(len(capabilities) == 1 and
                            capabilities[0].get("ranking_eligible") is True)
    opportunities: dict[str, tuple[dict[str, Any], State]] = {}
    choices: dict[str, dict[str, Any]] = {}
    invalid_references = 0
    last_step = -1
    for row in trace:
        event_step = row.get("step")
        if type(event_step) is int:
            if event_step < last_step:
                invalid_references += 1
            last_step = max(last_step, event_step)
        if row.get("type") == "opportunity_open":
            key, step = row.get("opportunity_id"), row.get("evidence_step")
            state = states.get(step)
            if (not isinstance(key, str) or key in opportunities or state is None or
                    row.get("legal_actions") != legal_actions(contract, state)):
                invalid_references += 1
            else:
                opportunities[key] = (row, state)
        elif row.get("type") == "choice":
            key = row.get("opportunity_id")
            opened = opportunities.get(key)
            timing_valid = bool(
                opened and type(row.get("step")) is int and
                row["step"] >= opened[0]["step"] and
                type(row.get("evidence_step")) is int and
                row["evidence_step"] >= opened[0]["evidence_step"] and
                row["evidence_step"] <= row["step"]
            )
            if (key not in opportunities or key in choices or
                    normalize_action(row.get("action")) is None or not timing_valid):
                invalid_references += 1
            else:
                choices[key] = row

    eligible = {key: value for key, value in opportunities.items()
                if any(action["type"] not in {"wait", "stop"}
                       for action in value[0]["legal_actions"])}
    correct = 0
    choice_error_seen = False
    for key, (opened, state) in eligible.items():
        action = normalize_action(choices.get(key, {}).get("action"))
        if action in legal_actions(contract, state):
            correct += 1
        elif action is not None:
            choice_error_seen = True

    recall_opportunities = {
        key: value for key, value in opportunities.items()
        if value[1].phase == "recall" and value[1].environment_ready
    }
    deliveries = [row for row in trace if row.get("type") == "evidence_delivery"]
    delivery = deliveries[0] if len(deliveries) == 1 else None
    delivery_valid = bool(
        delivery and type(delivery.get("step")) is int and
        isinstance(delivery.get("sim_time_s"), (int, float)) and
        set(delivery.get("frame_ids", {})) == {"front", "wrist"} and
        delivery.get("history_protocol") == "online_once" and
        all(opened[0]["step"] >= delivery["step"] and
            opened[0]["evidence_step"] >= delivery["step"]
            for opened in recall_opportunities.values())
    )
    if recall_opportunities and not delivery_valid:
        invalid_references += 1
    delivered_observations = [row for row in trace
                              if row.get("type") == "policy_observation_delivered"]
    delivered_ids: set[str] = set()
    for row in delivered_observations:
        observation_id = row.get("observation_id")
        if (not isinstance(observation_id, str) or observation_id in delivered_ids or
                row.get("history_replayed") is not False or
                set(row.get("frame_ids", {})) != {"front", "wrist"} or
                not isinstance(row.get("sim_time_s"), (int, float))):
            invalid_references += 1
        else:
            delivered_ids.add(observation_id)
    recall_reports = [row for row in trace if row.get("type") == "recall_report"]
    first_report = recall_reports[0] if recall_reports else None
    report_supported = any(row.get("type") == "policy_capabilities" and
                           "recall" in row.get("reports", []) for row in trace)
    report_opportunity = (None if first_report is None else
                          recall_opportunities.get(first_report.get("opportunity_id")))
    report_state = None if report_opportunity is None else report_opportunity[1]
    report_timing_valid = bool(
        first_report and report_opportunity and type(first_report.get("step")) is int and
        first_report["step"] >= report_opportunity[0]["step"]
    )
    if first_report is not None and not report_timing_valid:
        invalid_references += 1
    recall_correct = (None if not report_supported else bool(
        first_report and report_timing_valid and report_state and
        first_report.get("slot") == report_state.actual_placement_slot and
        first_report.get("slot") in report_state.occupied))
    retrieval_choices = [row for row in choices.values()
                         if normalize_action(row.get("action", {})) and
                         row["action"].get("type") == "retrieve"]
    def first_choice_for_source(source: str) -> dict[str, Any] | None:
        return next((row for row in retrieval_choices if row.get("source") == source), None)

    def choice_correct(row: dict[str, Any] | None) -> bool | None:
        if row is None:
            return None
        opened, choice_state = opportunities[row["opportunity_id"]]
        return normalize_action(row["action"]) in legal_actions(contract, choice_state)

    first_symbolic = first_choice_for_source("explicit_commitment")
    first_behavior = first_choice_for_source("behavior_approach")
    symbolic_correct = choice_correct(first_symbolic)
    behavior_correct = choice_correct(first_behavior)
    first_retrieval = retrieval_choices[0] if retrieval_choices else None
    first_retrieval_correct = choice_correct(first_retrieval)
    track_results = {
        "explicit_slot_report": recall_correct,
        "symbolic_slot_selection": symbolic_correct,
        "native_behavior_selection": behavior_correct,
    }
    slot_reasoning_success = track_results.get(scoring_track)
    checkpoint_mode = any(row.get("type") == "checkpoint_initialization" for row in trace)
    checkpoint_state_verified = any(
        row.get("event") == "checkpoint_state_verified" for row in physical)
    placement_success = any(row.get("event") in
                            {"placement_confirmed", "collection_placement_confirmed"}
                            for row in physical)
    retrieval_success = any(row.get("event") == "target_retrieved" for row in physical)
    physical_completion = task_complete(contract, final)
    logged_violations = [row for row in trace if row.get("type") == "violation"]
    environment_failures = ([row for row in logged_violations
                             if row.get("scope") == "environment_validity"] +
                            [row for row in physical
                             if row.get("type") == "violation" and
                             row.get("scope") == "environment_validity"])
    policy_violations = list(final.violations) + [str(row.get("kind")) for row in logged_violations
                                                  if row.get("scope") != "environment_validity"]
    evaluation_valid = bool(
        len(capabilities) == 1 and states and not environment_failures and
        invalid_references == 0 and
        scoring_track in SCORING_TRACKS and history_protocol == "online_once"
    )
    reached_selection_opportunity = bool(recall_opportunities)
    selection_identified = bool(reached_selection_opportunity and
                                slot_reasoning_success is not None)
    if not evaluation_valid:
        selection_outcome = "invalid"
    elif not reached_selection_opportunity:
        selection_outcome = "unreached"
    elif not selection_identified:
        selection_outcome = "unidentified"
    else:
        selection_outcome = "correct" if slot_reasoning_success else "wrong"
    reasoning_valid_success = bool(evaluation_valid and
                                   slot_reasoning_success is True)
    valid_success = bool(evaluation_valid and physical_completion and
                         not {"distractor_displaced", "distractor_touched"} & set(policy_violations))
    return {
        "evaluation_version": EVALUATION_VERSION,
        "scoring_version": SCORING_VERSION,
        "task_version": contract.task_version,
        "contract_hash": contract.hash,
        "difficulty": contract.difficulty,
        "condition_variant": contract.condition_variant,
        "scoring_track": scoring_track,
        "history_protocol": history_protocol,
        "episode_memory": episode_memory,
        "evaluation_track": ("verified_state_checkpoint" if checkpoint_mode
                             else "autonomous_rollout"),
        "ranking_eligible": ranking_eligible,
        "ranking_success": bool(ranking_eligible and reasoning_valid_success),
        "checkpoint_state_verified": checkpoint_state_verified,
        "evidence_delivery_valid": delivery_valid,
        "evaluation_valid": evaluation_valid,
        "invalid_reference_count": invalid_references,
        "reached_selection_opportunity": reached_selection_opportunity,
        "selection_identified": selection_identified,
        "selection_outcome": selection_outcome,
        "opportunity_count": len(opportunities),
        "scored_choice_count": len(eligible),
        "identified_choice_count": sum(key in choices for key in eligible),
        "online_next_choice_accuracy": correct / len(eligible) if eligible else None,
        "choice_identification_coverage": (sum(key in choices for key in eligible) / len(eligible)
                                           if eligible else None),
        "placement_success": placement_success,
        "placement_execution_success": placement_success,
        "placement_execution_applicable": not checkpoint_mode,
        "actual_placement_slot": final.actual_placement_slot,
        "recall_report_supported": report_supported,
        "recall_report_correct": recall_correct,
        "explicit_slot_report_correct": recall_correct,
        "symbolic_slot_selection_correct": symbolic_correct,
        "behavior_slot_selection_correct": behavior_correct,
        "report_action_consistent": (
            None if first_report is None or first_symbolic is None else
            first_report.get("slot") == first_symbolic.get("action", {}).get("slot")),
        "first_retrieval_choice_correct": first_retrieval_correct,
        "slot_reasoning_success": slot_reasoning_success,
        "slot_reasoning_evaluable": selection_identified,
        # The benchmark asks whether the policy identifies the correct local
        # slot after occlusion. Placement and retrieval are execution probes,
        # not prerequisites for credit on that reasoning outcome.
        "primary_metric": "slot_reasoning_success",
        "benchmark_success": reasoning_valid_success,
        "all_rollout_correct_selection": reasoning_valid_success,
        "reasoning_valid_success": reasoning_valid_success,
        "placement_skill_gate_applied": False,
        "retrieval_execution_success": retrieval_success,
        "physical_completion": physical_completion,
        "clean_completion": bool(valid_success and not final.failure_seen and
                                 not choice_error_seen),
        "recovered_completion": bool(valid_success and (final.failure_seen or
                                     choice_error_seen)),
        "policy_violations": sorted(set(policy_violations)),
        "environment_failure_count": len(environment_failures),
        "valid_success": valid_success,
    }


def compare_pairs(plan: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexed: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("pair_id"), row.get("member"))
        if key in indexed:
            raise ValueError(f"duplicate result for pair member: {key}")
        indexed[key] = row
    complete = 0
    both_correct = 0
    invalid = 0
    missing = []
    for pair in plan:
        pair_id = pair["pair_id"]
        expected = {(pair_id, member["member"]) for member in pair["members"]}
        actual = [indexed.get(key) for key in expected]
        if any(row is None for row in actual):
            missing.append(pair_id)
            continue
        expected_fields = {
            "task_version": pair["task_version"],
            "contract_hash": pair["contract_hash"],
            "scoring_version": pair.get("scoring_version", SCORING_VERSION),
            "scoring_track": pair.get("scoring_track"),
            "history_protocol": pair.get("history_protocol", "online_once"),
            "oracle_profile": pair.get("oracle_profile"),
        }
        if any(any(expected is not None and row.get(field) != expected
                   for field, expected in expected_fields.items()) for row in actual):
            raise ValueError(f"pair {pair_id} mixes task contracts")
        complete += 1
        pair_valid = all(row.get("evaluation_valid") is True for row in actual)
        invalid += int(not pair_valid)
        both_correct += int(pair_valid and all(
            row.get("benchmark_success") is True for row in actual))
    return {"planned_pair_count": len(plan), "complete_pair_count": complete,
            "invalid_pair_count": invalid,
            "missing_pair_ids": missing,
            "state_pair_accuracy": both_correct / len(plan) if plan else None}


def apply_runtime_validity(scores: dict[str, Any], runtime_status: str) -> dict[str, Any]:
    """Attach lifecycle validity without discarding diagnostic sub-results."""
    result = dict(scores)
    complete = runtime_status == "finished"
    result["runtime_status"] = runtime_status
    result["runtime_complete"] = complete
    if complete:
        return result
    result.update({
        "evaluation_valid": False,
        "selection_outcome": "invalid",
        "benchmark_success": False,
        "all_rollout_correct_selection": False,
        "reasoning_valid_success": False,
        "ranking_success": False,
        "valid_success": False,
        "lifecycle_invalid_reason": f"runtime_status={runtime_status}",
    })
    return result
