"""Evidence-timed online decisions and measurement-based replay for weighing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .core import Contract, Instance, Measurement, State, advance, legal_actions, normalize_action, task_complete


EVALUATION_VERSION = "labference/weighing-online-v1"


def signature(state: State, contract: Contract) -> tuple[Any, ...]:
    """Only a new load or a stable/terminal transition makes a fresh choice."""
    return (state.supported_weights_g, state.reading_stable,
            state.final_steps >= contract.final_steps, state.sensor_valid)


@dataclass
class OpportunityBook:
    contract: Contract
    instance: Instance
    interface: str
    current_id: str | None = None
    current_evidence_step: int | None = None
    current_signature: tuple[Any, ...] | None = None
    next_number: int = 0
    first_choice: dict[str, Any] | None = None

    def update(self, state: State, step: int, *, evidence_step: int,
               operation_in_flight: bool = False,
               trigger: str = "state_changed") -> dict[str, Any] | None:
        if operation_in_flight or evidence_step > step:
            return None
        current = signature(state, self.contract)
        if current == self.current_signature:
            return None
        self.current_signature = current
        self.current_id = f"opportunity-{self.next_number:04d}"
        self.current_evidence_step = evidence_step
        self.next_number += 1
        self.first_choice = None
        return {"type": "opportunity_open", "opportunity_id": self.current_id,
                "step": step, "evidence_step": evidence_step,
                "trigger": trigger, "interface": self.interface,
                "legal_actions": legal_actions(self.contract, self.instance, state)}

    def retry_after_failure(self, state: State, step: int, *, evidence_step: int) -> dict[str, Any]:
        self.current_signature = None
        opened = self.update(state, step, evidence_step=evidence_step,
                             trigger="observable_operation_failure")
        assert opened is not None
        return opened

    def record_choice(self, action: dict[str, Any], step: int, *, evidence_step: int,
                      source: str) -> dict[str, Any] | None:
        normalized = normalize_action(action)
        if self.current_id is None or normalized is None or evidence_step > step:
            return None
        if self.first_choice is None:
            self.first_choice = normalized
            return {"type": "choice", "opportunity_id": self.current_id,
                    "step": step, "evidence_step": evidence_step,
                    "source": source, "action": normalized}
        if self.first_choice != normalized:
            return {"type": "choice_revision", "opportunity_id": self.current_id,
                    "step": step, "evidence_step": evidence_step,
                    "source": source, "action": normalized}
        return None


def physical_choice(contacts: dict[int, dict[str, Any]], supported: tuple[int, ...]) -> dict[str, Any] | None:
    """Two finger contacts identify a weight being grasped; hovering has no label."""
    picks = [weight for weight, sample in contacts.items()
             if weight in (10, 20, 50, 100) and sample.get("left_finger_contact") and
             sample.get("right_finger_contact") and sample.get("gripper_closing")]
    if len(picks) != 1:
        return None
    return {"type": "remove" if picks[0] in supported else "add", "weight_g": picks[0]}


def execution_subgoal_confirmed(action: dict[str, Any], state: State,
                                *, observed_change: bool) -> bool:
    """Confirm a selected operation only after its physical load has settled."""
    chosen = normalize_action(action)
    if chosen is None or chosen["type"] not in {"add", "remove"}:
        return False
    supported = chosen["weight_g"] in state.supported_weights_g
    return bool(observed_change and state.sensor_valid and state.reading_stable and
                state.sample_supported and state.sample_untouched and state.released and
                state.no_external_support and state.no_hand_press and
                supported == (chosen["type"] == "add"))


def _measurement_from_row(row: dict[str, Any]) -> Measurement:
    data = row["measurement"]
    contacts = data.get("contacts")
    if contacts is not None:
        if not isinstance(contacts, dict) or "sample" not in contacts or "tray" not in contacts:
            raise ValueError("incomplete physical contact readback")
        expected = tuple(weight for weight in (10, 20, 50, 100)
                         if contacts.get(f"{weight}g", {}).get("tray") and
                         not contacts[f"{weight}g"].get("left") and
                         not contacts[f"{weight}g"].get("right") and
                         not contacts[f"{weight}g"].get("table"))
        sample_supported = bool(contacts["sample"].get("tray"))
        no_hand_press = not bool(contacts["tray"].get("franka_contact"))
        if (tuple(data["supported_weights_g"]) != expected or
                bool(data["sample_supported"]) != sample_supported or
                bool(data["no_hand_press"]) != no_hand_press):
            raise ValueError("measurement disagrees with contact readback")
    if data.get("sensor_source") is not None:
        if data["sensor_source"] != "physx_loadcell_joint":
            raise ValueError("weighing trace uses a non-physical sensor")
        if data.get("sensor_valid") and not data.get("overloaded", False):
            force = data.get("raw_force_n")
            zero = data.get("zero_force_n")
            if (not isinstance(force, (int, float)) or not isinstance(zero, (int, float)) or
                    not math.isfinite(force) or not math.isfinite(zero) or
                    abs((force - zero) * 1000.0 / 9.81 - float(data["measured_g"])) > .01):
                raise ValueError("display measurement disagrees with PhysX force")
    return Measurement(
        step=int(row["step"]), sensor_valid=bool(data["sensor_valid"]),
        measured_g=float(data["measured_g"]), displayed_g=float(data["displayed_g"]),
        supported_weights_g=tuple(sorted(int(w) for w in data["supported_weights_g"])),
        sample_supported=bool(data["sample_supported"]),
        sample_untouched=bool(data["sample_untouched"]), released=bool(data["released"]),
        no_external_support=bool(data["no_external_support"]),
        no_hand_press=bool(data["no_hand_press"]),
        max_linear_speed_m_s=float(data["max_linear_speed_m_s"]),
        max_angular_speed_rad_s=float(data["max_angular_speed_rad_s"]),
        overloaded=bool(data.get("overloaded", False)),
    )


def replay_states(contract: Contract, instance: Instance,
                  trace: list[dict[str, Any]]) -> tuple[dict[int, State], list[str]]:
    state = State()
    states: dict[int, State] = {}
    errors: list[str] = []
    for row in trace:
        if row.get("type") != "measurement":
            continue
        try:
            measured = _measurement_from_row(row)
            advance(contract, instance, state, measured)
            clone = State(**{key: value for key, value in vars(state).items() if key != "_window"})
            clone._window.extend(state._window)
            states[measured.step] = clone
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            errors.append(f"step {row.get('step')}: {exc}")
    return states, errors


def score_trace(contract: Contract, instance: Instance,
                trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit choices against reconstructed measurement state, never logged answers."""
    states, errors = replay_states(contract, instance, trace)
    opportunities: dict[str, dict[str, Any]] = {}
    choices: dict[str, dict[str, Any]] = {}
    revisions: list[dict[str, Any]] = []
    measurements = {row.get("step"): row for row in trace if row.get("type") == "measurement"}
    decisions = {row.get("step"): row for row in trace if row.get("type") == "decision"}
    operation_starts = {(row.get("step"), tuple(sorted(normalize_action(row.get("action", {})).items())))
                        for row in trace if row.get("type") == "operation_start" and
                        normalize_action(row.get("action")) is not None}
    last_open_signature = None
    failed_steps: set[int] = set()
    policy_failures = 0
    safety_violations = 0
    for row in trace:
        kind = row.get("type")
        if kind == "operation_result":
            action = normalize_action(row.get("action"))
            start_step, end_step = row.get("started_step"), row.get("step")
            before = states.get(start_step)
            after = states.get(end_step)
            # A physical grasp excludes the held weight from pan support in
            # that very measurement. Use the audited pre-grasp observation
            # associated with the choice, otherwise every removal looks as
            # though the weight was already absent before the operation.
            physical_choices = [choice for choice in choices.values()
                                if choice.get("source") == "physical_interaction"
                                and choice.get("step") == start_step
                                and normalize_action(choice.get("action")) == action]
            if len(physical_choices) == 1:
                opened = opportunities[physical_choices[0]["opportunity_id"]]
                before = states.get(opened["evidence_step"])
            if (action is None or action["type"] not in {"add", "remove"} or
                    not isinstance(start_step, int) or not isinstance(end_step, int) or
                    start_step >= end_step or before is None or after is None or
                    (start_step, tuple(sorted(action.items()))) not in operation_starts or
                    row.get("physical_supported_weights_g") != list(after.supported_weights_g)):
                errors.append(f"unverified operation result at step {end_step}")
                continue
            weight = action["weight_g"]
            changed = (weight in after.supported_weights_g) == (action["type"] == "add") and (
                (weight in before.supported_weights_g) != (weight in after.supported_weights_g))
            if row.get("status") == "confirmed":
                if not changed:
                    errors.append(f"false confirmed operation at step {end_step}")
            elif row.get("status") in {"failed", "dropped", "wrong_load"}:
                if changed:
                    errors.append(f"false failed operation at step {end_step}")
                else:
                    policy_failures += 1
                    failed_steps.add(end_step)
            else:
                errors.append(f"invalid operation status at step {end_step}")
        elif kind == "opportunity_open":
            key = row.get("opportunity_id")
            evidence_step = row.get("evidence_step")
            ref = states.get(evidence_step)
            if not isinstance(key, str) or key in opportunities or ref is None or (
                    row.get("legal_actions") != legal_actions(contract, instance, ref) or
                    row.get("step", -1) < evidence_step):
                errors.append(f"invalid opportunity: {key}")
            else:
                new_signature = signature(ref, contract)
                if new_signature == last_open_signature and not (
                        row.get("trigger") == "observable_operation_failure" and
                        any(failed <= row["step"] and failed >= evidence_step - contract.physics_hz * 5
                            for failed in failed_steps)):
                    errors.append(f"duplicated decision opportunity: {key}")
                    continue
                opportunities[key] = row
                last_open_signature = new_signature
        elif kind == "choice":
            key = row.get("opportunity_id")
            if key not in opportunities or key in choices or normalize_action(row.get("action")) is None:
                errors.append(f"invalid first choice: {key}")
            elif row.get("evidence_step", -1) < opportunities[key]["evidence_step"]:
                errors.append(f"choice predates evidence: {key}")
            elif row.get("source") not in {"explicit_commitment", "physical_interaction"}:
                errors.append(f"invalid choice source: {key}")
            else:
                if row["source"] == "physical_interaction":
                    sample = measurements.get(row.get("step"), {}).get("measurement", {})
                    contacts = {int(weight): value for weight, value in sample.get("finger_contacts", {}).items()}
                    ref_state = states[opportunities[key]["evidence_step"]]
                    if physical_choice(contacts, ref_state.supported_weights_g) != normalize_action(row["action"]):
                        errors.append(f"unverified physical choice: {key}")
                        continue
                elif normalize_action(decisions.get(row.get("step"), {}).get("action")) != normalize_action(row["action"]):
                    errors.append(f"unverified explicit commitment: {key}")
                    continue
                choices[key] = row
        elif kind == "choice_revision":
            if row.get("opportunity_id") in opportunities and normalize_action(row.get("action")):
                revisions.append(row)
        elif kind == "safety_violation":
            safety_violations += 1
    if any(not state.sample_untouched or not state.no_hand_press for state in states.values()):
        safety_violations += 1
    correct = 0
    identifiable = 0
    wrong = 0
    scored_opportunities = {
        key: row for key, row in opportunities.items()
        if row.get("interface") != "native" or any(
            action["type"] in {"add", "remove"} for action in row["legal_actions"])
    }
    first_ready = next((key for key, row in scored_opportunities.items() if any(
        action["type"] in {"add", "remove"} for action in row["legal_actions"])), None)
    first_ready_choice = choices.get(first_ready) if first_ready is not None else None
    first_ready_correct = (normalize_action(first_ready_choice["action"]) in
                           legal_actions(contract, instance,
                                         states[opportunities[first_ready]["evidence_step"]])) if first_ready_choice else None
    for key, opened in scored_opportunities.items():
        choice = choices.get(key)
        if choice is None:
            continue
        identifiable += 1
        action = normalize_action(choice["action"])
        reference = states[opened["evidence_step"]]
        if action in legal_actions(contract, instance, reference):
            correct += 1
        else:
            wrong += 1
    terminal = states[max(states)] if states else None
    completed = bool(terminal and task_complete(contract, instance, terminal))
    reported = any(row.get("interface") == "reported" for row in opportunities.values())
    terminal_stop = any(
        normalize_action(row.get("action")) == {"type": "stop"} and
        legal_actions(contract, instance, states[opportunities[row["opportunity_id"]]["evidence_step"]]) ==
        [{"type": "stop"}]
        for row in choices.values() if row.get("opportunity_id") in opportunities)
    protocol_complete = not reported or terminal_stop
    invalid_sensor = any(not state.sensor_valid for state in states.values())
    valid = not errors and bool(states) and not invalid_sensor
    denominator = len(scored_opportunities)
    return {
        "task_version": contract.task_version, "contract_hash": contract.hash,
        "evaluation_version": EVALUATION_VERSION,
        "evaluation_valid": valid, "audit_errors": errors,
        "opportunities": denominator, "wait_or_stop_opportunities": len(opportunities) - denominator,
        "identifiable_choices": identifiable,
        "recognition_coverage": identifiable / denominator if denominator else None,
        "online_legal_next_action_rate": correct / denominator if denominator else None,
        "identifiable_choice_accuracy": correct / identifiable if identifiable else None,
        "first_ready_choice_identified": first_ready_choice is not None,
        "first_ready_choice_correct": first_ready_correct,
        "wrong_first_choices": wrong, "choice_revisions": len(revisions),
        "task_completion": completed, "protocol_complete": protocol_complete,
        "clean_completion": completed and protocol_complete and valid and wrong == 0
        and policy_failures == 0 and safety_violations == 0,
        "recovered_completion": completed and (wrong > 0 or policy_failures > 0),
        "valid_success": completed and protocol_complete and valid and safety_violations == 0,
        "sensor_invalid_samples": sum(not state.sensor_valid for state in states.values()),
        "safety_violations": safety_violations,
        "final_reading_g": terminal.displayed_g if terminal else None,
        "final_supported_weights_g": list(terminal.supported_weights_g) if terminal else None,
        "final_stable_steps": terminal.final_steps if terminal else 0,
    }


def compare_pairs(plan: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A planned pair stays in the denominator when one or both runs are missing."""
    lookup = {(row.get("pair_id"), row.get("member"), row.get("condition")): row for row in rows}
    result = []
    for item in plan:
        pair_id = item["pair_id"]
        condition = item["condition"]
        left = lookup.get((pair_id, "a", condition))
        right = lookup.get((pair_id, "b", condition))
        if left and right:
            for key in ("task_version", "contract_hash", "scene_revision", "policy_id",
                        "sensor_condition", "action_space", "max_steps_budget"):
                if left.get(key) != right.get(key):
                    raise ValueError(f"incomparable paired runs: {pair_id}: {key}")
            if item.get("scene_revision") is not None and left.get("scene_revision") != item["scene_revision"]:
                raise ValueError(f"paired run scene differs from plan: {pair_id}")
            if left.get("scene_seed") is not None and left["scene_seed"] != right.get("scene_seed"):
                raise ValueError(f"paired run layouts differ: {pair_id}")
            if (left.get("scene_layout_hash") is not None or right.get("scene_layout_hash") is not None or
                    item.get("scene_layout_hash") is not None):
                expected = item.get("scene_layout_hash")
                if (not left.get("scene_layout_hash") or
                        left["scene_layout_hash"] != right.get("scene_layout_hash") or
                        expected is not None and left["scene_layout_hash"] != expected):
                    raise ValueError(f"paired run authored layouts differ: {pair_id}")
            for member in item.get("members", []):
                run = left if member["member"] == "a" else right
                if any(run.get(key) != member[key] for key in ("seed", "scene_seed", "mass_seed")):
                    raise ValueError(f"paired run seed differs from plan: {pair_id}")
            if left.get("instance") and right.get("instance") and (
                    left["instance"].get("target_g") != right["instance"].get("target_g") or
                    left["instance"].get("sample_mass_g") == right["instance"].get("sample_mass_g")):
                raise ValueError(f"paired run evidence did not change as planned: {pair_id}")
        both_correct = bool(left and right and left.get("first_choice_correct") is True and
                            right.get("first_choice_correct") is True)
        result.append({"pair_id": pair_id, "condition": condition,
                       "both_present": bool(left and right), "both_correct": both_correct})
    count = len(result)
    engineering_only = any(row.get("uses_evaluator_state") for row in rows)
    complete = bool(count and all(item["both_present"] for item in result))
    valid_runs = all(row.get("evaluation_valid") is True for row in rows)
    return {"planned_pairs": count, "present_pairs": sum(v["both_present"] for v in result),
            "engineering_only": engineering_only,
            "ranking_eligible": complete and valid_runs and not engineering_only,
            "state_pair_accuracy": sum(v["both_correct"] for v in result) / count if count else None,
            "pairs": result}


def compare_oracle_pairs(plan: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare matched online runs, with an explicit denominator for absent runs."""
    lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("task_version"), row.get("seed"), row.get("oracle_profile"))
        if key in lookup:
            raise ValueError(f"duplicate Oracle run: {key}")
        lookup[key] = row
    details = []
    for item in plan:
        if item.get("base_condition") != "none" or item.get("support_condition") not in {
                "semantic", "decision", "geometry", "skill"}:
            raise ValueError("Oracle plan requires none and one declared support condition")
        key = (item["task_version"], item["seed"])
        base = lookup.get((*key, "none"))
        support = lookup.get((*key, item["support_condition"]))
        pair = {"pair_id": item["pair_id"], "seed": item["seed"],
                "support_condition": item["support_condition"],
                "both_present": bool(base and support), "status": "matched" if base and support else "missing"}
        if base and support:
            if base.get("contract_hash") != item.get("contract_hash"):
                raise ValueError(f"Oracle pair {item['pair_id']} does not match the planned contract")
            if item.get("scene_revision") is not None and base.get("scene_revision") != item["scene_revision"]:
                raise ValueError(f"Oracle pair {item['pair_id']} does not match the planned scene")
            for field in ("task_version", "contract_hash", "seed", "scene_seed", "mass_seed",
                          "scene_revision", "scene_layout_hash", "policy_id", "sensor_condition", "action_space",
                          "max_steps_budget"):
                if base.get(field) != support.get(field):
                    raise ValueError(f"incomparable Oracle pair {item['pair_id']}: {field}")
            if item.get("scene_layout_hash") is not None and base.get("scene_layout_hash") != item["scene_layout_hash"]:
                raise ValueError(f"Oracle pair {item['pair_id']} does not match the planned layout")
            if base.get("uses_evaluator_state") or support.get("uses_evaluator_state"):
                pair["status"] = "diagnostic_expert"
            elif not base.get("evaluation_valid") or not support.get("evaluation_valid"):
                pair["status"] = "invalid_run"
            else:
                first = base.get("online_legal_next_action_rate")
                second = support.get("online_legal_next_action_rate")
                pair["choice_gain"] = (second - first if first is not None and second is not None else None)
                pair["completion_gain"] = int(bool(support.get("valid_success"))) - int(bool(base.get("valid_success")))
        details.append(pair)
    planned = len(details)
    matched = [row for row in details if row["status"] == "matched"]
    choice = [row["choice_gain"] for row in matched if row.get("choice_gain") is not None]
    return {"planned_pairs": planned, "matched_pairs": len(matched),
            "coverage": len(matched) / planned if planned else None,
            "official_gain_available": bool(planned and len(matched) == planned and len(choice) == planned),
            "paired_choice_gain": sum(choice) / planned if len(choice) == planned and planned else None,
            "paired_completion_gain": sum(row["completion_gain"] for row in matched) / planned
            if len(matched) == planned and planned else None,
            "pairs": details}
