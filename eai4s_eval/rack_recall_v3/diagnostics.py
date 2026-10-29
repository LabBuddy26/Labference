"""Policy boundary and Oracle diagnostics for rack-placement recall."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

from .core import (APPEARANCE_CONDITIONS, Contract, Instance, RackPose, SLOTS,
                   State, legal_actions, normalize_action)


SCHEMA_VERSION = "labference/rack-recall-diagnostic-v3"
PRIVATE_KEYS = {
    "target_tube", "target_tube_id", "actual_placement_slot", "correct_slot",
    "reference_slot", "evaluator_state", "success", "valid_success",
}


def _walk_private(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if normalized in PRIVATE_KEYS:
                raise ValueError(f"private evaluator field in policy input: {key}")
            _walk_private(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _walk_private(child)
    elif isinstance(value, str):
        lowered = value.lower()
        if any(re.search(rf"\b{re.escape(key)}\b\s*[:=]", lowered)
               for key in PRIVATE_KEYS):
            raise ValueError("serialized private evaluator field in policy input")


def policy_observation(images: Mapping[str, Any], joint_positions: list[float],
                       task_rule: str, authorized_history: list[dict[str, Any]],
                       *, evidence_step: int,
                       observation_id: str,
                       sim_time_s: float,
                       frame_ids: Mapping[str, str],
                       appearance_condition: str = "single",
                       history_delivery_mode: str = "online_once") -> dict[str, Any]:
    if not isinstance(images, Mapping) or not {"front", "wrist"} <= set(images):
        raise ValueError("front and wrist images are required")
    if not all(type(value) in (int, float) and math.isfinite(value)
               for value in joint_positions):
        raise ValueError("joint positions must be finite")
    if appearance_condition not in APPEARANCE_CONDITIONS:
        raise ValueError("unknown appearance condition")
    if history_delivery_mode not in {"online_once", "none"}:
        raise ValueError("unknown history delivery mode")
    if history_delivery_mode == "online_once" and authorized_history:
        raise ValueError("online_once observations must not replay historical frames")
    if (not isinstance(observation_id, str) or not observation_id or
            type(sim_time_s) not in (int, float) or not math.isfinite(sim_time_s) or
            sim_time_s < 0 or not isinstance(frame_ids, Mapping) or
            set(frame_ids) != set(images) or
            any(not isinstance(value, str) or not value for value in frame_ids.values())):
        raise ValueError("observation identity, time, and frame IDs are required")
    observation = {
        "schema_version": SCHEMA_VERSION,
        "images": dict(images),
        "joint_positions": list(map(float, joint_positions)),
        "task_rule": str(task_rule),
        "authorized_history": list(authorized_history),
        "evidence_step": int(evidence_step),
        "observation_id": observation_id,
        "sim_time_s": float(sim_time_s),
        "frame_ids": dict(frame_ids),
        "appearance_condition": appearance_condition,
        "history_delivery_mode": history_delivery_mode,
        "rack_convention": {
            "grid": "3x4", "rows": list("ABC"), "columns": [1, 2, 3, 4],
            "orientation_cues": {"row_A_long_edge": "blue",
                                 "column_1_short_edge": "yellow",
                                 "A1_corner": "black_white_checker"},
        },
    }
    check_policy_request(observation)
    return observation


def check_policy_request(request: Mapping[str, Any]) -> None:
    if not isinstance(request, Mapping) or request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid rack-recall policy request")
    allowed = {"schema_version", "images", "joint_positions", "task_rule",
               "authorized_history", "evidence_step", "rack_convention",
               "appearance_condition", "history_delivery_mode",
               "observation_id", "sim_time_s", "frame_ids",
               "oracle_context", "oracle_prompt", "selected_operation",
               "interaction_pose"}
    unknown = set(request) - allowed
    if unknown:
        raise ValueError(f"unknown policy fields: {sorted(unknown)}")
    if (request.get("history_delivery_mode") == "online_once" and
            request.get("authorized_history") != []):
        raise ValueError("online_once policy requests cannot replay history")
    images = request.get("images")
    frame_ids = request.get("frame_ids")
    if (not isinstance(images, Mapping) or not isinstance(frame_ids, Mapping) or
            set(frame_ids) != set(images) or
            not isinstance(request.get("observation_id"), str) or
            not request.get("observation_id") or
            type(request.get("sim_time_s")) not in (int, float) or
            not math.isfinite(request["sim_time_s"])):
        raise ValueError("policy request lacks auditable frame identity")
    _walk_private({key: value for key, value in request.items()
                   if key not in {"oracle_context", "oracle_prompt"}})
    if "oracle_context" in request:
        validate_oracle_context(request["oracle_context"])
    if "oracle_prompt" in request:
        prompt = request["oracle_prompt"]
        match = (None if not isinstance(prompt, str) else
                 re.fullmatch(r"<ORACLE(?: profile=([a-z]+))?>(.*)</ORACLE>",
                              prompt, flags=re.DOTALL))
        if match is None:
            raise ValueError("invalid serialized Oracle prompt")
        try:
            context = json.loads(match.group(2))
        except json.JSONDecodeError as exc:
            raise ValueError("invalid serialized Oracle payload") from exc
        validate_oracle_context(context)
        if match.group(1) is not None and context["profile"] != match.group(1):
            raise ValueError("serialized Oracle profile mismatch")


def visible_facts(state: State, instance: Instance | None = None) -> dict[str, Any]:
    """Encode visible occupancy without revealing which identical tube is the target."""
    color_by_slot = {}
    if instance is not None:
        colors = dict(instance.cap_colors)
        color_by_slot = {slot: colors[tube] for slot, tube in state.occupied.items()
                         if tube in colors}
    return {"phase": state.phase,
            "occupied_slots": sorted(state.occupied),
            "tube_count_by_slot": {slot: 1 for slot in sorted(state.occupied)},
            "visible_cap_color_by_slot": color_by_slot,
            "rack_orientation_cues_visible": state.environment_ready or state.phase == "placement"}


def candidate_geometry(contract: Contract, pose: RackPose) -> dict[str, Any]:
    return {"frame": "world", "units": "m",
            "rack_local_slot_labels": list(SLOTS),
            "slots": {slot: {"mouth_world_m": list(contract.slot_world_center(slot, pose)),
                              "insertion_axis_world": [0.0, 0.0, -1.0]}
                      for slot in SLOTS}}


def oracle_context(profile: str, contract: Contract, state: State,
                   pose: RackPose, selected_action: Mapping[str, Any] | None = None,
                   instance: Instance | None = None) -> dict[str, Any]:
    base = {"schema_version": SCHEMA_VERSION, "profile": profile}
    if profile == "semantic":
        base["facts"] = visible_facts(state, instance)
    elif profile == "decision":
        actions = legal_actions(contract, state)
        base["legal_operation"] = actions[0] if len(actions) == 1 else actions
    elif profile == "geometry":
        base["candidate_geometry"] = candidate_geometry(contract, pose)
    elif profile == "skill":
        if normalize_action(selected_action) is None:
            raise ValueError("Skill Oracle requires a selected operation")
        base["selected_operation"] = normalize_action(selected_action)
    elif profile != "none":
        raise ValueError(f"unknown Oracle profile: {profile}")
    validate_oracle_context(base)
    return base


def validate_oracle_context(context: Any) -> None:
    if not isinstance(context, Mapping) or context.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid rack-recall Oracle context")
    profile = context.get("profile")
    expected = {"none": set(), "semantic": {"facts"}, "decision": {"legal_operation"},
                "geometry": {"candidate_geometry"}, "skill": {"selected_operation"}}
    if profile not in expected or set(context) != {"schema_version", "profile"} | expected[profile]:
        raise ValueError("Oracle context fields do not match its profile")
    if profile == "semantic":
        try:
            _walk_private(context["facts"])
        except ValueError as exc:
            raise ValueError("Semantic Oracle must not identify the target tube") from exc
        facts = context["facts"]
        required_facts = {"phase", "occupied_slots", "tube_count_by_slot",
                          "visible_cap_color_by_slot",
                          "rack_orientation_cues_visible"}
        if (not isinstance(facts, Mapping) or set(facts) != required_facts or
                facts.get("phase") not in {"placement", "transition", "recall",
                                           "recovery", "complete"} or
                not isinstance(facts.get("occupied_slots"), list) or
                any(slot not in SLOTS for slot in facts["occupied_slots"]) or
                facts["occupied_slots"] != sorted(set(facts["occupied_slots"])) or
                facts.get("tube_count_by_slot") != {
                    slot: 1 for slot in facts["occupied_slots"]} or
                not isinstance(facts.get("visible_cap_color_by_slot"), Mapping) or
                not set(facts["visible_cap_color_by_slot"]) <= set(facts["occupied_slots"]) or
                type(facts.get("rack_orientation_cues_visible")) is not bool):
            raise ValueError("Semantic Oracle facts exceed the visible-state scope")
    if profile == "decision":
        operations = context["legal_operation"]
        operations = operations if isinstance(operations, list) else [operations]
        if not operations or any(normalize_action(operation) != operation
                                 for operation in operations):
            raise ValueError("Decision Oracle contains an invalid operation")
    if profile == "geometry":
        geometry = context["candidate_geometry"]
        required = {"frame", "units", "rack_local_slot_labels", "slots"}
        if (not isinstance(geometry, Mapping) or set(geometry) != required or
                geometry.get("frame") != "world" or geometry.get("units") != "m" or
                geometry.get("rack_local_slot_labels") != list(SLOTS) or
                set(geometry.get("slots", {})) != set(SLOTS)):
            raise ValueError("Geometry Oracle scope must expose all candidates without selection")
        for slot in SLOTS:
            sample = geometry["slots"][slot]
            if (not isinstance(sample, Mapping) or
                    set(sample) != {"mouth_world_m", "insertion_axis_world"} or
                    len(sample["mouth_world_m"]) != 3 or
                    len(sample["insertion_axis_world"]) != 3 or
                    not all(type(value) in (int, float) and math.isfinite(value)
                            for value in (*sample["mouth_world_m"],
                                          *sample["insertion_axis_world"]))):
                raise ValueError("Geometry Oracle contains an invalid candidate")
    if profile == "skill" and normalize_action(
            context["selected_operation"]) != context["selected_operation"]:
        raise ValueError("Skill Oracle contains an invalid selected operation")


def deliver_input(observation: Mapping[str, Any], context: Mapping[str, Any],
                  transport: str, supported: set[str]) -> dict[str, Any]:
    check_policy_request(observation)
    validate_oracle_context(context)
    if context["profile"] == "none":
        return dict(observation)
    if transport not in supported:
        raise ValueError(f"unsupported Oracle transport: {transport}")
    result = dict(observation)
    if "oracle_context" in result or "oracle_prompt" in result:
        raise ValueError("Oracle input is already present")
    if transport == "structured":
        result["oracle_context"] = dict(context)
    elif transport == "text":
        payload = json.dumps(context, sort_keys=True, separators=(",", ":"))
        result["oracle_prompt"] = f"<ORACLE profile={context['profile']}>{payload}</ORACLE>"
    else:
        raise ValueError("rack-recall runner supports structured or text transport")
    check_policy_request(result)
    return result


def score_report(report: Any, reference_slot: str, visible_slots: list[str]) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        return {"status": "format_error", "correct": False}
    slot = report.get("slot")
    if set(report) != {"slot"} or slot not in SLOTS:
        return {"status": "format_error", "correct": False}
    return {"status": "scored", "slot": slot,
            "correct": slot == reference_slot and slot in visible_slots}


class RackRecallOracleValidator:
    """Task-owned checks used by the shared intervention controller."""

    def validate_assist(self, profile: str, context: Mapping[str, Any],
                        observation: Mapping[str, Any]) -> None:
        validate_oracle_context(context)
        check_policy_request(observation)
        if context["profile"] != profile:
            raise ValueError("Oracle profile mismatch")

    def validate_replacement(self, profile: str, site: str, original: Any,
                             replacement: Any,
                             runtime_context: Mapping[str, Any]) -> None:
        if profile == "decision":
            normalized = normalize_action(replacement)
            if normalized is None:
                raise ValueError("invalid decision replacement")
            allowed = runtime_context.get("legal_actions")
            if allowed is not None and normalized not in allowed:
                raise ValueError("decision replacement is not legal in the current state")
        if profile == "semantic" and not isinstance(replacement, Mapping):
            raise ValueError("semantic replacement must be structured")
        if profile == "geometry" and not isinstance(replacement, Mapping):
            raise ValueError("geometry replacement must be structured")
        if profile == "skill" and normalize_action(runtime_context.get("selected_operation")) is None:
            raise ValueError("Skill replacement requires the selected operation")
