"""Input isolation, scoped Oracles, and accessible weighing checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Callable

from .core import Contract, Instance, ORACLES, State, WEIGHTS, legal_actions, normalize_action


SCHEMA_VERSION = "labference/weighing-diagnostic-v1"
LAYERS = ("semantic", "decision", "geometry", "execution")
BASE_FIELDS = frozenset({
    "images", "joint_positions", "task_rule", "visible_weight_labels_g",
    "display_reading_g", "stable_indicator", "history", "calibration",
})
LOCATIONS = frozenset({"pan", "table", "gripper"})
_PRIVATE_KEY = re.compile(
    r"(?:hidden|sample_mass|desired_weights|correct_weight|answer|ground_truth|"
    r"evaluator|success_label|task_success|legal_actions|target_(?:world_)?(?:pose|position|frame)|"
    r"goal_(?:pose|position|frame))", re.IGNORECASE
)
_PRIVATE_TEXT = re.compile(
    r"\b(?:hidden[_ ]mass|sample[_ ]mass|desired[_ ]weights|correct[_ ]weight|"
    r"ground[_ ]truth|evaluator[_ ]state|success[_ ]label|task[_ ]success|"
    r"legal[_ ]actions|target[_ ]world[_ ]pose)(?:_g)?\b", re.IGNORECASE
)
_SAMPLE_MASS_TEXT = re.compile(
    r"\b(?:sample|preload)\b.{0,32}\b(?:weighs?|mass|is)\b\s*[:=]?\s*"
    r"\d+(?:\.\d+)?\s*(?:g|grams)\b", re.IGNORECASE
)
_ANSWER_IN_RULE = re.compile(
    r"\b(?:add|remove|choose|select)\s+(?:the\s+)?(?:10|20|50|100)\s*g\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PolicyCapabilities:
    reports: frozenset[str]
    oracle_transports: frozenset[str]
    action_space: str = "array"
    resettable: bool = True

    def __post_init__(self) -> None:
        if not self.reports <= set(LAYERS) or not self.oracle_transports <= {"structured", "text"}:
            raise ValueError("unknown policy report or Oracle transport")
        if self.action_space not in {"array", "symbolic"}:
            raise ValueError("unknown policy action space")


@dataclass(frozen=True)
class VerifiedSkillExecutor:
    """A separately validated reference executor and independent physical check."""

    validation_id: str
    validated_operations: frozenset[str]
    execute: Callable[[dict[str, Any]], dict[str, Any]]
    verify_subgoal: Callable[[dict[str, Any]], bool]
    verified: bool = False

    def __post_init__(self) -> None:
        if (not self.verified or not isinstance(self.validation_id, str) or
                not self.validation_id.strip() or not isinstance(self.validated_operations, frozenset) or
                not self.validated_operations or not self.validated_operations <= {"add", "remove"} or
                not callable(self.execute) or not callable(self.verify_subgoal)):
            raise ValueError("Skill executor needs physical validation and a verifier")


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _weights(value: Any, *, all_weights: bool = False) -> list[int]:
    if (not isinstance(value, (list, tuple)) or any(type(weight) is not int or weight not in WEIGHTS
                                                    for weight in value) or len(set(value)) != len(value)):
        raise ValueError("weight labels must be distinct known integer grams")
    if all_weights and set(value) != set(WEIGHTS):
        raise ValueError("geometry must include every standard weight")
    return sorted(value)


def _inspect_private(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str) or _PRIVATE_KEY.search(key):
                raise ValueError(f"private evaluator field in policy input: {key}")
            _inspect_private(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _inspect_private(nested)
    elif isinstance(value, str) and _PRIVATE_TEXT.search(value):
        raise ValueError("private evaluator label in policy text")


def _vector(value: Any, count: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != count or not all(
            _finite_number(item) for item in value):
        raise ValueError(f"geometry needs {count} finite coordinates")
    return tuple(float(item) for item in value)


def _frame(value: Any) -> dict[str, list[float]]:
    if not isinstance(value, dict) or set(value) != {"position_world_m", "quaternion_wxyz"}:
        raise ValueError("geometry frame needs position_world_m and quaternion_wxyz")
    position = _vector(value["position_world_m"], 3)
    quaternion = _vector(value["quaternion_wxyz"], 4)
    if abs(math.sqrt(sum(item * item for item in quaternion)) - 1.0) > 1e-3:
        raise ValueError("geometry quaternion must be normalized")
    return {"position_world_m": list(position), "quaternion_wxyz": list(quaternion)}


def _candidate_geometry(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
            "frame", "units", "grasp_frames", "pan_frames", "return_frames"}:
        raise ValueError("geometry Oracle needs all grasp, pan, and return frames")
    if value["frame"] != "world" or value["units"] != "m":
        raise ValueError("candidate geometry must use world-frame meters")
    result: dict[str, Any] = {"frame": "world", "units": "m"}
    expected = {str(weight) for weight in WEIGHTS}
    for group in ("grasp_frames", "pan_frames", "return_frames"):
        frames = value[group]
        if not isinstance(frames, dict) or set(frames) != expected:
            raise ValueError(f"{group} must describe all standard weights")
        result[group] = {key: _frame(frames[key]) for key in sorted(expected, key=int)}
    return result


def _reference_geometry(candidates: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    kind, weight = selected["type"], str(selected["weight_g"])
    return {
        "operation": selected,
        "frame": "world", "units": "m",
        "grasp_frame": candidates["grasp_frames"][weight],
        "place_frame": candidates["pan_frames" if kind == "add" else "return_frames"][weight],
    }


def _geometry_report(value: Any, selected: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
            "operation", "frame", "units", "grasp_frame", "place_frame"}:
        raise ValueError("geometry report needs an operation and two frames")
    if (normalize_action(value["operation"]) != selected or value["frame"] != "world" or
            value["units"] != "m"):
        raise ValueError("geometry report changed the selected operation or frame")
    return {"operation": selected, "frame": "world", "units": "m",
            "grasp_frame": _frame(value["grasp_frame"]),
            "place_frame": _frame(value["place_frame"])}


def _visible_facts(observation: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(facts, dict) or set(facts) - {
            "visible_weight_locations_g", "sample_visible_on_pan",
            "display_reading_g", "stable_indicator"}:
        raise ValueError("semantic Oracle accepts only current visible facts")
    locations = facts.get("visible_weight_locations_g", {})
    if (not isinstance(locations, dict) or any(
            not isinstance(weight, str) or weight not in {str(w) for w in observation["visible_weight_labels_g"]} or
            location not in LOCATIONS for weight, location in locations.items())):
        raise ValueError("semantic locations must refer only to visible weights")
    reading = facts.get("display_reading_g", observation["display_reading_g"])
    stable = facts.get("stable_indicator", observation["stable_indicator"])
    if not (reading is None or _finite_number(reading)) or type(stable) is not bool:
        raise ValueError("semantic readout fact must have a finite reading and Boolean stability")
    result: dict[str, Any] = {
        "display_reading_g": reading,
        "stable_indicator": stable,
        "visible_weight_locations_g": {key: locations[key] for key in sorted(locations, key=int)},
    }
    if "sample_visible_on_pan" in facts:
        if type(facts["sample_visible_on_pan"]) is not bool:
            raise ValueError("sample visibility must be Boolean")
        result["sample_visible_on_pan"] = facts["sample_visible_on_pan"]
    return result


def policy_observation(
    images: dict[str, Any], joint_positions: list[float], task_rule: str,
    visible_weight_labels_g: list[int], display_reading_g: float | None,
    stable_indicator: bool | None, history: list[dict[str, Any]] | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete allowlisted native policy input, with no evaluator state."""
    request = {
        "images": images, "joint_positions": joint_positions, "task_rule": task_rule,
        "visible_weight_labels_g": visible_weight_labels_g,
        "display_reading_g": display_reading_g,
        "stable_indicator": stable_indicator,
        "history": history if history is not None else [],
        "calibration": calibration if calibration is not None else {},
    }
    check_policy_request(request)
    return request


def _check_oracle_context(context: Any) -> None:
    if not isinstance(context, dict) or context.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid Oracle schema")
    profile = context.get("profile")
    fields = {
        "semantic": {"facts"}, "decision": {"advice"},
        "geometry": {"candidate_geometry"},
        "skill": {"service", "requested_operation", "validation_id"},
    }
    if profile not in fields or set(context) != {"schema_version", "profile"} | fields[profile]:
        raise ValueError("Oracle contains unexpected or missing fields")
    if profile == "semantic":
        if (not isinstance(context["facts"], dict) or
                not {"display_reading_g", "stable_indicator", "visible_weight_locations_g"} <= set(context["facts"]) or
                set(context["facts"]) - {
                    "display_reading_g", "stable_indicator", "visible_weight_locations_g", "sample_visible_on_pan"}):
            raise ValueError("semantic Oracle contains non-visible facts")
        locations = context["facts"].get("visible_weight_locations_g")
        if (not isinstance(locations, dict) or any(
                not isinstance(weight, str) or weight not in {str(w) for w in WEIGHTS} or location not in LOCATIONS
                for weight, location in locations.items())):
            raise ValueError("invalid visible weight facts")
        if (not (context["facts"].get("display_reading_g") is None or
                 _finite_number(context["facts"]["display_reading_g"])) or
                type(context["facts"].get("stable_indicator")) is not bool):
            raise ValueError("invalid visible display facts")
        if "sample_visible_on_pan" in context["facts"] and type(
                context["facts"]["sample_visible_on_pan"]) is not bool:
            raise ValueError("invalid sample visibility")
    elif profile == "decision":
        if normalize_action(context["advice"]) is None:
            raise ValueError("invalid symbolic Oracle advice")
    elif profile == "geometry":
        _candidate_geometry(context["candidate_geometry"])
    elif (context["service"] != "verified_reference_executor" or
          not isinstance(context["validation_id"], str) or not context["validation_id"] or
          normalize_action(context["requested_operation"]) is None):
        raise ValueError("invalid Skill Oracle request")
    _inspect_private(context)


def _check_checkpoint_context(context: Any) -> None:
    if not isinstance(context, dict) or context.get("layer") not in LAYERS:
        raise ValueError("invalid checkpoint context")
    layer = context["layer"]
    if layer == "decision":
        if set(context) != {"layer", "standard_state"}:
            raise ValueError("decision checkpoint needs standard state")
        state = context["standard_state"]
        if (not isinstance(state, dict) or set(state) != {
                "display_reading_g", "stable_indicator", "supported_weight_labels_g"} or
                not _finite_number(state["display_reading_g"]) or
                type(state["stable_indicator"]) is not bool):
            raise ValueError("invalid standard state")
        _weights(state["supported_weight_labels_g"])
    elif layer in {"geometry", "execution"}:
        expected = {"layer", "selected_operation"}
        if layer == "execution":
            expected.add("reference_geometry")
        if set(context) != expected:
            raise ValueError("fixed-subgoal checkpoint has wrong fields")
        selected = normalize_action(context["selected_operation"])
        if selected is None or selected["type"] not in {"add", "remove"}:
            raise ValueError("checkpoint needs a selected add/remove operation")
        if layer == "execution":
            _geometry_report(context["reference_geometry"], selected)
    elif set(context) != {"layer"}:
        raise ValueError("semantic checkpoint has unexpected fields")
    _inspect_private(context)


def check_policy_request(request: dict[str, Any], *, allow_checkpoint: bool = False) -> None:
    """Validate the actual serialized boundary, including nested history and metadata."""
    if not isinstance(request, dict) or not BASE_FIELDS <= set(request):
        raise ValueError("policy request is missing required public fields")
    allowed = BASE_FIELDS | {"oracle_context", "oracle_prompt"}
    if allow_checkpoint:
        allowed |= {"checkpoint_context"}
    if set(request) - allowed or {"oracle_context", "oracle_prompt"} <= set(request):
        raise ValueError("unexpected field or duplicate Oracle delivery")
    if (not isinstance(request["images"], dict) or
            not {"front", "wrist"} <= set(request["images"]) or
        set(request["images"]) - {"front", "wrist", "display", "weights"}):
        raise ValueError("front and wrist images are required; display and weights are optional registered views")
    for image in request["images"].values():
        if isinstance(image, str):
            continue
        if (not isinstance(image, dict) or set(image) - {"base64", "mime_type", "width", "height"} or
                not isinstance(image.get("base64"), str)):
            raise ValueError("image must contain only base64 data and public dimensions")
        if ("mime_type" in image and image["mime_type"] not in {"image/png", "image/jpeg", "image/webp"}) or any(
                key in image and (type(image[key]) is not int or image[key] <= 0)
                for key in ("width", "height")):
            raise ValueError("invalid image metadata")
    joints = request["joint_positions"]
    if not isinstance(joints, (list, tuple)) or len(joints) < 8 or not all(
            _finite_number(value) for value in joints):
        raise ValueError("joint_positions need at least eight finite values")
    if not isinstance(request["task_rule"], str) or not request["task_rule"].strip():
        raise ValueError("task_rule must be nonempty text")
    if _SAMPLE_MASS_TEXT.search(request["task_rule"]) or _ANSWER_IN_RULE.search(request["task_rule"]):
        raise ValueError("task_rule contains an instance answer")
    _weights(request["visible_weight_labels_g"])
    reading = request["display_reading_g"]
    if reading is not None and not _finite_number(reading):
        raise ValueError("display reading must be finite or null")
    if request["stable_indicator"] is not None and type(request["stable_indicator"]) is not bool:
        raise ValueError("stable indicator must be Boolean or omitted from the direct sensor channel")
    if not isinstance(request["history"], list) or not isinstance(
            request["calibration"], dict):
        raise ValueError("invalid stable indicator, history, or calibration")
    _inspect_private({key: request[key] for key in BASE_FIELDS if key != "images"})
    if "checkpoint_context" in request:
        _check_checkpoint_context(request["checkpoint_context"])
    if "oracle_prompt" in request:
        prompt = request["oracle_prompt"]
        if not isinstance(prompt, str) or not prompt.startswith("<ORACLE>") or not prompt.endswith("</ORACLE>"):
            raise ValueError("invalid Oracle text transport")
        try:
            context = json.loads(prompt[len("<ORACLE>"):-len("</ORACLE>")])
        except json.JSONDecodeError as exc:
            raise ValueError("invalid Oracle JSON") from exc
    else:
        context = request.get("oracle_context")
    if context is not None:
        _check_oracle_context(context)
        if context["profile"] == "semantic":
            facts = context["facts"]
            if ((reading is not None and facts["display_reading_g"] != reading) or
                    (request["stable_indicator"] is not None and
                     facts["stable_indicator"] != request["stable_indicator"]) or
                    set(facts["visible_weight_locations_g"]) - {
                        str(weight) for weight in request["visible_weight_labels_g"]}):
                raise ValueError("semantic Oracle contradicts the current visible observation")
        if context["profile"] == "geometry" and set(request["visible_weight_labels_g"]) != set(WEIGHTS):
            raise ValueError("geometry Oracle cannot expose locations of unobserved weights")
    try:
        json.dumps(request, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("policy request must be finite JSON data") from exc


def oracle_context(
    profile: str, contract: Contract, instance: Instance, state: State,
    observation: dict[str, Any], visible_facts: dict[str, Any],
    candidate_geometry: dict[str, Any], selected_action: dict[str, Any] | None = None,
    *, validation_id: str | None = None,
) -> dict[str, Any]:
    if profile not in ORACLES:
        raise ValueError(f"unknown Oracle profile: {profile}")
    if profile == "none":
        return {}
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "profile": profile}
    if profile == "semantic":
        result["facts"] = _visible_facts(observation, visible_facts)
    elif profile == "decision":
        result["advice"] = legal_actions(contract, instance, state)[0]
    elif profile == "geometry":
        if set(observation["visible_weight_labels_g"]) != set(WEIGHTS):
            raise ValueError("geometry Oracle requires all standard weights to be visible")
        result["candidate_geometry"] = _candidate_geometry(candidate_geometry)
    else:
        selected = normalize_action(selected_action)
        if selected is None or selected["type"] not in {"add", "remove"} or not validation_id:
            raise ValueError("Skill Oracle needs an exact add/remove operation and verified executor")
        result.update(service="verified_reference_executor", requested_operation=selected,
                      validation_id=validation_id)
    _check_oracle_context(result)
    return result


def deliver_input(observation: dict[str, Any], context: dict[str, Any],
                  transport: str, supports: set[str], *,
                  checkpoint_context: dict[str, Any] | None = None) -> dict[str, Any]:
    if transport not in {"structured", "text"} or transport not in supports:
        raise ValueError(f"unsupported Oracle transport: {transport}")
    check_policy_request(observation)
    request = dict(observation)
    if checkpoint_context is not None:
        _check_checkpoint_context(checkpoint_context)
        request["checkpoint_context"] = checkpoint_context
    if context:
        _check_oracle_context(context)
        if transport == "structured":
            request["oracle_context"] = context
        else:
            request["oracle_prompt"] = "<ORACLE>" + json.dumps(
                context, sort_keys=True, separators=(",", ":"), allow_nan=False
            ) + "</ORACLE>"
    check_policy_request(request, allow_checkpoint=checkpoint_context is not None)
    return request


def _orientation_error_deg(first: dict[str, Any], second: dict[str, Any]) -> float:
    qa = first["quaternion_wxyz"]
    qb = second["quaternion_wxyz"]
    dot = abs(sum(a * b for a, b in zip(qa, qb)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def score_checkpoint(record: dict[str, Any]) -> dict[str, Any]:
    """Score only the accessible layer, never infer a model's hidden module state."""
    if not isinstance(record.get("instance_id"), str) or record.get("layer") not in LAYERS:
        raise ValueError("checkpoint needs an instance id and layer")
    layer = record["layer"]
    result: dict[str, Any] = {"instance_id": record["instance_id"], "layer": layer}
    for name in LAYERS:
        result[name] = "unsupported"
    if not record.get("supported", False):
        return result
    if layer == "semantic":
        expected = record.get("expected_semantic")
        report = record.get("semantic_report")
        result[layer] = "pass" if isinstance(report, dict) and report == expected else "fail"
    elif layer == "decision":
        chosen = normalize_action(record.get("committed_action"))
        legal = [normalize_action(action) for action in record.get("legal_actions", [])]
        result[layer] = "pass" if chosen is not None and chosen in legal else "fail"
        result["decision_type"] = "pass" if chosen is not None and any(
            action is not None and action["type"] == chosen["type"] for action in legal
        ) else "fail"
    elif layer == "geometry":
        selected = normalize_action(record.get("selected_action"))
        reference = record.get("reference_geometry")
        if selected is None or selected["type"] not in {"add", "remove"} or reference is None:
            result[layer] = "unavailable"
        else:
            try:
                actual = _geometry_report(record.get("geometry_report"), selected)
                expected = _geometry_report(reference, selected)
                for name in ("grasp", "place"):
                    first, second = actual[f"{name}_frame"], expected[f"{name}_frame"]
                    result[f"{name}_position_error_m"] = math.dist(
                        first["position_world_m"], second["position_world_m"]
                    )
                    result[f"{name}_orientation_error_deg"] = _orientation_error_deg(first, second)
                result[layer] = "pass" if all(
                    result[f"{name}_position_error_m"] <= record.get("position_tolerance_m", 0.01) and
                    result[f"{name}_orientation_error_deg"] <= record.get("orientation_tolerance_deg", 10.0)
                    for name in ("grasp", "place")
                ) else "fail"
            except (KeyError, TypeError, ValueError, OverflowError):
                result[layer] = "fail"
    else:
        observed = record.get("physical_subgoal_confirmed")
        result[layer] = "pass" if observed is True else "fail" if observed is False else "unavailable"
    return result


def summarize_checkpoint_scores(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    """Unsupported checkpoints never enter a layer's accuracy denominator."""
    summary: dict[str, Any] = {}
    for layer in LAYERS:
        counts = {name: 0 for name in ("pass", "fail", "unsupported", "unavailable")}
        for checkpoint in checkpoints:
            status = checkpoint.get(layer, "unsupported")
            if status not in counts:
                raise ValueError(f"unknown {layer} status: {status}")
            counts[status] += 1
        denominator = counts["pass"] + counts["fail"]
        summary[layer] = {**counts, "denominator": denominator,
                          "accuracy": counts["pass"] / denominator if denominator else None}
    return summary


def evaluate_policy_checkpoint(
    contract: Contract, instance: Instance, state: State, observation: dict[str, Any],
    visible_facts: dict[str, Any], candidate_geometry: dict[str, Any],
    policy: Callable[[dict[str, Any]], dict[str, Any]], capabilities: PolicyCapabilities,
    *, instance_id: str, layer: str, profile: str = "none", transport: str = "structured",
    selected_action: dict[str, Any] | None = None,
    physical_subgoal_confirmed: bool | None = None,
    reset_policy: Callable[[], None] | None = None,
    skill_executor: VerifiedSkillExecutor | None = None,
) -> dict[str, Any]:
    """Run one resettable, fixed-state checkpoint through a declared policy interface."""
    if layer not in LAYERS or profile not in ORACLES:
        raise ValueError("unknown checkpoint layer or Oracle profile")
    if profile == "skill" and layer != "execution":
        raise ValueError("Skill Oracle is only defined for physical execution")
    if profile != "none" and transport not in capabilities.oracle_transports:
        return {"instance_id": instance_id, "status": "unsupported", "reason": "oracle_transport"}
    if not capabilities.resettable or reset_policy is None:
        return {"instance_id": instance_id, "status": "unsupported", "reason": "reset"}
    if layer != "execution" and layer not in capabilities.reports:
        return {"instance_id": instance_id, "status": "unsupported", "reason": "report"}
    selected = normalize_action(selected_action)
    if layer in {"geometry", "execution"} or profile == "skill":
        if selected is None or selected["type"] not in {"add", "remove"}:
            return {"instance_id": instance_id, "status": "unsupported", "reason": "selected_action_required"}
    if profile == "skill" and (not isinstance(skill_executor, VerifiedSkillExecutor) or
                               selected["type"] not in skill_executor.validated_operations):
        return {"instance_id": instance_id, "status": "unsupported", "reason": "verified_executor_required"}
    check_policy_request(observation)
    candidates = _candidate_geometry(candidate_geometry) if layer in {"geometry", "execution"} or profile == "geometry" else None
    checkpoint_context: dict[str, Any] | None = {"layer": "semantic"} if layer == "semantic" else None
    if layer == "decision":
        checkpoint_context = {"layer": layer, "standard_state": {
            "display_reading_g": state.displayed_g,
            "stable_indicator": state.reading_stable,
            "supported_weight_labels_g": list(state.supported_weights_g),
        }}
    elif layer == "geometry":
        checkpoint_context = {"layer": layer, "selected_operation": selected}
    elif layer == "execution":
        checkpoint_context = {"layer": layer, "selected_operation": selected,
                              "reference_geometry": _reference_geometry(candidates, selected)}
    context = oracle_context(
        profile, contract, instance, state, observation, visible_facts,
        candidate_geometry, selected, validation_id=skill_executor.validation_id if profile == "skill" else None
    )
    request = (dict(observation, checkpoint_context=checkpoint_context) if profile == "none" and checkpoint_context
               else dict(observation) if profile == "none" else
               deliver_input(observation, context, transport, set(capabilities.oracle_transports),
                             checkpoint_context=checkpoint_context))
    check_policy_request(request, allow_checkpoint=checkpoint_context is not None)
    try:
        reset_policy()
    except Exception as exc:
        return {"instance_id": instance_id, "status": "service_error", "phase": "reset",
                "error_type": type(exc).__name__, "message": str(exc)}
    try:
        response = policy(request)
    except Exception as exc:
        return {"instance_id": instance_id, "status": "service_error", "phase": "policy",
                "error_type": type(exc).__name__, "message": str(exc)}
    if not isinstance(response, dict):
        response = {}
    reference_executor_success: bool | None = None
    if profile == "skill":
        if ("committed_action" in response and
                normalize_action(response["committed_action"]) != selected):
            return {"instance_id": instance_id, "status": "invalid_selected_operation",
                    "reason": "policy_commitment_differs_from_selected"}
        try:
            result = skill_executor.execute(dict(selected))
            if not isinstance(result, dict) or normalize_action(result.get("executed_action")) != selected:
                return {"instance_id": instance_id, "status": "invalid_executor_result",
                        "reason": "executed_operation_differs_from_selected"}
            observed = skill_executor.verify_subgoal(dict(selected))
            if type(observed) is not bool:
                raise ValueError("physical subgoal verifier must return bool")
            reference_executor_success = observed
        except Exception as exc:
            return {"instance_id": instance_id, "status": "service_error", "phase": "reference_executor",
                    "error_type": type(exc).__name__, "message": str(exc)}
    expected_semantic = _visible_facts(observation, visible_facts) if layer == "semantic" else None
    record = {
        "instance_id": instance_id, "layer": layer,
        "supported": layer == "execution" or layer in capabilities.reports,
        "expected_semantic": expected_semantic,
        "semantic_report": response.get("semantic_report"),
        "legal_actions": legal_actions(contract, instance, state),
        "committed_action": response.get("committed_action"),
        "selected_action": selected,
        "reference_geometry": _reference_geometry(candidates, selected) if layer == "geometry" else None,
        "geometry_report": response.get("geometry_report"),
        "physical_subgoal_confirmed": (reference_executor_success if profile == "skill"
                                       else physical_subgoal_confirmed),
    }
    return {"instance_id": instance_id, "status": "scored", "layer": layer, "profile": profile,
            "request": request, "response": response,
            "reference_executor_validation_id": skill_executor.validation_id if profile == "skill" else None,
            "reference_executor_success": reference_executor_success,
            "scores": score_checkpoint(record)}


__all__ = [
    "PolicyCapabilities", "VerifiedSkillExecutor", "policy_observation", "check_policy_request",
    "oracle_context", "deliver_input", "evaluate_policy_checkpoint", "score_checkpoint",
    "summarize_checkpoint_scores", "SCHEMA_VERSION", "LAYERS",
]
