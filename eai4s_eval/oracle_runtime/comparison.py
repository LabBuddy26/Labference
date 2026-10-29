"""Pre-registered matched comparisons for assist and replace conditions."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .contracts import InterventionSpec


MATCH_FIELDS = (
    "task_version", "contract_hash", "instance_id", "policy_profile_id",
    "policy_profile_hash", "policy_revision", "sensor_condition", "action_spec_hash",
    "toolset_hash", "reset_protocol_hash", "memory_revision", "budget_id",
    "scene_layout_hash", "initial_state_hash", "scene_seed", "mass_seed", "pair_member",
)


def condition_id(spec: InterventionSpec) -> str:
    if spec.profile == "none":
        return "base"
    suffix = f":{spec.transport}" if spec.mode == "assist" else ""
    return f"{spec.profile}:{spec.mode}:{spec.site}{suffix}"


def plan_matched_conditions(*, task_version: str, contract_hash: str,
                            instance_ids: Iterable[str], policy_profile_id: str,
                            policy_profile_hash: str, policy_revision: str,
                            sensor_condition: str, action_spec_hash: str,
                            toolset_hash: str, reset_protocol_hash: str,
                            memory_revision: str, budget_id: str,
                            interventions: Iterable[InterventionSpec],
                            instance_metadata: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    specs = list(interventions)
    if not specs or any(spec.profile == "none" for spec in specs):
        raise ValueError("condition plan receives non-base interventions only")
    identifiers = [condition_id(spec) for spec in specs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("intervention conditions must be unique")
    shared = {"task_version": task_version, "contract_hash": contract_hash,
              "policy_profile_id": policy_profile_id, "policy_profile_hash": policy_profile_hash,
              "policy_revision": policy_revision,
              "sensor_condition": sensor_condition, "action_spec_hash": action_spec_hash,
              "toolset_hash": toolset_hash, "reset_protocol_hash": reset_protocol_hash,
              "memory_revision": memory_revision,
              "budget_id": budget_id}
    pairs = []
    for instance_id in instance_ids:
        if not instance_id:
            raise ValueError("instance ids must be non-empty")
        metadata = dict((instance_metadata or {}).get(instance_id, {}))
        unknown = set(metadata) - set(MATCH_FIELDS)
        if unknown:
            raise ValueError(f"unknown instance match fields: {sorted(unknown)}")
        for spec in specs:
            pairs.append({"pair_id": f"{instance_id}:{condition_id(spec)}", **shared, **metadata,
                          "instance_id": instance_id, "base_condition": "base",
                          "support_condition": condition_id(spec),
                          "intervention": spec.__dict__})
    return {"schema_version": "labference/matched-intervention-plan-v1", "pairs": pairs}


def compare_matched_conditions(plan: Mapping[str, Any], runs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    if plan.get("schema_version") != "labference/matched-intervention-plan-v1":
        raise ValueError("unknown matched intervention plan")
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for run in runs:
        key = (str(run.get("instance_id")), str(run.get("condition_id")))
        if key in lookup:
            raise ValueError(f"duplicate condition run: {key}")
        lookup[key] = run
    details = []
    for pair in plan.get("pairs", []):
        base = lookup.get((pair["instance_id"], "base"))
        supported = lookup.get((pair["instance_id"], pair["support_condition"]))
        row = {"pair_id": pair["pair_id"], "instance_id": pair["instance_id"],
               "condition_id": pair["support_condition"], "intervention": pair["intervention"],
               "both_present": bool(base and supported), "status": "missing"}
        if base and supported:
            for field in MATCH_FIELDS:
                if base.get(field) != pair.get(field) or supported.get(field) != pair.get(field):
                    raise ValueError(f"incomparable matched pair {pair['pair_id']}: {field}")
            if not base.get("evaluation_valid") or not supported.get("evaluation_valid"):
                row["status"] = "invalid_run"
            elif supported.get("condition_supported") is not True:
                row["status"] = "unsupported"
            else:
                row["status"] = "matched"
                for metric in ("online_legal_next_action_rate", "physical_completion"):
                    left, right = base.get(metric), supported.get(metric)
                    row[f"{metric}_gain"] = right - left if left is not None and right is not None else None
        details.append(row)
    planned = len(details)
    matched = [row for row in details if row["status"] == "matched"]
    grouped: dict[str, dict[str, Any]] = {}
    for condition in sorted({row["condition_id"] for row in details}):
        rows = [row for row in details if row["condition_id"] == condition]
        valid = [row for row in rows if row["status"] == "matched"]
        grouped[condition] = {
            "planned_pairs": len(rows), "matched_pairs": len(valid),
            "coverage": len(valid) / len(rows) if rows else None,
            "paired_legal_choice_gain": (
                sum(row["online_legal_next_action_rate_gain"] for row in valid) / len(rows)
                if len(valid) == len(rows) and all(row.get("online_legal_next_action_rate_gain") is not None
                                                   for row in valid) else None),
            "paired_completion_gain": (
                sum(row["physical_completion_gain"] for row in valid) / len(rows)
                if len(valid) == len(rows) and all(row.get("physical_completion_gain") is not None
                                                   for row in valid) else None),
        }
    return {"planned_pairs": planned, "matched_pairs": len(matched),
            "coverage": len(matched) / planned if planned else None,
            "conditions": grouped, "pairs": details}
