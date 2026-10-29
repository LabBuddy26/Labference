from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


CATALOG_PATH = Path(__file__).resolve().parent / "configs" / "leaderboards" / "laboratory_benchmark_v1.json"

ATOMIC_ACTIONS = (
    ("grasp", "抓取"),
    ("place", "放置"),
    ("press", "按压"),
    ("pour", "倾倒"),
    ("open", "打开"),
    ("close", "关闭"),
    ("move", "移动"),
    ("stir", "搅拌"),
    ("shake", "晃动"),
    ("lift", "抬升"),
    ("twist", "旋拧"),
)

TRACK_ORDER = ("long_horizon", "reasoning_in_action")
SPLIT_PRIORITY = ("sim_randomized", "sim_clean", "sim_dev")
NON_LEADERBOARD_TASK_IDS: set[str] = set()

# Only exact semantic matches enter the formal L/R benchmark.  Similar legacy
# tasks remain visible in the EAI4S suite but do not inflate benchmark scores.
BENCHMARK_TASK_BINDINGS = {
    "level3_R01_scaledigital": "R01",
    "level2_L01_sampletransport": "L01",
    "level2_L02_sampleorganition": "L02",
    "level2_L03_singlepipetting": "L03",
    "level4_L04_LiquidMixing": "L04",
    "level2_L05_formulation_spotting": "L05",
    "level2_L06_wights": "L06",
    "level2_L07_centrifuge_load": "L07",
    "level2_L08_NMRInjection": "L08",
    "level4_L09_DeviceOperation": "L09",
    "level2_L10_cohort_analysis": "L10",
    "level4_L11_ConditionModule": "L11",
    "level4_L12_clearreset": "L12",
    "level2_R02_wights": "R02",
    "level3_R03_temperaturepanel": "R03",
    "level3_R04_centrifugepanel": "R04",
    "level3_R05_devicestatus": "R05",
    "level2_R06_centrifuge_load": "R06",
    "level2_R07_centrifuge_load": "R07",
    "level2_R08_NMRSample": "R08",
    "level2_R09_NMRRack": "R09",
    "level3_R10_rackplace": "R10",
    "level2_R11_plateposition": "R11",
    "level3_R12_devicecover_status": "R12",
}

SUPPORTED_TASK_SPECS = {
    "L03": "../tasks/B30_level2_L03_singlepipetting.yaml",
    "L05": "../tasks/B31_level2_L05_formulation_spotting.yaml",
    "L07": "../tasks/B14_level2_centrifuge_load.yaml",
    "R01": "../tasks/B28_level3_R01_scaledigital.yaml",
    "R03": "../tasks/B25_level3_R03_temperaturepanel.yaml",
    "R04": "../tasks/B29_level3_R04_centrifugepanel.yaml",
    "R05": "../tasks/B26_level3_R05_devicestatus.yaml",
    "R06": "../tasks/B32_level2_R06_centrifuge_opposite_slot.yaml",
    "R07": "../tasks/B33_level2_R07_centrifuge_symmetric_pair.yaml",
    "R08": "../tasks/B23_level2_R08_nmrsample.yaml",
    "R09": "../tasks/B24_level2_R09_nmrrack.yaml",
    "R11": "../tasks/B27_level2_R11_plateposition.yaml",
    "R12": "../tasks/B34_level3_R12_devicecover_status.yaml",
}

# Predicate-derived compatibility for existing result files.  New result
# producers may write explicit ``atomic_outcomes`` and bypass this table.
ATOMIC_PREDICATE_BINDINGS: dict[str, list[dict[str, Any]]] = {
    "level1_pick": [
        {"action": "grasp", "keys": ["grasped"]},
        {"action": "lift", "keys": ["lifted"]},
    ],
    "level1_pick_liquid": [
        {"action": "grasp", "keys": ["grasped"]},
        {"action": "lift", "keys": ["lifted"]},
    ],
    "level1_place": [
        {"action": "move", "keys": ["goal_pass"]},
        {"action": "place", "keys": ["goal_pass", "terminal_pass"]},
    ],
    "level1_pour": [{"action": "pour", "keys": ["sequence_pass"]}],
    "level2_pour": [{"action": "pour", "keys": ["sequence_pass"]}],
    "level1_open_door": [{"action": "open", "keys": ["goal_pass"]}],
    "level1_close_door": [{"action": "close", "keys": ["goal_pass"]}],
    "level1_stir": [{"action": "stir", "keys": ["goal_pass"]}],
    "level3_place": [
        {"action": "move", "keys": ["aligned"]},
        {"action": "place", "keys": ["inserted", "aligned"]},
    ],
    "level2_L01_sampletransport": [
        {"action": "grasp", "keys": ["goal_pass"]},
        {"action": "move", "keys": ["goal_pass"]},
        {"action": "place", "keys": ["goal_pass", "terminal_pass"]},
    ],
    "level4_L09_DeviceOperation": [
        {"action": "open", "keys": ["device_door_opened"]},
        {"action": "place", "keys": ["device_beaker_inside", "device_beaker3_placed"]},
        {"action": "press", "keys": ["button_pressed"]},
    ],
    "level2_L08_NMRInjection": [{"action": "place", "keys": ["goal_pass", "terminal_pass"]}],
    "level2_L07_centrifuge_load": [{"action": "place", "keys": ["goal_pass", "terminal_pass"]}],
    "level2_L10_cohort_analysis": [{"action": "place", "keys": ["goal_pass", "terminal_pass"]}],
    "level2_L02_sampleorganition": [{"action": "place", "keys": ["goal_pass", "terminal_pass"]}],
    "level2_L06_wights": [{"action": "place", "keys": ["goal_pass", "terminal_pass"]}],
    "level4_L11_ConditionModule": [
        {"action": "place", "keys": ["goal_pass"]},
        {"action": "press", "keys": ["button_pressed"]},
    ],
    "level4_L04_LiquidMixing": [{"action": "pour", "keys": ["goal_pass"]}],
    "level4_L12_clearreset": [{"action": "place", "keys": ["goal_pass", "terminal_pass"]}],
    "level2_R02_wights": [
        {"action": "grasp", "keys": ["goal_pass"]},
        {"action": "place", "keys": ["goal_pass", "terminal_pass"]},
    ],
    "level3_R10_rackplace": [
        {"action": "move", "keys": ["aligned"]},
        {"action": "place", "keys": ["inserted", "aligned"]},
    ],
    "T02_franka_pick_lift_proxy": [
        {"action": "grasp", "keys": ["grasped"]},
        {"action": "lift", "keys": ["lifted"]},
    ],
    "T03_tube_insert_rack": [
        {"action": "move", "keys": ["aligned"]},
        {"action": "place", "keys": ["inserted", "aligned"]},
    ],
}


def load_benchmark_catalog(path: str | Path | None = None) -> dict[str, Any]:
    catalog_path = Path(path) if path else CATALOG_PATH
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    # Catalog v1 included an aspirational Fine Manipulation track.  The public
    # benchmark now publishes only tasks for which collection and policy
    # training are supported.  Filtering at load time also keeps archived v1
    # result bundles readable without allowing F-task rows into current scores.
    data["tracks"] = [
        item for item in data.get("tracks") or []
        if item.get("track_id") in TRACK_ORDER
    ]
    data["tasks"] = [
        item for item in data.get("tasks") or []
        if item.get("track_id") in TRACK_ORDER
    ]
    for item in data["tasks"]:
        task_id = str(item.get("task_id") or "")
        if task_id in NON_LEADERBOARD_TASK_IDS:
            item["status"] = "planned"
        if task_id in SUPPORTED_TASK_SPECS:
            item["status"] = "implemented"
            item["task_spec"] = SUPPORTED_TASK_SPECS[task_id]
    data["version"] = "2.0.0"
    validate_benchmark_catalog(data)
    return data


def validate_benchmark_catalog(data: dict[str, Any]) -> None:
    tracks = data.get("tracks") or []
    tasks = data.get("tasks") or []
    actions = data.get("atomic_actions") or []
    errors: list[str] = []
    if [item.get("track_id") for item in tracks] != list(TRACK_ORDER):
        errors.append("tracks must use the canonical L/R order")
    task_ids = [str(item.get("task_id") or "") for item in tasks]
    expected_task_count = 12 * len(TRACK_ORDER)
    if len(tasks) != expected_task_count or len(set(task_ids)) != expected_task_count:
        errors.append(f"catalog must contain {expected_task_count} unique tasks")
    for prefix, track_id in (("L", "long_horizon"), ("R", "reasoning_in_action")):
        expected = {f"{prefix}{index:02d}" for index in range(1, 13)}
        actual = {item["task_id"] for item in tasks if item.get("track_id") == track_id}
        if actual != expected:
            errors.append(f"{track_id} must contain {prefix}01-{prefix}12")
    action_ids = [str(item.get("action_id") or "") for item in actions]
    if action_ids != [item[0] for item in ATOMIC_ACTIONS]:
        errors.append("atomic action ids/order do not match the canonical 11 actions")
    for task in tasks:
        component_weight = sum(float(item.get("weight") or 0) for item in task.get("score_components") or [])
        if abs(component_weight - 100.0) > 1e-6:
            errors.append(f"{task.get('task_id')} score weights total {component_weight:g}, expected 100")
        unknown = set(task.get("atomic_actions") or []) - set(action_ids)
        if unknown:
            errors.append(f"{task.get('task_id')} has unknown atomic actions: {sorted(unknown)}")
    if errors:
        raise ValueError("Invalid laboratory benchmark catalog: " + "; ".join(errors))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _model_id(model_name: str, checkpoint: str) -> str:
    digest = hashlib.sha1(f"{model_name}\0{checkpoint}".encode("utf-8")).hexdigest()[:12]
    return f"model-{digest}"


def _mean(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _count_values(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        value = row.get(key)
        if value not in (None, "", "none"):
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def score_task_episode(task: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Score one episode when every workbook-defined component is available.

    Metric bindings use normalized 0-100 values.  Missing metrics never receive
    an implicit score; this keeps partially implemented simulator truth from
    looking like an official quality score.
    """
    if row.get("hard_failure") or row.get("safety_valid") is False:
        return {"score_100": 0.0, "status": "hard_failure", "components": {}}
    quality = row.get("quality_metrics") if isinstance(row.get("quality_metrics"), dict) else {}
    details: dict[str, Any] = {}
    total = 0.0
    missing: list[str] = []
    for component in task.get("score_components") or []:
        label = str(component.get("component") or "component")
        metric = str(component.get("metric") or "")
        weight = float(component.get("weight") or 0.0)
        if metric == "success":
            normalized = 100.0 if row.get("success") else 0.0
        else:
            value = row.get(metric, quality.get(metric))
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                missing.append(metric or label)
                details[label] = {"weight": weight, "metric": metric or None, "score": None}
                continue
            normalized = max(0.0, min(100.0, float(value)))
        weighted = weight * normalized / 100.0
        total += weighted
        details[label] = {"weight": weight, "metric": metric, "score": normalized, "weighted_score": weighted}
    if missing:
        return {"score_100": None, "status": "metrics_unavailable", "components": details, "missing_metrics": sorted(set(missing))}
    return {"score_100": round(total, 4), "status": "evaluated", "components": details, "missing_metrics": []}


def score_episode_for_bound_task(row: dict[str, Any], catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    benchmark_id = BENCHMARK_TASK_BINDINGS.get(str(row.get("task_id") or ""))
    if not benchmark_id:
        # New-format runs may write the catalog id itself (L01..R12) as the
        # episode task id, so treat a direct catalog match as bound too.
        task_id = str(row.get("task_id") or "")
        if task_id and any(item["task_id"] == task_id for item in (catalog or load_benchmark_catalog()).get("tasks", [])):
            benchmark_id = task_id
    if not benchmark_id:
        return {"score_100": None, "status": "not_in_formal_catalog", "components": {}}
    catalog = catalog or load_benchmark_catalog()
    task = next(item for item in catalog["tasks"] if item["task_id"] == benchmark_id)
    return score_task_episode(task, row)


def _bool_from_episode(row: dict[str, Any], key: str) -> bool | None:
    if key in row and row[key] is not None:
        return bool(row[key])
    for group_name in ("goal_predicates", "terminal_predicates", "stability_predicates", "safety_predicates"):
        group = row.get(group_name)
        if isinstance(group, dict) and key in group:
            return bool(group[key])
    values = row.get("predicate_values")
    if isinstance(values, dict):
        value = values.get(key)
        if isinstance(value, dict) and "passed" in value:
            return bool(value["passed"])
    return None


def atomic_outcomes_for_episode(row: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = row.get("atomic_outcomes")
    if isinstance(explicit, list):
        outcomes = []
        for index, item in enumerate(explicit):
            if not isinstance(item, dict) or item.get("action") not in {value[0] for value in ATOMIC_ACTIONS}:
                continue
            outcomes.append({
                "instance_id": str(item.get("instance_id") or f"{item['action']}_{index}"),
                "action": item["action"],
                "attempted": bool(item.get("attempted", True)),
                "success": bool(item.get("success", False)),
                "source": str(item.get("source") or "native"),
                "failure_code": item.get("failure_code"),
            })
        return outcomes
    task_id = str(row.get("task_id") or "")
    outcomes = []
    for index, binding in enumerate(ATOMIC_PREDICATE_BINDINGS.get(task_id, [])):
        values = [_bool_from_episode(row, key) for key in binding.get("keys") or []]
        if not values or any(value is None for value in values):
            continue
        outcomes.append({
            "instance_id": f"{task_id}:{binding['action']}:{index}",
            "action": binding["action"],
            "attempted": True,
            "success": all(bool(value) for value in values),
            "source": "derived_legacy",
            "failure_code": None if all(bool(value) for value in values) else row.get("failure_code"),
        })
    return outcomes


def _preferred_split(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for split in SPLIT_PRIORITY:
        selected = [row for row in rows if row.get("split") == split]
        if selected:
            return selected
    return rows


def build_benchmark_dashboard(runs: list[dict[str, Any]], catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    catalog = catalog or load_benchmark_catalog()
    tasks_by_id = {item["task_id"]: item for item in catalog["tasks"]}
    # Keep all 24 catalogue rows visible, but only implemented tasks are
    # eligible for evaluation coverage and leaderboard scoring. Planned tasks
    # must not inflate the denominator before their simulator implementation
    # is available.
    eligible_task_ids = {
        task_id
        for task_id, task in tasks_by_id.items()
        if task.get("status") == "implemented" and task_id not in NON_LEADERBOARD_TASK_IDS
    }
    track_by_id = {item["track_id"]: item for item in catalog["tracks"]}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(str(run.get("model_name") or "unknown"), str(run.get("checkpoint") or "unknown"))].append(run)

    models: list[dict[str, Any]] = []
    for (model_name, checkpoint), model_runs in groups.items():
        benchmark_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        atomic_counts: dict[str, dict[str, Any]] = {
            action_id: {"attempts": 0, "successes": 0, "task_ids": set(), "sources": set(), "failure_codes": defaultdict(int)}
            for action_id, _ in ATOMIC_ACTIONS
        }
        cap_buckets: dict[str, list[float]] = {"partial_success": [], "efficiency": [], "safety": []}
        for run in model_runs:
            run_dir = Path(str(run.get("run_dir") or ""))
            episodes = _read_jsonl(run_dir / "episode_metrics.jsonl") if run_dir else []
            lb_overall = (run.get("leaderboard_row") or {}).get("overall") if isinstance(run.get("leaderboard_row"), dict) else {}
            lb_caps = (run.get("leaderboard_row") or {}).get("capabilities") if isinstance(run.get("leaderboard_row"), dict) else {}
            lb_overall = lb_overall if isinstance(lb_overall, dict) else {}
            lb_caps = lb_caps if isinstance(lb_caps, dict) else {}
            partial = lb_overall.get("partial_success_rate")
            if isinstance(partial, (int, float)) and not isinstance(partial, bool):
                cap_buckets["partial_success"].append(float(partial) * 100.0)
            elif lb_caps.get("partial_success") is not None:
                cap_buckets["partial_success"].append(float(lb_caps["partial_success"]))
            efficiency = lb_overall.get("efficiency_score")
            if isinstance(efficiency, (int, float)) and not isinstance(efficiency, bool):
                cap_buckets["efficiency"].append(float(efficiency))
            elif lb_caps.get("efficiency") is not None:
                cap_buckets["efficiency"].append(float(lb_caps["efficiency"]))
            safety = lb_overall.get("safety_score")
            if isinstance(safety, (int, float)) and not isinstance(safety, bool):
                cap_buckets["safety"].append(float(safety))
            elif lb_caps.get("safety") is not None:
                cap_buckets["safety"].append(float(lb_caps["safety"]))
            for episode in episodes:
                task_id = str(episode.get("task_id") or run.get("primary_task_id") or "")
                benchmark_id = BENCHMARK_TASK_BINDINGS.get(task_id)
                if benchmark_id is None and task_id in tasks_by_id:
                    benchmark_id = task_id
                if benchmark_id in eligible_task_ids:
                    benchmark_rows[benchmark_id].append(episode)
                if benchmark_id in NON_LEADERBOARD_TASK_IDS:
                    continue
                for outcome in atomic_outcomes_for_episode(episode):
                    if not outcome["attempted"]:
                        continue
                    bucket = atomic_counts[outcome["action"]]
                    bucket["attempts"] += 1
                    bucket["successes"] += int(outcome["success"])
                    bucket["task_ids"].add(task_id)
                    bucket["sources"].add(outcome["source"])
                    if not outcome["success"]:
                        bucket["failure_codes"][str(outcome.get("failure_code") or "unknown_failure")] += 1

        task_scores: list[dict[str, Any]] = []
        for task in catalog["tasks"]:
            # Keep all 24 catalogue rows visible for stable track denominators.
            # Explicitly excluded/planned tasks remain unevaluated even when an
            # archived or synthetic run happens to use the same task id.
            rows = (
                _preferred_split(benchmark_rows.get(task["task_id"], []))
                if task["task_id"] in eligible_task_ids
                else []
            )
            valid = [row for row in rows if row.get("valid_scene", True)]
            success_rate = _mean([1.0 if row.get("success") else 0.0 for row in valid])
            quality_values = [float(row["task_score_100"]) for row in valid if isinstance(row.get("task_score_100"), (int, float))]
            task_scores.append({
                **task,
                "evaluated": bool(valid),
                "num_trials": len(rows),
                "num_valid_trials": len(valid),
                "num_success": sum(1 for row in valid if row.get("success")),
                "success_rate": success_rate,
                "score_100": _mean(quality_values),
                "score_status": "evaluated" if quality_values else ("metrics_unavailable" if valid else task.get("status", "planned")),
                "split": valid[0].get("split") if valid else None,
                "stage_completion_mean": _mean(row.get("stage_completion") for row in valid),
                "inference_latency_p95_ms": _mean(row.get("inference_latency_p95_ms") for row in valid),
                "failure_category_distribution": _count_values(valid, "failure_category"),
                "failure_code_distribution": _count_values(valid, "failure_code"),
            })

        track_scores = []
        for track_id in TRACK_ORDER:
            items = [item for item in task_scores if item["track_id"] == track_id]
            evaluated = [item for item in items if item["success_rate"] is not None]
            scored = [item for item in items if item["score_100"] is not None]
            track_scores.append({
                **track_by_id[track_id],
                "task_count": len(items),
                "evaluated_task_count": len(evaluated),
                "coverage_rate": len(evaluated) / len(items) if items else None,
                "success_rate": _mean(item["success_rate"] for item in evaluated),
                "score_100": _mean(item["score_100"] for item in scored),
                "score_coverage_rate": len(scored) / len(items) if items else None,
            })

        atomic_scores = []
        for action_id, label in ATOMIC_ACTIONS:
            bucket = atomic_counts[action_id]
            attempts = int(bucket["attempts"])
            atomic_scores.append({
                "action_id": action_id,
                "label": label,
                "attempts": attempts,
                "successes": int(bucket["successes"]),
                "success_rate": bucket["successes"] / attempts if attempts else None,
                "task_count": len(bucket["task_ids"]),
                "task_ids": sorted(bucket["task_ids"]),
                "sources": sorted(bucket["sources"]),
                "failure_code_distribution": dict(sorted(bucket["failure_codes"].items())),
            })
        evaluated_tasks = [item for item in task_scores if item["success_rate"] is not None]
        scored_tasks = [item for item in task_scores if item["score_100"] is not None]
        evaluated_actions = [item for item in atomic_scores if item["success_rate"] is not None]
        if not evaluated_tasks and not evaluated_actions:
            continue
        model_id = _model_id(model_name, checkpoint)
        policy_providers = sorted({str(run.get("policy_provider")) for run in model_runs if run.get("policy_provider")})
        suite_ids = sorted({str(run.get("suite_id")) for run in model_runs if run.get("suite_id")})
        benchmark_versions = sorted({str(run.get("benchmark_version")) for run in model_runs if run.get("benchmark_version")})
        overall_success_rate = _mean(item["success_rate"] for item in evaluated_tasks)
        overall_score_100 = _mean(item["score_100"] for item in scored_tasks)
        atomic_success_rate = _mean(item["success_rate"] for item in evaluated_actions)
        failure_category_distribution: defaultdict[str, int] = defaultdict(int)
        failure_code_distribution: defaultdict[str, int] = defaultdict(int)
        for task in evaluated_tasks:
            for key, value in task.get("failure_category_distribution", {}).items():
                failure_category_distribution[key] += int(value)
            for key, value in task.get("failure_code_distribution", {}).items():
                failure_code_distribution[key] += int(value)
        task_total = len(eligible_task_ids)
        task_coverage = {"evaluated": len(evaluated_tasks), "total": task_total, "rate": len(evaluated_tasks) / task_total}
        score_coverage = {"evaluated": len(scored_tasks), "total": task_total, "rate": len(scored_tasks) / task_total}
        atomic_coverage = {"evaluated": len(evaluated_actions), "total": 11, "rate": len(evaluated_actions) / 11}
        capabilities = {
            "overall_success": None if overall_success_rate is None else overall_success_rate * 100.0,
            "partial_success": _mean(cap_buckets["partial_success"]),
            "efficiency": _mean(cap_buckets["efficiency"]),
            "safety": _mean(cap_buckets["safety"]),
            "atomic": None if atomic_success_rate is None else atomic_success_rate * 100.0,
            "task_coverage": task_coverage["rate"] * 100.0,
            "score_coverage": score_coverage["rate"] * 100.0,
        }
        models.append({
            "model_id": model_id,
            "model_name": model_name,
            "checkpoint": checkpoint,
            "policy_provider": policy_providers[0] if len(policy_providers) == 1 else ("mixed" if policy_providers else None),
            "suite_id": suite_ids[0] if len(suite_ids) == 1 else ("mixed" if suite_ids else None),
            "benchmark_version": benchmark_versions[0] if len(benchmark_versions) == 1 else ("mixed" if benchmark_versions else None),
            "overall_success_rate": overall_success_rate,
            "overall_score_100": overall_score_100,
            "task_coverage": task_coverage,
            "score_coverage": score_coverage,
            "atomic_success_rate": atomic_success_rate,
            "atomic_coverage": atomic_coverage,
            "num_trials": sum(int(item.get("num_valid_trials") or 0) for item in evaluated_tasks),
            "num_success": sum(int(item.get("num_success") or 0) for item in evaluated_tasks),
            "failure_category_distribution": dict(sorted(failure_category_distribution.items())),
            "failure_code_distribution": dict(sorted(failure_code_distribution.items())),
            "tracks": track_scores,
            "tasks": task_scores,
            "atomic_actions": atomic_scores,
            "capabilities": capabilities,
            "run_ids": sorted({str(run.get("run_id")) for run in model_runs if run.get("run_id")}),
        })
    models.sort(key=lambda item: (item["overall_success_rate"] is not None, item["overall_success_rate"] or -1), reverse=True)
    for rank, model in enumerate(models, start=1):
        model["rank"] = rank
    return {
        "schema_version": "laboratory_benchmark_dashboard/v1",
        "catalog_id": catalog.get("catalog_id"),
        "catalog_version": catalog.get("version"),
        "task_count": len(catalog["tasks"]),
        "track_count": len(catalog["tracks"]),
        "atomic_action_count": 11,
        "tracks": catalog["tracks"],
        "atomic_actions": catalog["atomic_actions"],
        "tasks": catalog["tasks"],
        "models": models,
    }


def benchmark_track_payload(dashboard: dict[str, Any], track_id: str) -> dict[str, Any]:
    if track_id not in TRACK_ORDER:
        raise KeyError(track_id)
    track = next(item for item in dashboard["tracks"] if item["track_id"] == track_id)
    models = []
    for model in dashboard["models"]:
        score = next(item for item in model["tracks"] if item["track_id"] == track_id)
        models.append({
            "model_id": model["model_id"],
            "model_name": model["model_name"],
            "checkpoint": model["checkpoint"],
            **score,
            "tasks": [item for item in model["tasks"] if item["track_id"] == track_id],
        })
    models.sort(key=lambda item: (item["success_rate"] is not None, item["success_rate"] or -1), reverse=True)
    for rank, model in enumerate(models, start=1):
        model["rank"] = rank
    return {"track": track, "models": models}
