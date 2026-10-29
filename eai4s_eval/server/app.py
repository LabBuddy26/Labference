"""EAI4S-bench web server — serves the leaderboard frontend and REST API.

Start with:
    python -m eai4s_eval.server.app --port 5000
    python -m eai4s_eval.server.app --outputs-root outputs/eval_runs --port 5000
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from eai4s_eval.benchmark_catalog import (
    benchmark_track_payload,
    build_benchmark_dashboard,
    load_benchmark_catalog,
)
from eai4s_eval.eval.model_identity import resolve_model_name, validate_checkpoint_name
from eai4s_eval.eval.task_suite import (
    DEFAULT_SUITE_ID,
    default_official_suite_id,
    load_official_evaluation_suites,
    load_task_catalog,
)
from eai4s_eval.public_site import (
    PUBLIC_SPLITS,
    build_public_snapshot,
    load_release_protocols,
    public_object,
    public_release_protocol,
    public_task,
)
from eai4s_eval.server.job_queue import PersistentEvalQueue

try:
    from flask import Flask, jsonify, render_template, request
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in minimal test environments.
    _FLASK_IMPORT_ERROR = exc

    class Flask:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError("Flask is required to start the EAI4S-bench web server. Install it with `pip install flask`.") from _FLASK_IMPORT_ERROR

    def jsonify(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError("Flask is required to use jsonify().") from _FLASK_IMPORT_ERROR

    def render_template(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError("Flask is required to render templates.") from _FLASK_IMPORT_ERROR

    class _MissingRequest:
        def get_json(self, *_args: Any, **_kwargs: Any) -> Any:
            raise ModuleNotFoundError("Flask is required to access requests.") from _FLASK_IMPORT_ERROR

    request = _MissingRequest()

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_PACKAGE_ROOT = _HERE.parent
_TEMPLATES = _HERE / "templates"
_BENCHMARK_ROOT = _PACKAGE_ROOT.parent
_DEFAULT_OUTPUTS_ROOT = _BENCHMARK_ROOT / "outputs" / "eval_runs"
_DEFAULT_RELEASE_OUTPUTS_ROOT = _BENCHMARK_ROOT / "outputs" / "release_collection"
_DEFAULT_DATASET_STATISTICS_PATH = _PACKAGE_ROOT / "data" / "dataset_statistics.json"
_DEFAULT_SIMULATION_STATISTICS_PATH = _HERE / "static" / "web_assets" / "dataset_stats_20260826.json"
_DATASET_STATISTICS_SCHEMA = "dataset_statistics/v1"
_DATASET_STATISTICS_SECTIONS = ("real_robot_teleoperation", "umi", "simulation")
_DATASET_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATASET_OPERATOR_RE = re.compile(r"^operator\d{2}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.=-]+$")
_SPLIT_ORDER = ("sim_dev", "sim_clean", "sim_randomized")
_CONFIG_ROOT = _PACKAGE_ROOT / "configs"


def _default_policy_profile(provider: str) -> Path:
    normalized_provider = str(provider).lower()
    filename = (
        "gr00t_franka_ee_pos_v1.yaml"
        if normalized_provider == "gr00t"
        else "openpi_franka_ee_pos_v1.yaml"
    )
    return _CONFIG_ROOT / "policies" / filename


_TASK_PRESETS: dict[str, dict[str, Any]] = {
    "t02": {
        "task_spec": str(_CONFIG_ROOT / "tasks" / "T02_franka_pick_lift_proxy.yaml"),
        "instruction": "pick glass bottle",
        "target_object": "vial",
        "max_steps": 300,
    },
    "t03": {
        "task_spec": str(_CONFIG_ROOT / "tasks" / "T03_tube_insert_rack_draft.yaml"),
        "instruction": "place the glass tube in the rack",
        "target_object": "tube",
        "max_steps": 300,
    },
}


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _dataset_statistics_contract_errors(payload: dict[str, Any]) -> list[str]:
    """Return precise nested contract errors for a dataset-statistics payload."""

    errors: list[str] = []
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at.strip():
        errors.append("generated_at")

    for section_name in _DATASET_STATISTICS_SECTIONS:
        section = payload.get(section_name)
        if not isinstance(section, dict):
            continue
        status = section.get("status")
        totals = section.get("totals")
        rows = section.get("rows")
        if not isinstance(status, str) or status not in {"active", "empty"}:
            errors.append(f"{section_name}.status")
        if not isinstance(totals, dict):
            errors.append(f"{section_name}.totals")
            totals = None
        if not isinstance(rows, list):
            errors.append(f"{section_name}.rows")
            rows = None

        is_simulation = section_name == "simulation"
        total_fields = (
            ("dataset_batch_count", "episode_count", "frame_count", "duration_seconds", "scene_count")
            if is_simulation
            else ("batch_count", "sample_count", "duration_seconds")
        )
        if totals is not None:
            for field in total_fields:
                if not _is_nonnegative_int(totals.get(field)):
                    errors.append(f"{section_name}.totals.{field}")

        if rows is None:
            continue
        numeric_fields = (
            ("dataset_batch_count", "episode_count", "frame_count", "duration_seconds", "scene_count")
            if is_simulation
            else ("batch_count", "sample_count", "duration_seconds")
        )
        string_fields = (
            ("simulator", "robot_type", "task_id", "task_name")
            if is_simulation
            else ("device_model", "task_id", "task_name")
        )
        date_fields = (
            ("generation_date_start", "generation_date_end")
            if is_simulation
            else ("collection_date_start", "collection_date_end")
        )
        for index, row in enumerate(rows):
            prefix = f"{section_name}.rows[{index}]"
            if not isinstance(row, dict):
                errors.append(prefix)
                continue
            for field in string_fields:
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{prefix}.{field}")
            for field in numeric_fields:
                if not _is_nonnegative_int(row.get(field)):
                    errors.append(f"{prefix}.{field}")
            parsed_dates: list[str] = []
            for field in date_fields:
                value = row.get(field)
                if not isinstance(value, str) or not _DATASET_DATE_RE.fullmatch(value):
                    errors.append(f"{prefix}.{field}")
                else:
                    try:
                        datetime.strptime(value, "%Y-%m-%d")
                    except ValueError:
                        errors.append(f"{prefix}.{field}")
                    else:
                        parsed_dates.append(value)
            if len(parsed_dates) == 2 and parsed_dates[0] > parsed_dates[1]:
                errors.append(f"{prefix}.{date_fields[0]}>{date_fields[1]}")
            if not is_simulation:
                operator_ids = row.get("operator_ids")
                valid_operator_items = (
                    isinstance(operator_ids, list)
                    and bool(operator_ids)
                    and all(
                        isinstance(item, str) and bool(_DATASET_OPERATOR_RE.fullmatch(item))
                        for item in operator_ids
                    )
                )
                if not valid_operator_items or len(operator_ids) != len(set(operator_ids)):
                    errors.append(f"{prefix}.operator_ids")

        if totals is not None and all(_is_nonnegative_int(totals.get(field)) for field in total_fields):
            additive_fields = (
                ("dataset_batch_count", "episode_count", "frame_count", "duration_seconds")
                if is_simulation
                else ("batch_count", "sample_count", "duration_seconds")
            )
            dict_rows = [row for row in rows if isinstance(row, dict)]
            for field in additive_fields:
                if all(_is_nonnegative_int(row.get(field)) for row in dict_rows):
                    if sum(row[field] for row in dict_rows) != totals[field]:
                        errors.append(f"{section_name}.totals.{field}:row_sum")
            if status == "empty" and (rows or any(totals[field] for field in total_fields)):
                errors.append(f"{section_name}.status:empty_mismatch")

    return errors


def _task_key_from_body(body: dict[str, Any]) -> str:
    raw_candidates = [body.get("task_id"), body.get("task_mode"), body.get("task")]
    task_spec = str(body.get("task_spec") or "")
    if "T03_tube_insert_rack" in task_spec:
        return "t03"
    if "T02_franka_pick_lift_proxy" in task_spec:
        return "t02"
    for raw in raw_candidates:
        value = str(raw or "").strip().lower()
        if value in {"t03", "tube_insert", "tube_rack_insert", "t03_tube_insert_rack"}:
            return "t03"
        if value in {"t02", "pick_lift", "t02_franka_pick_lift_proxy"}:
            return "t02"
    return "t02"


def _apply_eval_task_defaults(body: dict[str, Any]) -> dict[str, Any]:
    updated = dict(body)
    preset = _TASK_PRESETS[_task_key_from_body(updated)]
    for key in ("task_spec", "instruction", "target_object", "max_steps"):
        if updated.get(key) in {None, ""}:
            updated[key] = preset[key]
    return updated


def _safe_run_dir(outputs_root: Path, run_id: str) -> Path | None:
    """Resolve a run id under outputs_root and reject path traversal."""
    if not run_id or not _RUN_ID_RE.fullmatch(run_id):
        return None
    candidate = (outputs_root / run_id).resolve()
    try:
        candidate.relative_to(outputs_root)
    except ValueError:
        return None
    return candidate


def _shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _legacy_eval_group_id(run_id: str) -> str:
    for suffix in ("_sim_clean", "_sim_randomized"):
        if run_id.endswith(suffix):
            return run_id[: -len(suffix)]
    return run_id


def _weighted_mean(items: list[tuple[float | None, int]]) -> float | None:
    total_weight = sum(weight for value, weight in items if value is not None and weight > 0)
    if total_weight <= 0:
        return None
    return sum(float(value) * weight for value, weight in items if value is not None and weight > 0) / total_weight


def _mean(items: list[float | None]) -> float | None:
    values = [float(item) for item in items if item is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _merge_count_dicts(items: list[dict[str, Any] | None]) -> dict[str, int]:
    merged: defaultdict[str, int] = defaultdict(int)
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            try:
                merged[str(key)] += int(value)
            except (TypeError, ValueError):
                continue
    return dict(sorted(merged.items()))


def _first_nonempty(items: list[Any], default: Any = None) -> Any:
    for item in items:
        if item not in (None, "", [], {}):
            return item
    return default


def _read_json_path(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _scan_runs(outputs_root: Path) -> list[dict[str, Any]]:
    """Scan *outputs_root* for run directories and return shallow run summaries.

    Complete run directories contain at least ``leaderboard_row.json`` and
    ``summary.json``. Partial/in-progress directories are still returned with
    missing values so the frontend can show them as partial.
    """
    if not outputs_root.exists():
        return []

    runs: list[dict[str, Any]] = []
    for candidate in sorted(outputs_root.iterdir(), reverse=True):
        if not candidate.is_dir() or candidate.name.startswith("_"):
            continue
        if (candidate / "suite_leaderboard_row.json").exists():
            continue
        lb_path = candidate / "leaderboard_row.json"
        summary_path = candidate / "summary.json"
        manifest_path = candidate / "run_manifest.json"

        row: dict[str, Any] = {}
        if lb_path.exists():
            try:
                row = json.loads(lb_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        summary: dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        task_summary: list[dict[str, Any]] = []
        task_summary_path = candidate / "task_summary.json"
        if task_summary_path.exists():
            try:
                loaded = json.loads(task_summary_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    task_summary = loaded
            except (json.JSONDecodeError, OSError):
                pass
        family_summary: list[dict[str, Any]] = []
        family_summary_path = candidate / "family_summary.json"
        if family_summary_path.exists():
            try:
                loaded = json.loads(family_summary_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    family_summary = loaded
            except (json.JSONDecodeError, OSError):
                pass
        task_ids = [str(item.get("task_id")) for item in task_summary if item.get("task_id")]
        task_names = [str(item.get("task_name")) for item in task_summary if item.get("task_name")]
        task_families = [str(item.get("task_family")) for item in task_summary if item.get("task_family")]
        if not task_families:
            task_families = [str(item.get("task_family")) for item in family_summary if item.get("task_family")]
        primary_task = task_summary[0] if task_summary else {}
        primary_family = family_summary[0] if family_summary else {}
        task_success_rate = primary_task.get("success_rate")
        run_id = row.get("run_id", candidate.name)
        split = row.get("track", manifest.get("split", primary_task.get("split", "sim_dev")))
        eval_group_id = manifest.get("eval_group_id") or row.get("eval_group_id") or _legacy_eval_group_id(run_id)
        policy_backend = manifest.get("policy_backend") or (manifest.get("policy") or {}).get("backend")
        policy_provider = manifest.get("policy_provider")
        if not policy_provider:
            normalized_backend = str(policy_backend or "").lower()
            policy_provider = "gr00t" if "gr00t" in normalized_backend or normalized_backend == "zmq" else (
                "openpi" if "openpi" in normalized_backend or normalized_backend in {"websocket", "http"} else None
            )

        runs.append(
            {
                "run_id": run_id,
                "eval_group_id": eval_group_id,
                "model_name": row.get("model_name") or manifest.get("model_name") or "unknown",
                "checkpoint": manifest.get("checkpoint") or "unknown",
                "policy_provider": policy_provider,
                "policy_backend": policy_backend,
                "suite_id": manifest.get("suite_id"),
                "benchmark_version": row.get("benchmark_version", manifest.get("benchmark_version")),
                "track": split,
                "split": split,
                "task_ids": task_ids,
                "task_names": task_names,
                "task_families": task_families,
                "task_label": ", ".join(task_names or task_ids) if (task_names or task_ids) else "unknown_task",
                "primary_task_id": primary_task.get("task_id"),
                "primary_task_name": primary_task.get("task_name"),
                "primary_task_family": primary_task.get("task_family") or primary_family.get("task_family") or "basic",
                "leaderboard_group": primary_task.get("leaderboard_group") or primary_family.get("leaderboard_group"),
                "success_logic": primary_task.get("success_logic"),
                "task_success_rate": task_success_rate,
                "overall_success_rate": summary.get("success_rate"),
                "num_trials": row.get("num_trials", summary.get("num_episodes")),
                "num_success": summary.get("num_success"),
                "success_rate": task_success_rate if task_success_rate is not None else summary.get("success_rate"),
                "stage_completion_mean": summary.get("stage_completion_mean"),
                "inference_latency_p95_ms": summary.get("inference_latency_p95_ms"),
                "capability_index": (row.get("overall") or {}).get("capability_index_v0_1"),
                "failure_category_distribution": summary.get("failure_category_distribution") or primary_task.get("failure_category_distribution") or {},
                "failure_code_distribution": summary.get("failure_code_distribution") or primary_task.get("failure_code_distribution") or {},
                "drop_rate": summary.get("drop_rate") if summary.get("drop_rate") is not None else primary_task.get("drop_rate"),
                "collision_rate": summary.get("collision_rate") if summary.get("collision_rate") is not None else primary_task.get("collision_rate"),
                "wrong_place_rate": summary.get("wrong_place_rate") if summary.get("wrong_place_rate") is not None else primary_task.get("wrong_place_rate"),
                "timeout_rate": summary.get("timeout_rate") if summary.get("timeout_rate") is not None else primary_task.get("timeout_rate"),
                "created_at": manifest.get("created_at"),
                "run_dir": str(candidate),
                "leaderboard_row": row,
                "task_summary": task_summary,
                "family_summary": family_summary,
                "summary": summary,
            }
        )
    return runs


def _release_protocol_id(manifest: dict[str, Any]) -> str | None:
    """Map a release runner manifest to the public independent task id."""
    version = str(manifest.get("task_version") or "")
    evaluation = manifest.get("evaluation") if isinstance(manifest.get("evaluation"), dict) else {}
    instance = manifest.get("instance") if isinstance(manifest.get("instance"), dict) else {}
    difficulty = str(manifest.get("difficulty") or evaluation.get("difficulty") or "").lower()
    if version == "centrifuge-opposite/v2":
        return f"centrifuge_{difficulty}" if difficulty in {"easy", "hard"} else None
    if version == "weighing-feedback/v1":
        return f"weighing_{difficulty}" if difficulty in {"easy", "hard"} else None
    if version == "rack-placement-recall/v4":
        condition = str(
            manifest.get("condition_variant")
            or instance.get("condition_variant")
            or evaluation.get("condition_variant")
            or ""
        ).lower()
        if condition in {"single", "easy"}:
            return "rack_easy"
        if condition in {"same_cap", "distinct_cap"}:
            return f"rack_{condition}"
    if version == "device-operation-cycle/v1":
        return "device_operation_hard"
    return None


def _scan_release_runs(outputs_root: Path) -> list[dict[str, Any]]:
    """Read release-protocol run.json/scores.json pairs for public output."""
    if not outputs_root.exists():
        return []
    catalog = load_release_protocols()
    protocol_ids = {str(item["protocol_id"]) for item in catalog.get("protocols", [])}
    runs: list[dict[str, Any]] = []
    # A batch's terminal records supersede its raw worker artifacts. Do not
    # count both, and do not publish any of the optional ordinary results.
    batch_roots = []
    from eai4s_eval.eval.batch_results import read_json as read_batch_json
    for plan_path in outputs_root.rglob("plan.json"):
        plan = read_batch_json(plan_path)
        if plan.get("schema_version") != "labference_eval_plan/v1":
            continue
        batch_roots.append(plan_path.parent)
        for job in plan.get("jobs", []):
            if job.get("group") != "special":
                continue
            key = f"{job['task_id']}_{job['split']}_{job['seed']}"
            result = read_batch_json(plan_path.parent / "results" / f"{key}.json")
            if not result or any(result.get(k) != job.get(k) for k in ("task_id", "seed", "split", "mode", "model_name", "checkpoint")):
                continue
            runs.append({"run_id": f"{plan['run_id']}/{key}", "protocol_id": job['task_id'],
                         "model_name": job['model_name'], "checkpoint": job['checkpoint'],
                         "seed": job['seed'], "split": job['split'], "eval_group_id": plan['run_id'],
                         "execution_mode": job['mode'], "runtime_status": result.get('termination_reason'),
                         "ranking_eligible": result.get('ranking_eligible') is True,
                         "evaluation_valid": result.get('evaluation_valid') is True,
                         "planned_seeds": [j['seed'] for j in plan['jobs'] if j['task_id'] == job['task_id'] and j['split'] == job['split']],
                         "scores": {**result.get('diagnostics', {}), 'valid_success': result.get('success')}})
    for manifest_path in sorted(outputs_root.rglob("run.json"), reverse=True):
        candidate = manifest_path.parent
        if any(candidate.is_relative_to(batch_root) for batch_root in batch_roots):
            continue
        if any(part.startswith("_") for part in candidate.relative_to(outputs_root).parts):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(manifest, dict):
            continue
        protocol_id = _release_protocol_id(manifest)
        if protocol_id not in protocol_ids:
            continue
        scores = _read_json_path(candidate / "scores.json")
        if not isinstance(scores, dict):
            scores = {}
        policy_interface = manifest.get("policy_interface")
        if isinstance(policy_interface, dict):
            policy_interface_name = str(policy_interface.get("name") or "")
        else:
            policy_interface_name = str(policy_interface or manifest.get("policy") or "")
        policy_id = str(manifest.get("policy_id") or "").strip()
        is_reference = policy_interface_name == "diagnostic_rule_expert" or policy_id == "diagnostic_rule_expert"
        model_name = str(
            manifest.get("model_name")
            or manifest.get("policy_name")
            or ("Reference executor" if is_reference else policy_id)
            or policy_interface_name
            or "unknown model"
        )
        checkpoint = str(
            manifest.get("checkpoint")
            or manifest.get("policy_revision")
            or manifest.get("policy_profile_id")
            or "unknown"
        )
        created_at = manifest.get("created_at")
        if not created_at:
            try:
                created_at = datetime.fromtimestamp(manifest_path.stat().st_mtime).astimezone().isoformat()
            except OSError:
                created_at = None
        evaluation_valid = scores.get("evaluation_valid")
        if not isinstance(evaluation_valid, bool):
            evaluation_valid = manifest.get("evaluation_valid")
        runs.append(
            {
                "run_id": str(manifest.get("run_id") or candidate.name),
                "protocol_id": protocol_id,
                "task_version": manifest.get("task_version"),
                "model_name": model_name,
                "checkpoint": checkpoint,
                "policy_provider": manifest.get("policy_provider"),
                "policy_interface": policy_interface_name,
                "runtime_status": manifest.get("runtime_status"),
                "evaluation_valid": evaluation_valid,
                "ranking_eligible": bool(manifest.get("ranking_eligible") is True and not is_reference),
                "created_at": created_at,
                "run_dir": str(candidate),
                "scores": scores,
                "manifest": manifest,
            }
        )
    runs.sort(key=lambda item: (str(item.get("created_at") or ""), str(item.get("run_id") or "")), reverse=True)
    return runs


def _scan_suite_runs(outputs_root: Path) -> list[dict[str, Any]]:
    if not outputs_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for candidate in sorted(outputs_root.iterdir(), reverse=True):
        if not candidate.is_dir() or candidate.name.startswith("_"):
            continue
        suite_lb_path = candidate / "suite_leaderboard_row.json"
        if not suite_lb_path.exists():
            continue
        try:
            row = json.loads(suite_lb_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        suite_summary = {}
        task_score_table = row.get("task_score_table")
        family_score_table = row.get("family_score_table")
        loaded_summary = _read_json_path(candidate / "suite_summary.json")
        if isinstance(loaded_summary, dict):
            suite_summary = loaded_summary
        if not isinstance(task_score_table, list):
            task_score_table = _read_json_path(candidate / "task_score_table.json") or []
        if not isinstance(family_score_table, list):
            family_score_table = _read_json_path(candidate / "family_score_table.json") or []
        if isinstance(task_score_table, list):
            row["task_score_table"] = task_score_table
        if isinstance(family_score_table, list):
            row["family_score_table"] = family_score_table
        row["suite_summary"] = suite_summary or row.get("suite_summary") or {}
        row["run_dir"] = str(candidate)
        rows.append(row)
    rows.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def _query_runs(
    runs: list[dict[str, Any]],
    *,
    limit: int = 50,
    offset: int = 0,
    model_name: str = "",
    task_id: str = "",
    split: str = "",
) -> dict[str, Any]:
    """Filter and page run summaries without changing the legacy runs API."""
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    model_name = str(model_name or "").strip()
    task_id = str(task_id or "").strip()
    split = str(split or "").strip()

    ordered = sorted(
        runs,
        key=lambda item: (str(item.get("created_at") or ""), str(item.get("run_id") or "")),
        reverse=True,
    )
    available_filters = {
        "model_names": sorted({str(item.get("model_name")) for item in ordered if item.get("model_name")}),
        "task_ids": sorted({str(value) for item in ordered for value in (item.get("task_ids") or []) if value}),
        "splits": [value for value in _SPLIT_ORDER if any(item.get("split") == value for item in ordered)],
    }
    filtered = [
        item
        for item in ordered
        if (not model_name or item.get("model_name") == model_name)
        and (not task_id or task_id in (item.get("task_ids") or []))
        and (not split or item.get("split") == split)
    ]
    total = len(filtered)
    rows = filtered[offset : offset + limit]
    return {
        "schema_version": "eai4s.runs_page/v1",
        "rows": rows,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": total,
            "has_previous": offset > 0,
            "has_next": offset + len(rows) < total,
        },
        "available_filters": available_filters,
    }


def _aggregate_leaderboard_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        task_key = run.get("primary_task_id") or ",".join(run.get("task_ids") or []) or "unknown_task"
        key = (
            str(run.get("eval_group_id") or run.get("run_id")),
            str(run.get("model_name") or "unknown"),
            str(run.get("checkpoint") or "unknown"),
            str(task_key),
        )
        grouped[key].append(run)

    rows: list[dict[str, Any]] = []
    for (eval_group_id, model_name, checkpoint, _task_key), group in grouped.items():
        split_metrics: dict[str, dict[str, Any]] = {}
        run_ids_by_split: dict[str, str] = {}
        task_ids: list[str] = []
        task_names: list[str] = []
        task_families: list[str] = []
        task_summary_items: list[dict[str, Any]] = []
        family_summary_items: list[dict[str, Any]] = []
        success_logic = None
        created_values = [item.get("created_at") for item in group if item.get("created_at")]
        created_at = min(created_values) if created_values else None

        for item in group:
            split = str(item.get("split") or item.get("track") or "sim_dev")
            num_trials = int(item.get("num_trials") or 0)
            num_success = item.get("num_success")
            if num_success is None and item.get("success_rate") is not None:
                num_success = round(float(item["success_rate"]) * num_trials)
            split_metrics[split] = {
                "run_id": item.get("run_id"),
                "success_rate": item.get("task_success_rate") if item.get("task_success_rate") is not None else item.get("success_rate"),
                "num_trials": num_trials,
                "num_success": int(num_success or 0),
                "stage_completion_mean": item.get("stage_completion_mean"),
                "inference_latency_p95_ms": item.get("inference_latency_p95_ms"),
                "capability_index": item.get("capability_index"),
                "failure_category_distribution": item.get("failure_category_distribution") or {},
                "failure_code_distribution": item.get("failure_code_distribution") or {},
                "drop_rate": item.get("drop_rate"),
                "collision_rate": item.get("collision_rate"),
                "wrong_place_rate": item.get("wrong_place_rate"),
                "timeout_rate": item.get("timeout_rate"),
            }
            run_ids_by_split[split] = str(item.get("run_id"))
            task_ids.extend([task_id for task_id in item.get("task_ids", []) if task_id and task_id not in task_ids])
            task_names.extend([task_name for task_name in item.get("task_names", []) if task_name and task_name not in task_names])
            task_families.extend([family for family in item.get("task_families", []) if family and family not in task_families])
            task_summary_items.extend(item.get("task_summary") or [])
            family_summary_items.extend(item.get("family_summary") or [])
            success_logic = success_logic or item.get("success_logic")

        ordered_splits = [split for split in _SPLIT_ORDER if split in split_metrics]
        ordered_splits.extend(sorted(split for split in split_metrics if split not in ordered_splits))
        total_trials = sum(int(split_metrics[split].get("num_trials") or 0) for split in ordered_splits)
        total_success = sum(int(split_metrics[split].get("num_success") or 0) for split in ordered_splits)
        task_success_rate = total_success / total_trials if total_trials else None
        stage_completion = _weighted_mean(
            [(split_metrics[split].get("stage_completion_mean"), int(split_metrics[split].get("num_trials") or 0)) for split in ordered_splits]
        )
        latency = _mean([split_metrics[split].get("inference_latency_p95_ms") for split in ordered_splits])
        efficiency = None if latency is None else max(0.0, min(100.0, 100.0 - latency / 10.0))
        success_score = None if task_success_rate is None else task_success_rate * 100.0
        partial_score = None if stage_completion is None else stage_completion * 100.0
        capability_values = [value for value in (success_score, partial_score, efficiency) if value is not None]
        capability_index = sum(capability_values) / len(capability_values) if capability_values else None
        failure_category_distribution = _merge_count_dicts([split_metrics[split].get("failure_category_distribution") for split in ordered_splits])
        failure_code_distribution = _merge_count_dicts([split_metrics[split].get("failure_code_distribution") for split in ordered_splits])

        mini_by_task = []
        for task_item in task_summary_items:
            mini_by_task.append(
                {
                    "scope": "task",
                    "model_name": model_name,
                    "checkpoint": checkpoint,
                    "task_id": task_item.get("task_id"),
                    "task_name": task_item.get("task_name"),
                    "task_family": task_item.get("task_family") or "basic",
                    "leaderboard_group": task_item.get("leaderboard_group"),
                    "split": task_item.get("split"),
                    "success_rate": task_item.get("success_rate"),
                    "safe_success_rate": task_item.get("safe_success_rate"),
                    "num_trials": task_item.get("num_trials"),
                    "num_success": task_item.get("num_success"),
                    "failure_category_distribution": task_item.get("failure_category_distribution") or {},
                    "failure_examples": task_item.get("failure_examples") or [],
                }
            )
        mini_by_family = []
        for family_item in family_summary_items:
            mini_by_family.append(
                {
                    "scope": "family",
                    "model_name": model_name,
                    "checkpoint": checkpoint,
                    "task_family": family_item.get("task_family") or "basic",
                    "task_ids": family_item.get("task_ids") or [],
                    "leaderboard_group": family_item.get("leaderboard_group"),
                    "split": family_item.get("split"),
                    "success_rate": family_item.get("success_rate"),
                    "safe_success_rate": family_item.get("safe_success_rate"),
                    "num_trials": family_item.get("num_trials"),
                    "num_success": family_item.get("num_success"),
                    "failure_category_distribution": family_item.get("failure_category_distribution") or {},
                    "failure_examples": family_item.get("failure_examples") or [],
                }
            )

        by_family: dict[str, dict[str, Any]] = {}
        for family in task_families or ["basic"]:
            items = [item for item in mini_by_family if item.get("task_family") == family]
            if not items:
                items = [item for item in mini_by_task if item.get("task_family") == family]
            total_family_trials = sum(int(item.get("num_trials") or 0) for item in items)
            total_family_success = sum(int(item.get("num_success") or 0) for item in items)
            by_family[family] = {
                "task_family": family,
                "model_name": model_name,
                "checkpoint": checkpoint,
                "task_ids": sorted({task_id for item in items for task_id in (item.get("task_ids") or [item.get("task_id")]) if task_id}),
                "success_rate": total_family_success / total_family_trials if total_family_trials else None,
                "safe_success_rate": _weighted_mean([(item.get("safe_success_rate"), int(item.get("num_trials") or 0)) for item in items]),
                "num_trials": total_family_trials,
                "num_success": total_family_success,
                "failure_category_distribution": _merge_count_dicts([item.get("failure_category_distribution") for item in items]),
            }

        rows.append(
            {
                "leaderboard_id": f"{eval_group_id}_{task_ids[0] if task_ids else 'unknown_task'}",
                "run_id": eval_group_id,
                "eval_group_id": eval_group_id,
                "model_name": model_name,
                "checkpoint": checkpoint,
                "benchmark_version": group[0].get("benchmark_version"),
                "track": "multi_split" if len(ordered_splits) > 1 else (ordered_splits[0] if ordered_splits else group[0].get("track")),
                "available_splits": ordered_splits,
                "run_ids_by_split": {split: run_ids_by_split[split] for split in ordered_splits},
                "split_metrics": {split: split_metrics[split] for split in ordered_splits},
                "task_ids": task_ids,
                "task_names": task_names,
                "task_families": task_families,
                "task_label": ", ".join(task_names or task_ids) if (task_names or task_ids) else "unknown_task",
                "primary_task_id": task_ids[0] if task_ids else group[0].get("primary_task_id"),
                "primary_task_name": task_names[0] if task_names else group[0].get("primary_task_name"),
                "primary_task_family": _first_nonempty(task_families, group[0].get("primary_task_family") or "basic"),
                "leaderboard_group": group[0].get("leaderboard_group"),
                "success_logic": success_logic,
                "task_success_rate": task_success_rate,
                "overall_success_rate": task_success_rate,
                "success_rate": task_success_rate,
                "num_trials": total_trials,
                "num_success": total_success,
                "stage_completion_mean": stage_completion,
                "inference_latency_p95_ms": latency,
                "capability_index": capability_index,
                "failure_category_distribution": failure_category_distribution,
                "failure_code_distribution": failure_code_distribution,
                "drop_rate": _weighted_mean([(split_metrics[split].get("drop_rate"), int(split_metrics[split].get("num_trials") or 0)) for split in ordered_splits]),
                "collision_rate": _weighted_mean([(split_metrics[split].get("collision_rate"), int(split_metrics[split].get("num_trials") or 0)) for split in ordered_splits]),
                "wrong_place_rate": _weighted_mean([(split_metrics[split].get("wrong_place_rate"), int(split_metrics[split].get("num_trials") or 0)) for split in ordered_splits]),
                "timeout_rate": _weighted_mean([(split_metrics[split].get("timeout_rate"), int(split_metrics[split].get("num_trials") or 0)) for split in ordered_splits]),
                "by_family": by_family,
                "mini_leaderboards": {
                    "by_task": mini_by_task,
                    "by_family": mini_by_family,
                },
                "created_at": created_at,
                "overall": {
                    "sim_dev_macro_success_rate": split_metrics.get("sim_dev", {}).get("success_rate"),
                    "sim_clean_macro_success_rate": split_metrics.get("sim_clean", {}).get("success_rate"),
                    "sim_randomized_macro_success_rate": split_metrics.get("sim_randomized", {}).get("success_rate"),
                    "partial_success_rate": stage_completion,
                    "efficiency_score": efficiency,
                    "safety_score": 100.0,
                    "capability_index_v0_1": capability_index,
                },
                "capabilities": {
                    "success": success_score,
                    "partial_success": partial_score,
                    "efficiency": efficiency,
                    "safety": 100.0,
                },
            }
        )
    return rows


def _load_run_detail(run_dir: Path) -> dict[str, Any] | None:
    """Load the full detail blobs for a single run directory."""
    if not run_dir.exists():
        return None

    def _read_json(name: str) -> Any:
        p = run_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return None

    task_details = []
    static_dir = run_dir / "static_api"
    if static_dir.exists():
        for p in sorted(static_dir.glob("task_detail_*.json")):
            try:
                task_details.append(json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass

    return {
        "summary": _read_json("summary.json"),
        "suite_summary": _read_json("suite_summary.json"),
        "leaderboard_row": _read_json("leaderboard_row.json"),
        "suite_leaderboard_row": _read_json("suite_leaderboard_row.json"),
        "run_manifest": _read_json("run_manifest.json"),
        "suite_manifest": _read_json("suite_manifest.json"),
        "run_config": _read_json("run_config.json"),
        "task_summary": _read_json("task_summary.json"),
        "family_summary": _read_json("family_summary.json"),
        "task_score_table": _read_json("task_score_table.json"),
        "family_score_table": _read_json("family_score_table.json"),
        "model_detail": _read_json("static_api/model_detail.json"),
        "task_detail_T02": _read_json("static_api/task_detail_T02.json"),
        "task_details": task_details,
    }


def _load_episodes(run_dir: Path) -> list[dict[str, Any]]:
    """Return episode rows for a run (from episode_metrics.jsonl or episodes.json)."""
    episodes: list[dict[str, Any]] = []

    # Preferred: episode_metrics.jsonl (one JSON object per line)
    metrics_path = run_dir / "episode_metrics.jsonl"
    if metrics_path.exists():
        try:
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass

    if not episodes:
        # Fallback: static_api/episodes.json
        ep_path = run_dir / "static_api" / "episodes.json"
        if ep_path.exists():
            try:
                data = json.loads(ep_path.read_text(encoding="utf-8"))
                episodes = data.get("episodes", [])
            except (json.JSONDecodeError, OSError):
                pass

    return episodes


def _normalize_published_simulation(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize the lightweight published dataset snapshot to the unified section."""
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        return None
    rows = [
        {
            "simulator": "Isaac Sim",
            "robot_type": "Franka Panda",
            "task_id": str(item.get("name") or "unknown"),
            "task_name": str(item.get("task") or item.get("name") or "Unknown task"),
            "fps": int(item.get("fps") or 10),
            "dataset_batch_count": 1,
            "episode_count": int(item.get("episodes") or 0),
            "frame_count": int(item.get("steps") or 0),
            "duration_seconds": int(float(item.get("duration_s") or 0)),
            "scene_count": 1,
            "generation_date_start": "2026-08-26",
            "generation_date_end": "2026-08-26",
        }
        for item in datasets
        if isinstance(item, dict)
    ]
    return {
        "status": "active" if rows else "empty",
        "totals": {
            "dataset_batch_count": int(payload.get("dataset_count") or len(rows)),
            "episode_count": int(payload.get("total_episodes") or 0),
            "frame_count": int(payload.get("total_steps") or 0),
            "duration_seconds": sum(int(float(item.get("duration_s") or 0)) for item in datasets if isinstance(item, dict)),
            "scene_count": len(rows),
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------

def create_app(
    outputs_root: Path | None = None,
    release_outputs_root: Path | None = None,
    eval_python: str | None = None,
    eval_setup_cmd: str | None = None,
    dataset_statistics_path: Path | None = None,
    console_enabled: bool = True,
    public_base_url: str = "",
    public_media_enabled: bool | None = None,
) -> Flask:
    """Create and configure the Flask application.

    Parameters
    ----------
    outputs_root:
        Root directory under which per-run output folders live.
        Defaults to ``outputs/eval_runs`` relative to the current working
        directory, but can be pointed anywhere on the server filesystem.
    release_outputs_root:
        Root containing the independent release-protocol run directories.
        Defaults to ``outputs/release_collection`` so the public site can
        show release results while the official suite remains under
        ``outputs/eval_runs``.
    eval_python:
        Python executable used for Isaac evaluation subprocesses. This is often
        an Isaac Lab environment Python, while the Flask server may run in a
        lighter environment.
    eval_setup_cmd:
        Deprecated and ignored. Configure ``eval_python`` as an explicit Isaac
        Python/launcher executable so jobs never depend on shell concatenation.
    dataset_statistics_path:
        Unified dataset-statistics JSON served by ``/api/dataset_statistics``.
        The file is read on every request so regenerating it never requires a
        Flask restart.
    console_enabled:
        Expose the internal evaluation console and its ``/api/eval/*`` control
        endpoints. Public read-only deployments should disable this.
    public_base_url:
        Optional path prefix embedded in public-page links for static hosting,
        for example ``/LabBench`` on GitHub Pages.
    public_media_enabled:
        Override public preview media generation. ``None`` keeps configured
        media for internal deployments and disables it when the Console is
        disabled. Static exporters may explicitly enable reviewed media.
    """
    root = Path(outputs_root or _DEFAULT_OUTPUTS_ROOT).resolve()
    if release_outputs_root is not None:
        release_root = Path(release_outputs_root).resolve()
    elif outputs_root is None:
        release_root = _DEFAULT_RELEASE_OUTPUTS_ROOT.resolve()
    else:
        # Keep custom/test output roots isolated from the repository's default
        # release collection.  The normal ``outputs/eval_runs`` root still
        # resolves to its sibling ``outputs/release_collection`` directory.
        release_root = (root.parent / "release_collection").resolve()
    root.mkdir(parents=True, exist_ok=True)
    eval_python_cmd = eval_python or sys.executable
    merge_default_simulation = dataset_statistics_path is None
    statistics_path = Path(dataset_statistics_path or _DEFAULT_DATASET_STATISTICS_PATH).resolve()
    eval_queue = PersistentEvalQueue(root / "_jobs")

    app = Flask(
        __name__,
        template_folder=str(_TEMPLATES),
    )
    app.config["OUTPUTS_ROOT"] = str(root)
    app.config["RELEASE_OUTPUTS_ROOT"] = str(release_root)
    app.config["EVAL_PYTHON"] = eval_python_cmd
    app.config["EVAL_QUEUE"] = eval_queue
    app.config["EVAL_SETUP_CMD_IGNORED"] = eval_setup_cmd
    app.config["DATASET_STATISTICS_PATH"] = str(statistics_path)
    app.config["CONSOLE_ENABLED"] = bool(console_enabled)
    app.config["PUBLIC_BASE_URL"] = str(public_base_url).rstrip("/")
    app.config["PUBLIC_MEDIA_ENABLED"] = public_media_enabled
    if hasattr(app, "json") and hasattr(app.json, "ensure_ascii"):
        app.json.ensure_ascii = False

    @app.before_request
    def _guard_internal_control_api() -> Any:
        if not console_enabled and request.path.startswith("/api/") and not request.path.startswith("/api/public/"):
            return jsonify({"error": "internal results are not public"}), 404
        if not console_enabled and request.path.startswith("/api/eval/"):
            return jsonify({"error": "evaluation console is disabled"}), 404
        if not console_enabled and request.path.startswith("/static/web_assets/"):
            return "Public media is not distributed", 404
        return None

    @app.after_request
    def _force_utf8(response: Any) -> Any:
        if response.mimetype in {"text/html", "application/json"}:
            response.headers["Content-Type"] = f"{response.mimetype}; charset=utf-8"
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    # ------------------------------------------------------------------
    # Static frontend
    # ------------------------------------------------------------------

    def _public_dataset_statistics() -> dict[str, Any] | None:
        try:
            payload = json.loads(statistics_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") is None:
            simulation = _normalize_published_simulation(payload)
            payload = {
                "schema_version": _DATASET_STATISTICS_SCHEMA,
                "generated_at": str(payload.get("generated_at") or "2026-08-26"),
                "simulation": simulation,
            }
        elif merge_default_simulation:
            try:
                published = json.loads(_DEFAULT_SIMULATION_STATISTICS_PATH.read_text(encoding="utf-8"))
                simulation = _normalize_published_simulation(published)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                simulation = None
            if simulation is not None:
                payload = dict(payload)
                payload["simulation"] = simulation
                payload["generated_at"] = max(
                    str(payload.get("generated_at") or ""),
                    str(published.get("generated_at") or ""),
                )
        if not isinstance(payload, dict) or payload.get("schema_version") != _DATASET_STATISTICS_SCHEMA:
            return None
        if _dataset_statistics_contract_errors(payload):
            return None
        return payload

    def _public_snapshot() -> dict[str, Any]:
        media_override = public_media_enabled
        if media_override is None and not console_enabled:
            media_override = False
        release_runs = _scan_release_runs(root)
        if release_root != root:
            seen = {str(item.get("run_id")) for item in release_runs}
            release_runs.extend(
                item for item in _scan_release_runs(release_root)
                if str(item.get("run_id")) not in seen
            )
        return build_public_snapshot(
            _scan_runs(root),
            dataset_statistics=_public_dataset_statistics(),
            public_media_enabled=media_override,
            release_runs=release_runs,
        )

    def _format_duration(value: Any) -> str:
        seconds = max(0, int(value or 0))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:,}:{minutes:02d}:{seconds:02d}"

    def _public_context(snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        snapshot = snapshot or _public_snapshot()
        tasks = snapshot["catalog"]["tasks"]
        objects = snapshot["objects"]["objects"]
        track_titles = {
            "long_horizon": "Capability Leaderboard 1 · Long-Horizon Procedures",
            "reasoning_in_action": "Capability Leaderboard 2 · Reasoning in Action",
        }
        track_descriptions = {
            "long_horizon": "Measures reliable multi-stage laboratory procedures, ordering and state-dependent transitions.",
            "reasoning_in_action": "Measures visual and structured-state understanding with correct action selection.",
        }
        track_cards = []
        for track in snapshot["catalog"]["tracks"]:
            track_tasks = [task for task in tasks if task["track_id"] == track["track_id"]]
            track_cards.append(
                {
                    "track": track,
                    "title_en": track_titles[track["track_id"]],
                    "description_en": track_descriptions[track["track_id"]],
                    "available": sum(task["availability"] == "available" for task in track_tasks),
                    "planned": sum(task["availability"] == "planned" for task in track_tasks),
                    "task_count": len(track_tasks),
                }
            )
        track_task_counts = {
            track["track_id"]: sum(1 for task in tasks if task["track_id"] == track["track_id"])
            for track in snapshot["catalog"]["tracks"]
        }
        section_specs = (
            ("simulation", "Simulation collection", "Isaac Sim Simulation", "Physics-based Franka Panda task trajectories", "Episodes", "episode_count"),
        )
        statistics = snapshot.get("dataset_statistics") or {}
        dataset_sections = []
        for section_id, kicker, title, description, primary_label, primary_key in section_specs:
            section = statistics.get(section_id) if isinstance(statistics, dict) else None
            section = section if isinstance(section, dict) else {}
            totals = section.get("totals") if isinstance(section.get("totals"), dict) else {}
            dataset_sections.append(
                {
                    "kicker": kicker,
                    "title": title,
                    "description": description,
                    "primary_label": primary_label,
                    "primary_value": f"{int(totals.get(primary_key) or 0):,}",
                    "duration": _format_duration(totals.get("duration_seconds")),
                    "status": str(section.get("status") or "not published").replace("_", " ").title(),
                    "rows": section.get("rows") or [],
                }
            )
        link_labels = {
            "paper": "Paper", "code": "Code", "dataset": "Dataset",
            "documentation": "Documentation",
        }
        active_links = [
            (link_labels[key], value)
            for key, value in snapshot["site"].get("links", {}).items()
            if key in link_labels and value
        ]
        score_metric_labels = {
            "success": "Completion",
            "geometry_quality_score_100": "Geometry quality",
            "force_quality_score_100": "Force quality",
            "quantity_quality_score_100": "Quantity accuracy",
            "sequence_quality_score_100": "Sequence quality",
            "subgoal_quality_score_100": "Subgoal completion",
            "evidence_quality_score_100": "Evidence quality",
            "efficiency_score_100": "Efficiency",
        }
        available_tasks = [
            task for task in tasks if task["availability"] == "available"
        ]
        featured_tasks = [
            task
            for track in snapshot["catalog"]["tracks"]
            for task in [
                candidate
                for candidate in available_tasks
                if candidate["track_id"] == track["track_id"]
            ][:3]
        ]
        return {
            "snapshot": snapshot,
            "base_url": app.config["PUBLIC_BASE_URL"],
            "console_enabled": bool(console_enabled),
            "featured_tasks": featured_tasks,
            "featured_objects": sorted([item for item in objects if item["status"] == "available"],
                                       key=lambda item: not bool(item.get("special_families")))[:8],
            "track_cards": track_cards,
            "track_task_counts": track_task_counts,
            "track_title_en": track_titles,
            "track_description_en": track_descriptions,
            "dataset_sections": dataset_sections,
            "active_links": active_links,
            "score_metric_labels": score_metric_labels,
            "release_protocols": snapshot.get("release_protocols", []),
            "special_families": snapshot.get("special_families", []),
            "release_leaderboard": snapshot.get("release_leaderboard", {}),
        }

    @app.route("/")
    def index():
        return render_template("public_home.html", **_public_context())

    @app.route("/tasks/")
    def public_tasks_page():
        return render_template("public_tasks.html", **_public_context())

    @app.route("/tasks/<task_id>/")
    def public_task_page(task_id: str):
        snapshot = _public_snapshot()
        task = public_task(snapshot, task_id)
        if task is None:
            return render_template("public_not_found.html", kind="task", value=task_id, **_public_context(snapshot)), 404
        object_by_id = {item["object_id"]: item for item in snapshot["objects"]["objects"]}
        # Planned task definitions may mention roadmap objects that are
        # intentionally absent from the public object catalogue.  Keep the
        # task page renderable and show only currently published assets.
        related_objects = [
            object_by_id[value]
            for value in task.get("object_ids") or []
            if value in object_by_id
        ]
        return render_template(
            "public_task_detail.html", task=task, related_objects=related_objects,
            **_public_context(snapshot),
        )

    @app.route("/tasks/protocol/<protocol_id>/")
    def public_release_protocol_page(protocol_id: str):
        snapshot = _public_snapshot()
        protocol = public_release_protocol(snapshot, protocol_id)
        if protocol is None:
            return render_template(
                "public_not_found.html", kind="release protocol", value=protocol_id,
                **_public_context(snapshot),
            ), 404
        runs = [
            item for item in (_scan_release_runs(root) + ([] if release_root == root else _scan_release_runs(release_root)))
            if item.get("protocol_id") == protocol["protocol_id"]
        ]
        return render_template(
            "public_release_protocol_detail.html",
            protocol=protocol,
            public_family=next((item for item in snapshot["special_families"] if item["family"] == protocol["family"]), None),
            related_objects=[item for item in snapshot["objects"]["objects"] if any(
                family["family"] == protocol["family"] and item["object_id"] in family.get("object_ids", [])
                for family in snapshot["special_families"])],
            protocol_runs=runs,
            **_public_context(snapshot),
        )

    @app.route("/objects/")
    def public_objects_page():
        return render_template("public_objects.html", **_public_context())

    @app.route("/objects/<object_id>/")
    def public_object_page(object_id: str):
        snapshot = _public_snapshot()
        item = public_object(snapshot, object_id)
        if item is None:
            return render_template("public_not_found.html", kind="object", value=object_id, **_public_context(snapshot)), 404
        task_by_id = {task["task_id"]: task for task in snapshot["catalog"]["tasks"]}
        related_tasks = [task_by_id[value] for value in item.get("task_ids") or []]
        return render_template(
            "public_object_detail.html", object=item, related_tasks=related_tasks,
            **_public_context(snapshot),
        )

    @app.route("/dataset/")
    def public_dataset_page():
        return render_template("public_dataset.html", **_public_context())

    @app.route("/leaderboard/")
    def public_leaderboard_page():
        return render_template("public_leaderboard.html", **_public_context())

    @app.route("/docs/")
    def public_docs_page():
        return render_template("public_docs.html", **_public_context())

    @app.route("/console")
    def console():
        if not console_enabled:
            return "Console is disabled", 404
        return render_template("index.html")

    @app.route('/console/batches')
    def console_batches():
        if not console_enabled:
            return 'Console is disabled', 404
        reports = []
        seen = set()
        for folder in (root, release_root):
            for path in folder.rglob('summary.json'):
                if path in seen:
                    continue
                seen.add(path)
                payload = _read_json_path(path)
                if isinstance(payload, dict) and payload.get('schema_version') == 'labference_batch_summary/v1':
                    reports.append(payload)
        return render_template('batch_reports.html', reports=reports)

    @app.route("/api/public/site")
    def api_public_site():
        return jsonify(_public_snapshot())

    @app.route("/api/public/tasks")
    def api_public_tasks():
        snapshot = _public_snapshot()
        catalog = dict(snapshot["catalog"])
        catalog["release_protocols"] = snapshot.get("release_protocols", [])
        return jsonify(catalog)

    @app.route("/api/public/release-protocols")
    def api_public_release_protocols():
        snapshot = _public_snapshot()
        return jsonify({
            "schema_version": "labbench_release_protocols_public/v1",
            "suite_id": "labference_release_protocols_v1",
            "protocols": snapshot.get("release_protocols", []),
        })

    @app.route("/api/public/release-protocols/<protocol_id>")
    def api_public_release_protocol(protocol_id: str):
        protocol = public_release_protocol(_public_snapshot(), protocol_id)
        return jsonify(protocol) if protocol is not None else (jsonify({"error": "unknown release protocol"}), 404)

    @app.route("/api/public/tasks/<task_id>")
    def api_public_task(task_id: str):
        task = public_task(_public_snapshot(), task_id)
        return jsonify(task) if task is not None else (jsonify({"error": "unknown task"}), 404)

    @app.route("/api/public/objects")
    def api_public_objects():
        return jsonify(_public_snapshot()["objects"])

    @app.route("/api/public/objects/<object_id>")
    def api_public_object(object_id: str):
        item = public_object(_public_snapshot(), object_id)
        return jsonify(item) if item is not None else (jsonify({"error": "unknown object"}), 404)

    @app.route("/api/public/leaderboard")
    def api_public_leaderboard():
        return jsonify({"error": "ordinary_results_private", "replacement": "/api/public/release-leaderboard"}), 410

    @app.route("/api/public/release-leaderboard")
    def api_public_release_leaderboard():
        return jsonify(_public_snapshot().get("release_leaderboard", {}))

    # ------------------------------------------------------------------
    # REST API
    # ------------------------------------------------------------------

    @app.route("/api/runs")
    def api_runs():
        """List all evaluation runs (shallow summaries)."""
        return jsonify(_scan_runs(root))

    @app.route("/api/runs/query")
    def api_runs_query():
        """Return a filtered, bounded page of evaluation run summaries."""
        try:
            limit = int(request.args.get("limit", 50))
            offset = int(request.args.get("offset", 0))
        except (TypeError, ValueError):
            return jsonify({"error": "limit and offset must be integers"}), 400
        return jsonify(
            _query_runs(
                _scan_runs(root),
                limit=limit,
                offset=offset,
                model_name=request.args.get("model_name", ""),
                task_id=request.args.get("task_id", ""),
                split=request.args.get("split", ""),
            )
        )

    @app.route("/api/runs/<run_id>")
    def api_run_detail(run_id: str):
        """Full detail for a single run."""
        run_dir = _safe_run_dir(root, run_id)
        if run_dir is None:
            return jsonify({"error": f"Invalid run id: {run_id}"}), 400
        detail = _load_run_detail(run_dir)
        if detail is None:
            return jsonify({"error": f"Run '{run_id}' not found"}), 404
        return jsonify(detail)

    @app.route("/api/runs/<run_id>/episodes")
    def api_run_episodes(run_id: str):
        """Episode-level rows for a single run."""
        run_dir = _safe_run_dir(root, run_id)
        if run_dir is None:
            return jsonify({"error": f"Invalid run id: {run_id}"}), 400
        if not run_dir.exists():
            return jsonify({"error": f"Run '{run_id}' not found"}), 404
        return jsonify(_load_episodes(run_dir))

    @app.route("/api/leaderboard")
    def api_leaderboard():
        """Suite leaderboard when available, otherwise legacy single-task rows."""
        suite_rows = _scan_suite_runs(root)
        if suite_rows:
            runs = suite_rows
            mode = "suite"
        else:
            runs = _aggregate_leaderboard_rows(_scan_runs(root))
            runs.sort(key=lambda r: (r.get("capability_index") is not None, r.get("capability_index") or 0), reverse=True)
            mode = "single_task"

        for idx, run in enumerate(runs, start=1):
            run["rank"] = idx

        return jsonify(
            {
                "updated_at": datetime.now().astimezone().isoformat(),
                "total_runs": len(runs),
                "leaderboard_mode": mode,
                "rows": runs,
            }
        )

    def _laboratory_dashboard() -> dict[str, Any]:
        return build_benchmark_dashboard(_scan_runs(root), load_benchmark_catalog())

    @app.route("/api/benchmark/catalog")
    def api_benchmark_catalog():
        """Return the versioned L/R task and atomic-action catalog."""
        return jsonify(load_benchmark_catalog())

    @app.route("/api/benchmark/leaderboard")
    def api_benchmark_leaderboard():
        """Return model-level overall, track and atomic-action scores."""
        payload = _laboratory_dashboard()
        payload["updated_at"] = datetime.now().astimezone().isoformat()
        return jsonify(payload)

    @app.route("/api/benchmark/tracks/<track_id>")
    def api_benchmark_track(track_id: str):
        try:
            return jsonify(benchmark_track_payload(_laboratory_dashboard(), track_id))
        except KeyError:
            return jsonify({"error": f"Unknown benchmark track: {track_id}"}), 404

    @app.route("/api/benchmark/atomic-actions")
    def api_benchmark_atomic_actions():
        dashboard = _laboratory_dashboard()
        return jsonify(
            {
                "atomic_actions": dashboard["atomic_actions"],
                "models": [
                    {
                        "model_id": model["model_id"],
                        "model_name": model["model_name"],
                        "checkpoint": model["checkpoint"],
                        "atomic_success_rate": model["atomic_success_rate"],
                        "atomic_coverage": model["atomic_coverage"],
                        "atomic_actions": model["atomic_actions"],
                    }
                    for model in dashboard["models"]
                ],
            }
        )

    @app.route("/api/benchmark/models/<model_id>")
    def api_benchmark_model(model_id: str):
        dashboard = _laboratory_dashboard()
        model = next((item for item in dashboard["models"] if item["model_id"] == model_id), None)
        if model is None:
            return jsonify({"error": f"Unknown benchmark model: {model_id}"}), 404
        return jsonify(model)

    @app.route("/api/task_catalog")
    def api_task_catalog():
        suite_id = request.args.get("suite") or DEFAULT_SUITE_ID
        try:
            return jsonify(load_task_catalog(suite_id))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/task_scores")
    def api_task_scores():
        suite_run_id = request.args.get("suite_run_id")
        rows = _scan_suite_runs(root)
        if suite_run_id:
            rows = [row for row in rows if row.get("run_id") == suite_run_id or row.get("eval_group_id") == suite_run_id]
        if not rows:
            return jsonify({"suite_run_id": suite_run_id, "suite_summary": None, "tasks": [], "families": []})
        row = rows[0]
        return jsonify(
            {
                "suite_run_id": row.get("run_id"),
                "suite_id": row.get("suite_id"),
                "suite_summary": row.get("suite_summary") or {},
                "tasks": row.get("task_score_table") or [],
                "families": row.get("family_score_table") or [],
            }
        )

    @app.route("/api/dataset_statistics")
    def api_dataset_statistics():
        """Return the latest validated dataset-statistics snapshot.

        This endpoint deliberately performs no process-level caching: the XLSX
        build command can atomically replace the JSON while the dashboard is
        running, and the next poll will immediately see the new snapshot.
        """
        normalized = _public_dataset_statistics()
        if normalized is not None:
            return jsonify(normalized)
        try:
            payload = json.loads(statistics_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return (
                jsonify(
                    {
                        "error": {
                            "code": "dataset_statistics_not_found",
                            "message": "Dataset statistics JSON has not been generated.",
                            "path": str(statistics_path),
                        }
                    }
                ),
                404,
            )
        except json.JSONDecodeError as exc:
            return (
                jsonify(
                    {
                        "error": {
                            "code": "dataset_statistics_invalid_json",
                            "message": f"Dataset statistics JSON is invalid: {exc.msg}.",
                            "path": str(statistics_path),
                        }
                    }
                ),
                500,
            )
        except OSError as exc:
            return (
                jsonify(
                    {
                        "error": {
                            "code": "dataset_statistics_read_failed",
                            "message": f"Dataset statistics JSON could not be read: {exc}.",
                            "path": str(statistics_path),
                        }
                    }
                ),
                500,
            )

        missing_sections = [
            section
            for section in _DATASET_STATISTICS_SECTIONS
            if not isinstance(payload, dict) or not isinstance(payload.get(section), dict)
        ]
        contract_errors = (
            _dataset_statistics_contract_errors(payload)
            if isinstance(payload, dict)
            else ["root"]
        )
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _DATASET_STATISTICS_SCHEMA
            or missing_sections
            or contract_errors
        ):
            return (
                jsonify(
                    {
                        "error": {
                            "code": "dataset_statistics_schema_incompatible",
                            "message": (
                                f"Expected {_DATASET_STATISTICS_SCHEMA} with sections: "
                                f"{', '.join(_DATASET_STATISTICS_SECTIONS)}."
                            ),
                            "actual_schema": payload.get("schema_version") if isinstance(payload, dict) else None,
                            "missing_or_invalid_sections": missing_sections,
                            "invalid_contract_fields": contract_errors,
                            "path": str(statistics_path),
                        }
                    }
                ),
                422,
            )

        return jsonify(payload)

    # ------------------------------------------------------------------
    # Evaluation trigger (persistent single-worker FIFO)
    # ------------------------------------------------------------------

    def _benchmark_command(body: dict[str, Any], *, run_id: str, checkpoint: str, run_suite: bool) -> list[str]:
        module = "eai4s_eval.cli.run_benchmark_suite_eval" if run_suite else "eai4s_eval.cli.run_benchmark_eval"
        provider = str(body.get("policy_provider") or "openpi").lower()
        default_policy_port = 8765 if provider == "gr00t" else 8000
        cmd = [
            eval_python_cmd,
            "-m",
            module,
            "--benchmark-root",
            str(body.get("benchmark_root") or _BENCHMARK_ROOT),
            "--policy-profile",
            str(body.get("policy_profile") or _default_policy_profile(provider)),
            "--policy-provider",
            provider,
            "--policy-backend",
            str(body.get("policy_backend") or ("zmq" if body.get("policy_provider") == "gr00t" else "websocket")),
            "--policy-timeout",
            str(body.get("policy_timeout") or 30.0),
            "--vla-host",
            str(body.get("vla_host") or "127.0.0.1"),
            "--vla-port",
            str(body.get("vla_port") or default_policy_port),
            "--model-name",
            resolve_model_name(provider, body.get("model_name")),
            "--checkpoint",
            checkpoint,
            "--num-episodes",
            str(body.get("num_episodes") or 5),
            "--split",
            str(body.get("split") or "sim_dev"),
            "--run-id",
            run_id,
            "--output-root",
            str(root),
            "--backend",
            str(body.get("simulation_backend") or "numpy"),
        ]
        if run_suite:
            cmd.extend(["--task-suite", str(body.get("task_suite") or default_official_suite_id())])
            if body.get("multi_split"):
                cmd.append("--all-splits")
        else:
            cmd.extend(
                [
                    "--task-spec",
                    str(body.get("task_spec") or (_CONFIG_ROOT / "tasks" / "B01_level1_pick.yaml")),
                    "--action-horizon",
                    str(body.get("action_horizon") or 40),
                ]
            )
            if body.get("config_name"):
                cmd.extend(["--config-name", str(body["config_name"])])
        if body.get("headless", True):
            cmd.append("--headless")
        if body.get("save_video"):
            cmd.append("--save-video")
        for key, flag in (
            ("api_key", "--api-key"),
            ("seeds", "--seeds"),
            ("seed_manifest", "--seed-manifest"),
            ("checkpoint_hash", "--checkpoint-hash"),
            ("dataset_version", "--dataset-version"),
        ):
            if body.get(key) not in (None, ""):
                cmd.extend([flag, str(body[key])])
        return cmd

    def _legacy_command(body: dict[str, Any], *, run_id: str, checkpoint: str, run_suite: bool, task_suite: str) -> list[str]:
        cmd = [
            eval_python_cmd,
            "-m",
            "eai4s_eval.cli.run_task_suite_eval" if run_suite else "eai4s_eval.cli.run_pick_lift_eval",
            "--model_name", str(body.get("model_name") or "eai4s_model"),
            "--checkpoint", checkpoint,
            "--vla_host", str(body.get("vla_host", "127.0.0.1")),
            "--vla_port", str(body.get("vla_port", 8188)),
            "--robot", str(body.get("robot", "franka_panda")),
            "--num_episodes", str(body.get("num_episodes", 5)),
            "--instruction", str(body.get("instruction", "pick glass bottle")),
            "--target_object", str(body.get("target_object", "vial")),
            "--max_steps", str(body.get("max_steps", 300)),
            "--split", str(body.get("split", "sim_dev")),
            "--run_id", run_id,
            "--output_root", str(root),
            "--policy_api", str(body.get("policy_api", "auto")),
            "--action_horizon", str(body.get("action_horizon", 16)),
            "--reaction_interpolation_steps", str(body.get("reaction_interpolation_steps", 1)),
            "--sim_steps_per_action", str(body.get("sim_steps_per_action", 6)),
            "--initial_hold_steps", str(body.get("initial_hold_steps", 100)),
            "--stabilization_max_steps", str(body.get("stabilization_max_steps", 200)),
            "--stabilization_hold_steps", str(body.get("stabilization_hold_steps", 10)),
            "--stabilization_object_lin_vel_threshold", str(body.get("stabilization_object_lin_vel_threshold", 0.02)),
            "--stabilization_object_ang_vel_threshold", str(body.get("stabilization_object_ang_vel_threshold", 0.20)),
            "--stabilization_joint_vel_threshold", str(body.get("stabilization_joint_vel_threshold", 0.02)),
            "--control_hz", str(body.get("control_hz", 10)),
            "--predicate_log_interval", str(body.get("predicate_log_interval", 25)),
            "--shutdown_timeout_sec", str(body.get("shutdown_timeout_sec", 30.0)),
            "--max_delta_xyz", str(body.get("max_delta_xyz", 0.03)),
        ]
        if run_suite:
            cmd.extend(["--task_suite", task_suite])
        if body.get("headless", True):
            cmd.append("--headless")
        cmd.append("--enable_cameras")
        if body.get("use_policy_orientation"):
            cmd.append("--use_policy_orientation")
        for key, flag in (
            ("success_hold_steps", "--success_hold_steps"),
            ("success_height_threshold", "--success_height_threshold"),
            ("success_distance_threshold", "--success_distance_threshold"),
            ("success_grip_threshold", "--success_grip_threshold"),
            ("seeds", "--seeds"),
            ("device", "--device"),
            ("asset_root", "--asset_root"),
            ("task_spec", "--task_spec"),
            ("seed_manifest", "--seed_manifest"),
            ("checkpoint_hash", "--checkpoint_hash"),
            ("dataset_version", "--dataset_version"),
        ):
            if body.get(key) not in (None, ""):
                cmd.extend([flag, str(body[key])])
        if body.get("multi_split"):
            cmd.append("--all-splits")
        if body.get("skip_isaac_close"):
            cmd.append("--skip_isaac_close")
        return cmd

    @app.route("/api/eval/run", methods=["POST"])
    def api_eval_run():
        """Trigger a new evaluation run asynchronously.

        Expects JSON body with optional keys:
            model_name, checkpoint, vla_host, vla_port, robot, num_episodes,
            instruction, target_object, max_steps, split, seeds, asset_root,
            policy_api, action_horizon, initial_hold_steps, control_hz,
            reaction_interpolation_steps, sim_steps_per_action,
            success_hold_steps, max_delta_xyz, task_spec, seed_manifest
        """
        body = dict(request.get_json(silent=True) or {})

        eval_time_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        if "run_suite" in body:
            run_suite = bool(body.get("run_suite"))
        else:
            task_mode = str(body.get("task_mode") or body.get("task") or "").strip().lower()
            run_suite = task_mode in {"suite", "full_suite"} or bool(body.get("task_suite"))
        provider = str(body.get("policy_provider") or "openpi").lower()
        try:
            resolved_model_name = resolve_model_name(provider, body.get("model_name"))
            checkpoint_value = validate_checkpoint_name(body.get("checkpoint"), required=run_suite)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        model_name = re.sub(r"[^A-Za-z0-9_.=-]+", "_", resolved_model_name).strip("_")
        checkpoint = str(checkpoint_value or eval_time_id)
        body["model_name"] = resolved_model_name
        body["checkpoint"] = checkpoint
        if not run_suite:
            body = _apply_eval_task_defaults(body)
        task_suite = str(body.get("task_suite") or (default_official_suite_id() if run_suite else DEFAULT_SUITE_ID))
        run_id = f"{eval_time_id}_{model_name}_{task_suite}" if run_suite else f"{eval_time_id}_{model_name}"
        requested_backend = str(body.get("backend") or "")
        benchmark_mode = requested_backend == "eai4s_benchmark" or (
            not requested_backend and (task_suite.startswith("eai4s_benchmark") or bool(body.get("config_name")))
        )
        cmd = (
            _benchmark_command(body, run_id=run_id, checkpoint=checkpoint, run_suite=run_suite)
            if benchmark_mode
            else _legacy_command(body, run_id=run_id, checkpoint=checkpoint, run_suite=run_suite, task_suite=task_suite)
        )
        job = eval_queue.enqueue(
            run_id=run_id,
            command=cmd,
            cwd=_BENCHMARK_ROOT,
            metadata={"backend": "eai4s_benchmark" if benchmark_mode else "isaaclab", "task_suite": task_suite},
        )
        return jsonify({"job_id": job["job_id"], "run_id": run_id, "status": "queued", "command": _shell_join(cmd)}), 202

    @app.route("/api/eval/status/<run_id>")
    def api_eval_status(run_id: str):
        """Poll the status of an async evaluation run."""
        info = eval_queue.get(run_id)
        if info is None:
            return jsonify({"error": f"No such evaluation run: {run_id}"}), 404
        progress_path = root / run_id / "progress.json"
        if progress_path.exists():
            try:
                info["progress"] = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        return jsonify(info)

    @app.route("/api/eval/jobs")
    def api_eval_jobs():
        jobs = eval_queue.list()
        for job in jobs:
            progress_path = root / str(job.get("run_id") or "") / "progress.json"
            if progress_path.exists():
                try:
                    job["progress"] = json.loads(progress_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
        return jsonify({"jobs": jobs})

    @app.route("/api/evaluation_catalog")
    def api_evaluation_catalog():
        try:
            registry = load_official_evaluation_suites()
            suite_ids = registry["suite_ids"]
            suites = [load_task_catalog(suite_id) for suite_id in suite_ids]
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return jsonify({"error": f"Official evaluation catalog is invalid: {exc}"}), 500
        profiles = [
            {"profile_id": path.stem, "path": str(path)}
            for path in sorted((_CONFIG_ROOT / "policies").glob("*.yaml"))
        ]
        configs = [path.stem for path in sorted((_BENCHMARK_ROOT / "config").glob("*.yaml"))]
        return jsonify(
            {
                "backends": ["eai4s_benchmark", "isaaclab"],
                "policy_providers": [
                    {"id": "openpi", "protocol": "websocket", "default_port": 8000},
                    {"id": "gr00t", "protocol": "zmq", "default_port": 8765},
                ],
                "suites": suites,
                "default_suite_id": registry["default_suite_id"],
                "policy_profiles": profiles,
                "environment_configs": configs,
                "splits": list(_SPLIT_ORDER),
                "defaults": {
                    "debug_episodes": 5,
                    "leaderboard_episodes": 50,
                    "action_dim": 8,
                    "action_space": "ee_pos",
                    "openpi_policy_profile": str(_default_policy_profile("openpi")),
                    "gr00t_policy_profile": str(_default_policy_profile("gr00t")),
                },
            }
        )

    @app.route("/api/eval/tasks")
    def api_eval_tasks():
        suite_id = request.args.get("suite") or "eai4s_benchmark_core_v1"
        try:
            return jsonify(load_task_catalog(suite_id))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/eval/policy_profiles")
    def api_eval_policy_profiles():
        return jsonify(
            {
                "profiles": [
                    {"profile_id": path.stem, "path": str(path)}
                    for path in sorted((_CONFIG_ROOT / "policies").glob("*.yaml"))
                ]
            }
        )

    @app.route("/api/eval/policy/health")
    def api_policy_health():
        import socket

        provider = str(request.args.get("provider") or "openpi").lower()
        host = str(request.args.get("host") or "127.0.0.1")
        port = int(request.args.get("port") or (8765 if provider == "gr00t" else 8000))
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return jsonify({"status": "reachable", "provider": provider, "host": host, "port": port})
        except OSError as exc:
            return jsonify({"status": "unreachable", "provider": provider, "host": host, "port": port, "error": str(exc)}), 503

    @app.route("/api/eval/openpi/health")
    def api_openpi_health():
        # Backward-compatible route used by older dashboard clients.
        return api_policy_health()

    @app.route("/api/eval/preflight", methods=["POST"])
    def api_eval_preflight():
        import socket

        body = dict(request.get_json(silent=True) or {})
        provider = str(body.get("policy_provider") or "openpi").lower()
        host = str(body.get("vla_host") or "127.0.0.1")
        port = int(body.get("vla_port") or (8765 if provider == "gr00t" else 8000))
        profile_path = Path(
            str(body.get("policy_profile") or _default_policy_profile(provider))
        ).resolve()
        try:
            import yaml  # type: ignore

            profile = yaml.safe_load(profile_path.read_text(encoding="utf-8-sig")) or {}
            source_cameras = sorted(str(key) for key in (profile.get("observation_map") or {}))
            profile_check = {
                "ok": int(profile.get("action_dim", -1)) == 8 and bool(source_cameras),
                "path": str(profile_path),
                "action_dim": profile.get("action_dim"),
            }
        except Exception as exc:
            source_cameras = []
            profile_check = {"ok": False, "path": str(profile_path), "error": str(exc)}

        config_names: list[str] = []
        task_spec_value = body.get("task_spec")
        if task_spec_value:
            task_path = Path(str(task_spec_value))
            if not task_path.exists():
                task_path = _CONFIG_ROOT / "tasks" / task_path.name
            try:
                task_data = yaml.safe_load(task_path.read_text(encoding="utf-8-sig")) or {}
                config_names.append(str((task_data.get("runtime") or {}).get("env_config") or ""))
            except Exception:
                pass
        else:
            suite_id = str(body.get("task_suite") or "eai4s_benchmark_core_v1")
            try:
                config_names.extend(
                    str((yaml.safe_load(Path(task["task_spec"]).read_text(encoding="utf-8-sig")) or {}).get("runtime", {}).get("env_config") or "")
                    for task in load_task_catalog(suite_id).get("tasks", [])
                    if task.get("status") == "implemented"
                )
            except Exception:
                pass
        missing_by_config: dict[str, list[str]] = {}
        for config_name in sorted(set(name for name in config_names if name)):
            try:
                config = yaml.safe_load((_BENCHMARK_ROOT / "config" / f"{config_name}.yaml").read_text(encoding="utf-8-sig")) or {}
                available = {
                    f"{camera.get('name')}_{kind}"
                    for camera in config.get("cameras") or []
                    for kind in str(camera.get("image_type", "rgb")).split("+")
                }
                missing = sorted(set(source_cameras).difference(available))
                if missing:
                    missing_by_config[config_name] = missing
            except Exception as exc:
                missing_by_config[config_name] = [str(exc)]

        gpu_check: dict[str, Any]
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                probe = subprocess.run(
                    [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5, check=False,
                )
                gpu_check = {"ok": probe.returncode == 0, "devices": probe.stdout.strip(), "error": probe.stderr.strip()}
            except Exception as exc:
                gpu_check = {"ok": False, "error": str(exc)}
        else:
            gpu_check = {"ok": None, "message": "nvidia-smi is unavailable in the dashboard environment; Isaac will recheck GPU access."}
        checks: dict[str, Any] = {
            "benchmark_root": {"ok": _BENCHMARK_ROOT.is_dir(), "path": str(_BENCHMARK_ROOT)},
            "assets": {"ok": (_BENCHMARK_ROOT / "assets").exists(), "path": str(_BENCHMARK_ROOT / "assets")},
            "policy_profile": profile_check,
            "camera_mapping": {"ok": not missing_by_config and bool(source_cameras), "sources": source_cameras, "missing": missing_by_config},
            "gpu": gpu_check,
        }
        try:
            with socket.create_connection((host, port), timeout=1.0):
                checks["policy"] = {"ok": True, "provider": provider, "host": host, "port": port}
        except OSError as exc:
            checks["policy"] = {"ok": False, "provider": provider, "host": host, "port": port, "error": str(exc)}
        return jsonify({"ok": all(item.get("ok") is not False for item in checks.values()), "checks": checks})

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.route("/api/health")
    def api_health():
        return jsonify({"status": "ok", "outputs_root": str(root), "release_outputs_root": str(release_root), "eval_python": eval_python_cmd})

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EAI4S-bench web server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=5000, help="TCP port")
    parser.add_argument("--outputs-root", type=str, default=None, help="Directory containing eval run outputs")
    parser.add_argument(
        "--release-outputs-root",
        type=str,
        default=None,
        help="Directory containing independent release-protocol run outputs",
    )
    parser.add_argument(
        "--eval-python",
        type=str,
        default=None,
        help="Python executable used to launch Isaac evaluation subprocesses. Defaults to this server's Python.",
    )
    parser.add_argument(
        "--eval-setup-cmd",
        type=str,
        default=None,
        help="Deprecated; use --eval-python with an explicit Isaac launcher executable.",
    )
    parser.add_argument(
        "--disable-console",
        action="store_true",
        help="Serve the public portal without the internal evaluation Console route.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args(argv)

    app = create_app(
        outputs_root=args.outputs_root,
        release_outputs_root=args.release_outputs_root,
        eval_python=args.eval_python,
        eval_setup_cmd=args.eval_setup_cmd,
        console_enabled=not args.disable_console,
    )
    print(f"EAI4S-bench server starting on http://{args.host}:{args.port}")
    print(f"Outputs root: {app.config['OUTPUTS_ROOT']}")
    print(f"Eval Python: {app.config['EVAL_PYTHON']}")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
