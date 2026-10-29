from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eai4s_eval.eval.task_suite import DEFAULT_SUITE_ID, TaskSuiteAggregator, load_task_catalog


EVAL_GROUP_ID = "eval_20990101_000000_smoke_model"
MODEL_NAME = "smoke_model"
CHECKPOINT = "eval_20990101_000000"
TASK_ID = "T02_franka_pick_lift_proxy"
TASK_NAME = "Franka pick/lift proxy"
TASK_FAMILY = "pick_place"
LEADERBOARD_GROUP = "basic_pick_place"
SPLITS = ("sim_dev", "sim_clean", "sim_randomized")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _episode(split: str, index: int, *, success: bool, failure_category: str = "none", failure_code: str = "none") -> dict[str, Any]:
    termination = "success" if success else failure_code
    return {
        "run_id": _run_id_for_split(split),
        "episode_id": f"{TASK_ID}_{split}_fake_{index}",
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "task_family": TASK_FAMILY,
        "task_level": "base",
        "task_tags": ["pick", "lift", "single_object"],
        "leaderboard_group": LEADERBOARD_GROUP,
        "split": split,
        "seed": 990000 + index,
        "valid_scene": True,
        "success": success,
        "success_logic": "robust_semantic_v0",
        "goal_pass": success,
        "sequence_pass": True,
        "sequence_stage": "completed",
        "terminal_pass": True,
        "stability_pass": True,
        "safety_valid": success,
        "hard_failure": not success and failure_category in {"drop", "collision"},
        "latched_violations": [] if success else [failure_code],
        "termination_reason": termination,
        "failure_type": None if success else failure_code,
        "failure_category": failure_category,
        "failure_code": failure_code,
        "failure_source": "none" if success else "success_predicate",
        "failure_severity": "none" if success else "hard",
        "completed_stage": "success" if success else "failed",
        "stage_completion": 1.0 if success else 0.5,
        "quality_metrics": {},
        "episode_steps": 80 + index,
        "inference_latency_p95_ms": 90.0,
        "predicate_values": {"lifted": {"passed": success}},
        "goal_predicates": {"grasped": success, "lifted": success},
        "terminal_predicates": {},
        "stability_predicates": {},
        "safety_predicates": {},
    }


def _run_id_for_split(split: str) -> str:
    if split == "sim_dev":
        return EVAL_GROUP_ID
    return f"{EVAL_GROUP_ID}_{split}"


def _episodes_for_split(split: str) -> list[dict[str, Any]]:
    if split == "sim_dev":
        return [
            _episode(split, 1, success=True),
            _episode(split, 2, success=False, failure_category="timeout", failure_code="timeout"),
        ]
    if split == "sim_clean":
        return [
            _episode(split, 1, success=True),
            _episode(split, 2, success=True),
        ]
    if split == "sim_randomized":
        return [
            _episode(split, 1, success=False, failure_category="drop", failure_code="object_dropped"),
            _episode(split, 2, success=False, failure_category="collision", failure_code="forbidden_collision"),
        ]
    raise ValueError(f"Unknown split: {split}")


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("success"):
            continue
        value = str(row.get(key) or "unknown_failure")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _rate(rows: list[dict[str, Any]], category: str) -> float:
    return sum(1 for row in rows if row.get("failure_category") == category) / len(rows)


def _summaries(split: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    run_id = _run_id_for_split(split)
    successes = sum(1 for row in rows if row["success"])
    success_rate = successes / len(rows)
    failure_category_distribution = _distribution(rows, "failure_category")
    failure_code_distribution = _distribution(rows, "failure_code")
    task_summary = [
        {
            "run_id": run_id,
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "task_family": TASK_FAMILY,
            "task_level": "base",
            "task_tags": ["pick", "lift", "single_object"],
            "leaderboard_group": LEADERBOARD_GROUP,
            "split": split,
            "success_logic": "robust_semantic_v0",
            "num_trials": len(rows),
            "num_valid_trials": len(rows),
            "num_success": successes,
            "success_rate": success_rate,
            "goal_pass_rate": success_rate,
            "sequence_pass_rate": 1.0,
            "terminal_pass_rate": 1.0,
            "stability_pass_rate": 1.0,
            "safety_valid_rate": success_rate,
            "safe_success_rate": success_rate,
            "stage_completion_mean": sum(float(row["stage_completion"]) for row in rows) / len(rows),
            "failure_category_distribution": failure_category_distribution,
            "failure_code_distribution": failure_code_distribution,
            "drop_rate": _rate(rows, "drop"),
            "collision_rate": _rate(rows, "collision"),
            "wrong_place_rate": _rate(rows, "wrong_place"),
            "timeout_rate": _rate(rows, "timeout"),
            "quality_metric_means": {},
        }
    ]
    family_summary = [
        {
            **task_summary[0],
            "task_id": None,
            "task_ids": [TASK_ID],
            "task_name": TASK_FAMILY,
        }
    ]
    summary = {
        "num_episodes": len(rows),
        "num_valid_episodes": len(rows),
        "num_success": successes,
        "success_rate": success_rate,
        "sequence_pass_rate": 1.0,
        "safe_success_rate": success_rate,
        "stage_completion_mean": task_summary[0]["stage_completion_mean"],
        "inference_latency_p95_ms": 90.0,
        "failure_category_distribution": failure_category_distribution,
        "failure_code_distribution": failure_code_distribution,
        "drop_rate": _rate(rows, "drop"),
        "collision_rate": _rate(rows, "collision"),
        "wrong_place_rate": _rate(rows, "wrong_place"),
        "timeout_rate": _rate(rows, "timeout"),
        "task_summary_count": 1,
        "family_summary_count": 1,
    }
    leaderboard = {
        "run_id": run_id,
        "eval_group_id": EVAL_GROUP_ID,
        "model_name": MODEL_NAME,
        "benchmark_version": "smoke",
        "track": split,
        "task_families": [TASK_FAMILY],
        "num_trials": len(rows),
        "implemented_task_count": 1,
        "overall": {
            "sim_dev_macro_success_rate": success_rate if split == "sim_dev" else None,
            "sim_clean_macro_success_rate": success_rate if split == "sim_clean" else None,
            "sim_randomized_macro_success_rate": success_rate if split == "sim_randomized" else None,
            "partial_success_rate": task_summary[0]["stage_completion_mean"],
            "efficiency_score": 91.0,
            "safety_score": 100.0,
            "capability_index_v0_1": 80.0 + success_rate * 10.0,
        },
        "by_task": {TASK_ID: task_summary[0]},
        "by_family": {TASK_FAMILY: family_summary[0]},
        "mini_leaderboards": {
            "by_task": [{"scope": "task", **task_summary[0], "model_name": MODEL_NAME, "checkpoint": CHECKPOINT}],
            "by_family": [{"scope": "family", **family_summary[0], "model_name": MODEL_NAME, "checkpoint": CHECKPOINT}],
        },
        "capabilities": {"success": success_rate * 100.0, "partial_success": task_summary[0]["stage_completion_mean"] * 100.0, "efficiency": 91.0, "safety": 100.0},
    }
    return summary, task_summary, family_summary, leaderboard


def make_fake_eval_outputs(outputs_root: str | Path) -> dict[str, Any]:
    root = Path(outputs_root)
    root.mkdir(parents=True, exist_ok=True)
    run_ids: dict[str, str] = {}
    for split in SPLITS:
        run_id = _run_id_for_split(split)
        run_ids[split] = run_id
        run_dir = root / run_id
        rows = _episodes_for_split(split)
        summary, task_summary, family_summary, leaderboard = _summaries(split, rows)
        manifest = {
            "run_id": run_id,
            "eval_group_id": EVAL_GROUP_ID,
            "model_name": MODEL_NAME,
            "checkpoint": CHECKPOINT,
            "benchmark_version": "smoke",
            "robot": "franka_panda",
            "task_id": TASK_ID,
            "task_family": TASK_FAMILY,
            "task_level": "base",
            "task_tags": ["pick", "lift", "single_object"],
            "leaderboard_group": LEADERBOARD_GROUP,
            "failure_modes": [
                {"code": "timeout", "category": "timeout"},
                {"code": "object_dropped", "category": "drop"},
                {"code": "forbidden_collision", "category": "collision"},
            ],
            "split": split,
            "control_hz": 10,
            "action_chunk": 16,
            "created_at": "2099-01-01T00:00:00+00:00",
        }
        _write_json(run_dir / "run_manifest.json", manifest)
        _write_json(run_dir / "run_config.json", {"args": {}, "task_spec": {"task_id": TASK_ID}})
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "task_summary.json", task_summary)
        _write_json(run_dir / "family_summary.json", family_summary)
        _write_json(run_dir / "leaderboard_row.json", leaderboard)
        _write_jsonl(run_dir / "episode_metrics.jsonl", rows)
        _write_jsonl(run_dir / "failure_cases.jsonl", [row for row in rows if not row["success"]])
        _write_json(run_dir / "static_api" / "episodes.json", {"episodes": rows})
        _write_json(run_dir / "static_api" / "leaderboard.json", {"rows": [leaderboard]})
        _write_json(run_dir / "static_api" / "model_detail.json", {"leaderboard_row": leaderboard, "task_summary": task_summary, "family_summary": family_summary})
        _write_json(run_dir / "static_api" / f"task_detail_{TASK_ID}.json", {"task_spec": {"task_id": TASK_ID}, "task_summary": task_summary, "episodes": rows})
    catalog = load_task_catalog(DEFAULT_SUITE_ID)
    suite_dir = root / f"{EVAL_GROUP_ID}_{DEFAULT_SUITE_ID}"
    TaskSuiteAggregator(suite_dir).write_all(
        task_catalog=catalog,
        outputs_root=root,
        eval_group_id=EVAL_GROUP_ID,
        model_name=MODEL_NAME,
        checkpoint=CHECKPOINT,
        robot="franka_panda",
        benchmark_version="smoke",
    )
    return {"root": root, "eval_group_id": EVAL_GROUP_ID, "run_ids": run_ids, "suite_dir": suite_dir}
