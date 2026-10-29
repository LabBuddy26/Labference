"""Generate deterministic fake laboratory-benchmark evaluation runs for the dashboard.

The EAI4S-bench web dashboard shows an empty 综合模型榜 / 子任务榜 / 评测记录 until
real evaluation runs exist.  This CLI writes realistic-looking runs into the
outputs root (default ``outputs/eval_runs``) so the dashboard can be previewed
without running Isaac:

    python -m eai4s_eval.cli.generate_fake_benchmark
    python -m eai4s_eval.cli.generate_fake_benchmark --output-root /tmp/labbench_eval_runs

Every run covers a single catalog task (L01..R12) with a small number of
episodes, which is how the real ``run_benchmark_eval`` CLI writes runs.  The
model profiles use deterministic randomness so the generated numbers are
stable across invocations.
"""

from __future__ import annotations

import argparse
import json
import zlib
from collections import Counter
from pathlib import Path
from random import Random
from typing import Any

from eai4s_eval.benchmark_catalog import load_benchmark_catalog

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "openpi_prod": {
        "model_name": "openpi_prod",
        "checkpoint": "eval_20260101_prod",
        "track_success": {"long_horizon": 0.70, "reasoning_in_action": 0.78},
        "latency_ms": 90.0,
        "safety": 100.0,
    },
    "gr00t_candidate": {
        "model_name": "gr00t_candidate",
        "checkpoint": "eval_20260101_gr00t",
        "track_success": {"long_horizon": 0.54, "reasoning_in_action": 0.60},
        "latency_ms": 130.0,
        "safety": 97.0,
    },
    "scripted_baseline": {
        "model_name": "scripted_baseline",
        "checkpoint": "eval_20260101_base",
        "track_success": {"long_horizon": 0.40, "reasoning_in_action": 0.46},
        "latency_ms": 45.0,
        "safety": 88.0,
    },
}

CREATED_AT = "2026-01-01T08:00:00+00:00"
FAILURE_CODES = {
    "drop": "object_dropped",
    "collision": "forbidden_collision",
    "timeout": "timeout",
    "wrong_place": "wrong_target",
    "safety_violation": "safety_violation",
}
HARD_FAILURES = {"drop", "collision", "safety_violation"}
FAILURE_WEIGHTS = [("drop", 0.25), ("collision", 0.20), ("timeout", 0.20), ("wrong_place", 0.20), ("safety_violation", 0.15)]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _stable_seed(*parts: str) -> int:
    return zlib.crc32("\x1f".join(parts).encode("utf-8"))


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("success"):
            continue
        counts[str(row.get(key) or "unknown_failure")] += 1
    return dict(sorted(counts.items()))


def _rate(rows: list[dict[str, Any]], category: str) -> float:
    return sum(1 for row in rows if row.get("failure_category") == category) / len(rows)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean_bool(rows: list[dict[str, Any]], predicate: Any) -> float | None:
    values = [1.0 if predicate(row) else 0.0 for row in rows]
    return _mean(values)


def _quality_metrics(task: dict[str, Any], success: bool, rng: Random) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for comp in task.get("score_components") or []:
        metric = str(comp.get("metric") or "")
        if metric == "success":
            continue
        high = rng.randint(70, 100) if success else rng.randint(20, 68)
        metrics[metric] = float(high)
    return metrics


def _components(task: dict[str, Any], success: bool, rng: Random) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for comp in task.get("score_components") or []:
        metric = str(comp.get("metric") or "")
        weight = float(comp.get("weight") or 0.0)
        score = 100.0 if metric == "success" and success else (0.0 if metric == "success" else float(rng.randint(60, 100) if success else rng.randint(20, 65)))
        parts.append({
            "component": comp.get("component") or "component",
            "weight": weight,
            "metric": metric or None,
            "score": score,
            "weighted_score": round(score * weight / 100.0, 4),
        })
    return parts


def _atomic_outcomes(task: dict[str, Any], success: bool, failure_code: str, rng: Random) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for index, action in enumerate(task.get("atomic_actions") or []):
        ok = success or rng.random() < 0.4
        outcomes.append({
            "instance_id": f"{task['task_id']}:{action}:{index}",
            "action": action,
            "attempted": True,
            "success": ok,
            "source": "native",
            "failure_code": None if ok else failure_code,
        })
    return outcomes


def _make_episode(
    *,
    run_id: str,
    task: dict[str, Any],
    profile: dict[str, Any],
    track_rate: float,
    ep_index: int,
    episodes_per_task: int,
    seed: int,
) -> dict[str, Any]:
    rng = Random(_stable_seed(profile["model_name"], task["task_id"], str(ep_index), str(seed)))
    prob = min(0.97, max(0.05, track_rate + rng.uniform(-0.12, 0.12)))
    success = rng.random() < prob
    if success:
        failure_category = "none"
        failure_code = "none"
    else:
        names = [item[0] for item in FAILURE_WEIGHTS]
        weights = [item[1] for item in FAILURE_WEIGHTS]
        failure_category = rng.choices(names, weights=weights, k=1)[0]
        failure_code = FAILURE_CODES[failure_category]
    hard_failure = failure_category in HARD_FAILURES
    safety_valid = failure_category != "safety_violation"
    task_score = (
        78.0 + rng.randint(0, 21)
        if success
        else (0.0 if hard_failure else float(rng.randint(20, 58)))
    )
    return {
        "run_id": run_id,
        "episode_id": f"{run_id}_{ep_index + 1}",
        "task_id": task["task_id"],
        "task_name": task["task_name"],
        "task_family": task["subboard"],
        "task_level": "benchmark",
        "task_tags": [],
        "leaderboard_group": task["subboard"],
        "split": "sim_randomized",
        "seed": seed + ep_index,
        "valid_scene": True,
        "success": success,
        "success_logic": "robust_semantic_v0",
        "goal_pass": success,
        "sequence_pass": success,
        "terminal_pass": success,
        "stability_pass": success,
        "safety_valid": safety_valid,
        "hard_failure": hard_failure,
        "latched_violations": [] if success else [failure_code],
        "termination_reason": "success" if success else failure_code,
        "failure_type": None if success else failure_code,
        "failure_category": failure_category,
        "failure_code": failure_code,
        "failure_source": "none" if success else "success_predicate",
        "failure_severity": "none" if success else ("hard" if hard_failure else "soft"),
        "completed_stage": "success" if success else "failed",
        "stage_completion": 1.0 if success else round(rng.uniform(0.25, 0.75), 4),
        "episode_steps": 60 + ep_index * 3 + rng.randint(0, 20),
        "inference_latency_p95_ms": profile["latency_ms"],
        "quality_metrics": _quality_metrics(task, success, rng),
        "task_score_100": task_score,
        "task_score_components": _components(task, success, rng),
        "score_status": "evaluated",
        "atomic_outcomes": _atomic_outcomes(task, success, failure_code, rng),
        "predicate_values": {},
        "goal_predicates": {"goal_pass": success},
        "terminal_predicates": {"terminal_pass": success},
        "stability_predicates": {"stability_pass": success},
        "safety_predicates": {"safety_valid": safety_valid},
    }


def _task_summary(task: dict[str, Any], episodes: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    valid = [row for row in episodes if row.get("valid_scene", True)]
    successes = sum(1 for row in valid if row.get("success"))
    rate = successes / len(valid) if valid else None
    stage_mean = _mean([float(row.get("stage_completion") or 0.0) for row in valid])
    quality_means: dict[str, float] = {}
    for row in valid:
        for key, value in (row.get("quality_metrics") or {}).items():
            try:
                quality_means.setdefault(key, []).append(float(value))  # type: ignore[attr-defined]
            except (TypeError, ValueError):
                continue
    return {
        "run_id": episodes[0]["run_id"] if episodes else None,
        "task_id": task["task_id"],
        "task_name": task["task_name"],
        "task_family": task["subboard"],
        "task_level": "benchmark",
        "task_tags": [],
        "leaderboard_group": task["subboard"],
        "split": "sim_randomized",
        "success_logic": "robust_semantic_v0",
        "num_trials": len(valid),
        "num_valid_trials": len(valid),
        "num_success": successes,
        "success_rate": rate,
        "goal_pass_rate": rate,
        "sequence_pass_rate": rate,
        "terminal_pass_rate": rate,
        "stability_pass_rate": rate,
        "safety_valid_rate": _mean_bool(valid, lambda row: bool(row.get("safety_valid", True))),
        "safe_success_rate": _mean_bool(valid, lambda row: bool(row.get("success") and row.get("safety_valid", True))),
        "stage_completion_mean": stage_mean,
        "partial_success_rate": stage_mean,
        "task_score_100": _mean([float(row.get("task_score_100")) for row in valid if isinstance(row.get("task_score_100"), (int, float))]),
        "score_coverage_rate": 1.0,
        "failure_category_distribution": _distribution(valid, "failure_category"),
        "failure_code_distribution": _distribution(valid, "failure_code"),
        "drop_rate": _rate(valid, "drop"),
        "collision_rate": _rate(valid, "collision"),
        "wrong_place_rate": _rate(valid, "wrong_place"),
        "timeout_rate": _rate(valid, "timeout"),
        "safety_violation_rate": _rate(valid, "safety_violation"),
        "quality_metric_means": {key: _mean(values) for key, values in quality_means.items()},
        "failure_examples": [],
        "inference_latency_p95_ms": profile["latency_ms"],
        "success_definition": {},
    }


def _write_run(root: Path, *, profile: dict[str, Any], eval_group_id: str, task: dict[str, Any], episodes: list[dict[str, Any]], seed: int) -> str:
    run_id = f"eval_20260101_{profile['model_name']}_{task['task_id']}"
    run_dir = root / run_id
    summary_row = _task_summary(task, episodes, profile)
    family_summary = [
        {
            **summary_row,
            "task_id": None,
            "task_ids": [task["task_id"]],
            "task_name": task["subboard"],
        }
    ]
    valid = [row for row in episodes if row.get("valid_scene", True)]
    successes = sum(1 for row in valid if row.get("success"))
    rate = successes / len(valid) if valid else None
    stage_mean = _mean([float(row.get("stage_completion") or 0.0) for row in valid])
    latency = profile["latency_ms"]
    efficiency = max(0.0, min(100.0, 100.0 - latency / 10.0))
    success_score = None if rate is None else rate * 100.0
    partial_score = None if stage_mean is None else stage_mean * 100.0
    capability_values = [value for value in (success_score, partial_score, efficiency) if value is not None]
    capability_index = sum(capability_values) / len(capability_values) if capability_values else None

    manifest = {
        "run_id": run_id,
        "eval_group_id": eval_group_id,
        "model_name": profile["model_name"],
        "checkpoint": profile["checkpoint"],
        "benchmark_version": "laboratory_benchmark_v1",
        "robot": "franka_panda",
        "task_id": task["task_id"],
        "task_family": task["subboard"],
        "task_level": "benchmark",
        "task_tags": [],
        "leaderboard_group": task["subboard"],
        "split": "sim_randomized",
        "control_hz": 10,
        "action_chunk": 16,
        "created_at": CREATED_AT,
    }
    leaderboard_row = {
        "run_id": run_id,
        "eval_group_id": eval_group_id,
        "model_name": profile["model_name"],
        "checkpoint": profile["checkpoint"],
        "benchmark_version": "laboratory_benchmark_v1",
        "track": "sim_randomized",
        "task_families": [task["subboard"]],
        "num_trials": len(valid),
        "implemented_task_count": 1,
        "overall": {
            "sim_randomized_macro_success_rate": rate,
            "sim_clean_macro_success_rate": None,
            "sim_dev_macro_success_rate": None,
            "partial_success_rate": stage_mean,
            "efficiency_score": efficiency,
            "safety_score": profile["safety"],
            "capability_index_v0_1": capability_index,
        },
        "capabilities": {
            "success": success_score,
            "partial_success": partial_score,
            "efficiency": efficiency,
            "safety": profile["safety"],
        },
        "by_task": {task["task_id"]: summary_row},
        "by_family": {task["subboard"]: family_summary[0]},
        "mini_leaderboards": {
            "by_task": [{**summary_row, "model_name": profile["model_name"], "checkpoint": profile["checkpoint"]}],
            "by_family": [{**family_summary[0], "model_name": profile["model_name"], "checkpoint": profile["checkpoint"]}],
        },
    }
    summary = {
        "num_episodes": len(episodes),
        "num_valid_episodes": len(valid),
        "num_success": successes,
        "success_rate": rate,
        "sequence_pass_rate": rate,
        "safe_success_rate": _mean_bool(valid, lambda row: bool(row.get("success") and row.get("safety_valid", True))),
        "stage_completion_mean": stage_mean,
        "inference_latency_p95_ms": latency,
        "failure_category_distribution": _distribution(valid, "failure_category"),
        "failure_code_distribution": _distribution(valid, "failure_code"),
        "drop_rate": _rate(valid, "drop"),
        "collision_rate": _rate(valid, "collision"),
        "wrong_place_rate": _rate(valid, "wrong_place"),
        "timeout_rate": _rate(valid, "timeout"),
        "task_summary_count": 1,
        "family_summary_count": 1,
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_json(run_dir / "run_config.json", {"args": {}, "task_spec": {"task_id": task["task_id"], "task_name": task["task_name"]}})
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "task_summary.json", [summary_row])
    _write_json(run_dir / "family_summary.json", family_summary)
    _write_json(run_dir / "leaderboard_row.json", leaderboard_row)
    _write_jsonl(run_dir / "episode_metrics.jsonl", episodes)
    _write_jsonl(run_dir / "failure_cases.jsonl", [row for row in episodes if not row["success"]])
    _write_json(run_dir / "static_api" / "episodes.json", {"episodes": episodes})
    _write_json(run_dir / "static_api" / "leaderboard.json", {"rows": [leaderboard_row]})
    _write_json(run_dir / "static_api" / "model_detail.json", {"leaderboard_row": leaderboard_row, "task_summary": [summary_row], "family_summary": family_summary})
    return run_id


def generate_fake_benchmark(
    output_root: str | Path,
    catalog: dict[str, Any] | None = None,
    *,
    episodes_per_task: int = 3,
    seed: int = 1,
    models: list[str] | None = None,
) -> list[str]:
    """Write fake benchmark runs for the selected model profiles.

    Returns the list of written run ids.  Each model produces one run per
    catalog task, covering all 24 tasks so every subtask leaderboard has
    12 tasks with data for its score radar.
    """
    catalog = catalog or load_benchmark_catalog()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    selected = [key for key in (models or list(MODEL_PROFILES)) if key in MODEL_PROFILES]
    run_ids: list[str] = []
    for profile_key in selected:
        profile = MODEL_PROFILES[profile_key]
        eval_group_id = f"fake_{profile_key}_{seed}"
        for task in catalog["tasks"]:
            track_rate = profile["track_success"].get(task["track_id"], 0.5)
            episodes = [
                _make_episode(
                    run_id=f"eval_20260101_{profile['model_name']}_{task['task_id']}",
                    task=task,
                    profile=profile,
                    track_rate=track_rate,
                    ep_index=index,
                    episodes_per_task=episodes_per_task,
                    seed=seed,
                )
                for index in range(episodes_per_task)
            ]
            run_ids.append(_write_run(root, profile=profile, eval_group_id=eval_group_id, task=task, episodes=episodes, seed=seed))
    return run_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic fake laboratory-benchmark runs for the dashboard preview.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/eval_runs"))
    parser.add_argument("--episodes-per-task", type=int, default=3, help="Episodes to generate per catalog task (default 3).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODEL_PROFILES),
        help=f"Comma-separated profile keys: {', '.join(MODEL_PROFILES)}.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = [item for item in selected if item not in MODEL_PROFILES]
    if unknown:
        raise SystemExit(f"Unknown model profile(s): {', '.join(unknown)}; choose from {', '.join(MODEL_PROFILES)}.")
    catalog = load_benchmark_catalog()
    run_ids = generate_fake_benchmark(
        args.output_root,
        catalog,
        episodes_per_task=args.episodes_per_task,
        seed=args.seed,
        models=selected,
    )
    print(f"Wrote {len(run_ids)} fake benchmark runs ({len(selected)} model(s) x {len(catalog['tasks'])} tasks) to {Path(args.output_root).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
