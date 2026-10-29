from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import json

from eai4s_eval.benchmark_catalog import atomic_outcomes_for_episode, score_episode_for_bound_task
from eai4s_eval.utils.io import ensure_dir, try_write_parquet, write_csv, write_json, write_jsonl


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _mean(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _mean_bool(values: list[Any]) -> float | None:
    valid = [bool(v) for v in values if v is not None]
    if not valid:
        return None
    return sum(1 for value in valid if value) / len(valid)


def _predicate_pass_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        predicate_values = row.get("predicate_values") or {}
        if not isinstance(predicate_values, dict):
            continue
        for name, value in predicate_values.items():
            if isinstance(value, dict) and "passed" in value:
                counts[name].append(bool(value["passed"]))
            elif hasattr(value, "passed"):
                counts[name].append(bool(value.passed))
    return {name: sum(values) / len(values) for name, values in sorted(counts.items()) if values}


def _quality_metric_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    values_by_name: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        quality_metrics = row.get("quality_metrics") or {}
        if not isinstance(quality_metrics, dict):
            continue
        for name, value in quality_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values_by_name[str(name)].append(float(value))
    return {
        name: sum(values) / len(values)
        for name, values in sorted(values_by_name.items())
        if values
    }


def _atomic_action_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"attempts": 0, "successes": 0, "sources": set(), "failure_codes": Counter()}
    )
    for row in rows:
        outcomes = row.get("atomic_outcomes") or atomic_outcomes_for_episode(row)
        if not isinstance(outcomes, list):
            continue
        for outcome in outcomes:
            if not isinstance(outcome, dict) or not outcome.get("attempted", True) or not outcome.get("action"):
                continue
            bucket = grouped[str(outcome["action"])]
            bucket["attempts"] += 1
            bucket["successes"] += int(bool(outcome.get("success")))
            bucket["sources"].add(str(outcome.get("source") or "native"))
            if not outcome.get("success"):
                bucket["failure_codes"][str(outcome.get("failure_code") or "unknown_failure")] += 1
    return [
        {
            "action": action,
            "attempts": int(bucket["attempts"]),
            "successes": int(bucket["successes"]),
            "success_rate": bucket["successes"] / bucket["attempts"] if bucket["attempts"] else None,
            "sources": sorted(bucket["sources"]),
            "failure_code_distribution": dict(sorted(bucket["failure_codes"].items())),
        }
        for action, bucket in sorted(grouped.items())
    ]


def _counter_distribution(rows: list[dict[str, Any]], key: str, *, failures_only: bool = False) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if failures_only and row.get("success"):
            continue
        value = row.get(key)
        if value is None or value == "":
            value = "none" if row.get("success") else "unknown_failure"
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _merge_dict_counts(items: list[dict[str, Any] | None]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            try:
                counts[str(key)] += int(value)
            except (TypeError, ValueError):
                continue
    return dict(sorted(counts.items()))


def _category_rate(rows: list[dict[str, Any]], category: str) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row.get("failure_category") == category) / len(rows)


def _failure_examples(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        if row.get("success"):
            continue
        examples.append(
            {
                "episode_id": row.get("episode_id"),
                "split": row.get("split"),
                "seed": row.get("seed"),
                "failure_category": row.get("failure_category") or "unknown_failure",
                "failure_code": row.get("failure_code") or row.get("failure_type") or row.get("termination_reason"),
                "failure_source": row.get("failure_source") or "unknown",
                "failure_severity": row.get("failure_severity") or "soft",
                "termination_reason": row.get("termination_reason"),
                "episode_steps": row.get("episode_steps"),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _task_metadata_from_group(group: list[dict[str, Any]], task_spec: dict[str, Any]) -> dict[str, Any]:
    first = group[0] if group else {}
    return {
        "task_family": first.get("task_family") or task_spec.get("task_family") or "basic",
        "task_level": first.get("task_level") or task_spec.get("task_level") or "base",
        "task_tags": first.get("task_tags") or task_spec.get("task_tags") or [],
        "leaderboard_group": first.get("leaderboard_group") or task_spec.get("leaderboard_group") or first.get("task_family") or "basic",
        "failure_modes": task_spec.get("failure_modes") or [],
    }


def _summary_for_group(
    group: list[dict[str, Any]],
    *,
    task_spec: dict[str, Any],
    task_id: str | None = None,
    task_name: str | None = None,
    task_family: str | None = None,
    split: str | None = None,
    success_logic: str | None = None,
) -> dict[str, Any]:
    valid = [r for r in group if r.get("valid_scene", True)]
    successes = sum(1 for r in valid if r.get("success"))
    ci_low, ci_high = wilson_ci(successes, len(valid))
    metadata = _task_metadata_from_group(valid or group, task_spec)
    failure_category_distribution = _counter_distribution(valid, "failure_category", failures_only=True)
    failure_code_distribution = _counter_distribution(valid, "failure_code", failures_only=True)
    legacy_failure_distribution = _counter_distribution(valid, "failure_type", failures_only=True)
    return {
        "run_id": valid[0]["run_id"] if valid else group[0]["run_id"],
        "task_id": task_id or group[0].get("task_id"),
        "task_name": task_name or group[0].get("task_name"),
        "task_family": task_family or metadata["task_family"],
        "task_level": metadata["task_level"],
        "task_tags": metadata["task_tags"],
        "leaderboard_group": metadata["leaderboard_group"],
        "failure_modes": metadata["failure_modes"],
        "success_definition": task_spec.get("success") if task_spec.get("task_id") == (task_id or group[0].get("task_id")) else None,
        "success_logic": success_logic or group[0].get("success_logic", "robust_semantic_v0"),
        "metrics": task_spec.get("metrics") if task_spec.get("task_id") == (task_id or group[0].get("task_id")) else None,
        "split": split or group[0].get("split"),
        "num_trials": len(group),
        "num_valid_trials": len(valid),
        "num_success": successes,
        "success_rate": successes / len(valid) if valid else None,
        "success_ci_low": ci_low,
        "success_ci_high": ci_high,
        "goal_pass_rate": _mean_bool([r.get("goal_pass") for r in valid]),
        "sequence_pass_rate": _mean_bool([r.get("sequence_pass") for r in valid]),
        "terminal_pass_rate": _mean_bool([r.get("terminal_pass") for r in valid]),
        "stability_pass_rate": _mean_bool([r.get("stability_pass") for r in valid]),
        "safety_valid_rate": _mean_bool([r.get("safety_valid") for r in valid]),
        "safe_success_rate": _mean_bool([r.get("success") and r.get("safety_valid") for r in valid]),
        "predicate_pass_rates": _predicate_pass_rates(valid),
        "quality_metric_means": _quality_metric_means(valid),
        "task_score_100": _mean([r.get("task_score_100") for r in valid]),
        "score_coverage_rate": _mean_bool([r.get("task_score_100") is not None for r in valid]),
        "atomic_action_summary": _atomic_action_summary(valid),
        "partial_success_rate": _mean([r.get("stage_completion") for r in valid]),
        "stage_completion_mean": _mean([r.get("stage_completion") for r in valid]),
        "timeout_rate": _category_rate(valid, "timeout"),
        "drop_rate": _category_rate(valid, "drop"),
        "collision_rate": _category_rate(valid, "collision"),
        "wrong_place_rate": _category_rate(valid, "wrong_place"),
        "wrong_object_rate": _category_rate(valid, "wrong_object"),
        "safety_violation_rate": _category_rate(valid, "safety_violation"),
        "sequence_error_rate": _category_rate(valid, "sequence_error"),
        "policy_error_rate": _category_rate(valid, "policy_error"),
        "failure_category_distribution": failure_category_distribution,
        "failure_code_distribution": failure_code_distribution,
        "failure_type_distribution_json": legacy_failure_distribution,
        "failure_examples": _failure_examples(valid),
        "spill_rate": None,
        "hazard_recall": None,
        "false_alarm_rate": None,
        "trajectory_smoothness_mean": None,
        "path_efficiency_mean": None,
        "inference_latency_p95_ms": _mean([r.get("inference_latency_p95_ms") for r in valid]),
    }


class RunAggregator:
    def __init__(self, output_dir: str | Path):
        self.output_dir = ensure_dir(output_dir)

    def write_all(
        self,
        *,
        episode_rows: list[dict[str, Any]],
        run_manifest: dict[str, Any],
        task_spec: dict[str, Any],
    ) -> dict[str, Any]:
        for row in episode_rows:
            if "atomic_outcomes" not in row:
                row["atomic_outcomes"] = atomic_outcomes_for_episode(row)
            if "task_score_100" not in row:
                score = score_episode_for_bound_task(row)
                row["task_score_100"] = score["score_100"]
                row["task_score_components"] = score["components"]
                row["score_status"] = score["status"]
                if score.get("missing_metrics"):
                    row["score_missing_metrics"] = score["missing_metrics"]
        write_jsonl(self.output_dir / "episode_metrics.jsonl", episode_rows)
        write_csv(self.output_dir / "episode_metrics.csv", episode_rows)
        parquet_available = try_write_parquet(self.output_dir / "episode_metrics.parquet", episode_rows)

        task_summary = self._task_summary(episode_rows, task_spec)
        write_json(self.output_dir / "task_summary.json", task_summary)
        write_csv(self.output_dir / "task_summary.csv", task_summary)
        try_write_parquet(self.output_dir / "task_summary.parquet", task_summary)

        family_summary = self._family_summary(episode_rows, task_spec)
        write_json(self.output_dir / "family_summary.json", family_summary)
        write_csv(self.output_dir / "family_summary.csv", family_summary)
        try_write_parquet(self.output_dir / "family_summary.parquet", family_summary)

        summary = self._summary(episode_rows, task_summary, family_summary, parquet_available)
        leaderboard = self._leaderboard(run_manifest, episode_rows, task_summary, family_summary, summary)
        static_api = self._static_api(episode_rows, task_summary, family_summary, leaderboard, task_spec)

        write_json(self.output_dir / "summary.json", summary)
        write_json(self.output_dir / "leaderboard_row.json", leaderboard)
        summary_lines = self._summary_lines(summary, task_summary)
        summary_log = self.output_dir / "evaluation_summary.log"
        summary_log.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        static_dir = ensure_dir(self.output_dir / "static_api")
        for name, data in static_api.items():
            write_json(static_dir / name, data)
        return {
            "summary": summary,
            "task_summary": task_summary,
            "family_summary": family_summary,
            "leaderboard_row": leaderboard,
            "parquet_available": parquet_available,
            "summary_lines": summary_lines,
            "summary_log": str(summary_log),
        }

    @staticmethod
    def _summary_lines(summary: dict[str, Any], task_summary: list[dict[str, Any]]) -> list[str]:
        valid = int(summary.get("num_valid_episodes") or 0)
        successes = int(summary.get("num_success") or 0)
        overall_rate = successes / valid if valid else None
        rate_text = "N/A" if overall_rate is None else f"{overall_rate * 100.0:.2f}%"
        lines = [
            "=== Evaluation Summary ===",
            f"Overall success rate: {successes}/{valid} ({rate_text})",
            "Per-task success rates:",
        ]

        grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"success": 0, "valid": 0}
        )
        for item in task_summary:
            task_id = str(item.get("task_id") or "unknown_task")
            task_name = str(item.get("task_name") or task_id)
            grouped[(task_id, task_name)]["success"] += int(item.get("num_success") or 0)
            grouped[(task_id, task_name)]["valid"] += int(item.get("num_valid_trials") or 0)
        for (task_id, task_name), counts in sorted(grouped.items()):
            task_valid = counts["valid"]
            task_success = counts["success"]
            task_rate = "N/A" if not task_valid else f"{task_success / task_valid * 100.0:.2f}%"
            lines.append(
                f"- {task_id} ({task_name}): {task_success}/{task_valid} ({task_rate})"
            )
        return lines

    def _task_summary(self, rows: list[dict[str, Any]], task_spec: dict[str, Any]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["task_id"], row["split"], row.get("success_logic", "robust_semantic_v0"))].append(row)

        summaries = []
        for (task_id, split, success_logic), group in sorted(grouped.items()):
            summaries.append(
                _summary_for_group(
                    group,
                    task_spec=task_spec,
                    task_id=task_id,
                    task_name=task_spec.get("task_name") if task_spec.get("task_id") == task_id else group[0].get("task_name"),
                    task_family=group[0].get("task_family") or task_spec.get("task_family") or "basic",
                    split=split,
                    success_logic=success_logic,
                )
            )
        return summaries

    def _family_summary(self, rows: list[dict[str, Any]], task_spec: dict[str, Any]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[
                (
                    str(row.get("task_family") or task_spec.get("task_family") or "basic"),
                    str(row.get("split") or "sim_dev"),
                    str(row.get("success_logic") or "robust_semantic_v0"),
                )
            ].append(row)
        summaries = []
        for (task_family, split, success_logic), group in sorted(grouped.items()):
            summary = _summary_for_group(
                group,
                task_spec=task_spec,
                task_id=None,
                task_name=task_family,
                task_family=task_family,
                split=split,
                success_logic=success_logic,
            )
            task_ids = sorted({str(row.get("task_id")) for row in group if row.get("task_id")})
            summary["task_id"] = None
            summary["task_ids"] = task_ids
            summary["task_family"] = task_family
            summaries.append(summary)
        return summaries

    def _summary(
        self,
        rows: list[dict[str, Any]],
        task_summary: list[dict[str, Any]],
        family_summary: list[dict[str, Any]],
        parquet_available: bool,
    ) -> dict[str, Any]:
        valid = [r for r in rows if r.get("valid_scene", True)]
        successes = sum(1 for r in valid if r.get("success"))
        ci_low, ci_high = wilson_ci(successes, len(valid))
        return {
            "num_episodes": len(rows),
            "num_valid_episodes": len(valid),
            "num_success": successes,
            "success_rate": successes / len(valid) if valid else None,
            "success_ci_low": ci_low,
            "success_ci_high": ci_high,
            "sequence_pass_rate": _mean_bool([r.get("sequence_pass") for r in valid]),
            "safe_success_rate": _mean_bool([r.get("success") and r.get("safety_valid") for r in valid]),
            "quality_metric_means": _quality_metric_means(valid),
            "failure_category_distribution": _counter_distribution(valid, "failure_category", failures_only=True),
            "failure_code_distribution": _counter_distribution(valid, "failure_code", failures_only=True),
            "drop_rate": _category_rate(valid, "drop"),
            "collision_rate": _category_rate(valid, "collision"),
            "wrong_place_rate": _category_rate(valid, "wrong_place"),
            "timeout_rate": _category_rate(valid, "timeout"),
            "stage_completion_mean": _mean([r.get("stage_completion") for r in valid]),
            "inference_latency_p95_ms": _mean([r.get("inference_latency_p95_ms") for r in valid]),
            "parquet_available": parquet_available,
            "task_summary_count": len(task_summary),
            "family_summary_count": len(family_summary),
        }

    def _leaderboard(
        self,
        run_manifest: dict[str, Any],
        rows: list[dict[str, Any]],
        task_summary: list[dict[str, Any]],
        family_summary: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        split_rate_values: dict[str, list[float]] = defaultdict(list)
        for item in task_summary:
            if item.get("success_rate") is not None:
                split_rate_values[item["split"]].append(float(item["success_rate"]))
        split_rates = {
            split: sum(values) / len(values)
            for split, values in split_rate_values.items()
            if values
        }
        clean_sr = split_rates.get("sim_clean")
        randomized_sr = split_rates.get("sim_randomized")
        dev_sr = split_rates.get("sim_dev")
        main_sr = randomized_sr if randomized_sr is not None else clean_sr if clean_sr is not None else dev_sr
        partial = summary.get("stage_completion_mean")
        latency = summary.get("inference_latency_p95_ms")
        efficiency = None if latency is None else max(0.0, min(100.0, 100.0 - latency / 10.0))
        success_score = None if main_sr is None else main_sr * 100.0
        partial_score = None if partial is None else partial * 100.0
        capability_values = [v for v in [success_score, partial_score, efficiency] if v is not None]
        capability_index = sum(capability_values) / len(capability_values) if capability_values else None
        by_task = {}
        for item in task_summary:
            by_task[item["task_id"]] = {
                "task_name": item.get("task_name"),
                "task_family": item.get("task_family"),
                "task_level": item.get("task_level"),
                "task_tags": item.get("task_tags"),
                "leaderboard_group": item.get("leaderboard_group"),
                "split": item.get("split"),
                "success_rate": item.get("success_rate"),
                "num_trials": item.get("num_trials"),
                "num_success": item.get("num_success"),
                "stage_completion_mean": item.get("stage_completion_mean"),
                "metrics": item.get("metrics"),
                "success_definition": item.get("success_definition"),
                "success_logic": item.get("success_logic"),
                "predicate_pass_rates": item.get("predicate_pass_rates"),
                "sequence_pass_rate": item.get("sequence_pass_rate"),
                "safe_success_rate": item.get("safe_success_rate"),
                "quality_metric_means": item.get("quality_metric_means"),
                "failure_category_distribution": item.get("failure_category_distribution"),
                "failure_code_distribution": item.get("failure_code_distribution"),
                "drop_rate": item.get("drop_rate"),
                "collision_rate": item.get("collision_rate"),
                "wrong_place_rate": item.get("wrong_place_rate"),
                "timeout_rate": item.get("timeout_rate"),
            }
        by_family = {}
        for family in sorted({str(item.get("task_family") or "basic") for item in family_summary}):
            items = [item for item in family_summary if str(item.get("task_family") or "basic") == family]
            total_trials = sum(int(item.get("num_trials") or 0) for item in items)
            total_success = sum(int(item.get("num_success") or 0) for item in items)
            by_family[family] = {
                "task_family": family,
                "task_level": items[0].get("task_level") if items else "base",
                "task_tags": sorted({tag for item in items for tag in (item.get("task_tags") or [])}),
                "leaderboard_group": items[0].get("leaderboard_group") if items else family,
                "task_ids": sorted({task_id for item in items for task_id in (item.get("task_ids") or [])}),
                "split": ", ".join(sorted({str(item.get("split")) for item in items if item.get("split")})),
                "success_rate": total_success / total_trials if total_trials else None,
                "num_trials": total_trials,
                "num_success": total_success,
                "stage_completion_mean": _mean([item.get("stage_completion_mean") for item in items]),
                "sequence_pass_rate": _mean([item.get("sequence_pass_rate") for item in items]),
                "safe_success_rate": _mean([item.get("safe_success_rate") for item in items]),
                "quality_metric_means": _quality_metric_means([{"quality_metrics": item.get("quality_metric_means") or {}} for item in items]),
                "failure_category_distribution": _merge_dict_counts([item.get("failure_category_distribution") for item in items]),
                "failure_code_distribution": _merge_dict_counts([item.get("failure_code_distribution") for item in items]),
                "drop_rate": _mean([item.get("drop_rate") for item in items]),
                "collision_rate": _mean([item.get("collision_rate") for item in items]),
                "wrong_place_rate": _mean([item.get("wrong_place_rate") for item in items]),
                "timeout_rate": _mean([item.get("timeout_rate") for item in items]),
            }
        mini_by_task = [
            {
                "scope": "task",
                "task_id": item.get("task_id"),
                "task_name": item.get("task_name"),
                "task_family": item.get("task_family"),
                "leaderboard_group": item.get("leaderboard_group"),
                "split": item.get("split"),
                "success_rate": item.get("success_rate"),
                "safe_success_rate": item.get("safe_success_rate"),
                "num_trials": item.get("num_trials"),
                "num_success": item.get("num_success"),
                "failure_category_distribution": item.get("failure_category_distribution"),
                "failure_examples": item.get("failure_examples"),
            }
            for item in task_summary
        ]
        mini_by_family = [
            {
                "scope": "family",
                "task_family": item.get("task_family"),
                "task_ids": item.get("task_ids"),
                "leaderboard_group": item.get("leaderboard_group"),
                "split": item.get("split"),
                "success_rate": item.get("success_rate"),
                "safe_success_rate": item.get("safe_success_rate"),
                "num_trials": item.get("num_trials"),
                "num_success": item.get("num_success"),
                "failure_category_distribution": item.get("failure_category_distribution"),
                "failure_examples": item.get("failure_examples"),
            }
            for item in family_summary
        ]
        return {
            "run_id": run_manifest["run_id"],
            "eval_group_id": run_manifest.get("eval_group_id") or run_manifest["run_id"],
            "model_name": run_manifest.get("model_name"),
            "benchmark_version": run_manifest.get("benchmark_version"),
            "track": run_manifest.get("split", "sim_dev"),
            "task_families": sorted({item.get("task_family") for item in task_summary if item.get("task_family")}),
            "implemented_task_count": len({item["task_id"] for item in task_summary}),
            "full_task_count": 35,
            "num_trials": len(rows),
            "safety_valid": True,
            "overall": {
                "sim_randomized_macro_success_rate": randomized_sr,
                "sim_ood_macro_success_rate": None,
                "sim_clean_macro_success_rate": clean_sr,
                "sim_dev_macro_success_rate": dev_sr,
                "success_ci_low": summary.get("success_ci_low"),
                "success_ci_high": summary.get("success_ci_high"),
                "partial_success_rate": partial,
                "trajectory_smoothness": None,
                "safety_score": 100.0,
                "efficiency_score": efficiency,
                "semantic_score": None,
                "bimanual_score": None,
                "capability_index_v0_1": capability_index,
            },
            "by_leaderboard": {
                "startup": {"success_rate": main_sr},
                "A": {"success_rate": None},
                "D": {"success_rate": None},
            },
            "by_task": by_task,
            "by_family": by_family,
            "mini_leaderboards": {
                "by_task": mini_by_task,
                "by_family": mini_by_family,
            },
            "capabilities": {
                "success": success_score,
                "partial_success": partial_score,
                "smoothness": None,
                "safety": 100.0,
                "efficiency": efficiency,
                "semantic": None,
                "bimanual": None,
                "ood_robustness": None,
            },
        }

    def _static_api(
        self,
        rows: list[dict[str, Any]],
        task_summary: list[dict[str, Any]],
        family_summary: list[dict[str, Any]],
        leaderboard: dict[str, Any],
        task_spec: dict[str, Any],
    ) -> dict[str, Any]:
        task_id = str(task_spec.get("task_id", "task"))
        task_detail_name = f"task_detail_{task_id}.json"
        return {
            "leaderboard.json": {"rows": [leaderboard]},
            "model_detail.json": {"leaderboard_row": leaderboard, "task_summary": task_summary, "family_summary": family_summary},
            task_detail_name: {"task_spec": task_spec, "task_summary": task_summary, "episodes": rows},
            "task_details.json": {"tasks": [{"task_spec": task_spec, "task_summary": task_summary, "episodes": rows}]},
            "episodes.json": {"episodes": rows},
            "family_summary.json": {"family_summary": family_summary},
        }


# ---------------------------------------------------------------------------
# Cross-run aggregation (used by the web server)
# ---------------------------------------------------------------------------

def aggregate_leaderboard(outputs_root: str | Path) -> dict[str, Any]:
    """Scan all run directories under *outputs_root* and build an aggregated leaderboard.

    Each run directory is expected to contain ``leaderboard_row.json`` and
    ``summary.json``.  Rows are sorted by ``capability_index_v0_1`` descending.

    Returns a dict with keys ``rows`` (list of merged leaderboard entries, each
    enriched with ``rank``) and ``updated_at`` (ISO-8601 timestamp).
    """
    from datetime import datetime

    root = Path(outputs_root)
    if not root.exists():
        return {"rows": [], "updated_at": datetime.now().astimezone().isoformat()}

    rows: list[dict[str, Any]] = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        lb_path = candidate / "leaderboard_row.json"
        summary_path = candidate / "summary.json"
        manifest_path = candidate / "run_manifest.json"

        if not lb_path.exists():
            continue

        try:
            row = json.loads(lb_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        # Enrich with summary data if available
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                row["success_rate"] = summary.get("success_rate")
                row["stage_completion_mean"] = summary.get("stage_completion_mean")
                row["num_episodes"] = summary.get("num_episodes")
                row["num_success"] = summary.get("num_success")
            except (json.JSONDecodeError, OSError):
                pass

        # Enrich with manifest data if available
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                row.setdefault("checkpoint", manifest.get("checkpoint"))
                row.setdefault("created_at", manifest.get("created_at"))
            except (json.JSONDecodeError, OSError):
                pass

        rows.append(row)

    # Sort by capability_index descending (None sorts last)
    rows.sort(
        key=lambda r: (
            (r.get("overall") or {}).get("capability_index_v0_1") is not None,
            (r.get("overall") or {}).get("capability_index_v0_1") or 0,
        ),
        reverse=True,
    )

    # Assign ranks
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return {
        "rows": rows,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
