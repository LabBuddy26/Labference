from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from eai4s_eval.utils.io import ensure_dir, write_csv, write_json


SPLIT_ORDER = ("sim_dev", "sim_clean", "sim_randomized")
DEFAULT_SUITE_ID = "eai4s_chem_basic_v0_1"
OFFICIAL_SUITES_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "task_suites"
    / "official_evaluation_suites_v1.json"
)


def load_official_evaluation_suites(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path else OFFICIAL_SUITES_PATH
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "labbench_official_evaluation_suites/v1":
        raise ValueError("unsupported official evaluation-suite registry schema")
    suite_ids = payload.get("suite_ids")
    default_suite_id = str(payload.get("default_suite_id") or "").strip()
    if not isinstance(suite_ids, list) or not all(isinstance(item, str) and item.strip() for item in suite_ids):
        raise ValueError("official evaluation-suite registry requires non-empty suite ids")
    normalized = [item.strip() for item in suite_ids]
    if len(normalized) != len(set(normalized)):
        raise ValueError("official evaluation-suite registry contains duplicate suite ids")
    if default_suite_id not in normalized:
        raise ValueError("default_suite_id must be present in suite_ids")
    return {**payload, "suite_ids": normalized, "default_suite_id": default_suite_id}


def default_official_suite_id() -> str:
    return str(load_official_evaluation_suites()["default_suite_id"])


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value.replace("'", '"'))
        except json.JSONDecodeError:
            return [item.strip() for item in value[1:-1].split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a small YAML/JSON mapping.

    PyYAML is used when available.  The fallback deliberately extracts only
    top-level scalar/list metadata, which is enough for catalog and scoreboard
    rendering in lightweight server/test environments.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        pass
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        pass

    result: dict[str, Any] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or line[:1].isspace() or ":" not in line:
            i += 1
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            result[key] = _parse_scalar(raw_value)
            i += 1
            continue

        values: list[Any] = []
        j = i + 1
        while j < len(lines):
            child = lines[j]
            child_stripped = child.strip()
            if not child_stripped or child_stripped.startswith("#"):
                j += 1
                continue
            if not child[:1].isspace():
                break
            if child_stripped.startswith("- "):
                values.append(_parse_scalar(child_stripped[2:]))
            j += 1
        result[key] = values if values else {}
        i = j
    return result


def resolve_task_suite_path(task_suite: str | Path | None) -> Path:
    value = str(task_suite or DEFAULT_SUITE_ID)
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    if candidate.suffix:
        named = _package_root() / "configs" / "task_suites" / candidate.name
    else:
        named = _package_root() / "configs" / "task_suites" / f"{value}.yaml"
    return named.resolve()


def resolve_task_spec_path(task_spec: str | Path, suite_path: Path | None = None) -> Path:
    candidate = Path(task_spec)
    if candidate.exists():
        return candidate.resolve()
    search_roots = []
    if suite_path is not None:
        search_roots.append(suite_path.parent)
    search_roots.append(_package_root() / "configs" / "tasks")
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return candidate.resolve()


def load_task_suite(task_suite: str | Path | None = None) -> dict[str, Any]:
    suite_path = resolve_task_suite_path(task_suite)
    data = _load_yaml_mapping(suite_path)
    data.setdefault("suite_id", suite_path.stem)
    data.setdefault("suite_name", data["suite_id"])
    data.setdefault("suite_version", "0.1.0")
    data.setdefault("tasks", [])
    data["_path"] = str(suite_path)
    if not isinstance(data["tasks"], list):
        raise ValueError(f"Task suite tasks must be a list: {suite_path}")
    return data


def _task_id_from_filename(path: Path) -> str:
    name = path.stem
    if name.endswith("_draft"):
        name = name[: -len("_draft")]
    return name


def load_task_catalog(task_suite: str | Path | dict[str, Any] | None = None) -> dict[str, Any]:
    suite = task_suite if isinstance(task_suite, dict) else load_task_suite(task_suite)
    suite_path = Path(str(suite.get("_path", ""))) if suite.get("_path") else None
    tasks = []
    for index, entry in enumerate(suite.get("tasks") or []):
        if not isinstance(entry, dict):
            continue
        spec_path = resolve_task_spec_path(entry.get("task_spec") or entry.get("path") or "", suite_path)
        spec = _load_yaml_mapping(spec_path) if spec_path.exists() else {}
        task_id = str(entry.get("task_id") or spec.get("task_id") or _task_id_from_filename(spec_path))
        task_name = str(entry.get("task_name") or spec.get("task_name") or task_id)
        task_family = str(entry.get("task_family") or spec.get("task_family") or "basic")
        leaderboard_group = str(entry.get("leaderboard_group") or spec.get("leaderboard_group") or task_family)
        status = str(entry.get("status") or "planned")
        tasks.append(
            {
                "order": int(entry.get("order") or index + 1),
                "task_id": task_id,
                # Task-suite entries may use a short leaderboard id (L01/R01)
                # while the task result keeps the benchmark task-spec id
                # (for example level2_L01_sampletransport).  Keep both ids so
                # aggregation can join the result without rewriting it.
                "source_task_id": str(spec.get("task_id") or task_id),
                "task_name": task_name,
                "task_family": task_family,
                "task_level": str(entry.get("task_level") or spec.get("task_level") or "base"),
                "task_tags": entry.get("task_tags") or spec.get("task_tags") or [],
                "leaderboard_group": leaderboard_group,
                "scene_id": entry.get("scene_id") or spec.get("scene_id"),
                "robot": entry.get("robot") or spec.get("robot"),
                "status": status,
                "skip_reason": entry.get("skip_reason"),
                "eligible": status == "implemented",
                "task_spec": str(spec_path),
                "max_steps": entry.get("max_steps") or spec.get("max_steps"),
                "instruction": spec.get("instruction"),
                "success": spec.get("success"),
            }
        )
    tasks.sort(key=lambda item: (int(item.get("order") or 0), str(item.get("task_id") or "")))
    return {
        "suite_id": suite.get("suite_id"),
        "suite_name": suite.get("suite_name"),
        "suite_version": suite.get("suite_version"),
        "description": suite.get("description"),
        "tasks": tasks,
        "suite_task_count": len(tasks),
    }


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _merge_count_dicts(items: list[dict[str, Any] | None]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            try:
                counts[str(key)] += int(value)
            except (TypeError, ValueError):
                continue
    return dict(sorted(counts.items()))


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _scan_task_runs(outputs_root: str | Path, *, eval_group_id: str, suite_id: str | None = None) -> list[dict[str, Any]]:
    root = Path(outputs_root)
    if not root.exists():
        return []
    runs: list[dict[str, Any]] = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / "suite_leaderboard_row.json").exists():
            continue
        manifest = _read_json(candidate / "run_manifest.json") or {}
        if manifest.get("eval_group_id") != eval_group_id:
            continue
        if suite_id and manifest.get("suite_id") not in {suite_id, None, ""}:
            continue
        task_summary = _read_json(candidate / "task_summary.json") or []
        summary = _read_json(candidate / "summary.json") or {}
        if not isinstance(task_summary, list):
            task_summary = []
        runs.append({"run_dir": str(candidate), "manifest": manifest, "task_summary": task_summary, "summary": summary})
    return runs


class TaskSuiteAggregator:
    def __init__(self, output_dir: str | Path):
        self.output_dir = ensure_dir(output_dir)

    def write_all(
        self,
        *,
        task_catalog: dict[str, Any],
        outputs_root: str | Path,
        eval_group_id: str,
        model_name: str,
        checkpoint: str,
        robot: str | None = None,
        benchmark_version: str = "EAI4S-bench-0.1.0",
    ) -> dict[str, Any]:
        suite_id = str(task_catalog.get("suite_id") or DEFAULT_SUITE_ID)
        runs = _scan_task_runs(outputs_root, eval_group_id=eval_group_id, suite_id=suite_id)
        task_score_table = self._task_score_table(task_catalog, runs)
        family_score_table = self._family_score_table(task_score_table)
        suite_summary = self._suite_summary(task_catalog, task_score_table, family_score_table)
        suite_manifest = {
            "run_id": eval_group_id,
            "eval_group_id": eval_group_id,
            "run_type": "task_suite",
            "suite_id": suite_id,
            "suite_name": task_catalog.get("suite_name"),
            "suite_version": task_catalog.get("suite_version"),
            "model_name": model_name,
            "checkpoint": checkpoint,
            "robot": robot,
            "benchmark_version": benchmark_version,
            "created_at": datetime.now().astimezone().isoformat(),
            "task_ids": [task.get("task_id") for task in task_catalog.get("tasks", [])],
        }
        leaderboard_row = self._suite_leaderboard_row(
            suite_manifest=suite_manifest,
            task_score_table=task_score_table,
            family_score_table=family_score_table,
            suite_summary=suite_summary,
        )

        write_json(self.output_dir / "suite_manifest.json", suite_manifest)
        write_json(self.output_dir / "suite_summary.json", suite_summary)
        write_json(self.output_dir / "task_score_table.json", task_score_table)
        write_csv(self.output_dir / "task_score_table.csv", task_score_table)
        write_json(self.output_dir / "family_score_table.json", family_score_table)
        write_csv(self.output_dir / "family_score_table.csv", family_score_table)
        write_json(self.output_dir / "suite_leaderboard_row.json", leaderboard_row)
        summary_lines = self._summary_lines(suite_summary, task_score_table)
        summary_log = self.output_dir / "evaluation_summary.log"
        summary_log.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        static_dir = ensure_dir(self.output_dir / "static_api")
        write_json(static_dir / "suite_leaderboard.json", {"rows": [leaderboard_row]})
        write_json(static_dir / "task_scores.json", {"tasks": task_score_table, "families": family_score_table})
        return {
            "suite_manifest": suite_manifest,
            "suite_summary": suite_summary,
            "task_score_table": task_score_table,
            "family_score_table": family_score_table,
            "suite_leaderboard_row": leaderboard_row,
            "summary_lines": summary_lines,
            "summary_log": str(summary_log),
        }

    @staticmethod
    def _summary_lines(
        suite_summary: dict[str, Any], task_score_table: list[dict[str, Any]]
    ) -> list[str]:
        trials = int(suite_summary.get("num_trials") or 0)
        successes = int(suite_summary.get("num_success") or 0)
        micro_rate = successes / trials if trials else None
        macro_rate = suite_summary.get("covered_task_macro_success_rate")
        lines = [
            "=== Task Suite Evaluation Summary ===",
            (
                f"Overall episode success rate: {successes}/{trials} "
                f"({'N/A' if micro_rate is None else f'{micro_rate * 100.0:.2f}%'})"
            ),
            (
                "Overall task-macro success rate: "
                + ("N/A" if macro_rate is None else f"{float(macro_rate) * 100.0:.2f}%")
            ),
            "Per-task success rates:",
        ]
        for item in sorted(task_score_table, key=lambda value: int(value.get("order") or 0)):
            task_id = str(item.get("task_id") or "unknown_task")
            task_name = str(item.get("task_name") or task_id)
            task_trials = int(item.get("num_trials") or 0)
            task_success = int(item.get("num_success") or 0)
            task_rate = item.get("task_success_rate")
            status = str(item.get("score_status") or "unknown")
            rate_text = "N/A" if task_rate is None else f"{float(task_rate) * 100.0:.2f}%"
            lines.append(
                f"- {task_id} ({task_name}): {task_success}/{task_trials} "
                f"({rate_text}), status={status}"
            )
        return lines

    def _task_score_table(self, task_catalog: dict[str, Any], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summaries_by_task: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in runs:
            for item in run.get("task_summary") or []:
                task_id = item.get("task_id")
                if task_id:
                    enriched = dict(item)
                    enriched["run_id"] = run.get("manifest", {}).get("run_id") or item.get("run_id")
                    summaries_by_task[str(task_id)].append(enriched)

        rows: list[dict[str, Any]] = []
        for task in task_catalog.get("tasks", []):
            task_id = str(task.get("task_id"))
            source_task_id = str(task.get("source_task_id") or task_id)
            items = list(summaries_by_task.get(task_id, []))
            if source_task_id != task_id:
                items.extend(summaries_by_task.get(source_task_id, []))
            split_metrics = {split: None for split in SPLIT_ORDER}
            buckets = {split: {} for split in SPLIT_ORDER}
            for item in items:
                split = str(item.get("split") or "sim_dev")
                if split not in buckets:
                    continue
                # Several seed-specific runs may belong to one suite. Merge
                # counts; do not overwrite the previous run for this split.
                buckets[split][str(item.get('run_id') or item.get('episode_id') or len(buckets[split]))] = item
            for split, by_run in buckets.items():
                if not by_run:
                    continue
                selected = list(by_run.values())
                trials = sum(int(item.get('num_valid_trials', item.get('num_trials')) or 0) for item in selected)
                successes = sum(int(item.get('num_success') or 0) for item in selected)
                split_metrics[split] = {
                    "run_id": selected[-1].get("run_id"),
                    "run_ids": list(by_run),
                    "success_rate": successes / trials if trials else None,
                    "num_trials": trials,
                    "num_attempted": sum(int(item.get('num_trials') or 0) for item in selected),
                    "num_success": successes,
                    "failure_category_distribution": _merge_count_dicts([item.get('failure_category_distribution') for item in selected]),
                }
            total_trials = sum(metric["num_trials"] for metric in split_metrics.values() if metric)
            total_success = sum(metric["num_success"] for metric in split_metrics.values() if metric)
            task_success_rate = total_success / total_trials if total_trials else None
            if total_trials:
                score_status = "evaluated"
            elif task.get("status") == "implemented":
                score_status = "not_run"
            else:
                score_status = "not_implemented"
            rows.append(
                {
                    "order": task.get("order"),
                    "task_id": task_id,
                    "task_name": task.get("task_name"),
                    "task_family": task.get("task_family"),
                    "leaderboard_group": task.get("leaderboard_group"),
                    "scene_id": task.get("scene_id"),
                    "status": task.get("status"),
                    "score_status": score_status,
                    "sim_dev_success_rate": split_metrics["sim_dev"]["success_rate"] if split_metrics["sim_dev"] else None,
                    "sim_clean_success_rate": split_metrics["sim_clean"]["success_rate"] if split_metrics["sim_clean"] else None,
                    "sim_randomized_success_rate": split_metrics["sim_randomized"]["success_rate"] if split_metrics["sim_randomized"] else None,
                    "task_success_rate": task_success_rate,
                    "num_trials": total_trials,
                    "num_success": total_success,
                    "split_metrics": split_metrics,
                    "failure_category_distribution": _merge_count_dicts(
                        [metric.get("failure_category_distribution") for metric in split_metrics.values() if metric]
                    ),
                }
            )
        return rows

    def _family_score_table(self, task_score_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in task_score_table:
            grouped[str(row.get("task_family") or "basic")].append(row)
        rows = []
        for family, items in sorted(grouped.items()):
            evaluated = [item for item in items if item.get("task_success_rate") is not None]
            split_metrics = {}
            for split in SPLIT_ORDER:
                metrics = [item.get("split_metrics", {}).get(split) for item in items]
                metrics = [metric for metric in metrics if isinstance(metric, dict)]
                trials = sum(int(metric.get("num_trials") or 0) for metric in metrics)
                success = sum(int(metric.get("num_success") or 0) for metric in metrics)
                split_metrics[split] = {
                    "success_rate": success / trials if trials else None,
                    "num_trials": trials,
                    "num_success": success,
                    "failure_category_distribution": _merge_count_dicts(
                        [metric.get("failure_category_distribution") for metric in metrics]
                    ),
                }
            rows.append(
                {
                    "task_family": family,
                    "leaderboard_group": items[0].get("leaderboard_group") if items else family,
                    "task_ids": [item.get("task_id") for item in items],
                    "suite_task_count": len(items),
                    "evaluated_task_count": len(evaluated),
                    "coverage_rate": len(evaluated) / len(items) if items else None,
                    "sim_dev_success_rate": _mean([item.get("sim_dev_success_rate") for item in evaluated]),
                    "sim_clean_success_rate": _mean([item.get("sim_clean_success_rate") for item in evaluated]),
                    "sim_randomized_success_rate": _mean([item.get("sim_randomized_success_rate") for item in evaluated]),
                    "family_success_rate": _mean([item.get("task_success_rate") for item in evaluated]),
                    "split_metrics": split_metrics,
                    "num_trials": sum(int(item.get("num_trials") or 0) for item in items),
                    "num_success": sum(int(item.get("num_success") or 0) for item in items),
                    "failure_category_distribution": _merge_count_dicts([item.get("failure_category_distribution") for item in items]),
                }
            )
        return rows

    def _suite_summary(
        self,
        task_catalog: dict[str, Any],
        task_score_table: list[dict[str, Any]],
        family_score_table: list[dict[str, Any]],
    ) -> dict[str, Any]:
        evaluated = [item for item in task_score_table if item.get("task_success_rate") is not None]
        suite_task_count = len(task_score_table)
        evaluated_task_count = len(evaluated)
        return {
            "suite_id": task_catalog.get("suite_id"),
            "suite_name": task_catalog.get("suite_name"),
            "suite_version": task_catalog.get("suite_version"),
            "suite_task_count": suite_task_count,
            "evaluated_task_count": evaluated_task_count,
            "coverage_rate": evaluated_task_count / suite_task_count if suite_task_count else None,
            "covered_task_macro_success_rate": _mean([item.get("task_success_rate") for item in evaluated]),
            "sim_dev_macro_success_rate": _mean([item.get("sim_dev_success_rate") for item in evaluated]),
            "sim_clean_macro_success_rate": _mean([item.get("sim_clean_success_rate") for item in evaluated]),
            "sim_randomized_macro_success_rate": _mean([item.get("sim_randomized_success_rate") for item in evaluated]),
            "num_trials": sum(int(item.get("num_trials") or 0) for item in task_score_table),
            "num_success": sum(int(item.get("num_success") or 0) for item in task_score_table),
            "failure_category_distribution": _merge_count_dicts([item.get("failure_category_distribution") for item in task_score_table]),
            "family_count": len(family_score_table),
        }

    def _suite_leaderboard_row(
        self,
        *,
        suite_manifest: dict[str, Any],
        task_score_table: list[dict[str, Any]],
        family_score_table: list[dict[str, Any]],
        suite_summary: dict[str, Any],
    ) -> dict[str, Any]:
        success_rate = suite_summary.get("covered_task_macro_success_rate")
        success_score = None if success_rate is None else success_rate * 100.0
        capability_index = success_score
        return {
            "row_type": "suite",
            "leaderboard_id": f"{suite_manifest['eval_group_id']}_{suite_manifest['suite_id']}",
            "run_id": suite_manifest["run_id"],
            "eval_group_id": suite_manifest["eval_group_id"],
            "suite_id": suite_manifest["suite_id"],
            "suite_name": suite_manifest.get("suite_name"),
            "suite_version": suite_manifest.get("suite_version"),
            "model_name": suite_manifest.get("model_name"),
            "checkpoint": suite_manifest.get("checkpoint"),
            "benchmark_version": suite_manifest.get("benchmark_version"),
            "track": "task_suite",
            "task_label": suite_manifest.get("suite_name") or suite_manifest.get("suite_id"),
            "task_ids": [item.get("task_id") for item in task_score_table],
            "task_names": [item.get("task_name") for item in task_score_table],
            "task_families": sorted({str(item.get("task_family")) for item in task_score_table if item.get("task_family")}),
            "suite_task_count": suite_summary.get("suite_task_count"),
            "evaluated_task_count": suite_summary.get("evaluated_task_count"),
            "coverage_rate": suite_summary.get("coverage_rate"),
            "task_success_rate": success_rate,
            "covered_task_macro_success_rate": success_rate,
            "overall_success_rate": success_rate,
            "success_rate": success_rate,
            "num_trials": suite_summary.get("num_trials"),
            "num_success": suite_summary.get("num_success"),
            "capability_index": capability_index,
            "failure_category_distribution": suite_summary.get("failure_category_distribution"),
            "task_score_table": task_score_table,
            "family_score_table": family_score_table,
            "overall": {
                "sim_dev_macro_success_rate": suite_summary.get("sim_dev_macro_success_rate"),
                "sim_clean_macro_success_rate": suite_summary.get("sim_clean_macro_success_rate"),
                "sim_randomized_macro_success_rate": suite_summary.get("sim_randomized_macro_success_rate"),
                "capability_index_v0_1": capability_index,
            },
            "capabilities": {
                "success": success_score,
                "task_macro_success": success_score,
                "coverage": None if suite_summary.get("coverage_rate") is None else suite_summary["coverage_rate"] * 100.0,
            },
            "created_at": suite_manifest.get("created_at"),
        }
