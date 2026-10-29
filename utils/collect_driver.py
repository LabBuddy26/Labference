"""Automated tmux-based data collection driver for LabBench.

Collects the tasks listed in a collection-list file (one YAML config name per
line) across at most two concurrent Isaac Sim terminals.  Each task collects
``--success-episodes`` successful episodes; a task is abandoned when it exceeds
``--consecutive-failures`` consecutive failed episodes or the
``--task-timeout-minutes`` wall clock.  The final ``collection_report.{json,md}``
records every task's config name, collected success count and duration.

Workflow::

    python -m utils.collect_driver start --task-list collection_tasks.txt

creates a detached tmux session with one worker pane per terminal, then
monitors until every queued task has a result.  Each worker self-schedules from
the shared queue (atomic ``os.rename`` claim), runs ``main.py`` for one task at
a time, parses the ``COLLECT_RESULT`` summary line printed by ``main.py`` and
writes ``results/<order>_<config>.json``.

This module intentionally imports only the standard library so it can be
imported and tested without Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Maximum concurrent collection terminals.  More Isaac instances than this
# overflow the GPU memory of the dev machine (single RTX 5090).
MAX_WORKERS = 4

COLLECT_RESULT_MARKER = "COLLECT_RESULT"
COLLECT_RESULT_RE = re.compile(r"^\s*COLLECT_RESULT\s+(.*?)\s*$")

# Extra time granted to main.py beyond its own task timeout for a clean Isaac
# shutdown (HDF5 flush + SimulationApp close) before the worker hard-kills it.
SHUTDOWN_GRACE_SECONDS = 300.0

DEFAULT_SESSION = "labbench_collect_new"

STATUS_LABELS = {
    "completed": "已完成",
    "consecutive_failures": "连续失败切换",
    "timeout": "超时",
    "error": "错误",
    "skipped": "跳过(预演)",
    "not_run": "未采集",
}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_json_atomic(path: str | Path, data: Any) -> None:
    """Write JSON via a temporary file + atomic replace (job_queue pattern)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_yaml_name(config_path: str | Path) -> str:
    """Return the ``name:`` field of a task YAML (fallback: config stem)."""
    path = Path(config_path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return path.stem
    match = re.search(r"(?m)^name\s*:\s*(.*?)\s*$", text)
    if not match:
        return path.stem
    name = match.group(1).strip()
    if len(name) >= 2 and name[0] == name[-1] and name[0] in "\"'":
        name = name[1:-1]
    return name


def parse_task_list(path: str | Path, config_dir: str | Path = "config") -> list[dict[str, Any]]:
    """Parse a collection list into [{config, name}] task descriptors.

    Blank lines and ``#`` comments are ignored; a trailing ``.yaml`` suffix is
    stripped; every config must exist under ``config_dir``.  The ``name`` field
    is the YAML's display name used to locate its run directory later.
    """
    config_dir = Path(config_dir)
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    source = Path(path)
    for lineno, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        config = line.split()[0]
        if config.endswith(".yaml"):
            config = config[:-5]
        if config in seen:
            continue
        config_path = config_dir / f"{config}.yaml"
        if not config_path.is_file():
            raise ValueError(f"{source}:{lineno}: config {config!r} not found at {config_path}")
        seen.add(config)
        tasks.append({"config": config, "name": _read_yaml_name(config_path)})
    if not tasks:
        raise ValueError(f"{source}: no tasks found")
    return tasks


def load_task_targets(path: str | Path | None, overrides: list[str] | None = None) -> dict[str, int]:
    """Load per-task episode targets from JSON and ``CONFIG=COUNT`` options."""
    targets: dict[str, int] = {}
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("--task-targets must contain a JSON object mapping config names to counts")
        for config, count in data.items():
            targets[str(config).removesuffix(".yaml")] = int(count)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"invalid --task-success-episodes value {item!r}; expected CONFIG=COUNT")
        config, count = item.split("=", 1)
        targets[config.removesuffix(".yaml")] = int(count)
    if any(value <= 0 for value in targets.values()):
        raise ValueError("per-task episode targets must be positive")
    return targets


def select_tasks_for_targets(
    tasks: list[dict[str, Any]], targets: dict[str, int]
) -> list[dict[str, Any]]:
    """Return only tasks explicitly named by a task-targets JSON mapping.

    The task list remains the allow-list and supplies display names/order, but
    unspecified task-list entries must not silently inherit the global target.
    """
    by_config = {str(task["config"]): task for task in tasks}
    missing = [config for config in targets if config not in by_config]
    if missing:
        raise ValueError(
            "--task-targets references configs absent from --task-list: "
            + ", ".join(missing)
        )
    selected = [task for task in tasks if task["config"] in targets]
    if not selected:
        raise ValueError("--task-targets did not select any tasks")
    return selected


def format_collect_result(
    *,
    config: str,
    success_count: int,
    total_episodes: int,
    consecutive_failures: int,
    status: str,
    duration_seconds: int,
    run_dir: str,
) -> str:
    """Format the machine-readable summary line printed by main.py."""
    return (
        f"{COLLECT_RESULT_MARKER} config={config} success_count={success_count} "
        f"total_episodes={total_episodes} consecutive_failures={consecutive_failures} "
        f"status={status} duration_seconds={duration_seconds} run_dir={run_dir}"
    )


_NUMERIC_KEYS = ("success_count", "total_episodes", "consecutive_failures", "duration_seconds")


def parse_collect_result(line: str) -> dict[str, Any] | None:
    """Parse a ``COLLECT_RESULT`` line into a dict, or return None."""
    match = COLLECT_RESULT_RE.match(line)
    if not match:
        return None
    fields: dict[str, Any] = {}
    for token in match.group(1).split():
        if "=" in token:
            key, value = token.split("=", 1)
            if key in _NUMERIC_KEYS:
                try:
                    value = int(value)
                except ValueError:
                    pass
            fields[key] = value
    return fields or None


def find_collect_result(log_path: str | Path) -> dict[str, Any] | None:
    """Locate the first COLLECT_RESULT line inside a run log."""
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        parsed = parse_collect_result(line)
        if parsed:
            return parsed
    return None


# ---------------------------------------------------------------------------
# Queue / claim protocol
# ---------------------------------------------------------------------------

def build_queue(work_dir: str | Path, tasks: list[dict[str, Any]]) -> int:
    """Materialize the task queue as one JSON file per task (idempotent)."""
    work_dir = Path(work_dir)
    for name in ("queue", "claimed", "results", "logs"):
        (work_dir / name).mkdir(parents=True, exist_ok=True)
    for order, task in enumerate(tasks):
        entry = {"order": order, **task}
        path = work_dir / "queue" / f"{order:03d}_{task['config']}.json"
        if not path.exists():
            _write_json_atomic(path, entry)
    _write_json_atomic(work_dir / "tasks.json", tasks)
    return len(tasks)


def claim_next(work_dir: str | Path) -> dict[str, Any] | None:
    """Atomically claim the lowest-order queued task, or None when empty.

    Claiming is an ``os.rename`` into ``claimed/`` on the same filesystem, so
    concurrent workers never double-claim: the loser of a race gets a
    ``FileNotFoundError`` and moves on.
    """
    queue_dir = Path(work_dir) / "queue"
    claimed_dir = Path(work_dir) / "claimed"
    while True:
        entries = sorted(queue_dir.glob("*.json"), key=lambda path: path.name)
        if not entries:
            return None
        for entry in entries:
            target = claimed_dir / entry.name
            try:
                os.rename(entry, target)
            except FileNotFoundError:
                continue  # another worker claimed it first
            return json.loads(target.read_text(encoding="utf-8"))
        time.sleep(0.05)  # every entry raced; re-list


# ---------------------------------------------------------------------------
# Success counting fallback
# ---------------------------------------------------------------------------

def count_success_episodes(name: str, outputs_root: str | Path = "outputs", *, since: float | None = None) -> int:
    """Count successful episodes in the newest run dir matching ``*_<name>``.

    ``main.py`` appends one line per successful episode to
    ``<run_dir>/dataset/meta/episode.jsonl`` synchronously, so this is a
    reliable fallback when the COLLECT_RESULT line is unavailable.
    """
    collect_root = Path(outputs_root) / "collect"
    if not collect_root.is_dir():
        return 0
    candidates: list[Path] = []
    for date_dir in collect_root.iterdir():
        if not date_dir.is_dir():
            continue
        candidates.extend(date_dir.glob(f"*_{name}/dataset/meta/episode.jsonl"))
    if not candidates:
        return 0
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    if since is not None and latest.stat().st_mtime < since:
        return 0
    count = 0
    for line in latest.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            count += 1
    return count


def _fallback_run_dir(log_path: Path) -> str | None:
    """Extract the run directory from a collection log when the marker is absent."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    matches = re.findall(r"Writing episode .*? to (.+?)/dataset/episode_\d+\.h5(?:\.tmp)?", text)
    return matches[-1].strip() if matches else None


def _fallback_summary(task: dict[str, Any], log_path: Path, options: argparse.Namespace,
                      started_ts: float, returncode: int | None) -> dict[str, Any]:
    """Recover a useful result from data/logs when COLLECT_RESULT is missing."""
    # The report root is outputs/collect_reports; collected episodes live under
    # the sibling outputs/collect tree.  Try both roots for older invocations.
    roots = [Path(options.project_dir) / "outputs"]
    report_root = Path(options.outputs_root)
    if report_root not in roots:
        roots.append(report_root)
    count = max((count_success_episodes(str(task.get("name", task["config"])), r, since=started_ts)
                 for r in roots), default=0)
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    target = int(task.get("target_episodes", options.success_episodes))
    if count >= target:
        status = "completed"
    elif "consecutive failure" in text.lower() or "consecutive_failures" in text.lower():
        status = "consecutive_failures"
    elif "wall-clock timeout" in text.lower() or returncode in (-9, 124):
        status = "timeout"
    else:
        status = "error"
    return {
        "status": status,
        "success_count": count,
        "total_episodes": count,
        "consecutive_failures": 0,
        "run_dir": _fallback_run_dir(log_path),
        "error": None if status == "completed" else "missing COLLECT_RESULT; recovered from log/data",
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def build_main_command(task: dict[str, Any], options: argparse.Namespace) -> list[str]:
    target_episodes = int(task.get("target_episodes", options.success_episodes))
    command = [
        str(options.python),
        "main.py",
        "--config-name",
        task["config"],
        "--config-dir",
        str(options.config_dir),
        "--backend",
        str(options.backend),
        "--max-success-episodes",
        str(target_episodes),
        "--max-consecutive-failures",
        str(options.consecutive_failures),
    ]
    if options.task_timeout_minutes:
        command += ["--task-timeout-minutes", str(options.task_timeout_minutes)]
    if options.headless:
        command.append("--headless")
    if not options.video:
        command.append("--no-video")
    run_dir = task.get("run_dir")
    if run_dir:
        command += ["--run-dir", str(run_dir)]
    return command


def _dataset_run_dir(task: dict[str, Any], work_dir: Path,
                     options: argparse.Namespace) -> Path | None:
    """Return the external dataset run directory for one queued task."""
    root = getattr(options, "dataset_root", None)
    if not root:
        return None
    name = str(task.get("name", task["config"]))
    return (Path(root).expanduser().resolve() / "collect" /
            work_dir.parent.name / f"{work_dir.name}_{name}")


def run_one(task: dict[str, Any], work_dir: str | Path, worker_index: int, options: argparse.Namespace) -> dict[str, Any]:
    """Run one collection task as a subprocess and write its result file."""
    work_dir = Path(work_dir)
    config = str(task["config"])
    order = int(task.get("order", 0))
    log_rel = f"logs/{order:03d}_{config}_w{worker_index}.log"
    log_path = work_dir / log_rel
    started_at = _now()
    started_ts = time.time()
    result: dict[str, Any] = {
        "order": order,
        "config": config,
        "name": task.get("name", config),
        "status": "error",
        "success_count": 0,
        "total_episodes": 0,
        "consecutive_failures": 0,
        "started_at": started_at,
        "finished_at": started_at,
        "duration_seconds": 0.0,
        "returncode": None,
        "run_dir": None,
        "log": log_rel,
    }

    dataset_run_dir = _dataset_run_dir(task, work_dir, options)
    command_task = dict(task)
    if dataset_run_dir is not None:
        dataset_run_dir.mkdir(parents=True, exist_ok=True)
        command_task["run_dir"] = str(dataset_run_dir)
    command = build_main_command(command_task, options)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(options.gpu_id)

    if options.dry_run:
        print(f"[worker {worker_index}] [dry-run] would run: {' '.join(command)}", flush=True)
        result["status"] = "skipped"
        write_result(work_dir, result)
        return result

    timeout_seconds: float | None = None
    if options.task_timeout_minutes:
        timeout_seconds = float(options.task_timeout_minutes) * 60.0 + SHUTDOWN_GRACE_SECONDS

    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            log.write(f"$ {' '.join(command)}\n")
            log.flush()
            proc = subprocess.Popen(
                command,
                cwd=str(options.project_dir),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                returncode = proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                # main.py's own wall-clock check cannot fire while world.step
                # is stuck, so hard-kill from outside as the backstop.
                print(f"[worker {worker_index}] {config}: exceeded {timeout_seconds}s, terminating PID {proc.pid}", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                returncode = -9
    except OSError as exc:
        result["finished_at"] = _now()
        result["duration_seconds"] = round(time.time() - started_ts, 1)
        result["error"] = f"failed to start subprocess: {exc}"
        write_result(work_dir, result)
        print(f"[worker {worker_index}] {config}: {result['error']}", flush=True)
        return result

    summary = find_collect_result(log_path)
    if summary is not None:
        result.update(
            status=str(summary.get("status") or "error"),
            success_count=int(summary.get("success_count") or 0),
            total_episodes=int(summary.get("total_episodes") or 0),
            consecutive_failures=int(summary.get("consecutive_failures") or 0),
            run_dir=summary.get("run_dir"),
        )
    else:
        result.update(_fallback_summary(task, log_path, options, started_ts, returncode))
        if result.get("error"):
            result["error"] = f"{result['error']}; returncode={returncode}"

    result["finished_at"] = _now()
    result["duration_seconds"] = round(time.time() - started_ts, 1)
    result["returncode"] = returncode
    write_result(work_dir, result)
    print(
        f"[worker {worker_index}] {config}: {result['status']} success={result['success_count']} "
        f"({result['duration_seconds']}s) log={log_rel}",
        flush=True,
    )
    return result


def write_result(work_dir: str | Path, result: dict[str, Any]) -> None:
    order = int(result.get("order", 0))
    config = str(result["config"])
    _write_json_atomic(Path(work_dir) / "results" / f"{order:03d}_{config}.json", result)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def aggregate(work_dir: str | Path) -> dict[str, Any]:
    """Aggregate per-task results into the final collection report."""
    work_dir = Path(work_dir)
    tasks = json.loads((work_dir / "tasks.json").read_text(encoding="utf-8")) if (work_dir / "tasks.json").is_file() else []
    results: dict[str, dict[str, Any]] = {}
    for path in sorted((work_dir / "results").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("config"):
            results[str(data["config"])] = data

    rows: list[dict[str, Any]] = []
    for task in tasks:
        result = results.get(str(task["config"]))
        if result is None:
            rows.append(
                {
                    "order": task.get("order", 0),
                    "config": task["config"],
                    "name": task.get("name", task["config"]),
                    "status": "not_run",
                    "success_count": 0,
                    "started_at": None,
                    "finished_at": None,
                    "duration_seconds": 0.0,
                    "run_dir": None,
                    "returncode": None,
                    "log": None,
                }
            )
            continue
        rows.append(result)
    rows.sort(key=lambda item: (int(item.get("order", 0)), str(item.get("config", ""))))

    summary: dict[str, int] = {status: 0 for status in STATUS_LABELS}
    for row in rows:
        summary[str(row.get("status") or "not_run")] = summary.get(str(row.get("status") or "not_run"), 0) + 1
    report = {
        "generated_at": _now(),
        "summary": {
            "total_tasks": len(rows),
            "completed": summary.get("completed", 0),
            "consecutive_failures": summary.get("consecutive_failures", 0),
            "timeout": summary.get("timeout", 0),
            "error": summary.get("error", 0),
            "skipped": summary.get("skipped", 0),
            "not_run": summary.get("not_run", 0),
            "total_success_episodes": sum(int(row.get("success_count") or 0) for row in rows),
        },
        "rows": rows,
    }
    _write_json_atomic(work_dir / "collection_report.json", report)
    write_markdown_report(work_dir, report)
    return report


def write_markdown_report(work_dir: str | Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# 采集报告",
        "",
        f"- 生成时间: {report['generated_at']}",
        (
            f"- 任务总数: {summary['total_tasks']} | 已完成: {summary['completed']} | "
            f"连续失败切换: {summary['consecutive_failures']} | 超时: {summary['timeout']} | "
            f"错误: {summary['error']} | 未采集: {summary['not_run']}"
        ),
        f"- 成功 episode 总数: {summary['total_success_episodes']}",
        "",
        "| 任务 | 配置 | 采集数量 | 状态 | 开始时间 | 结束时间 | 采集时长 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        status = str(row.get("status") or "not_run")
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                row.get("name", "-"),
                row.get("config", "-"),
                row.get("success_count", 0),
                STATUS_LABELS.get(status, status),
                row.get("started_at") or "-",
                row.get("finished_at") or "-",
                format_duration(float(row.get("duration_seconds") or 0.0)),
            )
        )
    lines.append("")
    (Path(work_dir) / "collection_report.md").write_text("\n".join(lines), encoding="utf-8")


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("=" * 72)
    print("采集完成汇总")
    print("=" * 72)
    print(f"任务总数: {summary['total_tasks']} | 已完成: {summary['completed']} | "
          f"连续失败切换: {summary['consecutive_failures']} | 超时: {summary['timeout']} | "
          f"错误: {summary['error']} | 未采集: {summary['not_run']}")
    print(f"成功 episode 总数: {summary['total_success_episodes']}")
    print("-" * 72)
    for row in report["rows"]:
        status = str(row.get("status") or "not_run")
        print(
            f"  {row.get('config', '-'):<28} 数量={row.get('success_count', 0):<4} "
            f"{STATUS_LABELS.get(status, status):<8} 时长={format_duration(float(row.get('duration_seconds') or 0.0))}"
        )
    print("=" * 72)


# ---------------------------------------------------------------------------
# tmux orchestration
# ---------------------------------------------------------------------------

def _tmux(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["tmux", *args], capture_output=True, text=True, check=check)


def _worker_argv(options: argparse.Namespace, index: int, gpu_id: str, work_dir: Path) -> list[str]:
    argv = [
        str(options.python),
        "-m",
        "utils.collect_driver",
        "worker",
        "--work-dir",
        str(work_dir),
        "--index",
        str(index),
        "--gpu-id",
        str(gpu_id),
        "--success-episodes",
        str(options.success_episodes),
        "--consecutive-failures",
        str(options.consecutive_failures),
        "--backend",
        str(options.backend),
        "--python",
        str(options.python),
        "--config-dir",
        str(options.config_dir),
        "--outputs-root",
        str(options.outputs_root),
        "--project-dir",
        str(options.project_dir),
    ]
    if getattr(options, "dataset_root", None):
        argv += ["--dataset-root", str(options.dataset_root)]
    if options.task_timeout_minutes:
        argv += ["--task-timeout-minutes", str(options.task_timeout_minutes)]
    if options.headless:
        argv.append("--headless")
    if options.video:
        argv.append("--video")
    if options.dry_run:
        argv.append("--dry-run")
    return argv


def _monitor(work_dir: Path, *, poll_interval: float, stale_after: float) -> None:
    """Block until every queued task has a result; recover stuck claims."""
    queue_dir = work_dir / "queue"
    claimed_dir = work_dir / "claimed"
    results_dir = work_dir / "results"
    # Authoritative task count from the queue snapshot; queue/ may already be
    # drained when re-attaching via `monitor`.
    total = 0
    tasks_json = work_dir / "tasks.json"
    if tasks_json.is_file():
        try:
            total = len(json.loads(tasks_json.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            total = 0
    if not total:
        total = len(list(queue_dir.glob("*.json")))
    started = time.time()
    while True:
        claimed = sorted(claimed_dir.glob("*.json"))
        for claimed_path in claimed:
            if time.time() - claimed_path.stat().st_mtime > stale_after:
                # A worker crashed mid-task (main.py's own stops could not
                # fire).  Record the task as failed so the queue drains.
                try:
                    task = json.loads(claimed_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    task = None
                if task is not None and isinstance(task, dict):
                    result = {
                        "order": int(task.get("order", 0)),
                        "config": str(task.get("config", "?")),
                        "name": str(task.get("name", task.get("config", "?"))),
                        "status": "timeout",
                        "success_count": 0,
                        "total_episodes": 0,
                        "consecutive_failures": 0,
                        "started_at": _now(),
                        "finished_at": _now(),
                        "duration_seconds": stale_after,
                        "returncode": None,
                        "run_dir": None,
                        "log": None,
                        "error": "worker did not finish the claimed task before shutdown/stale deadline",
                    }
                    write_result(work_dir, result)
                claimed_path.unlink(missing_ok=True)

        done = len(list(results_dir.glob("*.json")))
        remaining_queued = len(list(queue_dir.glob("*.json")))
        remaining_claimed = len(list(claimed_dir.glob("*.json")))
        elapsed = int(time.time() - started)
        print(
            f"[monitor] {done}/{total} tasks done, {remaining_claimed} running, "
            f"{remaining_queued} queued (elapsed {format_duration(elapsed)})",
            flush=True,
        )
        if done >= total and remaining_queued == 0 and remaining_claimed == 0:
            return
        time.sleep(poll_interval)


def cmd_start(args: argparse.Namespace) -> int:
    if args.workers < 1 or args.workers > MAX_WORKERS:
        raise SystemExit(f"--workers must be between 1 and {MAX_WORKERS}")
    if shutil.which("tmux") is None:
        raise SystemExit("tmux is required but was not found on PATH")

    if args.dataset_root:
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        dataset_root.mkdir(parents=True, exist_ok=True)
        if not os.access(dataset_root, os.W_OK):
            raise SystemExit(f"dataset root is not writable: {dataset_root}")
        free_bytes = shutil.disk_usage(dataset_root).free
        if free_bytes < 10 * 1024**3:
            raise SystemExit(
                f"dataset root has less than 10 GiB free: {dataset_root} "
                f"({free_bytes / 1024**3:.1f} GiB)"
            )

    tasks = parse_task_list(args.task_list, args.config_dir)
    json_targets = load_task_targets(args.task_targets) if args.task_targets else {}
    task_targets = load_task_targets(args.task_targets, args.task_success_episodes)
    if args.task_targets:
        override_only = sorted(set(task_targets) - set(json_targets))
        if override_only:
            raise ValueError(
                "--task-success-episodes cannot add tasks outside --task-targets: "
                + ", ".join(override_only)
            )
        tasks = select_tasks_for_targets(tasks, json_targets)
    for task in tasks:
        task["target_episodes"] = int(task_targets.get(task["config"], args.success_episodes))

    probe = _tmux(["has-session", "-t", args.session], check=False)
    if probe.returncode == 0:
        print(
            f"tmux session {args.session!r} already exists.\n"
            f"Attach with: tmux attach -t {args.session}",
            flush=True,
        )
        try:
            answer = input(
                f"Kill the existing session {args.session!r} and start a new collection? [y/N]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExisting tmux session was kept; collection not started.", flush=True)
            return 1
        if answer not in {"y", "yes"}:
            print("Existing tmux session was kept; collection not started.", flush=True)
            return 1
        _tmux(["kill-session", "-t", args.session], check=False)
        print(f"Killed existing tmux session {args.session!r}; starting a new one.", flush=True)

    gpu_ids = [item.strip() for item in str(args.gpu_ids).split(",") if item.strip()]
    if not gpu_ids:
        gpu_ids = ["0"]
    while len(gpu_ids) < args.workers:
        gpu_ids.append(gpu_ids[-1])

    stamp = datetime.now().strftime("%Y.%m.%d/%H.%M.%S")
    work_dir = Path(args.outputs_root) / stamp
    build_queue(work_dir, tasks)
    print(f"Queue built: {len(tasks)} tasks -> {work_dir}", flush=True)
    print(f"Starting tmux session {args.session!r} with {args.workers} worker(s)...", flush=True)

    _tmux(["new-session", "-d", "-s", args.session, "-x", "240", "-y", "60"])
    _tmux(["set-option", "-t", args.session, "remain-on-exit", "on"])
    for index in range(args.workers):
        command = shlex.join(_worker_argv(args, index, gpu_ids[index], work_dir))
        shell_line = f"cd {shlex.quote(str(args.project_dir))} && {command}; echo '[worker {index}] exited'"
        if index == 0:
            _tmux(["send-keys", "-t", f"{args.session}:0.0", shell_line, "Enter"])
        else:
            _tmux(["split-window", "-h", "-t", f"{args.session}:0"])
            _tmux(["send-keys", "-t", f"{args.session}:0.{index}", shell_line, "Enter"])
    _tmux(["select-layout", "-t", f"{args.session}:0", "tiled"])
    _tmux(["select-pane", "-t", f"{args.session}:0.0"])

    print(f"Attach with: tmux attach -t {args.session}", flush=True)
    stale_after = (
        float(args.task_timeout_minutes) * 60.0 + SHUTDOWN_GRACE_SECONDS + 900.0
        if args.task_timeout_minutes
        else 24 * 3600.0
    )
    _monitor(work_dir, poll_interval=args.poll_interval, stale_after=stale_after)
    print("\nAll tasks finished. Final report:", flush=True)
    print_report(aggregate(work_dir))
    print(f"Report written to {work_dir / 'collection_report.json'} and {work_dir / 'collection_report.md'}", flush=True)
    print(f"The tmux session is kept alive for inspection: tmux attach -t {args.session}", flush=True)
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    print(
        f"[worker {args.index}] starting (gpu={args.gpu_id}, work_dir={args.work_dir}, "
        f"success_target={args.success_episodes}, consecutive_failures={args.consecutive_failures}, "
        f"timeout={args.task_timeout_minutes}min)",
        flush=True,
    )
    while True:
        task = claim_next(args.work_dir)
        if task is None:
            print(f"[worker {args.index}] queue empty; worker done.", flush=True)
            return 0
        run_one(task, args.work_dir, args.index, args)
        # Release the claim so the monitor's queue-drain condition can fire.
        claim_path = (
            Path(args.work_dir)
            / "claimed"
            / f"{int(task.get('order', 0)):03d}_{task['config']}.json"
        )
        claim_path.unlink(missing_ok=True)


def cmd_monitor(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir)
    stale_after = (
        float(args.task_timeout_minutes) * 60.0 + SHUTDOWN_GRACE_SECONDS + 900.0
        if args.task_timeout_minutes
        else 24 * 3600.0
    )
    _monitor(work_dir, poll_interval=args.poll_interval, stale_after=stale_after)
    print("\nFinal report:", flush=True)
    print_report(aggregate(work_dir))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python interpreter that can import isaacsim (default: this interpreter)")
    parser.add_argument("--config-dir", type=str, default="config",
                        help="Hydra config directory with the task YAMLs")
    parser.add_argument("--outputs-root", type=str, default="outputs/collect_reports",
                        help="Root directory for work dirs and reports")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Optional root for episode runs (for example /mnt/VLA_datasets/sim_datasets)",
    )
    parser.add_argument("--project-dir", type=str,
                        default=str(Path(__file__).resolve().parents[1]),
                        help="Repository root where main.py lives (default: derived from this file)")
    parser.add_argument("--success-episodes", type=int, default=200,
                        help="Successful episodes to collect per task (default: 200)")
    parser.add_argument("--task-targets", type=str, default=None,
                        help="JSON mapping config names to targets; when provided, only these tasks are queued")
    parser.add_argument("--task-success-episodes", action="append", default=[], metavar="CONFIG=COUNT",
                        help="Override one task target; may be repeated (for example level1_pick=50)")
    parser.add_argument("--consecutive-failures", type=int, default=20,
                        help="Consecutive failed episodes after which a task is abandoned (default: 20)")
    parser.add_argument("--task-timeout-minutes", type=float, default=300.0,
                        help="Per-task wall-clock timeout in minutes; 0 disables it (default: 300 = 5 hours)")
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "gpu"])
    parser.add_argument("--headless", dest="headless", action=argparse.BooleanOptionalAction, default=True,
                        help="Run Isaac headless (default: on)")
    parser.add_argument("--video", dest="video", action=argparse.BooleanOptionalAction, default=False,
                        help="Save per-episode camera videos (default: off to save disk)")
    parser.add_argument("--poll-interval", type=float, default=10.0,
                        help="Seconds between monitor progress polls")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Build the queue, launch tmux workers and monitor to completion")
    start.add_argument("--task-list", required=True, help="Collection list file (one config name per line)")
    start.add_argument("--workers", type=int, default=2, help=f"Concurrent terminals (1-{MAX_WORKERS}, default 2)")
    start.add_argument("--session", type=str, default=DEFAULT_SESSION, help="tmux session name")
    start.add_argument("--gpu-ids", type=str, default="0,0", help="Comma-separated CUDA device per worker")
    start.add_argument("--dry-run", action="store_true", help="Workers print commands instead of running Isaac")
    _add_common_arguments(start)

    worker = subparsers.add_parser("worker", help="Internal: claim-and-run loop for one tmux pane")
    worker.add_argument("--work-dir", required=True, help="Driver work directory")
    worker.add_argument("--index", type=int, required=True, help="Worker index (0-based)")
    worker.add_argument("--gpu-id", type=str, default="0", help="CUDA device for this worker")
    worker.add_argument("--dry-run", action="store_true", help="Print the would-be command and mark the task skipped")
    _add_common_arguments(worker)

    monitor = subparsers.add_parser("monitor", help="Re-attach monitoring to an existing work dir and write the report")
    monitor.add_argument("--work-dir", required=True, help="Driver work directory")
    _add_common_arguments(monitor)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "start":
        return cmd_start(args)
    if args.command == "worker":
        return cmd_worker(args)
    if args.command == "monitor":
        return cmd_monitor(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
