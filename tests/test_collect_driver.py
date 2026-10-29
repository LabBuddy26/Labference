"""Tests for the tmux collection driver (utils/collect_driver.py).

Run with::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collect_driver.py -q
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path

import pytest

from utils.collect_driver import (
    MAX_WORKERS,
    STATUS_LABELS,
    aggregate,
    build_queue,
    claim_next,
    count_success_episodes,
    find_collect_result,
    format_collect_result,
    parse_collect_result,
    parse_task_list,
    select_tasks_for_targets,
    write_result,
)


def test_collection_driver_allows_four_workers():
    assert MAX_WORKERS == 4


# ---------------------------------------------------------------------------
# COLLECT_RESULT formatting / parsing
# ---------------------------------------------------------------------------

def test_collect_result_round_trip():
    line = format_collect_result(
        config="level1_pick",
        success_count=100,
        total_episodes=121,
        consecutive_failures=3,
        status="completed",
        duration_seconds=3725,
        run_dir="outputs/collect/2026.08.21/10.00.00_level1_pick",
    )
    assert line.startswith("COLLECT_RESULT ")
    parsed = parse_collect_result(line)
    assert parsed is not None
    assert parsed["config"] == "level1_pick"
    assert parsed["success_count"] == 100
    assert parsed["total_episodes"] == 121
    assert parsed["consecutive_failures"] == 3
    assert parsed["status"] == "completed"
    assert parsed["duration_seconds"] == 3725
    assert parsed["run_dir"].startswith("outputs/collect/")


def test_parse_collect_result_ignores_other_lines():
    assert parse_collect_result("some random log line") is None
    assert parse_collect_result("") is None
    assert parse_collect_result("  COLLECT_RESULT config=a success_count=5 status=timeout duration_seconds=1 run_dir=x")["config"] == "a"


def test_find_collect_result_in_log(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "$ python main.py --config-name level1_pour\n"
        "switching to phase 2...\n"
        "COLLECT_RESULT config=level1_pour success_count=7 total_episodes=27 "
        "consecutive_failures=20 status=consecutive_failures duration_seconds=900 run_dir=x\n"
        "trailing noise\n",
        encoding="utf-8",
    )
    result = find_collect_result(log)
    assert result is not None
    assert result["config"] == "level1_pour"
    assert result["status"] == "consecutive_failures"
    assert result["success_count"] == 7
    assert find_collect_result(tmp_path / "missing.log") is None


# ---------------------------------------------------------------------------
# parse_task_list
# ---------------------------------------------------------------------------

def _make_config(config_dir: Path, config: str, name: str | None = None) -> None:
    content = f"name: {name or config}\nrobot:\n  type: franka\n"
    (config_dir / f"{config}.yaml").write_text(content, encoding="utf-8")


def test_parse_task_list_basic(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _make_config(config_dir, "level1_pick", name="Pick the beaker")
    _make_config(config_dir, "level1_place")
    task_list = tmp_path / "tasks.txt"
    task_list.write_text(
        "# 注释行\n"
        "\n"
        "level1_pick\n"
        "level1_place.yaml\n"
        "level1_pick  # 重复行应被去重\n",
        encoding="utf-8",
    )
    tasks = parse_task_list(task_list, config_dir)
    assert [t["config"] for t in tasks] == ["level1_pick", "level1_place"]
    assert tasks[0]["name"] == "Pick the beaker"
    assert tasks[1]["name"] == "level1_place"  # fallback to config stem


def test_parse_task_list_missing_config_raises(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _make_config(config_dir, "level1_pick")
    task_list = tmp_path / "tasks.txt"
    task_list.write_text("level1_pick\nlevel1_nonexistent\n", encoding="utf-8")
    with pytest.raises(ValueError, match="level1_nonexistent"):
        parse_task_list(task_list, config_dir)


def test_parse_task_list_empty_raises(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    task_list = tmp_path / "tasks.txt"
    task_list.write_text("# only comments\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no tasks"):
        parse_task_list(task_list, config_dir)


def test_task_targets_select_only_explicit_json_tasks():
    tasks = [
        {"config": "level1_pick", "name": "Pick"},
        {"config": "level1_stir", "name": "Stir"},
        {"config": "level2_L07_centrifuge_load", "name": "Centrifuge"},
    ]
    selected = select_tasks_for_targets(
        tasks,
        {"level1_stir": 5, "level2_L07_centrifuge_load": 200},
    )
    assert [task["config"] for task in selected] == [
        "level1_stir",
        "level2_L07_centrifuge_load",
    ]


def test_task_targets_reject_config_absent_from_task_list():
    with pytest.raises(ValueError, match="absent from --task-list"):
        select_tasks_for_targets(
            [{"config": "level1_pick", "name": "Pick"}],
            {"level1_stir": 5},
        )


# ---------------------------------------------------------------------------
# Queue / claim exclusivity
# ---------------------------------------------------------------------------

def test_claim_exclusivity_under_concurrency(tmp_path):
    work_dir = tmp_path / "work"
    tasks = [{"config": f"task_{i}", "name": f"Task {i}"} for i in range(10)]
    build_queue(work_dir, tasks)

    claimed: list[dict] = []
    lock = threading.Lock()

    def worker():
        while True:
            task = claim_next(work_dir)
            if task is None:
                return
            time.sleep(0.001)
            with lock:
                claimed.append(task)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker) for _ in range(4)]
        concurrent.futures.wait(futures)

    assert len(claimed) == 10
    configs = sorted(t["config"] for t in claimed)
    assert configs == [f"task_{i}" for i in range(10)]  # no duplicates, no loss
    # Queue is drained; each claim leaves exactly one marker file (the worker
    # unlinks it after finishing, which cmd_worker does).
    assert list((work_dir / "queue").glob("*.json")) == []
    assert sorted(p.name for p in (work_dir / "claimed").glob("*.json")) == [
        f"{i:03d}_task_{i}.json" for i in range(10)
    ]


def test_claim_next_returns_none_when_empty(tmp_path):
    work_dir = tmp_path / "work"
    build_queue(work_dir, [{"config": "only", "name": "Only"}])
    assert claim_next(work_dir) is not None
    assert claim_next(work_dir) is None


# ---------------------------------------------------------------------------
# Success-count fallback (episode.jsonl line counting)
# ---------------------------------------------------------------------------

def test_count_success_episodes_picks_newest_run(tmp_path):
    outputs = tmp_path / "outputs"
    older = outputs / "collect" / "2026.08.20" / "09.00.00_level1_pick" / "dataset" / "meta"
    newer = outputs / "collect" / "2026.08.21" / "10.00.00_level1_pick" / "dataset" / "meta"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "episode.jsonl").write_text("a\nb\n", encoding="utf-8")
    (newer / "episode.jsonl").write_text("a\nb\nc\n\n", encoding="utf-8")

    now = time.time()
    os.utime(newer / "episode.jsonl", (now, now))
    os.utime(older / "episode.jsonl", (now - 1000, now - 1000))

    assert count_success_episodes("level1_pick", outputs_root=tmp_path / "outputs") == 3
    # since= excludes runs older than the given timestamp.
    assert count_success_episodes("level1_pick", outputs_root=tmp_path / "outputs", since=now + 10) == 0


def test_count_success_episodes_no_runs(tmp_path):
    assert count_success_episodes("nope", outputs_root=tmp_path / "outputs") == 0


# ---------------------------------------------------------------------------
# Results + aggregate report
# ---------------------------------------------------------------------------

def _sample_tasks():
    return [
        {"order": 0, "config": "level1_pick", "name": "Pick"},
        {"order": 1, "config": "level1_pour", "name": "Pour"},
        {"order": 2, "config": "level2_nmr_injection", "name": "NMR"},
    ]


def _sample_result(order, config, name, status, success, duration):
    return {
        "order": order,
        "config": config,
        "name": name,
        "status": status,
        "success_count": success,
        "total_episodes": success + 5,
        "consecutive_failures": 0,
        "started_at": "2026-08-21 10:00:00",
        "finished_at": "2026-08-21 11:00:00",
        "duration_seconds": duration,
        "returncode": 0,
        "run_dir": f"outputs/collect/2026.08.21/10.00.00_{name}",
        "log": f"logs/000_{config}_w0.log",
    }


def test_aggregate_report_json_and_md(tmp_path):
    work_dir = tmp_path / "work"
    build_queue(work_dir, _sample_tasks())
    write_result(
        work_dir,
        _sample_result(0, "level1_pick", "Pick", "completed", 100, 3600),
    )
    write_result(
        work_dir,
        _sample_result(1, "level1_pour", "Pour", "timeout", 42, 7800),
    )
    # task 2 has no result -> not_run

    report = aggregate(work_dir)
    assert report["summary"]["total_tasks"] == 3
    assert report["summary"]["completed"] == 1
    assert report["summary"]["timeout"] == 1
    assert report["summary"]["not_run"] == 1
    assert report["summary"]["total_success_episodes"] == 142

    rows = {row["config"]: row for row in report["rows"]}
    assert rows["level1_pour"]["duration_seconds"] == 7800
    assert rows["level2_nmr_injection"]["status"] == "not_run"

    # JSON report file exists and round-trips.
    json_report = json.loads((work_dir / "collection_report.json").read_text(encoding="utf-8"))
    assert json_report["summary"]["total_success_episodes"] == 142

    # Markdown report: header columns, duration formatting, status labels.
    md = (work_dir / "collection_report.md").read_text(encoding="utf-8")
    assert "| 任务 | 配置 | 采集数量 | 状态 | 开始时间 | 结束时间 | 采集时长 |" in md
    assert "| Pick | level1_pick | 100 |" in md
    assert STATUS_LABELS["timeout"] in md
    assert "2h 10m 0s" in md  # 7800s
    assert "1h 0m 0s" in md  # 3600s


def test_format_duration():
    from utils.collect_driver import format_duration

    assert format_duration(0) == "0s"
    assert format_duration(59) == "59s"
    assert format_duration(60) == "1m 0s"
    assert format_duration(3725) == "1h 2m 5s"
