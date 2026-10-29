"""Run complete seeded special-task demonstrations and require a success streak."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

TASKS = ("level3_L07_hard_centrifuge_load", "level4_weighing_feedback_hard",
         "level3_R10_hard_rack_recall", "level4_L09_hard_DeviceOperation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument("--output", required=True)
    parser.add_argument("--streak", type=int, default=5)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-attempts", type=int, default=20)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for task in args.task or TASKS:
        streak = successes = 0
        attempts = []
        for index in range(args.max_attempts):
            seed = args.seed + index
            out = root / task / f"seed_{seed}"
            out.mkdir(parents=True, exist_ok=False)
            command = [sys.executable, "main.py", "--config-name", task,
                       "--mode", "collect", "--backend", "numpy", "--headless", "--no-video",
                       "--seed", str(seed), "--max-success-episodes", "1",
                       "--max-consecutive-failures", "1", "--task-timeout-minutes", "25",
                       "--run-dir", str(out)]
            with (out / "worker.log").open("w") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            files = list((out / "dataset").glob("episode_*.h5"))
            metadata = out / "dataset/meta/episode.jsonl"
            rows = [json.loads(line) for line in metadata.read_text().splitlines()] if metadata.exists() else []
            accepted = bool(completed.returncode == 0 and len(files) == 1 and len(rows) == 1
                            and rows[0].get("task_properties", {}).get("physical_success") is True)
            attempts.append(dict(seed=seed, accepted=accepted, returncode=completed.returncode, run_dir=str(out)))
            successes += int(accepted)
            streak = streak + 1 if accepted else 0
            result = dict(task=task, success_count=successes, total_attempts=len(attempts),
                          consecutive_successes=streak, passed=streak >= args.streak, attempts=attempts)
            (root / f"{task}.json").write_text(json.dumps(result, indent=2))
            print(f"VALIDATION task={task} seed={seed} success={accepted} streak={streak}/{args.streak}", flush=True)
            if streak >= args.streak:
                break
        results.append(result)
    (root / "validation_report.json").write_text(json.dumps({
        "generated_at": datetime.now().isoformat(), "results": results,
        "passed": all(row["passed"] for row in results)}, indent=2))
    return 0 if all(row["passed"] for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
