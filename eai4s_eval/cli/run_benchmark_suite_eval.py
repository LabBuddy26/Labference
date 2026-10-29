from __future__ import annotations

import argparse
import re
import subprocess
import sys
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from eai4s_eval.cli.run_benchmark_eval import (
    _DEFAULT_BENCHMARK_ROOT,
    _DEFAULT_OUTPUT_ROOT,
    _DEFAULT_PROFILE,
)
from eai4s_eval.eval.task_suite import TaskSuiteAggregator, load_task_catalog, load_task_suite
from eai4s_eval.eval.task_suite import default_official_suite_id
from eai4s_eval.eval.model_identity import resolve_model_name, validate_checkpoint_name
from eai4s_eval.utils.io import ensure_dir, write_json


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_") or "eai4s_openpi"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an EAI4S benchmark task suite using one Isaac subprocess at a time.")
    parser.add_argument("--task-suite", "--task_suite", dest="task_suite", default=default_official_suite_id())
    parser.add_argument("--benchmark-root", dest="benchmark_root", default=str(_DEFAULT_BENCHMARK_ROOT))
    parser.add_argument("--policy-profile", "--policy_profile", dest="policy_profile", default=str(_DEFAULT_PROFILE))
    parser.add_argument("--vla-host", "--vla_host", dest="vla_host", default="127.0.0.1")
    parser.add_argument(
        "--vla-port", "--vla_port", dest="vla_port", type=int, default=None,
        help="Policy server port (default: OpenPI 8000, GR00T 5555).",
    )
    parser.add_argument("--api-key", "--api_key", dest="api_key", default=None)
    parser.add_argument("--policy-provider", "--policy_provider", dest="policy_provider",
                        choices=("openpi", "gr00t"), default="openpi")
    parser.add_argument("--policy-backend", "--policy_backend", dest="policy_backend",
                        choices=("websocket", "http", "zmq"), default=None)
    parser.add_argument("--policy-timeout", "--policy_timeout", dest="policy_timeout", type=float, default=30.0)
    parser.add_argument("--model-name", "--model_name", dest="model_name", default=None)
    parser.add_argument("--checkpoint", required=True,
                        help="Stable model-version identifier used by the official leaderboard.")
    parser.add_argument("--checkpoint-hash", "--checkpoint_hash", dest="checkpoint_hash", default=None)
    parser.add_argument("--dataset-version", "--dataset_version", dest="dataset_version", default=None)
    parser.add_argument("--split", choices=("sim_clean", "sim_dev", "sim_randomized"), default="sim_dev")
    parser.add_argument("--all-splits", "--all_splits", dest="all_splits", action="store_true")
    parser.add_argument("--num-episodes", "--num_episodes", dest="num_episodes", type=int, default=50)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--seed-manifest", "--seed_manifest", dest="seed_manifest", default=None)
    parser.add_argument("--run-id", "--run_id", dest="run_id", default=None)
    parser.add_argument("--output-root", "--output_root", dest="output_root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--eval-python", "--eval_python", dest="eval_python", default=sys.executable)
    parser.add_argument("--backend", choices=("numpy", "gpu"), default="numpy")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-video", "--save_video", dest="save_video", action="store_true")
    parser.add_argument(
        "--progress-interval-sec", "--progress_interval_sec",
        dest="progress_interval_sec", type=float, default=1.0,
    )
    parser.add_argument(
        "--progress-log-format", "--progress_log_format",
        dest="progress_log_format", choices=("both", "stdout", "jsonl"), default="both",
    )
    parser.add_argument("--continue-on-error", "--continue_on_error", dest="continue_on_error", action="store_true")
    parser.add_argument("--suite-only", "--suite_only", dest="suite_only", action="store_true")
    parser.add_argument('--evaluation-group', choices=('special', 'ordinary', 'both'))
    parser.add_argument('--tasks', help='Optional comma-separated task/protocol IDs for the versioned batch runner.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--task-timeout-seconds', type=float, default=1200)
    return parser


def _append(cmd: list[str], flag: str, value: object) -> None:
    if value not in (None, ""):
        cmd.extend([flag, str(value)])


def _redacted_command(cmd: Sequence[str]) -> list[str]:
    result = list(cmd)
    for index, value in enumerate(result[:-1]):
        if value == "--api-key":
            result[index + 1] = "***"
    return result


def _write_diagnostics(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, {"schema_version": "labbench_suite_diagnostics/v1", **payload})


def _run_task_process(
    cmd: list[str], *, cwd: Path, log_path: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    """Run and tee one task while persisting failures that occur before run creation."""
    ensure_dir(log_path.parent)
    started_at = datetime.now().astimezone()
    tail: deque[str] = deque(maxlen=200)
    result = {
        **identity,
        "command": _redacted_command(cmd),
        "cwd": str(cwd),
        "log_path": str(log_path),
        "started_at": started_at.isoformat(),
        "status": "running",
        "returncode": None,
    }
    proc: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"started_at: {started_at.isoformat()}\n")
            log.write("command: " + subprocess.list2cmdline(_redacted_command(cmd)) + "\n\n")
            log.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                tail.append(line.rstrip())
            returncode = proc.wait()
        result.update(
            status="completed" if returncode == 0 else "failed",
            returncode=returncode,
            output_tail=list(tail),
        )
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        result.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "launch_failed",
            returncode=130 if isinstance(exc, KeyboardInterrupt) else 127,
            exception_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
            output_tail=list(tail),
        )
    finished_at = datetime.now().astimezone()
    result["finished_at"] = finished_at.isoformat()
    result["duration_sec"] = round((finished_at - started_at).total_seconds(), 3)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.evaluation_group or args.task_suite == 'labference_special_featured_v1':
        if args.backend != 'numpy' or Path(args.eval_python).resolve() != Path(sys.executable).resolve():
            raise ValueError('Versioned batches currently use the current Python interpreter and numpy backend; launch this command with the intended environment Python and --backend numpy.')
        if args.all_splits or args.seed_manifest or args.api_key:
            raise ValueError('Versioned batches require explicit --seeds and one --split; authenticated adapters must use the existing single-task entrypoint.')
        from eai4s_eval.cli.run_labference_eval import main as run_batch
        group = args.evaluation_group or 'special'
        run_id = args.run_id or f'eval_{datetime.now():%Y%m%d_%H%M%S}_{group}'
        command = ['summarize' if args.suite_only else 'run', '--output', str(Path(args.output_root)/run_id),
                   '--group', group, '--checkpoint', args.checkpoint, '--split', args.split,
                   '--mode', 'reference' if args.reference else 'native',
                   '--model-name', args.model_name or ('reference-engineering' if args.reference else args.policy_provider),
                   '--seeds', args.seeds or ','.join(str(101+i) for i in range(args.num_episodes)),
                   '--policy-profile', args.policy_profile, '--policy-provider', args.policy_provider,
                   '--vla-host', args.vla_host, '--policy-timeout', str(args.policy_timeout),
                   '--task-timeout-seconds', str(args.task_timeout_seconds)]
        for flag, value in (('--tasks', args.tasks), ('--max-steps', args.max_steps),
                            ('--vla-port', args.vla_port), ('--policy-backend', args.policy_backend)):
            _append(command, flag, value)
        command.append('--headless' if args.headless else '--no-headless')
        if args.resume:
            command.append('--resume')
        if args.save_video:
            command.append('--save-video')
        return run_batch(command)
    args.model_name = resolve_model_name(args.policy_provider, args.model_name)
    args.checkpoint = validate_checkpoint_name(args.checkpoint, required=True)
    if args.policy_backend is None:
        args.policy_backend = "zmq" if args.policy_provider == "gr00t" else "websocket"
    if args.vla_port is None:
        args.vla_port = 5555 if args.policy_provider == "gr00t" else 8000
    suite = load_task_suite(args.task_suite)
    catalog = load_task_catalog(suite)
    suite_id = str(catalog["suite_id"])
    checkpoint = str(args.checkpoint)
    suite_run_id = args.run_id or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_safe(args.model_name)}_{suite_id}"
    splits = ("sim_clean", "sim_dev", "sim_randomized") if args.all_splits else (args.split,)
    suite_output = ensure_dir(Path(args.output_root) / suite_run_id)
    diagnostics_path = suite_output / "suite_diagnostics.json"
    diagnostics: dict[str, Any] = {
        "suite_run_id": suite_run_id,
        "suite_id": suite_id,
        "model_name": args.model_name,
        "checkpoint": checkpoint,
        "policy_provider": args.policy_provider,
        "policy_backend": args.policy_backend,
        "requested_splits": list(splits),
        "started_at": datetime.now().astimezone().isoformat(),
        "status": "running",
        "task_attempts": [],
        "failures": [],
    }
    _write_diagnostics(diagnostics_path, diagnostics)
    failures: list[dict[str, Any]] = []

    if not args.suite_only:
        for task in catalog.get("tasks", []):
            if task.get("status") != "implemented":
                continue
            for split in splits:
                task_id = str(task["task_id"])
                run_id = f"{suite_run_id}_{task_id}_{split}"
                cmd = [
                    str(args.eval_python), "-m", "eai4s_eval.cli.run_benchmark_eval",
                    "--benchmark-root", str(args.benchmark_root),
                    "--task-spec", str(task["task_spec"]),
                    "--policy-profile", str(args.policy_profile),
                    "--vla-host", str(args.vla_host),
                    "--vla-port", str(args.vla_port),
                    "--model-name", str(args.model_name),
                    "--checkpoint", str(checkpoint),
                    "--split", split,
                    "--num-episodes", str(args.num_episodes),
                    "--run-id", run_id,
                    "--eval-group-id", suite_run_id,
                    "--suite-id", suite_id,
                    "--run-type", "task_run",
                    "--output-root", str(args.output_root),
                    "--backend", str(args.backend),
                    "--progress-interval-sec", str(args.progress_interval_sec),
                    "--progress-log-format", str(args.progress_log_format),
                ]
                _append(cmd, "--api-key", args.api_key)
                _append(cmd, "--policy-provider", args.policy_provider)
                _append(cmd, "--policy-backend", args.policy_backend)
                _append(cmd, "--policy-timeout", args.policy_timeout)
                _append(cmd, "--seeds", args.seeds)
                _append(cmd, "--seed-manifest", args.seed_manifest)
                _append(cmd, "--checkpoint-hash", args.checkpoint_hash)
                _append(cmd, "--dataset-version", args.dataset_version)
                if args.headless:
                    cmd.append("--headless")
                if args.save_video:
                    cmd.append("--save-video")
                print(f"=== {task_id} / {split} ===", flush=True)
                attempt = _run_task_process(
                    cmd,
                    cwd=Path(__file__).resolve().parents[2],
                    log_path=suite_output / "task_logs" / f"{task_id}_{split}.log",
                    identity={"task_id": task_id, "source_task_id": task.get("source_task_id"), "split": split, "run_id": run_id},
                )
                diagnostics["task_attempts"].append(attempt)
                if attempt["returncode"]:
                    failures.append(attempt)
                    diagnostics["failures"] = failures
                    if attempt["status"] == "interrupted":
                        args.continue_on_error = False
                _write_diagnostics(diagnostics_path, diagnostics)
                if attempt["returncode"]:
                    print(
                        f"Task failed: {task_id}/{split}, returncode={attempt['returncode']}; "
                        f"see {attempt['log_path']}",
                        flush=True,
                    )
                    if not args.continue_on_error:
                        break
            if failures and not args.continue_on_error:
                break

    try:
        TaskSuiteAggregator(suite_output).write_all(
            task_catalog=catalog,
            outputs_root=args.output_root,
            eval_group_id=suite_run_id,
            model_name=str(args.model_name),
            checkpoint=str(checkpoint),
            robot="franka",
        )
    except BaseException as exc:
        diagnostics["aggregation_error"] = {
            "exception_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()
        }
        diagnostics["status"] = "aggregation_failed"
        diagnostics["finished_at"] = datetime.now().astimezone().isoformat()
        _write_diagnostics(diagnostics_path, diagnostics)
        raise
    diagnostics["status"] = "completed_with_failures" if failures else "completed"
    diagnostics["finished_at"] = datetime.now().astimezone().isoformat()
    _write_diagnostics(diagnostics_path, diagnostics)
    print(f"Suite outputs: {suite_output}", flush=True)
    if failures:
        print(
            "Task failures: "
            + str([(item["task_id"], item["split"], item["returncode"]) for item in failures]),
            flush=True,
        )
        print(f"Diagnostics: {diagnostics_path}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
