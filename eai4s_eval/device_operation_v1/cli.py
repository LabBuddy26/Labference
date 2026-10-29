"""CLI adapter for the hard drying-oven collection and model evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from . import TASK_VERSION


def _load(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    evaluation = payload.get("evaluation") if isinstance(payload, dict) else None
    if not isinstance(evaluation, dict) or evaluation.get("task_version") != TASK_VERSION:
        raise ValueError(f"config must declare evaluation.task_version={TASK_VERSION!r}")
    if evaluation.get("mode") not in {"reference", "native"}:
        raise ValueError("evaluation.mode must be reference or native")
    if payload.get("legacy_scene") != "config/level4_L09_hard_DeviceOperation.yaml":
        raise ValueError("device-operation protocol must use the registered hard L09 scene")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dataset_valid(root: Path, *, newer_than_ns: int | None = None) -> tuple[bool, list[str]]:
    try:
        import h5py
    except ImportError:
        return False, []
    files = sorted(root.rglob("*.hdf5")) + sorted(root.rglob("*.h5"))
    required = {"actions", "actions_ee", "action_gripper"}
    for path in files:
        if newer_than_ns is not None and path.stat().st_mtime_ns < newer_than_ns:
            continue
        try:
            with h5py.File(path, "r") as stream:
                names = set()
                stream.visit(names.add)
                leaves = {name.rsplit("/", 1)[-1] for name in names}
                cameras = {name for name in leaves if name in {"camera_1_rgb", "camera_3_rgb"}}
                # A schema-valid HDF5 is not automatically a physically
                # successful demonstration.  The device controller writes
                # this attestation only after all placement/retrieval
                # predicates have passed.
                physical_success = False
                if "task_properties" in stream:
                    raw = stream["task_properties"][()]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    try:
                        properties = json.loads(str(raw))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        properties = {}
                    physical_success = bool(
                        properties.get("physical_success") is True
                        and properties.get("placement_pass") is True
                        and properties.get("retrieval_return_pass") is True
                    )
                if (required <= leaves and cameras == {"camera_1_rgb", "camera_3_rgb"}
                        and physical_success):
                    return True, sorted(leaves)
        except OSError:
            continue
    return False, []


def run(config: str, output: str, seed: int, headless: bool, save_video: bool,
        dataset_output: str | None, max_steps: int | None = None) -> dict:
    payload = _load(config)
    evaluation = payload["evaluation"]
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    requested_data_root = (Path(dataset_output).resolve()
                           if dataset_output else out / "dataset")
    data_root = requested_data_root
    if evaluation["mode"] == "reference" and requested_data_root.exists():
        existing_episodes = (list(requested_data_root.rglob("*.h5"))
                             + list(requested_data_root.rglob("*.hdf5")))
        if existing_episodes:
            invocation = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
            data_root = requested_data_root / invocation
            print(
                f"Existing dataset episodes found; writing this invocation to {data_root}",
                flush=True,
            )
    worker_started_ns = time.time_ns()
    command = [sys.executable, "main.py", "--config-name", "level4_L09_hard_DeviceOperation",
               "--seed", str(seed)]
    if headless:
        command.append("--headless")
    if not save_video:
        command.append("--no-video")
    if evaluation["mode"] == "reference":
        command += ["--mode", "collect", "--max-success-episodes", "1",
                    "--max-consecutive-failures", "1", "--run-dir", str(data_root)]
    else:
        options = payload.get('labbench_runtime', {})
        interface = evaluation.get('policy_interface', {})
        command = [sys.executable, '-m', 'eai4s_eval.cli.run_benchmark_eval',
                   '--task-spec', 'eai4s_eval/configs/tasks/level4_L09_hard_device_operation.yaml',
                   '--output-root', str(out/'eval_runs'), '--run-id', 'episode',
                   '--num-episodes', '1', '--seeds', str(seed),
                   '--vla-host', str(interface.get('host', '127.0.0.1')),
                   '--vla-port', str(interface.get('port', 8000)),
                   '--max-steps', str(max_steps or options.get('max_action_steps') or evaluation['max_steps'])]
        if options.get('policy_profile'):
            command += ['--policy-profile', options['policy_profile']]
        if options.get('backend'):
            command += ['--policy-backend', options['backend']]
        if headless:
            command.append('--headless')
        if save_video:
            command.append('--save-video')
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=False)
    dataset_ok, streams = (_dataset_valid(data_root, newer_than_ns=worker_started_ns)
                           if evaluation["mode"] == "reference" else (False, []))
    native_summaries = sorted((out / "eval_runs").rglob("summary.json"))
    native_summary = (json.loads(native_summaries[-1].read_text(encoding="utf-8"))
                      if native_summaries else {})
    evaluation_valid = bool(native_summary.get("num_valid_episodes", native_summary.get('num_valid_trials', 0)))
    native_ok = bool(completed.returncode == 0 and evaluation_valid and
                     native_summary.get("success_rate", 0.0) > 0.0)
    valid_success = dataset_ok if evaluation["mode"] == "reference" else native_ok
    scores = {
        "task_version": TASK_VERSION,
        "valid_success": bool(valid_success),
        "evaluation_valid": bool(completed.returncode == 0 and
                                 (dataset_ok if evaluation["mode"] == "reference" else evaluation_valid)),
    }
    # Unknown diagnostics remain absent. Overall success is not evidence for
    # five independent measurements, especially on policy failure paths.
    native_metrics = sorted((out/'eval_runs').rglob('episode_metrics.jsonl'))
    if evaluation['mode'] == 'native' and native_metrics:
        for line in native_metrics[-1].read_text().splitlines():
            item = json.loads(line)
            if item.get('seed') == seed:
                for key in ('sequence_pass', 'terminal_pass', 'safety_valid'):
                    if isinstance(item.get(key), bool):
                        scores[key] = item[key]
                for key, value in (item.get('quality_metrics') or {}).items():
                    if isinstance(value, (bool, int, float)):
                        scores[key] = value
    manifest = {
        "task_version": TASK_VERSION, "difficulty": "hard", "seed": seed,
        "protocol_id": "device_operation_hard",
        "runtime_status": "finished" if completed.returncode == 0 else "error",
        "worker_exit_code": completed.returncode,
        "ranking_eligible": bool(evaluation["mode"] == "native" and evaluation_valid),
        "policy_interface": ("diagnostic_rule_expert" if evaluation["mode"] == "reference"
                             else evaluation.get("policy_interface")),
        "dataset_output": str(data_root) if evaluation["mode"] == "reference" else None,
        "dataset_streams": streams,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": evaluation,
    }
    _write_json(out / "run.json", manifest)
    _write_json(out / "scores.json", scores)
    if completed.returncode:
        raise RuntimeError(f"device-operation worker exited {completed.returncode}; see {out / 'run.json'}")
    if evaluation["mode"] == "reference" and not dataset_ok:
        raise RuntimeError(
            f"device-operation worker did not produce a new physically successful HDF5; "
            f"see {out / 'run.json'}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard device-operation release protocol")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--config", required=True)
    launch = sub.add_parser("run")
    launch.add_argument("--config", required=True)
    launch.add_argument("--output", required=True)
    launch.add_argument("--seed", type=int, default=101)
    launch.add_argument("--headless", action="store_true")
    launch.add_argument("--save-video", action="store_true")
    launch.add_argument("--dataset-output")
    launch.add_argument("--max-steps", type=int)
    score = sub.add_parser("score")
    score.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(_load(args.config), indent=2))
    elif args.command == "score":
        print((Path(args.run) / "scores.json").read_text(encoding="utf-8"))
    else:
        print(json.dumps(run(args.config, args.output, args.seed, args.headless,
                             args.save_video, args.dataset_output, args.max_steps), indent=2))


if __name__ == "__main__":
    main()
