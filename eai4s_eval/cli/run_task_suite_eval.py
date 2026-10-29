from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from eai4s_eval.cli.run_pick_lift_eval import build_parser as build_single_task_parser
from eai4s_eval.eval.task_suite import (
    DEFAULT_SUITE_ID,
    TaskSuiteAggregator,
    load_task_catalog,
    load_task_suite,
)


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_")
    return cleaned or "eai4s_model"


def build_parser() -> argparse.ArgumentParser:
    parser = build_single_task_parser()
    parser.description = "Run an EAI4S-bench task suite and aggregate per-task scores."
    parser.add_argument("--task_suite", type=str, default=DEFAULT_SUITE_ID,
                        help="Task suite id or path, e.g. eai4s_chem_basic_v0_1.")
    parser.add_argument("--eval_python", type=str, default=sys.executable,
                        help="Python executable used to launch single-task Isaac runs.")
    parser.add_argument("--suite_only", action="store_true",
                        help="Only write suite placeholder/aggregation outputs; do not launch Isaac task runs.")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue remaining implemented tasks if one task subprocess fails.")
    return parser


def _append_value(cmd: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    cmd.extend([flag, str(value)])


def _single_task_command(args: argparse.Namespace, unknown: list[str], *, suite_run_id: str, suite_id: str, task: dict[str, Any]) -> list[str]:
    task_id = str(task["task_id"])
    run_id = f"{suite_run_id}_{task_id}"
    cmd = [
        str(args.eval_python),
        "-m",
        "eai4s_eval.cli.run_pick_lift_eval",
        "--run_id",
        run_id,
        "--eval_group_id",
        suite_run_id,
        "--suite_id",
        suite_id,
        "--run_type",
        "task_run",
        "--task_spec",
        str(task["task_spec"]),
        "--output_root",
        str(args.output_root),
        "--model_name",
        str(args.model_name),
        "--checkpoint",
        str(args.checkpoint),
        "--robot",
        str(args.robot),
        "--num_episodes",
        str(args.num_episodes),
        "--vla_host",
        str(args.vla_host),
        "--vla_port",
        str(args.vla_port),
        "--policy_api",
        str(args.policy_api),
        "--split",
        str(args.split),
    ]

    instruction = task.get("instruction")
    if isinstance(instruction, dict):
        canonical = instruction.get("canonical")
    else:
        canonical = None
    _append_value(cmd, "--instruction", canonical or args.instruction)
    _append_value(cmd, "--target_object", task.get("target_object") or args.target_object)
    _append_value(cmd, "--asset_root", args.asset_root)
    _append_value(cmd, "--num_envs", args.num_envs)
    _append_value(cmd, "--policy_timeout_sec", args.policy_timeout_sec)
    _append_value(cmd, "--max_steps", task.get("max_steps") or args.max_steps)
    _append_value(cmd, "--action_horizon", args.action_horizon)
    _append_value(cmd, "--reaction_interpolation_steps", args.reaction_interpolation_steps)
    _append_value(cmd, "--sim_steps_per_action", args.sim_steps_per_action)
    _append_value(cmd, "--initial_hold_steps", args.initial_hold_steps)
    _append_value(cmd, "--stabilization_max_steps", args.stabilization_max_steps)
    _append_value(cmd, "--stabilization_hold_steps", args.stabilization_hold_steps)
    _append_value(cmd, "--stabilization_object_lin_vel_threshold", args.stabilization_object_lin_vel_threshold)
    _append_value(cmd, "--stabilization_object_ang_vel_threshold", args.stabilization_object_ang_vel_threshold)
    _append_value(cmd, "--stabilization_joint_vel_threshold", args.stabilization_joint_vel_threshold)
    _append_value(cmd, "--control_hz", task.get("control_hz") or args.control_hz)
    _append_value(cmd, "--predicate_log_interval", args.predicate_log_interval)
    _append_value(cmd, "--policy_log_interval", args.policy_log_interval)
    _append_value(cmd, "--max_delta_xyz", args.max_delta_xyz)
    _append_value(cmd, "--seeds", args.seeds)
    _append_value(cmd, "--seed_manifest", args.seed_manifest)
    _append_value(cmd, "--checkpoint_hash", args.checkpoint_hash)
    _append_value(cmd, "--dataset_version", args.dataset_version)
    _append_value(cmd, "--shutdown_timeout_sec", args.shutdown_timeout_sec)
    _append_value(cmd, "--success_hold_steps", args.success_hold_steps)
    _append_value(cmd, "--success_height_threshold", args.success_height_threshold)
    _append_value(cmd, "--success_distance_threshold", args.success_distance_threshold)
    _append_value(cmd, "--success_grip_threshold", args.success_grip_threshold)
    if args.all_splits:
        cmd.append("--all-splits")
    if args.save_video:
        cmd.append("--save_video")
    if args.use_policy_orientation:
        cmd.append("--use_policy_orientation")
    if args.skip_isaac_close:
        cmd.append("--skip_isaac_close")
    if args.success_require_gripper_closed is True:
        cmd.append("--success_require_gripper_closed")
    if args.success_require_gripper_closed is False:
        cmd.append("--success_ignore_gripper_closed")
    if args.success_require_stable is True:
        cmd.append("--success_require_stable")
    if args.success_require_stable is False:
        cmd.append("--success_ignore_stable")
    cmd.extend(unknown)
    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    eval_time_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not args.checkpoint:
        args.checkpoint = eval_time_id
    suite = load_task_suite(args.task_suite)
    catalog = load_task_catalog(suite)
    suite_id = str(catalog.get("suite_id") or DEFAULT_SUITE_ID)
    suite_run_id = args.run_id or f"{eval_time_id}_{_safe_id(str(args.model_name))}_{suite_id}"
    suite_output_dir = Path(args.output_root) / suite_run_id

    errors: list[tuple[str, int]] = []
    for task in catalog.get("tasks", []):
        status = str(task.get("status") or "planned")
        task_id = str(task.get("task_id"))
        if status != "implemented":
            print(f"Skipping {task_id}: status={status}; it will appear as not_implemented in the suite score table.", flush=True)
            continue
        if args.suite_only:
            print(f"Suite-only mode: not launching implemented task {task_id}.", flush=True)
            continue
        cmd = _single_task_command(args, unknown, suite_run_id=suite_run_id, suite_id=suite_id, task=task)
        print(f"\n=== Running suite task {task_id} ===", flush=True)
        print(" ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[2]))
        if proc.returncode != 0:
            errors.append((task_id, proc.returncode))
            if not args.continue_on_error:
                break

    result = TaskSuiteAggregator(suite_output_dir).write_all(
        task_catalog=catalog,
        outputs_root=args.output_root,
        eval_group_id=suite_run_id,
        model_name=str(args.model_name),
        checkpoint=str(args.checkpoint),
        robot=str(args.robot),
    )
    print("", flush=True)
    for line in result.get("summary_lines", []):
        print(line, flush=True)
    print(f"Summary log: {result.get('summary_log')}", flush=True)
    print(f"Task suite evaluation complete: {suite_output_dir}", flush=True)
    if errors:
        print(f"Task subprocess failures: {errors}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
