from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Sequence

from eai4s_eval.robots import available_robots, get_robot_adapter
from eai4s_eval.robots.base import RobotAdapterNotConfigured


def default_config_path(*parts: str) -> str:
    return str(Path(__file__).resolve().parents[1] / "configs" / Path(*parts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EAI4S-bench Franka/UR5-extensible pick/lift evaluation.")
    parser.add_argument("--robot", type=str, default="franka_panda", choices=available_robots())
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--vla_host", type=str, default="127.0.0.1", help="VLA model inference host")
    parser.add_argument("--vla_port", type=int, default=8188, help="VLA model inference port")
    parser.add_argument("--policy_api", type=str, default="auto", choices=["auto", "act", "predict_action"])
    parser.add_argument("--policy_timeout_sec", type=float, default=30.0)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--reaction_interpolation_steps", type=int, default=1)
    parser.add_argument("--sim_steps_per_action", type=int, default=6)
    parser.add_argument("--use_policy_orientation", action="store_true")
    parser.add_argument("--initial_hold_steps", type=int, default=100)
    parser.add_argument("--stabilization_max_steps", type=int, default=200)
    parser.add_argument("--stabilization_hold_steps", type=int, default=10)
    parser.add_argument("--stabilization_object_lin_vel_threshold", type=float, default=0.02)
    parser.add_argument("--stabilization_object_ang_vel_threshold", type=float, default=0.20)
    parser.add_argument("--stabilization_joint_vel_threshold", type=float, default=0.02)
    parser.add_argument("--control_hz", type=int, default=10)
    parser.add_argument("--success_hold_steps", type=int, default=None)
    parser.add_argument("--success_height_threshold", type=float, default=None)
    parser.add_argument("--success_distance_threshold", type=float, default=None)
    parser.add_argument("--success_grip_threshold", type=float, default=None)
    parser.add_argument("--success_require_gripper_closed", dest="success_require_gripper_closed", action="store_true", default=None)
    parser.add_argument("--success_ignore_gripper_closed", dest="success_require_gripper_closed", action="store_false")
    parser.add_argument("--success_require_stable", dest="success_require_stable", action="store_true", default=None)
    parser.add_argument("--success_ignore_stable", dest="success_require_stable", action="store_false")
    parser.add_argument("--predicate_log_interval", type=int, default=25)
    parser.add_argument(
        "--policy_log_interval",
        type=int,
        default=10,
        help="Print policy batch diagnostics every N control steps. Use 0 to disable.",
    )
    parser.add_argument("--max_delta_xyz", type=float, default=0.03)
    parser.add_argument("--instruction", type=str, default="pick glass bottle")
    parser.add_argument("--target_object", type=str, default="vial")
    parser.add_argument("--asset_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="outputs/eval_runs")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--eval_group_id", type=str, default=None,
                        help="Shared evaluation group id. Suite runs use this to group per-task outputs.")
    parser.add_argument("--suite_id", type=str, default=None,
                        help="Task suite id when this single-task run is launched by a suite evaluator.")
    parser.add_argument("--run_type", type=str, default="single_task",
                        choices=["single_task", "task_run"],
                        help="Output role for downstream aggregation.")
    parser.add_argument("--model_name", type=str, default="eai4s_model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint/version tag. Defaults to the evaluation timestamp, e.g. eval_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument("--checkpoint_hash", type=str, default=None)
    parser.add_argument("--dataset_version", type=str, default=None)
    parser.add_argument("--split", type=str, default="sim_dev")
    parser.add_argument("--all-splits", action="store_true",
                        help="Run evaluation sequentially across sim_dev, sim_clean, and sim_randomized")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list, e.g. 100001,100002")
    parser.add_argument("--seed_manifest", type=str, default=None,
                        help="Path to seed manifest JSON. Auto-resolved from --split when not provided.")
    parser.add_argument("--task_spec", type=str, default=default_config_path("tasks", "T02_franka_pick_lift_proxy.yaml"))
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument(
        "--shutdown_timeout_sec",
        type=float,
        default=30.0,
        help="Seconds to wait for Isaac Sim to close before force-ending the evaluator process. Use <=0 to disable.",
    )
    parser.add_argument(
        "--skip_isaac_close",
        action="store_true",
        help="Skip SimulationApp.close() and exit immediately after outputs are written. Useful on headless servers where Isaac/Kit shutdown hangs.",
    )
    return parser


def close_simulation_app(simulation_app: object, timeout_sec: float, exit_code: int, *, skip_close: bool = False) -> None:
    if skip_close:
        print(
            "Skipping Isaac Sim graceful shutdown; evaluation outputs have already been written.",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

    if timeout_sec <= 0:
        simulation_app.close()
        return

    state: dict[str, object] = {"done": False, "error": None}

    def _close() -> None:
        try:
            simulation_app.close()
        except Exception as exc:  # pragma: no cover - depends on Isaac runtime.
            state["error"] = exc
        finally:
            state["done"] = True

    thread = threading.Thread(target=_close, name="isaac-sim-close", daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if state["error"] is not None:
        raise RuntimeError("Isaac Sim failed during shutdown") from state["error"]  # type: ignore[arg-type]
    if state["done"]:
        return

    print(
        f"Isaac Sim shutdown did not finish within {timeout_sec:.1f}s; "
        "forcing evaluator process exit.",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    # The VLA policy consumes camera observations, so this evaluator should
    # enable Isaac Lab camera rendering unless the entry point is customized.
    parser.set_defaults(enable_cameras=True)
    args = parser.parse_args(argv)
    adapter = get_robot_adapter(args.robot)
    if not adapter.is_configured:
        try:
            adapter.build_robot_cfg("{ENV_REGEX_NS}/Robot")
        except RobotAdapterNotConfigured as exc:
            raise SystemExit(str(exc)) from exc

    try:
        import json_numpy  # type: ignore

        json_numpy.patch()
    except Exception:
        pass

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    exit_code = 0
    try:
        from eai4s_eval.eval.runner import TaskEvalRunner

        runner = TaskEvalRunner(args=args, simulation_app=simulation_app, robot_adapter=adapter)
        if args.all_splits:
            runner.run_all_splits()
        else:
            runner.run()
    except Exception:
        exit_code = 1
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        close_simulation_app(
            simulation_app,
            float(args.shutdown_timeout_sec),
            exit_code,
            skip_close=bool(args.skip_isaac_close) or exit_code != 0,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
