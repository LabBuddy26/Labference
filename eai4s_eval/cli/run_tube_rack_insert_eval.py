from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Sequence

from eai4s_eval.cli.run_pick_lift_eval import build_parser as build_base_parser
from eai4s_eval.robots import get_robot_adapter
from eai4s_eval.robots.base import RobotAdapterNotConfigured


def _task_spec_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "configs" / "tasks" / "T03_tube_insert_rack_draft.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()
    parser.description = "Run EAI4S-bench T03 tube-rack insertion evaluation."
    parser.set_defaults(
        task_spec=_task_spec_path(),
        instruction="place the glass tube in the rack",
        target_object="tube",
        max_steps=300,
        reaction_interpolation_steps=1,
        sim_steps_per_action=6,
    )
    return parser


def _close_simulation_app(simulation_app: object, timeout_sec: float, exit_code: int, *, skip_close: bool = False) -> None:
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
        _close_simulation_app(
            simulation_app,
            float(args.shutdown_timeout_sec),
            exit_code,
            skip_close=bool(args.skip_isaac_close) or exit_code != 0,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
