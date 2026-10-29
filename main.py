import os
import sys
import time
import argparse
import traceback

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='LabSim Simulation Environment')
    parser.add_argument('--backend', type=str, default='numpy', 
                       choices=['numpy', 'gpu'], 
                       help='Backend choice: numpy (CPU) or gpu')
    parser.add_argument('--headless', action='store_true', 
                       help='Run in headless mode (default is with GUI)')
    parser.add_argument('--no-video', action='store_true', 
                       help='Disable video display and saving')
    parser.add_argument('--config-name', type=str, default='level3_Heat_Liquid',
                       help='Configuration file name (without .yaml extension)')
    parser.add_argument('--config-dir', type=str, default='config',
                       help='Configuration directory path (default: config)')
    parser.add_argument('--mode', choices=['collect', 'eval', 'infer'], default=None,
                       help='Override the YAML mode; infer is a deprecated alias for eval')
    parser.add_argument('--execution-horizon', '--execution_horizon', type=int, default=None,
                       help='Execute this many direct policy waypoints per request (default: full chunk)')
    parser.add_argument('--max-success-episodes', type=int, default=None,
                       help='Collect this many successful episodes (default: cfg.max_episodes)')
    parser.add_argument('--sample-hz', type=int, choices=[10, 60], default=None,
                       help='Override collector.sample_hz for collection (default: YAML, normally 10)')
    parser.add_argument('--require-consecutive-successes', type=int, default=0,
                       help='Require a final streak of this many successful release episodes')
    parser.add_argument('--max-consecutive-failures', type=int, default=20,
                       help='Stop the task after this many consecutive failed episodes (collect mode)')
    parser.add_argument('--task-timeout-minutes', type=float, default=None,
                       help='Wall-clock timeout for the whole task run in minutes (collect mode)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Seed the task before constructing the simulation runtime')
    parser.add_argument('--run-dir', type=str, default=None,
                       help='Override multi_run.run_dir')
    parser.add_argument('--eval-output-root', type=str, default=None,
                       help='Override evaluation.output_root')
    parser.add_argument('--eval-num-episodes', type=int, default=None,
                       help='Override evaluation.num_episodes')
    parser.add_argument('--policy-url', type=str, default=None,
                       help='HTTP native policy endpoint for versioned release evaluation')
    parser.add_argument('--policy-action-space', choices=['ee_pos', 'joint'], default=None,
                       help='Explicit native release model action convention (default: branch EE pose)')
    parser.add_argument('--eval-max-action-steps', type=int, default=None,
                       help='Override the special protocol model-action budget (default: 500)')
    parser.add_argument('--vla-host', default=None)
    parser.add_argument('--vla-port', type=int, default=None)
    return parser.parse_args()

# Get command line arguments
args = parse_args()

from utils.special_protocol_driver import dispatch as dispatch_special_protocol
special_exit_code = dispatch_special_protocol(args)
if special_exit_code is not None:
    sys.exit(special_exit_code)

from isaacsim import SimulationApp

# Set up simulation app based on arguments
simulation_config = {
    "headless": args.headless,
    "extra_args": ["--/rtx/raytracing/fractionalCutoutOpacity=true"],
}

simulation_app = SimulationApp(simulation_config)

import hydra
from omegaconf import OmegaConf, open_dict
import cv2
import numpy as np

import omni
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
import omni.usd
from isaacsim.core.utils import extensions
from isaacsim.core.utils.types import ArticulationAction

extensions.enable_extension("omni.physx.bundle")
extensions.enable_extension("omni.usdphysics.ui")

from evaluation.runtime import BenchmarkRuntime
from utils.collection_startup import (
    CollectionStartupConfig,
    CollectionStartupGate,
    task_startup_readiness,
)
from utils.collect_driver import format_collect_result


_active_task_controller = None
_simulation_app_closed = False


def _close_simulation_app() -> None:
    """Close the shared SimulationApp at most once across all exit paths."""
    global _simulation_app_closed
    if not _simulation_app_closed:
        simulation_app.close()
        _simulation_app_closed = True


def _startup_object_positions(task):
    """Return every known task object position for the startup stability check."""
    paths = set()
    for item in getattr(task, "obj_configs", []) or []:
        path = item.get("path") if isinstance(item, dict) else None
        if path:
            paths.add(str(path))
    for attr in (
        "current_obj_path",
        "source_obj",
        "target_obj",
        "target_path",
        "sub_object",
        "glass_rod",
        "target_beaker",
    ):
        path = getattr(task, attr, None)
        if path:
            paths.add(str(path))

    positions = []
    for path in sorted(paths):
        position = task.object_utils.get_geometry_center(object_path=path)
        if position is None:
            position = task.object_utils.get_object_xform_position(object_path=path)
        if position is not None:
            positions.append(position)
    return positions


def _build_collection_startup_gate(cfg):
    if str(getattr(cfg, "mode", "")) != "collect" or str(cfg.robot.type) != "franka":
        return None
    startup_cfg = cfg.get("collection_startup")
    raw = OmegaConf.to_container(startup_cfg, resolve=True) if startup_cfg is not None else {}
    return CollectionStartupGate(CollectionStartupConfig.from_mapping(raw))

def main():
    global _active_task_controller
    hydra.initialize(config_path=args.config_dir, job_name=args.config_name)
    cfg = hydra.compose(config_name=args.config_name)

    if args.seed is not None:
        from evaluation.config import seed_everything
        seed_everything(int(args.seed))
    if args.run_dir is not None:
        OmegaConf.update(cfg, "multi_run.run_dir", os.path.abspath(args.run_dir), merge=False)
    if args.eval_output_root is not None:
        OmegaConf.update(cfg, "evaluation.output_root", os.path.abspath(args.eval_output_root), merge=False)
    if args.eval_num_episodes is not None:
        if args.eval_num_episodes <= 0:
            raise ValueError("--eval-num-episodes must be positive")
        OmegaConf.update(cfg, "evaluation.num_episodes", int(args.eval_num_episodes), merge=False)

    if args.mode is not None:
        with open_dict(cfg):
            cfg.mode = args.mode
    if str(getattr(cfg, "mode", "collect")) == "infer":
        print("[deprecated] mode 'infer' now runs the complete evaluation pipeline; use mode 'eval'.")
        with open_dict(cfg):
            cfg.mode = "eval"
    if args.execution_horizon is not None:
        if args.execution_horizon <= 0:
            raise ValueError("--execution-horizon must be positive")
        OmegaConf.update(
            cfg,
            "evaluation.policy.execution_horizon",
            int(args.execution_horizon),
            merge=False,
            force_add=True,
        )

    mode = str(getattr(cfg, "mode", "collect"))
    if mode == "eval":
        # Evaluation owns its run directory and writes the resolved task YAML
        # into outputs/eval_runs.  It shares this process' SimulationApp and
        # the normal BenchmarkRuntime/controllers with collection.
        from eai4s_eval.configured_runner import run_configured_eval

        try:
            run_configured_eval(
                cfg=cfg,
                simulation_app=simulation_app,
                backend=args.backend,
                headless=args.headless,
                no_video=args.no_video,
            )
        finally:
            _close_simulation_app()
            cv2.destroyAllWindows()
        return
    if mode != "collect":
        raise ValueError(f"Unsupported mode {mode!r}; expected 'collect' or 'eval'.")

    if args.sample_hz is not None:
        OmegaConf.update(
            cfg,
            "collector.sample_hz",
            int(args.sample_hz),
            merge=False,
            force_add=True,
        )
        print(f"Collecting synchronized data at {cfg.collector.sample_hz} Hz.", flush=True)

    # Override cfg.max_episodes BEFORE constructing the runtime: the collector
    # caps its own writes at cfg.max_episodes (silently dropping beyond it) and
    # BaseTask derives episodes_per_obj from it, so the main-loop success check
    # alone would never reach the requested target.
    if args.max_success_episodes is not None:
        if args.max_success_episodes <= 0:
            raise ValueError("--max-success-episodes must be positive")
        with open_dict(cfg):
            cfg.max_episodes = int(args.max_success_episodes)
        print(f"Collecting up to {cfg.max_episodes} successful episodes.", flush=True)

    os.makedirs(cfg.multi_run.run_dir, exist_ok=True)
    OmegaConf.save(cfg, cfg.multi_run.run_dir + "/config.yaml")

    # Override configuration based on command line arguments
    if args.no_video:
        save_video = False
        # show_video = False
    else:
        save_video = True
        # show_video = True

    runtime = BenchmarkRuntime(
        cfg=cfg,
        simulation_app=simulation_app,
        backend=args.backend,
        render=True,
    )
    world = runtime.world
    robot = runtime.robot
    task = runtime.task
    task_controller = runtime.controller
    # Keep the controller reachable from the process-level finally block. This
    # guarantees that an unexpected collection exception still drains the
    # non-daemon HDF5 writer before Isaac Sim exits.
    _active_task_controller = task_controller
    startup_gate = _build_collection_startup_gate(cfg)
    
    video_writer = None
    collect_started_at = time.monotonic()
    consecutive_failures = 0
    counted_episodes = 0

    def _finish_collect(code: int, status: str) -> None:
        """Flush collection, emit the result marker, then close Isaac.

        The marker must be written before ``SimulationApp.close()``.  Isaac
        can tear down the Python/stdio runtime as part of close(), which used
        to make successful runs look like parser errors to collect_driver.
        """
        cleanup_error = None
        try:
            task_controller.close()
        except Exception as exc:
            cleanup_error = exc
            print(f"Collection writer cleanup failed: {exc}", file=sys.stderr, flush=True)
        finally:
            # HDF5 has been flushed by controller.close() (or attempted) at
            # this point, so the summary is safe to consume immediately.
            marker_status = status if cleanup_error is None else "error"
            print(
                format_collect_result(
                    config=args.config_name,
                    success_count=task_controller.episode_num(),
                    total_episodes=counted_episodes,
                    consecutive_failures=consecutive_failures,
                    status=marker_status,
                    duration_seconds=int(time.monotonic() - collect_started_at),
                    run_dir=str(cfg.multi_run.run_dir),
                ),
                flush=True,
            )
            # Defer SimulationApp teardown to the process-level ``finally``.
            # Closing Kit here and then unwinding through ``finally`` used to
            # race Carbonite worker shutdown and trigger a native mutex
            # ownership abort (especially on the short R12 runs).
        if cleanup_error is not None:
            raise cleanup_error
        sys.exit(code)

    task.reset()
    if startup_gate is not None:
        startup_gate.reset()

    simulation_steps = 0
    while simulation_app.is_running():
        simulation_steps += 1
        world.step(render=True)

        if world.is_stopped():
            task_controller.reset_needed = True

        if world.is_playing():
            if (
                args.task_timeout_minutes is not None
                and time.monotonic() - collect_started_at > args.task_timeout_minutes * 60.0
            ):
                print(
                    f"Task wall-clock timeout exceeded ({args.task_timeout_minutes} minutes); "
                    "switching task.",
                    flush=True,
                )
                _finish_collect(3, "timeout")
            if task_controller.need_reset() or task.need_reset():
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                task._video_sample_origin_step = None

                # Episode outcome is only recorded for episodes that actually
                # completed (controller done or task frame-limit timeout);
                # startup-gate rejections and physics stops leave it None.
                outcome = getattr(task, "episode_outcome", None)
                if outcome is not None:
                    counted_episodes += 1
                    if outcome:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= args.max_consecutive_failures:
                            print(
                                f"Reached {consecutive_failures} consecutive failures; switching task.",
                                flush=True,
                            )
                            _finish_collect(2, "consecutive_failures")

                task_controller.reset()
                if task_controller.episode_num() >= cfg.max_episodes:
                    _finish_collect(0, "completed")
                task.reset()
                if startup_gate is not None:
                    startup_gate.reset()

                continue

            if startup_gate is not None and not startup_gate.is_ready:
                if startup_gate.config.drive_robot:
                    robot.get_articulation_controller().apply_action(
                        ArticulationAction(joint_positions=startup_gate.ready_joint_positions)
                    )
                progress = startup_gate.observe(
                    joint_velocities=robot.get_joint_velocities(),
                    object_positions=_startup_object_positions(task),
                )
                if progress.ready:
                    task_ready, task_detail = task_startup_readiness(task)
                    if not task_ready:
                        print(
                            "Collection startup rejected by task geometry: "
                            f"{task_detail}; resetting the episode.",
                            flush=True,
                        )
                        task_controller.reset_needed = True
                    else:
                        task.frame_idx = 0
                        # The startup gate drives joints directly. Reset RMPFlow
                        # once so its internal motion state starts from the final
                        # stabilized articulation state.
                        task_controller.rmp_controller.reset()
                        print(
                            "Collection startup ready after "
                            f"{progress.steps} physics steps "
                            f"({progress.stable_frames} stable frames)."
                        )
                elif progress.timed_out:
                    print(
                        "Collection startup timed out after "
                        f"{progress.steps} physics steps; resetting the episode."
                    )
                    task_controller.reset_needed = True
                continue

            state = task.step()
            if state is None:
                continue
            state["sim_step"] = int(task.frame_idx)
            
            action, done, is_success = task_controller.step(state)
            if action is not None:
                robot.get_articulation_controller().apply_action(action)
            if done:
                task_controller.print_failure_reason()
                task.episode_outcome = bool(is_success)
                task.on_task_complete(is_success)
                continue
            
            if save_video:
                sample_hz = int(cfg.collector.get("sample_hz", 10))
                sample_stride = 60 // sample_hz
                if getattr(task, "_video_sample_origin_step", None) is None:
                    task._video_sample_origin_step = int(task.frame_idx)
                should_write_video = (
                    int(task.frame_idx) - int(task._video_sample_origin_step)
                ) % sample_stride == 0
                camera_images = []
                if should_write_video:
                    for _, image_data in state['camera_display'].items():
                        display_img = cv2.cvtColor(image_data.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                        camera_images.append(display_img)

                if should_write_video and camera_images:
                    combined_img = np.hstack(camera_images)
                    total_width = 0
                    for idx, img in enumerate(camera_images):
                        label = f"Camera {idx+1} ({cfg.cameras[idx].image_type})"
                        cv2.putText(combined_img, label, (total_width + 2, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                        total_width += img.shape[1]
                    # if show_video:
                    #     cv2.imshow('Camera Views', combined_img)
                    #     cv2.waitKey(1)
                    if save_video:
                        output_dir = os.path.join(cfg.multi_run.run_dir, "video")
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"episode_{task_controller._episode_num}.mp4")
                        if video_writer is None:
                            height, width = combined_img.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(
                                output_path, fourcc, float(sample_hz), (width, height)
                            )
                        video_writer.write(combined_img)

    # Reaching this point during collection is never a successful run: normal
    # completion leaves through _finish_collect().  Preserve a machine-readable
    # result instead of silently returning 0 and making the release wrapper
    # report only that no HDF5 happened to appear.
    try:
        timeline = omni.timeline.get_timeline_interface()
        timeline_state = {
            "playing": bool(timeline.is_playing()),
            "stopped": bool(timeline.is_stopped()),
        }
    except Exception as exc:
        timeline_state = {"error": repr(exc)}
    print(
        "Simulation App stopped before collection completed: "
        f"loop_steps={simulation_steps}, timeline={timeline_state}",
        file=sys.stderr,
        flush=True,
    )
    _finish_collect(4, "simulation_app_stopped")


if __name__ == "__main__":
    try:
        main()
    except Exception as exit_error:
        # A library-level ``exit()`` previously made failed collection look
        # like a clean worker return.  Keep the originating Python stack in
        # the Kit log so any remaining premature exit is actionable.
        print(
            "Unexpected control-flow exit while collecting: "
            f"type={type(exit_error).__name__}, value={exit_error!r}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        primary_error = sys.exc_info()[1]
        controller_cleanup_error = None
        if _active_task_controller is not None:
            try:
                _active_task_controller.close()
            except Exception as cleanup_error:
                if primary_error is None:
                    controller_cleanup_error = cleanup_error
                else:
                    print(
                        f"Controller cleanup also failed: {cleanup_error}",
                        file=sys.stderr,
                        flush=True,
                    )
        try:
            _close_simulation_app()
        finally:
            cv2.destroyAllWindows()
        if controller_cleanup_error is not None:
            raise controller_cleanup_error
