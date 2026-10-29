from __future__ import annotations

"""Discriminating probe: run a single policy episode from a custom initial joint pose.

This is the *step-1 discriminator* for diagnosing why the gr00t pick policy fails.
The hypothesis is that the model does not emit a usable *approach* reach from the
benchmark's home pose (which sits on the saturated edge of the training state
distribution).  If we instead start the arm at a **mid-reach** pose along the
training trajectory (j3 ~= -1.52, j5 ~= 1.23), a healthy pick policy should
*continue* reaching down to the beaker, close the gripper, and lift it.  A model
that still hovers / drifts from mid-reach is broken beyond the start-state edge
case (e.g. it never learned a reliable grasp, or the eval visual domain is off).

It reuses the exact benchmark pipeline (BenchmarkRuntime + EAI4SBenchmarkBackend
+ OpenPIJointPolicyAdapter) so the scene, cameras, state convention and action
parsing are identical to a real eval run.  The only difference is that, right
after ``reset_episode``, the robot joints are overwritten with the requested pose
and the physics is settled a few frames before the policy loop starts.

Usage
-----
.. code-block:: bash

    python -m eai4s_eval.cli.run_discriminate_initial_pose \
        --benchmark-root ../EAI4S_benchmark \
        --task-spec eai4s_eval/configs/tasks/B01_level1_pick.yaml \
        --policy-profile eai4s_eval/configs/policies/gr00t_franka_joint8_v1.yaml \
        --policy-backend http --vla-host 127.0.0.1 --vla-port 8765 \
        --split sim_dev --seeds 210000 \
        --pose mid_reach --max-steps 1200 --headless

Pose presets (7 arm + 2 finger joints, taken from the training episode):
    home      -> [ 0.0044, -0.2231,  0.0,   -0.2328, -0.0005,  0.3028,  0.2973,  0.0,   0.0  ]
    mid_reach -> [ 0.0111, -0.2729, -0.0099, -1.5238,  0.0112,  1.2281,  0.4751,  0.04,  0.04 ]
    at_grasp  -> [-0.1113, -0.0629, -0.0764, -2.2764,  2.2355,  2.2711,  1.4487,  0.04,  0.04 ]

Or pass an explicit 9-value pose with ``--initial-joint-pos "a,b,...,i"``.
"""

import argparse
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from eai4s_eval.cli.run_benchmark_eval import build_parser, preflight
from eai4s_eval.eval.recorder import EpisodeRecorder
from eai4s_eval.eval.success import TaskSuccessEvaluator
from eai4s_eval.eval.task_spec import success_spec_from_task_spec
from eai4s_eval.utils.io import ensure_dir

# Presets come from training episode_0000 agent_pose (7 arm + 2 finger joints).
POSE_PRESETS: dict[str, list[float]] = {
    "home": [0.0044, -0.2231, 0.0, -0.2328, -0.0005, 0.3028, 0.2973, 0.0, 0.0],
    "mid_reach": [0.0111, -0.2729, -0.0099, -1.5238, 0.0112, 1.2281, 0.4751, 0.04, 0.04],
    "at_grasp": [-0.1113, -0.0629, -0.0764, -2.2764, 2.2355, 2.2711, 1.4487, 0.04, 0.04],
}


def build_probe_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    # Keep each 10 Hz model waypoint fixed for six 60 Hz simulator frames.
    parser.set_defaults(sim_steps_per_action=6, reaction_interpolation_steps=1)
    parser.add_argument(
        "--pose",
        choices=tuple(POSE_PRESETS),
        default="mid_reach",
        help="Named starting pose preset (training-derived joint config).",
    )
    parser.add_argument(
        "--initial-joint-pos",
        "--initial_joint_pos",
        dest="initial_joint_pos",
        type=str,
        default=None,
        help="Explicit 9-value starting joint pose, comma separated (overrides --pose).",
    )
    parser.add_argument(
        "--probe-steps",
        "--probe_steps",
        dest="probe_steps",
        type=int,
        default=1200,
        help="Max policy steps for the probe episode (default: 1200).",
    )
    return parser


def _resolve_pose(args: argparse.Namespace) -> np.ndarray:
    if args.initial_joint_pos:
        values = [float(item.strip()) for item in args.initial_joint_pos.split(",") if item.strip()]
        if len(values) != 9:
            raise ValueError(f"--initial-joint-pos needs exactly 9 values, got {len(values)}")
        pose = np.asarray(values, dtype=np.float32)
    else:
        pose = np.asarray(POSE_PRESETS[args.pose], dtype=np.float32)
    return pose


def _apply_initial_pose(runtime: Any, pose: np.ndarray, settle_steps: int = 6) -> None:
    """Overwrite the robot joints with *pose* and settle physics briefly.

    Called after ``backend.reset_episode()`` so the sim is at the benchmark home
    state; setting joints here makes the first policy observation see *pose*
    without the controller fighting the change (in infer mode the controller only
    applies actions from the injected trajectory engine).
    """
    robot = runtime.robot
    robot.set_joint_positions(pose)
    robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=np.float32))
    for _ in range(settle_steps):
        runtime.world.step(render=False)
    # Re-read to confirm the pose actually took.
    actual = np.asarray(robot.get_joint_positions(), dtype=np.float32).reshape(-1)
    print(
        f"[probe] initial pose applied  -> {np.round(pose, 4).tolist()}\n"
        f"[probe] sim joints after settle -> {np.round(actual, 4).tolist()}",
        flush=True,
    )


def _run_probe_episode(
    *,
    backend: Any,
    success_spec: Any,
    recorder: EpisodeRecorder,
    episode_id: str,
    episode_index: int,
    instruction: str,
    max_steps: int,
    seed: int,
    save_video: bool = False,
) -> dict[str, Any]:
    evaluator = TaskSuccessEvaluator(success_spec)
    evaluator.reset()
    trajectory: list[dict[str, Any]] = []
    video_frames: list[np.ndarray] = []
    predicate_result = None
    termination_reason = "timeout"
    sim_steps = 0
    started = time.time()

    while sim_steps <= int(max_steps):
        try:
            step = backend.step()
        except Exception as exc:
            termination_reason = f"policy_error: {exc}"
            print(f"[probe] policy error: {exc}", flush=True)
            break
        sim_steps += 1
        if step.state is None or step.snapshot is None:
            continue
        predicate_result = evaluator.update(step.snapshot)
        stats = backend.policy_stats()
        last_chunk = stats.get("last_action_chunk") or []
        current_target = last_chunk[0] if last_chunk else []
        trajectory.append(
            {
                "step": sim_steps,
                "seed": seed,
                "vla_action": {"joint_positions": current_target},
                "actual_state": backend.actual_state(),
                "predicates": asdict(predicate_result),
                "policy_calls": stats.get("policy_calls", 0),
                "vla_anchor_offset": stats.get("last_anchor_offset") or [],
            }
        )
        if save_video:
            display = step.state.get("camera_display") or {}
            frame = next(iter(display.values()), None)
            if frame is not None:
                array = np.asarray(frame)
                if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
                    array = np.moveaxis(array[:3], 0, -1)
                if array.ndim == 3:
                    video_frames.append(array.astype(np.uint8, copy=False))
        if predicate_result.failure_type:
            termination_reason = f"failure: {predicate_result.failure_type}"
            break
        if predicate_result.task_success:
            termination_reason = "success"
            break
        if step.legacy_done:
            termination_reason = step.termination_reason or "legacy_controller_failure"
            break

    if predicate_result is None:
        print("[probe] no task state observed", flush=True)
        return {"trajectory": trajectory, "predicate": None, "termination_reason": termination_reason, "sim_steps": sim_steps}

    paths = recorder.save_episode(
        episode_id=episode_id,
        episode_index=episode_index,
        instruction=instruction,
        action_horizon=16,
        max_steps=max_steps,
        success_result=predicate_result,
        trajectory=trajectory,
    )
    if video_frames:
        duration = max(time.time() - started, 0.1)
        recorder.save_video(
            episode_id=episode_id,
            success=bool(predicate_result.task_success),
            frames=video_frames,
            fps=max(1.0, len(video_frames) / duration),
        )
    return {
        "trajectory": trajectory,
        "predicate": predicate_result,
        "paths": paths,
        "termination_reason": termination_reason,
        "sim_steps": sim_steps,
    }


def _analyze(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    if not trajectory:
        return {}
    ee = np.asarray([t["actual_state"]["ee_pos"] for t in trajectory], dtype=np.float32)
    joints = np.asarray([t["actual_state"]["joint_pos"] for t in trajectory], dtype=np.float32)
    grip = np.asarray([t["actual_state"]["gripper_pos"] for t in trajectory], dtype=np.float32)
    preds = [t.get("predicates") or {} for t in trajectory]
    hand_obj = np.asarray(
        [float(p.get("hand_object_distance", float("nan"))) for p in preds], dtype=np.float32
    )
    height_change = np.asarray(
        [float(p.get("object_height_change", float("nan"))) for p in preds], dtype=np.float32
    )
    joint_ranges = {}
    for i in range(min(7, joints.shape[1])):
        joint_ranges[i] = (float(np.nanmin(joints[:, i])), float(np.nanmax(joints[:, i])))
    return {
        "n_steps": len(trajectory),
        "ee_z": (float(np.nanmin(ee[:, 2])), float(np.nanmax(ee[:, 2]))),
        "gripper": (float(np.nanmin(grip)), float(np.nanmax(grip))),
        "hand_obj_min": float(np.nanmin(hand_obj)) if hand_obj.size else None,
        "hand_obj_last": float(hand_obj[-1]) if hand_obj.size else None,
        "height_change_max": float(np.nanmax(height_change)) if height_change.size else None,
        "joint_ranges": joint_ranges,
    }


def _print_verdict(analysis: dict[str, Any], pose_name: str, termination_reason: str) -> None:
    print("\n" + "=" * 72, flush=True)
    print(f"[verdict] start pose: {pose_name}", flush=True)
    print(f"[verdict] termination: {termination_reason}", flush=True)
    if not analysis:
        print("[verdict] no trajectory captured", flush=True)
        return
    jr = analysis["joint_ranges"]
    print(f"[verdict] steps={analysis['n_steps']}", flush=True)
    print(
        f"[verdict] wrist sweep    j3 {jr.get(3, (0, 0))[0]: .3f} -> {jr.get(3, (0, 0))[1]: .3f}   "
        f"(training grasp j3 ~= -2.44)",
        flush=True,
    )
    print(
        f"[verdict] wrist sweep    j5 {jr.get(5, (0, 0))[0]: .3f} -> {jr.get(5, (0, 0))[1]: .3f}   "
        f"(training grasp j5 ~= 2.29)",
        flush=True,
    )
    print(
        f"[verdict] wrist sweep    j4 {jr.get(4, (0, 0))[0]: .3f} -> {jr.get(4, (0, 0))[1]: .3f}   "
        f"(training grasp j4 ~= 2.47)",
        flush=True,
    )
    print(f"[verdict] ee height z  {analysis['ee_z'][0]: .3f} -> {analysis['ee_z'][1]: .3f} m", flush=True)
    print(
        f"[verdict] gripper       {analysis['gripper'][0]: .4f} -> {analysis['gripper'][1]: .4f} "
        f"(closed >= 0.02)",
        flush=True,
    )
    print(
        f"[verdict] hand-object dist min={analysis['hand_obj_min'] if analysis['hand_obj_min'] is not None else float('nan'): .3f} m "
        f"(grasp threshold < 0.15)",
        flush=True,
    )
    hc = analysis.get("height_change_max")
    if hc is not None:
        print(f"[verdict] object height change max={hc: .3f} m (lifted > 0.1)", flush=True)
    if analysis["hand_obj_min"] is not None and analysis["hand_obj_min"] < 0.15:
        print("[verdict] >>> ARM REACHED the beaker (<0.15m). Model CAN continue the approach from this pose.", flush=True)
    else:
        print(
            "[verdict] >>> ARM did NOT reach the beaker (<0.15m). Model does not continue a grasp reach "
            "even from mid-approach.",
            flush=True,
        )
    print("=" * 72, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_probe_parser().parse_args(argv)
    pose = _resolve_pose(args)
    pose_name = args.pose if not args.initial_joint_pos else "custom"
    print(f"[probe] starting pose preset: {pose_name}", flush=True)

    task_spec, _raw_cfg, policy, benchmark_root, config_name = preflight(args)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(args.headless),
            "extra_args": ["--/rtx/raytracing/fractionalCutoutOpacity=true"],
        }
    )
    backend = None
    try:
        import hydra
        from omegaconf import open_dict

        with hydra.initialize_config_dir(config_dir=str(benchmark_root / "config")):
            cfg = hydra.compose(config_name=config_name)
        with open_dict(cfg):
            cfg.mode = "infer"
            cfg.max_episodes = int(args.num_episodes)
            cfg.infer.type = "external"
            cfg.infer.host = str(args.vla_host)
            cfg.infer.port = int(args.vla_port)

        from evaluation.runtime import BenchmarkRuntime
        from eai4s_eval.eval.benchmark_backend import EAI4SBenchmarkBackend

        runtime = BenchmarkRuntime(
            cfg=cfg,
            simulation_app=simulation_app,
            backend=args.backend,
            render=bool(args.save_video or not args.headless),
        )
        backend = EAI4SBenchmarkBackend(
            runtime=runtime,
            policy=policy,
            sim_steps_per_action=args.sim_steps_per_action,
            reaction_interpolation_steps=args.reaction_interpolation_steps,
            reanchor_chunks=args.reanchor_chunks,
        )
        instruction = str((task_spec.get("instruction") or {}).get("canonical", ""))
        success_spec = success_spec_from_task_spec(task_spec)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"probe_{now}_{pose_name}_{args.model_name}_{task_spec['task_id']}"
        output_dir = ensure_dir(Path(args.output_root) / run_id)
        recorder = EpisodeRecorder(output_dir)
        print(f"[probe] output: {output_dir}", flush=True)

        seed = int(args.seeds.split(",")[0].strip()) if args.seeds else 210000
        episode_id = f"{task_spec['task_id']}_{args.split}_seed{seed}_{pose_name}"

        initial_conditions = backend.reset_episode(seed=seed, split=args.split)
        print(f"[probe] object pos: {initial_conditions.get('object_position')}", flush=True)
        _apply_initial_pose(runtime, pose)

        result = _run_probe_episode(
            backend=backend,
            success_spec=success_spec,
            recorder=recorder,
            episode_id=episode_id,
            episode_index=0,
            instruction=instruction,
            max_steps=int(getattr(args, "probe_steps", 1200)),
            seed=seed,
            save_video=bool(args.save_video),
        )
        analysis = _analyze(result.get("trajectory") or [])
        _print_verdict(analysis, pose_name, result.get("termination_reason", "?"))
        if result.get("paths"):
            print(f"[probe] saved: {result['paths']}", flush=True)
        return 0
    finally:
        if backend is not None:
            backend.close()
        else:
            policy.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
