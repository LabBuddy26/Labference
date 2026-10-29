from __future__ import annotations

"""Step-2 discriminator: separate *state conditioning* from *vision conditioning*.

Why this exists
---------------
The step-1 probe (``run_discriminate_initial_pose.py``) showed the gr00t pick policy
only works when the arm *physically* starts at a mid-reach pose; from the benchmark
home pose it emits a near-identity "wobble" chunk and never approaches the beaker.
Two competing explanations remain:

  (a) **state-edge saturation** — the home joint values sit on the boundary of the
      training distribution, so a model *state* at home cannot emit a reach; give it
      a mid-reach *state* and it would reach.
  (b) **vision/phase gap** — the model is driven mainly by the *camera* frame and
      never learned the home->approach phase, so no matter what *state* you feed it,
      a home camera view yields only a wobble.

This script isolates the two by freezing the arm at a chosen **physical pose** (so the
camera shows that pose) and feeding the model an **injected state** that may differ
from reality.  Because the arm never moves, the camera frame is held constant and only
the joint state fed to the model varies -- a clean ablation.

The key cell is ``--physical-pose home --inject mid_reach``.  Classification is based on
where chunk0 *lands* relative to the camera pose, the injected state, and the grasp target:

    REACH           -> chunk0 near the grasp/approach target: state conditioning is
                       intact, the home *camera* is the blocker.
    CAMERA_ANCHORED -> chunk0 near the *camera* pose (home) even though the state says
                       mid_reach: the model anchors its action to the visual pose and
                       ignores the state -- the gap is vision/phase, not state-edge
                       saturation.
    IDENTITY        -> chunk0 stayed at the injected state (no motion commanded).
    WOBBLE          -> chunk0 moved somewhere that is neither the camera pose nor the
                       grasp target.

Usage
-----
.. code-block:: bash

    python -m eai4s_eval.cli.run_discriminate_state_vision \
        --benchmark-root ../EAI4S_benchmark \
        --task-spec eai4s_eval/configs/tasks/B01_level1_pick.yaml \
        --policy-profile eai4s_eval/configs/policies/gr00t_franka_joint8_v1.yaml \
        --policy-backend http --vla-host 127.0.0.1 --vla-port 8765 \
        --split sim_dev --seeds 210000 \
        --physical-pose home --inject home,mid_reach,at_grasp \
        --repeats 3 --headless

Matrix / controls
-----------------
* ``--physical-pose home``   + inject home      -> control (expect WOBBLE, the bug).
* ``--physical-pose home``   + inject mid_reach -> *the decisive cell*.
* ``--physical-pose home``   + inject at_grasp  -> does a grasp-*state* alone trigger
  anything?
* ``--physical-pose mid_reach`` + inject actual/mid_reach -> positive control
  (expect REACH, reproducing the step-1 no-reanchor success).

Extra options
-------------
* ``--inject actual`` uses the real (physical) joints -- i.e. no injection.
* ``--inject custom:a,b,...,i`` feeds an explicit 9-value pose.
* ``--sweep start:end:steps`` overrides ``--inject`` with ``steps`` states
  interpolated from preset ``start`` to preset ``end`` (e.g. ``home:mid_reach:5``
  traces the approach manifold while the camera is fixed).
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from eai4s_eval.cli.run_benchmark_eval import build_parser, preflight
from eai4s_eval.cli.run_discriminate_initial_pose import POSE_PRESETS, _apply_initial_pose
from eai4s_eval.utils.io import ensure_dir, write_json

# Reference arm poses (7-D) used as the anchor/target for classifying where chunk0 lands.
# The model's action convention is actions[i] = agent_pose[i+1], so a healthy chunk[0]
# should land near the *next* training pose (mid_reach when starting at home, at_grasp
# when starting at mid_reach).
_HOME = np.asarray(POSE_PRESETS["home"], dtype=np.float32)[:7]
_MID = np.asarray(POSE_PRESETS["mid_reach"], dtype=np.float32)[:7]
_GRASP = np.asarray(POSE_PRESETS["at_grasp"], dtype=np.float32)[:7]
_NEXT_TARGET: dict[str, np.ndarray] = {
    "home": _MID,
    "mid_reach": _GRASP,
    "at_grasp": _GRASP,
}


def _resolve_pose(value: str) -> np.ndarray:
    """Resolve 'preset' or 'custom:a,b,...,i' into a 9-value float32 pose."""
    if value.startswith("custom:"):
        raw = [float(x.strip()) for x in value.split(":", 1)[1].split(",") if x.strip()]
        if len(raw) != 9:
            raise ValueError(f"custom pose needs 9 values, got {len(raw)}: {value}")
        return np.asarray(raw, dtype=np.float32)
    if value not in POSE_PRESETS:
        raise ValueError(f"unknown pose {value!r}; use a preset or custom:a,b,...,i")
    return np.asarray(POSE_PRESETS[value], dtype=np.float32)


def _parse_inject(values: list[str]) -> dict[str, np.ndarray]:
    """Build {name: 9-value pose} from the --inject list. 'actual' is resolved later."""
    poses: dict[str, np.ndarray] = {}
    for item in values:
        if not item.strip():
            continue
        if item.strip() == "actual":
            poses["actual"] = None  # filled in after the first capture
        else:
            poses[item.strip()] = _resolve_pose(item.strip())
    return poses


def _resolve_sweep(spec: str) -> dict[str, np.ndarray]:
    """Resolve 'start:end:steps' into interpolated states (replaces --inject)."""
    start_name, end_name, steps_s = (part.strip() for part in spec.split(":"))
    steps = int(steps_s)
    if steps < 2:
        raise ValueError("--sweep needs at least 2 steps")
    start = _resolve_pose(start_name)[:7]
    end = _resolve_pose(end_name)[:7]
    out: dict[str, np.ndarray] = {}
    for i in range(steps):
        ratio = i / float(steps - 1)
        arm = start + (end - start) * ratio
        pose = np.concatenate([arm, np.asarray([0.04, 0.04], dtype=np.float32)])
        out[f"{start_name}->{end_name}:{ratio:0.2f}"] = pose
    return out


def _capture_observation(runtime: Any, physical_pose: np.ndarray) -> dict[str, Any]:
    """One physics+render step WITHOUT the controller, so the arm never moves.

    Re-asserts the physical pose first (in case of drift), steps the world so the
    cameras refresh, then reads the raw task state.  The controller is bypassed so no
    trajectory is ever generated/executed.
    """
    runtime.robot.set_joint_positions(physical_pose)
    runtime.robot.set_joint_velocities(np.zeros(runtime.robot.num_dof, dtype=np.float32))
    state: dict[str, Any] | None = None
    for _ in range(10):
        runtime.world.step(render=runtime.render)
        state = runtime.task.step()
        if state is not None:
            break
    if state is None:
        raise RuntimeError("capture failed: task.step() never returned a state")
    runtime._apply_observation_contract(state)
    return state


def _probe_policy(
    policy: Any,
    camera_state: dict[str, Any],
    injected_pose: np.ndarray,
    instruction: str,
    repeats: int,
) -> list[np.ndarray]:
    """Feed (frozen home camera + injected state) to the policy ``repeats`` times."""
    observation = {
        "camera_data": camera_state.get("camera_data") or {},
        "agent_pose": injected_pose[:8],  # 7 arm + 1 finger, the gr00t state contract
        "language_instruction": instruction,
    }
    chunks: list[np.ndarray] = []
    for _ in range(max(1, repeats)):
        chunk = policy.predict(observation)
        chunks.append(np.asarray(chunk.joint_positions, dtype=np.float32))
    return chunks


def _nearest_state(chunk0_arm: np.ndarray) -> str:
    dists = {
        name: float(np.linalg.norm(chunk0_arm - np.asarray(POSE_PRESETS[name], dtype=np.float32)[:7]))
        for name in POSE_PRESETS
    }
    return min(dists, key=dists.get)


def _next_target_for_pose(pose: np.ndarray) -> np.ndarray | None:
    """Healthy 'next-state' target for an injected pose, based on its nearest preset."""
    return _NEXT_TARGET.get(_nearest_state(pose[:7]))


def _classify_chunk(d_cam: float, d_base: float, d_target: float | None) -> str:
    """Classify where chunk0 landed relative to the camera pose / injected state / target."""
    if d_cam <= 0.35:
        return "CAMERA_ANCHORED"
    if d_target is not None and d_target <= 0.35:
        return "REACH"
    if d_base <= 0.35:
        return "IDENTITY"
    return "WOBBLE"


def _analyze(
    injected_name: str,
    injected_pose: np.ndarray,
    physical_pose: np.ndarray,
    physical_name: str,
    chunks: list[np.ndarray],
) -> dict[str, Any]:
    if not chunks:
        return {}
    chunk0 = np.asarray([c[0] for c in chunks], dtype=np.float32)      # [R, 8]
    chunk_last = np.asarray([c[-1] for c in chunks], dtype=np.float32)  # [R, 8]
    chunk0_mean = chunk0.mean(axis=0)
    chunk0_arm = chunk0_mean[:7]
    base_arm = injected_pose[:7]
    physical_arm = physical_pose[:7]
    target_arm = _next_target_for_pose(injected_pose)
    d_cam = float(np.linalg.norm(chunk0_arm - physical_arm))
    d_base = float(np.linalg.norm(chunk0_arm - base_arm))
    d_target = None if target_arm is None else float(np.linalg.norm(chunk0_arm - target_arm))
    cls = _classify_chunk(d_cam, d_base, d_target)
    return {
        "injected_name": injected_name,
        "injected_joints": injected_pose.tolist(),
        "chunk0": chunk0_mean.tolist(),
        "chunk_last": chunk_last.mean(axis=0).tolist(),
        "reach_rel_injected": (chunk0_arm - base_arm).tolist(),
        "motion_from_camera_pose": (chunk0_arm - physical_arm).tolist(),
        "d_to_camera_pose": d_cam,
        "d_to_injected": d_base,
        "d_to_target": d_target,
        "gripper_cmd": float(chunk0_mean[7]),
        "class": cls,
        "nearest_state": _nearest_state(chunk0_arm),
        "physical_name": physical_name,
        "repeats": len(chunks),
    }


def _fmt_motion(r: dict[str, Any]) -> str:
    m = r["motion_from_camera_pose"]
    return f"j3 {m[3]:+.2f}  j4 {m[4]:+.2f}  j5 {m[5]:+.2f}  |{r['d_to_camera_pose']:.2f} rad from camera pose"


def _print_table(results: dict[str, dict[str, Any]], physical_name: str) -> None:
    print("\n" + "=" * 84, flush=True)
    print(f"[table] physical (camera) pose = {physical_name}", flush=True)
    for name, r in results.items():
        if not r:
            continue
        d_t = " -- " if r["d_to_target"] is None else f"{r['d_to_target']:.3f}"
        print(f"\n  injected state = {name:24s} -> {r['class']:16s}  nearest-state {r['nearest_state']}", flush=True)
        print(f"      chunk0        {np.round(r['chunk0'], 3).tolist()}", flush=True)
        print(f"      chunk_last    {np.round(r['chunk_last'], 3).tolist()}", flush=True)
        print(f"      motion        {_fmt_motion(r)}", flush=True)
        print(
            f"      dist-to       camera {r['d_to_camera_pose']:.3f}   injected {r['d_to_injected']:.3f}   "
            f"grasp-target {d_t}",
            flush=True,
        )
        print(f"      gripper cmd   {r['gripper_cmd']:.3f}   (repeats {r['repeats']})", flush=True)
    print("=" * 84, flush=True)


def _print_discrimination(results: dict[str, dict[str, Any]], physical_name: str) -> None:
    cell = results.get("mid_reach")
    if not cell:
        print("[verdict] decisive cell (inject=mid_reach) not present; add 'mid_reach' to --inject", flush=True)
        return
    cls = cell["class"]
    d_cam = cell["d_to_camera_pose"]
    nearest = cell["nearest_state"]
    print("\n" + "=" * 84, flush=True)
    print(
        f"[verdict] physical={physical_name}, injected=mid_reach -> {cls}  "
        f"(chunk0 {d_cam:.2f} rad from camera pose, nearest-state {nearest})",
        flush=True,
    )
    if physical_name == "home":
        if cls == "REACH":
            print(
                "[verdict] >>> STATE CONDITIONING IS INTACT: with the home camera and a mid_reach state,\n"
                "          chunk0 lands near the grasp/approach target. The blocker is the home CAMERA view\n"
                "          (visual/phase gap). Add home-view frames paired with approach actions to training.",
                flush=True,
            )
        elif cls == "CAMERA_ANCHORED":
            print(
                "[verdict] >>> CAMERA-ANCHORED: even told it is at mid_reach, the home camera pinned chunk0\n"
                "          back to the home pose. The model anchors its action to the visually-observed pose\n"
                "          and ignores the injected state. -> the approach gap is vision/phase-specific, NOT\n"
                "          state-edge saturation. Feeding a mid_reach state cannot unlock the approach; the\n"
                "          model needs home->approach demonstrations whose observations are the home frames.",
                flush=True,
            )
        elif cls == "IDENTITY":
            print(
                "[verdict] >>> IDENTITY: chunk0 stayed at the injected mid_reach state (no motion). State alone\n"
                "          does not trigger an approach; the camera frame gates the reach.",
                flush=True,
            )
        else:
            print(
                f"[verdict] >>> {cls}: chunk0 moved somewhere that is neither the camera pose nor the grasp\n"
                "          target. The approach gap is not explained by state-edge saturation.",
                flush=True,
            )
    elif physical_name == "mid_reach":
        if cls == "REACH":
            print(
                "[verdict] >>> Positive control OK: mid_reach camera + mid_reach state -> approach reach.\n"
                "          The model CAN reach when both inputs are in-distribution.",
                flush=True,
            )
        else:
            print(
                f"[verdict] >>> Unexpected: mid_reach camera + mid_reach state -> {cls}. Check the model server /\n"
                "          profile before trusting any other cell.",
                flush=True,
            )
    print("=" * 84, flush=True)


def build_discriminator_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.set_defaults(sim_steps_per_action=6, reaction_interpolation_steps=1)
    parser.add_argument(
        "--physical-pose",
        "--physical_pose",
        dest="physical_pose",
        type=str,
        default="home",
        help="Pose the arm physically settles at (the camera shows this). Preset or custom:...",
    )
    parser.add_argument(
        "--inject",
        type=str,
        default="home,mid_reach,at_grasp",
        help="Comma-separated state poses to feed the model (may include 'actual'). "
        "Each may be a preset, 'actual', or 'custom:a,b,...,i'.",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Optional 'start:end:steps' interpolation (overrides --inject), e.g. home:mid_reach:5.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Policy calls per injected state (same frozen camera frame) to gauge output stability.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_discriminator_parser().parse_args(argv)
    physical_pose = _resolve_pose(args.physical_pose)
    physical_name = args.physical_pose if not args.physical_pose.startswith("custom") else "custom"
    if args.sweep:
        injected = _resolve_sweep(args.sweep)
    else:
        injected = _parse_inject([item for item in (args.inject or "").split(",") if item.strip()])
    if not injected:
        raise ValueError("no injected states to test; use --inject or --sweep")
    print(
        f"[probe] physical={physical_name} joints={np.round(physical_pose, 4).tolist()}\n"
        f"[probe] injected states: {', '.join(injected)}",
        flush=True,
    )

    task_spec, _raw_cfg, policy, benchmark_root, config_name = preflight(args)
    instruction = str((task_spec.get("instruction") or {}).get("canonical", ""))

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
            cfg.max_episodes = 1
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
            sim_steps_per_action=6,
            reaction_interpolation_steps=1,
            reanchor_chunks=False,  # raw absolute chunks only; reanchor corrupts this model
        )

        seed = int(args.seeds.split(",")[0].strip()) if args.seeds else 210000
        backend.reset_episode(seed=seed, split=args.split)
        _apply_initial_pose(runtime, physical_pose)

        capture = _capture_observation(runtime, physical_pose)

        # 'actual' injection resolves to the real (physical) joints.
        for name in list(injected):
            if injected[name] is None:
                injected[name] = np.asarray(capture.get("joint_positions", []), dtype=np.float32).reshape(-1)
                if injected[name].size < 8:
                    raise RuntimeError(f"'actual' joint state too short: {injected[name].shape}")

        results: dict[str, dict[str, Any]] = {}
        for name, pose in injected.items():
            chunks = _probe_policy(policy, capture, pose, instruction, args.repeats)
            results[name] = _analyze(name, pose, physical_pose, physical_name, chunks)

        _print_table(results, physical_name)
        _print_discrimination(results, physical_name)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"probe_{now}_statevision_{physical_name}_{args.model_name}_{task_spec['task_id']}"
        output_dir = ensure_dir(Path(args.output_root) / run_id)
        report_path = write_json(
            output_dir / "report_state_vision.json",
            {
                "physical_pose": physical_name,
                "physical_joints": physical_pose.tolist(),
                "task_id": task_spec["task_id"],
                "instruction": instruction,
                "results": results,
            },
        )
        print(f"[probe] saved report: {report_path}", flush=True)
        return 0
    finally:
        if backend is not None:
            backend.close()
        else:
            policy.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
