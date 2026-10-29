from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from eai4s_eval.eval.runner import load_task_spec
from eai4s_eval.eval.model_identity import resolve_model_name, validate_checkpoint_name
from eai4s_eval.eval.task_suite import resolve_task_spec_path
from eai4s_eval.policy import Gr00tPolicyClient, JointPolicyAdapter, load_openpi_profile


_BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _BENCHMARK_ROOT / "eai4s_eval"
_DEFAULT_BENCHMARK_ROOT = _BENCHMARK_ROOT
_DEFAULT_OUTPUT_ROOT = _DEFAULT_BENCHMARK_ROOT / "outputs" / "eval_runs"
_DEFAULT_PROFILE = _PACKAGE_ROOT / "configs" / "policies" / "openpi_franka_ee_pos_v1.yaml"
_DEFAULT_GR00T_PROFILE = _PACKAGE_ROOT / "configs" / "policies" / "gr00t_franka_ee_pos_v1.yaml"
_DEFAULT_TASK_SPEC = _PACKAGE_ROOT / "configs" / "tasks" / "B01_level1_pick.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenPI or GR00T 8D EE policy in the canonical LabBench runtime."
    )
    parser.add_argument("--benchmark-root", type=str, default=str(_DEFAULT_BENCHMARK_ROOT))
    parser.add_argument("--reference", action="store_true", help="Engineering controller smoke test; never a model submission.")
    parser.add_argument("--config-name", type=str, default=None,
                        help="Hydra environment config. Defaults to TaskSpec runtime.env_config.")
    parser.add_argument("--task-spec", "--task_spec", dest="task_spec", type=str, default=str(_DEFAULT_TASK_SPEC))
    parser.add_argument("--policy-profile", "--policy_profile", dest="policy_profile", type=str, default=str(_DEFAULT_PROFILE))
    parser.add_argument("--vla-host", "--vla_host", dest="vla_host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--vla-port", "--vla_port", dest="vla_port", type=int, default=None,
        help="Policy server port (default: OpenPI 8000, GR00T 5555).",
    )
    parser.add_argument("--api-key", "--api_key", dest="api_key", type=str, default=None)
    parser.add_argument("--policy-provider", "--policy_provider", dest="policy_provider",
                        choices=("openpi", "gr00t"), default="openpi")
    parser.add_argument("--policy-backend", "--policy_backend", dest="policy_backend",
                        choices=("websocket", "http", "zmq"), default="websocket")
    parser.add_argument(
        "--policy-timeout", "--policy_timeout", dest="policy_timeout", type=float, default=30.0,
        help="OpenPI/GR00T connection and inference timeout in seconds (default: 30).",
    )
    parser.add_argument("--model-name", "--model_name", dest="model_name", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-hash", "--checkpoint_hash", dest="checkpoint_hash", type=str, default=None)
    parser.add_argument("--dataset-version", "--dataset_version", dest="dataset_version", type=str, default=None)
    parser.add_argument("--split", choices=("sim_clean", "sim_dev", "sim_randomized"), default="sim_dev")
    parser.add_argument("--num-episodes", "--num_episodes", dest="num_episodes", type=int, default=5)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--seed-manifest", "--seed_manifest", dest="seed_manifest", type=str, default=None)
    parser.add_argument("--run-id", "--run_id", dest="run_id", type=str, default=None)
    parser.add_argument("--eval-group-id", "--eval_group_id", dest="eval_group_id", type=str, default=None)
    parser.add_argument("--suite-id", "--suite_id", dest="suite_id", type=str, default=None)
    parser.add_argument("--run-type", "--run_type", dest="run_type", type=str, default="single_task")
    parser.add_argument("--output-root", "--output_root", dest="output_root", type=str, default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--max-steps", "--max_steps", dest="max_steps", type=int, default=500,
        help="Maximum episode/control steps (chunk waypoints); default: 500.",
    )
    parser.add_argument("--action-horizon", "--action_horizon", dest="action_horizon", type=int, default=40)
    parser.add_argument("--sim-steps-per-action", "--sim_steps_per_action", dest="sim_steps_per_action", type=int, default=None,
                        help="Optional physics steps per model waypoint; default derives physics_hz/control_hz (6 for 60/10 Hz).")
    parser.add_argument("--reaction-interpolation-steps", "--reaction_interpolation_steps",
                        dest="reaction_interpolation_steps", type=int, default=1,
                        help="Interpolation sub-divisions within each original step, capped by sim-steps-per-action; does NOT inflate the step count (default: 1).")
    parser.add_argument("--reanchor-chunks", "--reanchor_chunks", dest="reanchor_chunks",
                        action="store_true", default=False,
                        help="Re-anchor each policy chunk to the current joint position (default: disabled). "
                             "Anchoring removes the rollback step at chunk boundaries when the policy's waypoint 0 "
                             "does not align with the actual robot state; enable only when the policy emits stale anchors.")
    parser.add_argument("--no-reanchor-chunks", dest="reanchor_chunks", action="store_false",
                        help="Disable chunk re-anchoring (only for debugging policies that require a fixed world anchor).")
    parser.add_argument("--arm-smoothing-steps", "--arm_smoothing_steps", dest="arm_smoothing_steps", type=int, default=4,
                        help="Physics-frame subdivisions for each arm target (default: 4).")
    parser.add_argument("--gripper-arm-reach-tolerance", "--gripper_arm_reach_tolerance",
                        dest="gripper_arm_reach_tolerance", type=float, default=0.08,
                        help="Maximum per-joint error before a close command is allowed (rad).")
    parser.add_argument("--gripper-close-threshold", "--gripper_close_threshold",
                        dest="gripper_close_threshold", type=float, default=0.02,
                        help="Physical gripper target at or below which the command is treated as close.")
    parser.add_argument("--gripper-reach-gate", "--gripper_reach_gate", dest="gripper_reach_gate",
                        action="store_true", default=False,
                        help="Hold close until the arm reaches the target joint pose (default: disabled).")
    parser.add_argument("--chunk-boundary-guard", "--chunk_boundary_guard", dest="chunk_boundary_guard",
                        action="store_true", default=False,
                        help="Skip stale initial waypoints at chunk boundaries without shifting the chunk (default: disabled; this changes strict policy execution).")
    parser.add_argument("--no-chunk-boundary-guard", dest="chunk_boundary_guard", action="store_false",
                        help="Disable stale-waypoint filtering at chunk boundaries.")
    parser.add_argument("--chunk-boundary-search-steps", "--chunk_boundary_search_steps",
                        dest="chunk_boundary_search_steps", type=int, default=4,
                        help="Number of initial waypoints inspected by the boundary guard (default: 4).")
    parser.add_argument("--chunk-boundary-error-threshold", "--chunk_boundary_error_threshold",
                        dest="chunk_boundary_error_threshold", type=float, default=0.03,
                        help="Minimum max-joint error before stale initial waypoints are skipped (rad).")
    parser.add_argument("--control-dt", "--control_dt", dest="control_dt", type=float, default=1.0 / 60.0,
                        help="Physics control timestep used for motion limits (default: 1/60 s).")
    parser.add_argument("--max-joint-velocity", "--max_joint_velocity", dest="max_joint_velocity", type=float, default=2.0,
                        help="Maximum commanded arm joint velocity (rad/s, default: 2.0).")
    parser.add_argument("--max-joint-acceleration", "--max_joint_acceleration", dest="max_joint_acceleration", type=float, default=10.0,
                        help="Maximum commanded arm joint acceleration (rad/s^2, default: 10.0).")
    parser.add_argument("--execution-horizon", "--execution_horizon", dest="execution_horizon", type=int, default=None,
                        help="Direct actions per request; default executes the complete model chunk.")
    parser.add_argument("--backend", choices=("numpy", "gpu"), default="numpy")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-video", "--save_video", dest="save_video", action="store_true")
    parser.add_argument(
        "--progress-interval-sec", "--progress_interval_sec",
        dest="progress_interval_sec", type=float, default=1.0,
        help="Periodic success-diagnostic interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--progress-log-format", "--progress_log_format",
        dest="progress_log_format", choices=("both", "stdout", "jsonl"), default="both",
        help="Write evaluation progress to stdout, JSONL, or both (default: both).",
    )
    return parser


def _resolve_task_spec(value: str) -> Path:
    path = resolve_task_spec_path(value)
    if not path.exists():
        raise FileNotFoundError(f"TaskSpec not found: {path}")
    return path


def preflight(args: argparse.Namespace) -> tuple[dict, object, JointPolicyAdapter, Path, str]:
    """Validate files, camera/action contracts and the selected policy server."""

    args.model_name = resolve_model_name(args.policy_provider, args.model_name)
    args.checkpoint = validate_checkpoint_name(args.checkpoint, required=False)
    if args.vla_port is None:
        args.vla_port = 5555 if args.policy_provider == "gr00t" else 8000
    if float(args.progress_interval_sec) <= 0:
        raise ValueError("--progress-interval-sec must be greater than zero")
    if float(args.policy_timeout) <= 0:
        raise ValueError("--policy-timeout must be greater than zero")

    benchmark_root = Path(args.benchmark_root).resolve()
    if not (benchmark_root / "config").is_dir():
        raise FileNotFoundError(f"invalid EAI4S_benchmark root: {benchmark_root}")
    task_path = _resolve_task_spec(args.task_spec)
    task_spec = load_task_spec(str(task_path), "franka", "", args.max_steps, None)
    # The standalone CLI's max_steps is the episode/control-step budget.  Do
    # not let the collection-oriented TaskSpec default (often 2000+) override
    # the CLI default of 500 or an explicit --max_steps value.
    task_spec["max_steps"] = int(args.max_steps)
    runtime = task_spec.get("runtime") or {}
    config_name = str(args.config_name or runtime.get("env_config") or "")
    config_path = benchmark_root / "config" / f"{config_name}.yaml"
    if not config_name or not config_path.exists():
        raise FileNotFoundError(f"benchmark environment config not found: {config_path}")

    if str(benchmark_root) not in sys.path:
        sys.path.insert(0, str(benchmark_root))
    from evaluation.config import validate_camera_contract

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required for the benchmark preflight") from exc
    raw_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    validate_camera_contract(raw_cfg)

    # The parser historically defaulted to the OpenPI profile.  Selecting
    # ``--policy-provider gr00t`` without also spelling out a profile would
    # therefore send ``observation/front_image`` keys to a GR00T server whose
    # ``new_embodiment`` transform expects ``video.main_camera``.  Choose the
    # matching profile automatically while still honoring an explicit custom
    # profile path.
    profile_path = Path(args.policy_profile)
    if args.policy_provider == "gr00t" and profile_path.resolve() == _DEFAULT_PROFILE.resolve():
        profile_path = _DEFAULT_GR00T_PROFILE
        args.policy_profile = str(profile_path)
    profile = load_openpi_profile(profile_path)
    available_streams = {
        f"{camera.get('name')}_{kind}"
        for camera in raw_cfg.get("cameras") or []
        for kind in str(camera.get("image_type", "rgb")).split("+")
    }
    missing_profile_streams = sorted(set(profile.observation_map).difference(available_streams))
    if missing_profile_streams:
        raise ValueError(
            f"policy profile {profile.profile_id} requires camera streams missing from {config_name}: "
            + ", ".join(missing_profile_streams)
        )
    instruction = str((task_spec.get("instruction") or {}).get("canonical", ""))
    client = None
    if args.policy_provider == "gr00t":
        if args.policy_backend != "zmq":
            raise ValueError("GR00T requires --policy-backend zmq")
        client = Gr00tPolicyClient(
            host=args.vla_host,
            port=args.vla_port,
            api_token=args.api_key,
        )
    policy = JointPolicyAdapter(
        host=args.vla_host,
        port=args.vla_port,
        profile=profile,
        api_key=args.api_key,
        client=client,
        backend="gr00t_zmq" if client is not None else args.policy_backend,
        default_instruction=instruction,
        timeout_sec=float(args.policy_timeout),
    )
    return task_spec, raw_cfg, policy, benchmark_root, config_name


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.reference:
        from eai4s_eval.eval.reference_worker import run_reference
        return run_reference(args)
    from utils.special_protocol_driver import dispatch_benchmark_cli
    release_exit_code = dispatch_benchmark_cli(args)
    if release_exit_code is not None:
        return release_exit_code
    task_spec, _raw_cfg, policy, benchmark_root, config_name = preflight(args)
    print(f"{args.policy_provider} preflight passed: {policy.health()}", flush=True)

    # SimulationApp must exist before importing the remaining Isaac modules.
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
            # Keep the externally invoked CLI on the same controller path as
            # ``main.py`` with ``mode: eval``.
            cfg.mode = "eval"
            cfg.max_episodes = int(args.num_episodes)
            cfg.infer.type = "external"
            cfg.infer.host = str(args.vla_host)
            cfg.infer.port = int(args.vla_port)

        from evaluation.runtime import BenchmarkRuntime
        from eai4s_eval.eval.benchmark_backend import EAI4SBenchmarkBackend
        from eai4s_eval.eval.benchmark_runner import BenchmarkTaskEvalRunner

        runtime = BenchmarkRuntime(
            cfg=cfg,
            simulation_app=simulation_app,
            backend=args.backend,
            # RGB observations are required by both OpenPI and GR00T.  A
            # headless run with ``render=False`` leaves Isaac camera buffers
            # empty even when video recording is disabled, causing the first
            # policy request to fail before the episode can start.
            render=True,
        )
        backend = EAI4SBenchmarkBackend(
            runtime=runtime,
            policy=policy,
            execution_horizon=args.execution_horizon,
            control_hz=int(task_spec.get("control_hz", policy.profile.control_hz)),
            sim_steps_per_action=args.sim_steps_per_action,
        )
        runner = BenchmarkTaskEvalRunner(
            args=args,
            backend=backend,
            task_spec=task_spec,
            policy_metadata=policy.health(),
        )
        print("[eval] Starting benchmark evaluation...", flush=True)
        print(f"[eval] Task: {task_spec.get('task_id')}, Split: {args.split}, Episodes: {args.num_episodes}", flush=True)
        print(f"[eval] Output: {runner.output_dir}", flush=True)
        try:
            runner.run()
        except Exception as exc:
            print(f"[eval] ERROR: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        print("[eval] Evaluation completed", flush=True)
        return 0
    finally:
        if backend is not None:
            backend.close()
        else:
            policy.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
