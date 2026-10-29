"""Run the canonical evaluator directly from one benchmark task YAML."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

from eai4s_eval.cli.run_benchmark_eval import build_parser
from eai4s_eval.eval.runner import load_task_spec
from eai4s_eval.eval.task_suite import resolve_task_spec_path
from eai4s_eval.policy import Gr00tPolicyClient, JointPolicyAdapter, load_openpi_profile


class EvaluationConfigError(ValueError):
    """Raised when ``mode: eval`` lacks the configuration it needs."""


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "items"):
        return {str(key): item for key, item in value.items()}
    raise EvaluationConfigError("evaluation must be a YAML mapping")


def _resolve_path(benchmark_root: Path, value: Any, label: str) -> Path:
    if not value:
        raise EvaluationConfigError(f"evaluation.{label} is required when mode is 'eval'")
    path = Path(str(value))
    if not path.is_absolute():
        path = benchmark_root / path
    path = path.resolve()
    if not path.exists():
        raise EvaluationConfigError(f"evaluation.{label} does not exist: {path}")
    return path


def _available_camera_streams(cfg: Any) -> set[str]:
    streams: set[str] = set()
    cameras = cfg.get("cameras", []) if hasattr(cfg, "get") else []
    for camera in cameras or []:
        name = camera.get("name") if hasattr(camera, "get") else None
        image_type = camera.get("image_type", "rgb") if hasattr(camera, "get") else "rgb"
        if name:
            streams.update(f"{name}_{kind}" for kind in str(image_type).split("+"))
    return streams


def _args_from_yaml(cfg: Any, *, benchmark_root: Path, backend: str, headless: bool, no_video: bool) -> Namespace:
    settings = _mapping(cfg.get("evaluation") if hasattr(cfg, "get") else None)
    policy_settings = _mapping(settings.get("policy")) if settings.get("policy") is not None else {}
    # Compatibility with the original flattened evaluation fields and the
    # pre-merge ``infer`` mapping. Canonical YAMLs use evaluation.policy.
    legacy_infer = _mapping(cfg.get("infer")) if hasattr(cfg, "get") and cfg.get("infer") is not None else {}
    args = build_parser().parse_args([])
    task_spec = _resolve_path(benchmark_root, settings.get("task_spec"), "task_spec")
    profile_value = policy_settings.get("profile", settings.get("policy_profile"))
    profile = _resolve_path(benchmark_root, profile_value, "policy.profile")
    output_root = Path(str(settings.get("output_root", "outputs/eval_runs")))
    if not output_root.is_absolute():
        output_root = benchmark_root / output_root

    args.benchmark_root = str(benchmark_root)
    args.config_name = None
    args.task_spec = str(task_spec)
    args.policy_profile = str(profile)
    args.policy_provider = str(policy_settings.get("provider", "openpi")).lower()
    args.vla_host = str(policy_settings.get("host", settings.get("vla_host", legacy_infer.get("host", "127.0.0.1"))))
    default_port = 8765 if args.policy_provider == "gr00t" else 8000
    args.vla_port = int(policy_settings.get("port", settings.get("vla_port", legacy_infer.get("port", default_port))))
    args.api_key = policy_settings.get("api_key", settings.get("api_key"))
    args.policy_backend = str(policy_settings.get("protocol", settings.get("policy_backend", "websocket")))
    args.policy_timeout = float(policy_settings.get("timeout", legacy_infer.get("timeout", 30)))
    args.model_name = str(settings.get("model_name", "eai4s_openpi"))
    args.checkpoint = settings.get("checkpoint")
    args.checkpoint_hash = settings.get("checkpoint_hash")
    args.dataset_version = settings.get("dataset_version")
    args.split = str(settings.get("split", "sim_dev"))
    args.num_episodes = int(settings.get("num_episodes", 5))
    args.seeds = settings.get("seeds")
    args.seed_manifest = settings.get("seed_manifest")
    args.run_id = settings.get("run_id")
    args.eval_group_id = settings.get("eval_group_id")
    args.suite_id = settings.get("suite_id")
    args.run_type = str(settings.get("run_type", "single_task"))
    args.output_root = str(output_root)
    # Evaluation has its own budget.  Never fall back to the collection
    # ``task.max_steps`` value from the environment YAML; when omitted, the
    # TaskSpec's documented budget is used after it is loaded below.
    args.eval_max_steps_explicit = "max_steps" in settings
    # Evaluation max_steps is the episode/control-step budget.  Keep the
    # default bounded at 500; explicit YAML/CLI values remain user overrides.
    args.max_steps = int(settings.get("max_steps", 500))
    args.action_horizon = int(settings.get("action_horizon", 40))
    args.control_hz = int(policy_settings.get("control_hz", 10))
    raw_sim_steps = settings.get("sim_steps_per_action")
    args.sim_steps_per_action = None if raw_sim_steps is None else int(raw_sim_steps)
    args.reaction_interpolation_steps = int(settings.get("reaction_interpolation_steps", 1))
    # Re-anchoring is opt-in.  If tracking has already overshot the first
    # waypoint, shifting the whole new chunk by the actual error can amplify
    # the overshoot and make the arm walk past the target.
    args.reanchor_chunks = bool(settings.get("reanchor_chunks", False))
    args.arm_smoothing_steps = int(settings.get("arm_smoothing_steps", 4))
    args.gripper_arm_reach_tolerance = float(settings.get("gripper_arm_reach_tolerance", 0.08))
    args.gripper_close_threshold = float(settings.get("gripper_close_threshold", 0.02))
    args.gripper_reach_gate = bool(settings.get("gripper_reach_gate", False))
    # Strict evaluation executes every selected model waypoint.  Boundary
    # filtering is useful as an explicit deployment option, but enabling it by
    # default would conditionally discard policy outputs and change the policy
    # being measured.
    args.chunk_boundary_guard = bool(settings.get("chunk_boundary_guard", False))
    args.chunk_boundary_search_steps = int(settings.get("chunk_boundary_search_steps", 4))
    args.chunk_boundary_error_threshold = float(settings.get("chunk_boundary_error_threshold", 0.03))
    args.control_dt = float(settings.get("control_dt", 1.0 / 60.0))
    args.max_joint_velocity = float(settings.get("max_joint_velocity", 2.0))
    args.max_joint_acceleration = float(settings.get("max_joint_acceleration", 10.0))
    raw_execution_horizon = policy_settings.get("execution_horizon", settings.get("execution_horizon"))
    args.execution_horizon = None if raw_execution_horizon in {None, "", "null"} else int(raw_execution_horizon)
    args.backend = backend
    args.headless = bool(headless)
    args.save_video = bool(settings.get("save_video", False)) and not bool(no_video)
    args.progress_interval_sec = float(settings.get("progress_interval_sec", 1.0))
    args.progress_log_format = str(settings.get("progress_log_format", "both"))
    if args.progress_interval_sec <= 0:
        raise EvaluationConfigError("evaluation.progress_interval_sec must be greater than zero")
    if args.progress_log_format not in {"both", "stdout", "jsonl"}:
        raise EvaluationConfigError(
            "evaluation.progress_log_format must be one of: both, stdout, jsonl"
        )
    return args


def run_configured_eval(
    *,
    cfg: Any,
    simulation_app: Any,
    backend: str,
    headless: bool,
    no_video: bool = False,
) -> dict[str, Any]:
    """Execute ``mode: eval`` using the same loaded Hydra task configuration."""

    if str(getattr(cfg, "mode", "")) != "eval":
        raise EvaluationConfigError("run_configured_eval requires mode: eval")
    robot_type = str(cfg.robot.get("type", "")) if hasattr(cfg.robot, "get") else ""
    if robot_type != "franka":
        raise EvaluationConfigError(
            "mode: eval currently accepts the Franka 8-D joint-action contract only; "
            f"task robot.type is {robot_type!r}"
        )

    benchmark_root = Path(__file__).resolve().parents[1]
    args = _args_from_yaml(
        cfg,
        benchmark_root=benchmark_root,
        backend=backend,
        headless=headless,
        no_video=no_video,
    )
    task_path = resolve_task_spec_path(args.task_spec)
    task_spec = load_task_spec(str(task_path), "franka", "", args.max_steps, None)
    if args.eval_max_steps_explicit:
        # The task YAML's ``evaluation.max_steps`` is authoritative for this
        # launch and is deliberately independent from collection ``task``.
        task_spec["max_steps"] = int(args.max_steps)
    else:
        # A missing evaluation override uses the bounded 500-step default;
        # never fall back to the collection-oriented TaskSpec budget.
        task_spec["max_steps"] = int(args.max_steps)

    from evaluation.config import validate_camera_contract
    from evaluation.runtime import BenchmarkRuntime
    from eai4s_eval.eval.benchmark_backend import EAI4SBenchmarkBackend
    from eai4s_eval.eval.benchmark_runner import BenchmarkTaskEvalRunner
    from omegaconf import open_dict

    validate_camera_contract(cfg)
    profile = load_openpi_profile(args.policy_profile)
    task_control_hz = int(task_spec.get("control_hz", 10))
    if args.control_hz != 10:
        raise EvaluationConfigError(
            f"evaluation.policy.control_hz must be 10 for the published model contract, got {args.control_hz}"
        )
    if profile.control_hz != args.control_hz or task_control_hz != args.control_hz:
        raise EvaluationConfigError(
            "10 Hz evaluation contract mismatch: "
            f"yaml={args.control_hz}, profile={profile.control_hz}, task_spec={task_control_hz}"
        )
    missing_streams = sorted(set(profile.observation_map).difference(_available_camera_streams(cfg)))
    if missing_streams:
        raise EvaluationConfigError(
            "policy profile requires cameras missing from this task YAML: " + ", ".join(missing_streams)
        )
    instruction = str((task_spec.get("instruction") or {}).get("canonical", ""))
    policy_client = None
    if args.policy_provider == "gr00t":
        if args.policy_backend not in {"zmq", "zeromq"}:
            raise EvaluationConfigError("GR00T evaluation.policy.protocol must be 'zmq'")
        policy_client = Gr00tPolicyClient(
            host=args.vla_host,
            port=args.vla_port,
            timeout_ms=int(args.policy_timeout * 1000),
            api_token=args.api_key,
        )
    elif args.policy_provider != "openpi":
        raise EvaluationConfigError(
            f"unsupported evaluation.policy.provider {args.policy_provider!r}; expected 'openpi' or 'gr00t'"
        )
    policy = JointPolicyAdapter(
        host=args.vla_host,
        port=args.vla_port,
        profile=profile,
        api_key=args.api_key,
        client=policy_client,
        backend="gr00t_zmq" if policy_client is not None else args.policy_backend,
        default_instruction=instruction,
        timeout_sec=float(args.policy_timeout),
    )
    eval_backend = None
    try:
        # Existing task controllers initialize an inference engine.  The
        # external type is a deferred placeholder and is immediately replaced
        # below by the evaluator's validated policy trajectory bridge.
        with open_dict(cfg):
            cfg.infer.type = "external"
            cfg.infer.host = args.vla_host
            cfg.infer.port = args.vla_port
        runtime = BenchmarkRuntime(
            cfg=cfg,
            simulation_app=simulation_app,
            backend=backend,
            # Policy evaluation always consumes RGB observations.  Isaac RTX
            # cameras return an empty frame when the world is stepped with
            # ``render=False``; tying rendering to ``save_video`` made a
            # headless/no-video evaluation fail on its first request with
            # camera shape errors.  Keep rendering enabled for inference even
            # when video recording is disabled.
            render=True,
        )
        eval_backend = EAI4SBenchmarkBackend(
            runtime=runtime,
            policy=policy,
            execution_horizon=args.execution_horizon,
            control_hz=args.control_hz,
            sim_steps_per_action=args.sim_steps_per_action,
        )
        runner = BenchmarkTaskEvalRunner(
            args=args,
            backend=eval_backend,
            task_spec=task_spec,
            policy_metadata=policy.health(),
        )
        from omegaconf import OmegaConf

        # Preserve the exact resolved scene/controller configuration beside
        # the evaluator's own run_config.json (which records runner args and
        # TaskSpec).  This makes an eval run reproducible from one YAML.
        OmegaConf.save(cfg, str(runner.output_dir / "environment_config.yaml"))
        print(f"[eval] Task: {task_spec.get('task_id')}, split: {args.split}, episodes: {args.num_episodes}", flush=True)
        print(f"[eval] Output: {runner.output_dir}", flush=True)
        return runner.run()
    finally:
        if eval_backend is not None:
            eval_backend.close()
        else:
            policy.close()


__all__ = ["EvaluationConfigError", "run_configured_eval"]
