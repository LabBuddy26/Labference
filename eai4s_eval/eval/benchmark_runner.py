from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from eai4s_eval import __version__
from eai4s_eval.eval.aggregator import RunAggregator
from eai4s_eval.eval.metric_manager import TaskMetricManager
from eai4s_eval.eval.recorder import EpisodeRecorder
from eai4s_eval.eval.success import EvaluationProgressReporter, SuccessEvaluation, TaskSuccessEvaluator
from eai4s_eval.eval.success.failure_taxonomy import classify_failure
from eai4s_eval.eval.task_spec import success_spec_from_task_spec, task_metadata_from_task_spec
from eai4s_eval.policy import PolicyOutputInvalid, PolicyClientError
from eai4s_eval.utils.io import ensure_dir


def _json_safe(obj: Any) -> Any:
    """Recursively convert OmegaConf DictConfig etc. to JSON-safe types."""
    if hasattr(obj, "_content"):  # OmegaConf DictConfig / ListConfig
        from omegaconf import OmegaConf
        return OmegaConf.to_container(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_benchmark_seeds(
    *,
    task_id: str,
    split: str,
    count: int,
    explicit: str | None = None,
    manifest_path: str | Path | None = None,
) -> list[int]:
    if explicit:
        seeds = [int(item.strip()) for item in explicit.split(",") if item.strip()]
        if len(seeds) < count:
            raise ValueError(f"only {len(seeds)} explicit seeds were provided for {count} episodes")
        return seeds[:count]
    path = Path(manifest_path) if manifest_path else Path(__file__).resolve().parents[1] / "configs" / "benchmark_seed_manifest.yaml"
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load benchmark seed manifests") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    split_table = (payload.get("splits") or {}).get(split) or {}
    if task_id not in split_table:
        raise ValueError(f"seed manifest {path} has no entry for task={task_id}, split={split}")
    available = int(payload.get("count", 100))
    if count > available:
        raise ValueError(f"requested {count} episodes but seed manifest provides {available}")
    base = int(split_table[task_id])
    return [base + offset for offset in range(count)]


class BenchmarkTaskEvalRunner:
    """File-compatible EAI4S evaluator for the canonical collection runtime."""

    def __init__(
        self,
        *,
        args: Any,
        backend: Any,
        task_spec: dict[str, Any],
        policy_metadata: dict[str, Any],
    ):
        self.args = args
        self.backend = backend
        self.task_spec = task_spec
        self.task_metadata = task_metadata_from_task_spec(task_spec)
        self.success_spec = success_spec_from_task_spec(task_spec)
        self.policy_metadata = policy_metadata
        self._ee_action_space = (
            str(self.policy_metadata.get("action_space", "joint")).lower() == "ee_pos"
        )
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not getattr(args, "checkpoint", None):
            args.checkpoint = f"eval_{now}"
        self.run_id = args.run_id or f"eval_{now}_{args.model_name}_{task_spec['task_id']}"
        self.eval_group_id = getattr(args, "eval_group_id", None) or self.run_id
        self.output_dir = ensure_dir(Path(args.output_root) / self.run_id)
        self.recorder = EpisodeRecorder(self.output_dir)
        self.progress_reporter = EvaluationProgressReporter(
            run_dir=self.output_dir,
            run_id=self.run_id,
            task_id=str(task_spec["task_id"]),
            spec=self.success_spec,
            predicate_hz=float(task_spec.get("predicate_hz", task_spec.get("control_hz", 10))),
            interval_sec=float(getattr(args, "progress_interval_sec", 1.0)),
            output_format=str(getattr(args, "progress_log_format", "both")),
        )
        self.metric_manager = TaskMetricManager()
        explicit = getattr(args, "seeds", None)
        if explicit:
            self.seeds = [int(item.strip()) for item in explicit.split(",") if item.strip()]
            if len(self.seeds) < int(args.num_episodes):
                raise ValueError("explicit seed list is shorter than the requested valid episode count")
        else:
            self.seeds = load_benchmark_seeds(
                task_id=str(task_spec["task_id"]),
                split=str(args.split),
                count=100,
                manifest_path=getattr(args, "seed_manifest", None),
            )
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("benchmark seed pool contains duplicates")

    def run(self) -> dict[str, Any]:
        manifest = self._run_manifest()
        self.recorder.write_run_manifest(manifest)
        self.recorder.write_run_config({"args": vars(self.args), "task_spec": self.task_spec})
        rows: list[dict[str, Any]] = []
        valid_completed = 0
        valid_successes = 0
        desired_valid = int(self.args.num_episodes)
        for episode_index, seed in enumerate(self.seeds):
            episode_id = f"{self.task_spec['task_id']}_{self.args.split}_seed{seed}"
            print(f"=== Running {episode_id} (valid {valid_completed}/{desired_valid}) ===", flush=True)
            row = self._run_episode_with_retries(episode_index, seed, episode_id)
            rows.append(row)
            print(
                "[eval] Step counts: "
                f"action_steps={int(row.get('episode_steps', 0))}/"
                f"{int(self.task_spec.get('max_steps', self.args.max_steps))}, "
                f"physics_steps={int(row.get('sim_steps', 0))}, "
                f"policy_chunks={int(row.get('policy_calls', 0))}",
                flush=True,
            )
            if row.get("valid_scene", True):
                valid_completed += 1
                valid_successes += int(bool(row.get("success")))
                status = "SUCCESS" if row.get("success") else "FAILED"
                rate = valid_successes / valid_completed * 100.0
                print(
                    f"[eval] Episode result: {status} | {episode_id} | "
                    f"official running success rate: "
                    f"{valid_successes}/{valid_completed} ({rate:.2f}%)",
                    flush=True,
                )
            else:
                print(f"[eval] Episode result: INVALID_SCENE | {episode_id}", flush=True)
            if not row.get("success") and row.get("valid_scene", True):
                self.recorder.append_failure_case(row)
            self._write_progress(valid_completed, desired_valid, row, attempted=episode_index + 1)
            if valid_completed >= desired_valid:
                break
        result = RunAggregator(self.output_dir).write_all(
            episode_rows=rows,
            run_manifest=manifest,
            task_spec=self.task_spec,
        )
        if valid_completed < desired_valid:
            raise RuntimeError(
                f"seed pool exhausted after {len(rows)} scenes: obtained {valid_completed}/{desired_valid} valid episodes"
            )
        print("", flush=True)
        for line in result.get("summary_lines", []):
            print(line, flush=True)
        print(f"Summary log: {result.get('summary_log')}", flush=True)
        print(f"Evaluation complete: {self.output_dir}", flush=True)
        self.progress_reporter.close()
        return result

    def _run_episode_with_retries(self, episode_index: int, seed: int, episode_id: str) -> dict[str, Any]:
        last_row: dict[str, Any] | None = None
        for attempt in range(3):
            row = self._run_episode(episode_index, seed, episode_id, attempt)
            last_row = row
            if row.get("valid_scene", True):
                return row
            if row.get('termination_reason') in {'runtime_error', 'policy_request_failed'}:
                return row
            print(f"Invalid scripted precondition for {episode_id}; reset attempt {attempt + 1}/3", flush=True)
        assert last_row is not None
        return last_row

    def _run_episode(self, episode_index: int, seed: int, episode_id: str, attempt: int) -> dict[str, Any]:
        initial_conditions = self.backend.reset_episode(seed=seed, split=self.args.split)
        initial_conditions = _json_safe(initial_conditions)
        self.progress_reporter.start_episode(
            episode_id=episode_id,
            episode_index=episode_index,
            split=str(self.args.split),
            seed=seed,
            attempt=attempt + 1,
        )
        evaluator = TaskSuccessEvaluator(self.success_spec)
        evaluator.reset()
        trajectory: list[dict[str, Any]] = []
        video_frames: list[np.ndarray] = []
        predicate_result: SuccessEvaluation | None = None
        termination_reason = "timeout"
        failure_type: str | None = "timeout"
        legacy_success = False
        terminal_hold_steps = 0
        in_terminal_hold = False
        sim_steps = 0
        task_steps = 0
        max_task_steps = int(self.task_spec.get("max_steps", self.args.max_steps))
        control_hz = int(self.task_spec.get("control_hz", 10))
        physics_hz = int(self.task_spec.get("physics_hz", 60))
        physics_steps_per_action = max(1, physics_hz // control_hz)
        print(
            "[eval] Step accounting: "
            f"max_steps={max_task_steps} counts {control_hz} Hz action steps; "
            f"physics runs at {physics_hz} Hz "
            f"({physics_steps_per_action} physics steps/action); "
            "startup stabilization is excluded",
            flush=True,
        )
        # Safety guard only.  The task budget below is in model/control steps;
        # smooth trajectory frames are allowed without consuming it.
        max_sim_safety_steps = max((max_task_steps + 10) * 100, max_task_steps + 100)
        started = time.time()

        # ``task_steps`` counts original policy waypoints/control actions.  Do
        # not add smoothing frames (or an extra +10 grace window) to this
        # budget: once the 500th action has been processed, the episode is
        # allowed to finish through success/terminal logic but no 501st action
        # is requested.  ``max_sim_safety_steps`` remains an independent
        # guard against a backend that stops advancing the action clock.
        while task_steps < max_task_steps and sim_steps <= max_sim_safety_steps:
            try:
                step = self.backend.step()
            except PolicyOutputInvalid as exc:
                termination_reason = "policy_output_invalid"
                failure_type = "policy_output_invalid"
                print(f"Policy output invalid: {exc}", flush=True)
                break
            except Exception as exc:
                termination_reason = 'policy_request_failed' if isinstance(exc, PolicyClientError) else 'runtime_error'
                failure_type = termination_reason
                print(f"Evaluation step failed ({termination_reason}): {exc}", flush=True)
                import traceback
                traceback.print_exc()
                break
            sim_steps += 1
            if step.state is None or step.snapshot is None:
                continue
            stats = self.backend.policy_stats()
            executed_action_steps = stats.get("executed_action_steps")
            if executed_action_steps is None:
                # Backends without a chunk-aware clock (for example replay)
                # have one control action per backend step.
                task_steps += 1
            else:
                # The merged Franka backend exposes the number of original
                # chunk waypoints started by the policy engine.  Scripted
                # preconditions do not have a policy chunk, so their physical
                # frames remain in ``sim_steps`` but do not consume this
                # policy/task-step budget.
                task_steps = int(executed_action_steps)
            if in_terminal_hold:
                terminal_hold_steps += 1
            predicate_result = evaluator.update(step.snapshot)
            self.progress_reporter.observe(
                predicate_result,
                action_step=task_steps,
                sim_step=sim_steps,
                policy_calls=int(stats.get("policy_calls", 0)),
            )
            # Record the waypoint that is actually being executed.  Logging
            # chunk[0] on every frame hid boundary skips and made a forward
            # place trajectory look like repeated rollback in diagnostics.
            current_target = stats.get("current_action_target") or []
            if self._ee_action_space and current_target:
                target = np.asarray(current_target, dtype=np.float32)
                vla_action = {
                    "ee_pos": target[:3].tolist(),
                    "ee_quat": target[3:7].tolist(),
                    "gripper_raw": float(target[7]),
                }
            else:
                vla_action = {"joint_positions": current_target}
            trajectory.append(
                {
                    "step": task_steps,
                    "task_step": task_steps,
                    "sim_step": sim_steps,
                    "seed": seed,
                    "vla_action": vla_action,
                    "actual_state": self.backend.actual_state(),
                    "predicates": asdict(predicate_result),
                    "policy_calls": stats.get("policy_calls", 0),
                    "vla_anchor_offset": stats.get("last_anchor_offset") or [],
                }
            )
            video_stride = max(
                1,
                int(self.task_spec.get("physics_hz", 60))
                // int(self.task_spec.get("control_hz", 10)),
            )
            if self.args.save_video and (sim_steps - 1) % video_stride == 0:
                display = step.state.get("camera_display") or {}
                frame = next(iter(display.values()), None)
                if frame is not None:
                    array = np.asarray(frame)
                    if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
                        array = np.moveaxis(array[:3], 0, -1)
                    if array.ndim == 3:
                        video_frames.append(array.astype(np.uint8, copy=False))
            legacy_success = legacy_success or bool(step.legacy_success)
            if predicate_result.failure_type:
                termination_reason = "catastrophic_failure"
                failure_type = predicate_result.failure_type
                break
            if predicate_result.task_success:
                termination_reason = "success"
                failure_type = None
                break
            if step.legacy_done:
                if step.legacy_success:
                    self.backend.enter_terminal_hold()
                    in_terminal_hold = True
                    terminal_hold_steps = 1
                    if terminal_hold_steps > self.success_spec.hold_steps + 5:
                        termination_reason = "legacy_success_official_failure"
                        failure_type = "unknown_failure"
                        break
                else:
                    policy_calls = int(stats.get("policy_calls", 0))
                    scripted = str((self.task_spec.get("runtime") or {}).get("policy_phase", "")) == "after_scripted_pick"
                    if scripted and policy_calls == 0:
                        self.progress_reporter.observe(
                            predicate_result,
                            action_step=task_steps,
                            sim_step=sim_steps,
                            policy_calls=policy_calls,
                            force=True,
                            event_type="invalid_scene",
                            termination_reason="invalid_scripted_precondition",
                        )
                        return self._invalid_scene_row(
                            episode_index=episode_index,
                            episode_id=episode_id,
                            seed=seed,
                            attempt=attempt,
                            initial_conditions=initial_conditions,
                        )
                    termination_reason = step.termination_reason or "legacy_controller_failure"
                    failure_type = "timeout" if termination_reason == "timeout" else "unknown_failure"
                    break
            if in_terminal_hold and terminal_hold_steps > self.success_spec.hold_steps + 5:
                termination_reason = "legacy_success_official_failure"
                failure_type = "unknown_failure"
                break

        if termination_reason == 'runtime_error':
            row = self._invalid_scene_row(episode_index=episode_index, episode_id=episode_id,
                                          seed=seed, attempt=attempt, initial_conditions=initial_conditions)
            row.update(termination_reason='runtime_error', failure_type='runtime_error',
                       failure_code='runtime_error', failure_source='environment')
            return row
        if predicate_result is None:
            if failure_type in {"policy_output_invalid", "policy_request_failed"}:
                stats = self.backend.policy_stats()
                self.progress_reporter.runtime_failure(
                    termination_reason=termination_reason,
                    action_step=task_steps,
                    sim_step=sim_steps,
                    policy_calls=int(stats.get("policy_calls", 0)),
                )
                return self._policy_error_row(
                    episode_index=episode_index,
                    episode_id=episode_id,
                    seed=seed,
                    initial_conditions=initial_conditions,
                    termination_reason=termination_reason,
                    failure_type=failure_type,
                    episode_steps=task_steps,
                    sim_steps=sim_steps,
                )
            # No valid task state without a policy error is a setup condition.
            stats = self.backend.policy_stats()
            self.progress_reporter.runtime_failure(
                termination_reason="invalid_scripted_precondition",
                action_step=task_steps,
                sim_step=sim_steps,
                policy_calls=int(stats.get("policy_calls", 0)),
            )
            return self._invalid_scene_row(
                episode_index=episode_index,
                episode_id=episode_id,
                seed=seed,
                attempt=attempt,
                initial_conditions=initial_conditions,
            )

        final_stats = self.backend.policy_stats()
        self.progress_reporter.observe(
            predicate_result,
            action_step=task_steps,
            sim_step=sim_steps,
            policy_calls=int(final_stats.get("policy_calls", 0)),
            force=True,
            event_type="success" if predicate_result.task_success else "failure",
            termination_reason=termination_reason,
        )

        paths = self.recorder.save_episode(
            episode_id=episode_id,
            episode_index=episode_index,
            instruction=str((self.task_spec.get("instruction") or {}).get("canonical", "")),
            action_horizon=int(getattr(self.args, "action_horizon", 40)),
            max_steps=int(self.task_spec.get("max_steps", self.args.max_steps)),
            success_result=predicate_result,
            trajectory=trajectory,
            episode_steps=task_steps,
            sim_steps=sim_steps,
        )
        video_uri = None
        if video_frames:
            video_uri = self.recorder.save_video(
                episode_id=episode_id,
                success=bool(predicate_result.task_success),
                frames=video_frames,
                fps=float(self.task_spec.get("control_hz", 10)),
            )
        stats = self.backend.policy_stats()
        latency_metrics = self.metric_manager.latency_metrics(stats.get("latencies_ms") or [])
        trajectory_metrics = self.metric_manager.trajectory_metrics(trajectory)
        classification = classify_failure(
            success=bool(predicate_result.task_success),
            termination_reason=termination_reason,
            failure_type=failure_type,
            success_result=predicate_result,
        )
        success = bool(predicate_result.task_success)
        return {
            "run_id": self.run_id,
            "episode_id": episode_id,
            "task_id": self.task_spec["task_id"],
            "task_name": self.task_spec.get("task_name"),
            "task_family": self.task_metadata["task_family"],
            "task_level": self.task_metadata["task_level"],
            "task_tags": self.task_metadata["task_tags"],
            "leaderboard_group": self.task_metadata["leaderboard_group"],
            "split": self.args.split,
            "seed": seed,
            "valid_scene": termination_reason not in {'runtime_error', 'policy_request_failed'},
            "success": success,
            "official_success": success,
            "legacy_success": bool(legacy_success),
            "success_disagreement": success != bool(legacy_success),
            "success_logic": predicate_result.success_logic,
            "goal_pass": predicate_result.goal_pass,
            "sequence_pass": predicate_result.sequence_pass,
            "sequence_stage": predicate_result.sequence_stage,
            "terminal_pass": predicate_result.terminal_pass,
            "stability_pass": predicate_result.stability_pass,
            "safety_valid": predicate_result.safety_valid,
            "hard_failure": predicate_result.hard_failure,
            "latched_violations": predicate_result.latched_violations,
            "termination_reason": termination_reason,
            "failure_type": failure_type,
            **classification.to_dict(),
            "completed_stage": predicate_result.completed_stage,
            "stage_completion": predicate_result.stage_completion,
            "quality_metrics": predicate_result.quality_metrics,
            "predicate_values": asdict(predicate_result).get("predicate_values", {}),
            "goal_predicates": predicate_result.goal_predicates,
            "terminal_predicates": predicate_result.terminal_predicates,
            "stability_predicates": predicate_result.stability_predicates,
            "safety_predicates": predicate_result.safety_predicates,
            "episode_steps": task_steps,
            "sim_steps": sim_steps,
            "policy_calls": stats.get("policy_calls", 0),
            "episode_time_sec": round(time.time() - started, 4),
            "inference_latency_p50_ms": latency_metrics.get("inference_latency_p50_ms"),
            "inference_latency_p95_ms": latency_metrics.get("inference_latency_p95_ms"),
            "path_length_m": trajectory_metrics.get("path_length_m"),
            "eef_jerk_rms": trajectory_metrics.get("eef_jerk_rms"),
            "joint_jerk_rms": trajectory_metrics.get("joint_jerk_rms"),
            "env_config": (self.task_spec.get("runtime") or {}).get("env_config"),
            "policy_profile": self.policy_metadata.get("profile_id"),
            "policy_server_metadata": self.policy_metadata.get("server_metadata", {}),
            "initial_conditions": initial_conditions,
            "video_uri": video_uri,
            "trajectory_uri": paths["trajectory_npz"],
            "legacy_log_uri": paths["legacy_json"],
            "object_lifted": predicate_result.object_lifted,
            "grasped": predicate_result.grasped,
            "gripper_closed": predicate_result.gripper_closed,
            "object_near_hand": predicate_result.object_near_hand,
            "stable": predicate_result.stable,
            "hand_object_distance": predicate_result.hand_object_distance,
            "object_height_change": predicate_result.object_height_change,
            "grasp_attempts": 0,
        }

    def _policy_error_row(
        self,
        *,
        episode_index: int,
        episode_id: str,
        seed: int,
        initial_conditions: dict[str, Any],
        termination_reason: str,
        failure_type: str,
        episode_steps: int,
        sim_steps: int,
    ) -> dict[str, Any]:
        stats = self.backend.policy_stats()
        return {
            "run_id": self.run_id,
            "episode_id": episode_id,
            "episode_index": episode_index,
            "task_id": self.task_spec["task_id"],
            "task_name": self.task_spec.get("task_name"),
            "task_family": self.task_metadata["task_family"],
            "task_level": self.task_metadata["task_level"],
            "task_tags": self.task_metadata["task_tags"],
            "leaderboard_group": self.task_metadata["leaderboard_group"],
            "split": self.args.split,
            "seed": seed,
            "valid_scene": termination_reason != 'policy_request_failed',
            "success": False,
            "official_success": False,
            "legacy_success": False,
            "success_disagreement": False,
            "success_logic": self.success_spec.logic,
            "termination_reason": termination_reason,
            "failure_type": failure_type,
            "failure_category": "policy_error",
            "failure_code": failure_type,
            "failure_source": "policy",
            "failure_severity": "episode",
            "episode_steps": episode_steps,
            "sim_steps": sim_steps,
            "policy_calls": stats.get("policy_calls", 0),
            "env_config": (self.task_spec.get("runtime") or {}).get("env_config"),
            "policy_profile": self.policy_metadata.get("profile_id"),
            "policy_server_metadata": self.policy_metadata.get("server_metadata", {}),
            "initial_conditions": initial_conditions,
        }

    def _invalid_scene_row(
        self,
        *,
        episode_index: int,
        episode_id: str,
        seed: int,
        attempt: int,
        initial_conditions: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "episode_id": episode_id,
            "task_id": self.task_spec["task_id"],
            "task_name": self.task_spec.get("task_name"),
            "task_family": self.task_metadata["task_family"],
            "task_level": self.task_metadata["task_level"],
            "task_tags": self.task_metadata["task_tags"],
            "leaderboard_group": self.task_metadata["leaderboard_group"],
            "split": self.args.split,
            "seed": seed,
            "valid_scene": False,
            "success": False,
            "official_success": False,
            "legacy_success": False,
            "success_disagreement": False,
            "success_logic": self.success_spec.logic,
            "termination_reason": "invalid_scripted_precondition",
            "failure_type": "invalid_scene",
            "failure_category": "unknown_failure",
            "failure_code": "invalid_scripted_precondition",
            "failure_source": "environment",
            "failure_severity": "invalid",
            "episode_steps": 0,
            "sim_steps": 0,
            "policy_calls": 0,
            "attempt": attempt + 1,
            "env_config": (self.task_spec.get("runtime") or {}).get("env_config"),
            "policy_profile": self.policy_metadata.get("profile_id"),
            "initial_conditions": initial_conditions,
        }

    def _run_manifest(self) -> dict[str, Any]:
        runtime = self.task_spec.get("runtime") or {}
        return {
            "run_id": self.run_id,
            "eval_group_id": self.eval_group_id,
            "run_type": getattr(self.args, "run_type", "single_task"),
            "suite_id": getattr(self.args, "suite_id", None),
            "benchmark_version": "EAI4S-benchmark-1.0.0",
            "labbench_code_version": __version__,
            "backend": "eai4s_benchmark",
            "env_config": runtime.get("env_config"),
            "model_name": self.args.model_name,
            "checkpoint": self.args.checkpoint,
            "policy_provider": getattr(self.args, "policy_provider", None),
            "policy_backend": getattr(self.args, "policy_backend", None),
            "checkpoint_hash": getattr(self.args, "checkpoint_hash", None),
            "dataset_version": getattr(self.args, "dataset_version", None),
            "robot": self.task_spec.get("robot", "franka"),
            "task_id": self.task_spec["task_id"],
            "task_family": self.task_metadata["task_family"],
            "task_level": self.task_metadata["task_level"],
            "task_tags": self.task_metadata["task_tags"],
            "leaderboard_group": self.task_metadata["leaderboard_group"],
            "failure_modes": self.task_metadata["failure_modes"],
            "control_mode": "ee_position_8d" if self._ee_action_space else "joint_position_8d",
            "control_hz": self.task_spec.get("control_hz", 10),
            "physics_hz": self.task_spec.get("physics_hz", 60),
            "predicate_hz": self.task_spec.get("predicate_hz", 60),
            "progress_logging": {
                "interval_sec": float(getattr(self.args, "progress_interval_sec", 1.0)),
                "format": str(getattr(self.args, "progress_log_format", "both")),
                "schema_version": "eai4s.eval_progress.v1",
            },
            "sim_steps_per_action": int(
                self.task_spec.get("physics_hz", 60)
                // self.task_spec.get("control_hz", 10)
            ),
            "observation_track": "eai4s_hdf5_v1",
            "action_chunk": getattr(self.args, "action_horizon", 40),
            "success_predicates": self.success_spec.to_dict(),
            "policy": self.policy_metadata,
            "split": self.args.split,
            "seeds": self.seeds,
            "seed_pool": self.seeds,
            "requested_valid_episodes": int(self.args.num_episodes),
            "created_at": datetime.now().astimezone().isoformat(),
            "tuned_on_eval": False,
        }

    def _write_progress(self, completed: int, total: int, row: dict[str, Any], *, attempted: int) -> None:
        payload = {
            "run_id": self.run_id,
            "status": "running" if completed < total else "completed",
            "completed_episodes": completed,
            "total_episodes": total,
            "attempted_scenes": attempted,
            "last_episode_id": row.get("episode_id"),
            "last_episode_success": row.get("success"),
            "updated_at": datetime.now().astimezone().isoformat(),
        }
        (self.output_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
