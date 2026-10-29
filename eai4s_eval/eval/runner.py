from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from eai4s_eval import __version__
from eai4s_eval.eval.action_manager import ActionManager
from eai4s_eval.eval.aggregator import RunAggregator
from eai4s_eval.eval.metric_manager import TaskMetricManager
from eai4s_eval.eval.observation_manager import ObservationManager
from eai4s_eval.eval.recorder import EpisodeRecorder
from eai4s_eval.eval.success import SuccessEvaluation, TaskSuccessEvaluator
from eai4s_eval.eval.success.failure_taxonomy import classify_failure
from eai4s_eval.eval.task_spec import apply_success_cli_overrides_to_spec, success_spec_from_task_spec, task_metadata_from_task_spec
from eai4s_eval.policy import PolicyOutputInvalid, VLAHttpPolicyClient
from eai4s_eval.robots.base import RobotAdapter
from eai4s_eval.scenes.registry import get_scene_runtime
from eai4s_eval.utils.io import ensure_dir


def default_task_spec(robot_id: str, instruction: str, max_steps: int, success_hold_steps: int | None) -> dict[str, Any]:
    hold_steps = success_hold_steps or 10
    return {
        "task_id": "T02_franka_pick_lift_proxy",
        "task_name": "Franka pick/lift proxy",
        "task_version": "0.1.0",
        "suite": "eai4s_eval_v0_1",
        "scene_id": "tabletop_pick_lift_proxy",
        "robot": robot_id,
        "task_family": "pick_place",
        "task_level": "base",
        "task_tags": ["pick", "lift", "single_object"],
        "leaderboard_group": "basic_pick_place",
        "control_hz": 10,
        "max_steps": max_steps,
        "success_hold_steps": hold_steps,
        "instruction": {"canonical": instruction, "variants": ["pick the vial", "grasp and lift the glass bottle"]},
        "success": {
            "logic": "robust_semantic_v0",
            "hold_steps": hold_steps,
            "goal": {
                "all": [
                    {"predicate": "grasped", "args": {"grip_threshold": 0.04, "distance_threshold_m": 0.15}},
                    {"predicate": "lifted", "args": {"height_threshold_m": 0.05}},
                ],
            },
            "terminal": {"required": False, "all": []},
            "stability": {
                "required": False,
                "all": [
                    {
                        "predicate": "object_velocity_below",
                        "args": {"linear_velocity_threshold_mps": 0.05, "angular_velocity_threshold_radps": 0.50},
                    }
                ],
            },
            "safety": {"hard_fail": [{"predicate": "object_out_of_workspace"}]},
        },
        "failure_modes": [
            {"code": "timeout", "category": "timeout", "description": "Episode exceeded max_steps."},
            {"code": "object_out_of_workspace", "category": "drop", "description": "Target object left the valid workspace."},
            {"code": "policy_output_invalid", "category": "policy_error", "description": "Policy returned an invalid action."},
            {"code": "policy_request_failed", "category": "policy_error", "description": "Policy server request failed."},
        ],
    }


def load_task_spec(path: str | None, robot_id: str, instruction: str, max_steps: int, success_hold_steps: int | None) -> dict[str, Any]:
    if not path:
        return default_task_spec(robot_id, instruction, max_steps, success_hold_steps)
    task_path = Path(path)
    if not task_path.exists():
        return default_task_spec(robot_id, instruction, max_steps, success_hold_steps)
    try:
        import yaml  # type: ignore

        with task_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            data.setdefault("robot", robot_id)
            data.setdefault("max_steps", max_steps)
            data.setdefault("success_hold_steps", success_hold_steps or 10)
            return data
    except Exception:
        pass
    return default_task_spec(robot_id, instruction, max_steps, success_hold_steps)


def parse_seeds(seeds: str | None, seed_manifest: str | None, split: str) -> list[int]:
    if seeds:
        return [int(item.strip()) for item in seeds.split(",") if item.strip()]
    # Auto-resolve seed manifest from split if not explicitly provided
    resolved = seed_manifest
    if not resolved:
        auto_path = Path(__file__).resolve().parents[1] / "configs" / f"seed_manifest_{split}.json"
        if auto_path.exists():
            resolved = str(auto_path)
    if resolved and Path(resolved).exists():
        with Path(resolved).open("r", encoding="utf-8-sig") as f:
            manifest = json.load(f)
        for task in manifest.get("tasks", []):
            task_seeds = [int(item["seed"]) for item in task.get("seeds", []) if item.get("valid_scene", True)]
            if task_seeds:
                return task_seeds
    if split == "sim_dev":
        return [100001, 100002, 100003, 100004, 100005]
    return [100001]


def _resolve_seed_manifest_for_split(split: str) -> str | None:
    manifest_path = Path(__file__).resolve().parents[1] / "configs" / f"seed_manifest_{split}.json"
    if manifest_path.exists():
        return str(manifest_path)
    fallback = Path(__file__).resolve().parents[1] / "configs" / "seed_manifest_sim_dev.json"
    return str(fallback) if fallback.exists() else None


class TaskEvalRunner:
    def __init__(self, *, args: Any, simulation_app: Any, robot_adapter: RobotAdapter):
        self.args = args
        self.simulation_app = simulation_app
        self.adapter = robot_adapter
        self.device = getattr(args, "device", "cuda:0")
        self.eval_time_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not getattr(args, "checkpoint", None):
            self.args.checkpoint = self.eval_time_id
        self.run_id = args.run_id or f"{self.eval_time_id}_{args.model_name}"
        self.eval_group_id = getattr(args, "eval_group_id", None) or self.run_id
        self.output_dir = ensure_dir(Path(args.output_root) / self.run_id)
        self.task_spec = load_task_spec(
            args.task_spec,
            args.robot,
            args.instruction,
            args.max_steps,
            args.success_hold_steps,
        )
        self.task_metadata = task_metadata_from_task_spec(self.task_spec)
        self.scene_runtime = get_scene_runtime(self.task_spec.get("scene_id"))
        self.target_object_name = self.scene_runtime.resolve_target_object_name(getattr(args, "target_object", None))
        self.success_spec = apply_success_cli_overrides_to_spec(success_spec_from_task_spec(self.task_spec), args)
        self.success_criteria = self.success_spec
        self.seeds = parse_seeds(args.seeds, args.seed_manifest, args.split)[: args.num_episodes]
        self.recorder = EpisodeRecorder(self.output_dir)
        self.policy_client = VLAHttpPolicyClient(
            host=args.vla_host,
            port=args.vla_port,
            api=args.policy_api,
            timeout_sec=args.policy_timeout_sec,
        )
        self.metric_manager = TaskMetricManager()
        self.predicate_manager = TaskSuccessEvaluator(self.success_spec)
        self.sim = None
        self.scene = None
        self.resolved_robot = None
        self.robot = None
        self.target_object = None
        self.initial_object_pos = None
        self.original_object_pos = None
        self.original_object_quat = None
        self.action_manager = None
        self.observation_manager = None

        if int(self.args.control_hz) != 10:
            raise ValueError(
                f"legacy evaluation control_hz must be 10, got {self.args.control_hz}"
            )
        effective_physics_steps = int(self.args.reaction_interpolation_steps) * int(
            self.args.sim_steps_per_action
        )
        if effective_physics_steps != 6:
            raise ValueError(
                "10 Hz evaluation over 60 Hz physics requires "
                "reaction_interpolation_steps * sim_steps_per_action == 6, got "
                f"{effective_physics_steps}"
            )

    def run(self) -> dict[str, Any]:
        self._setup_sim()
        return self._run_episodes_for_current_split()

    def run_all_splits(self) -> dict[str, Any]:
        splits = ["sim_dev", "sim_clean", "sim_randomized"]
        base_run_id = self.run_id
        self.eval_group_id = getattr(self.args, "eval_group_id", None) or base_run_id
        all_results: dict[str, Any] = {}
        for idx, split in enumerate(splits):
            self.args.split = split
            self.args.seed_manifest = _resolve_seed_manifest_for_split(split)
            self.seeds = parse_seeds(self.args.seeds, self.args.seed_manifest, self.args.split)[: self.args.num_episodes]
            self.run_id = base_run_id if idx == 0 else f"{base_run_id}_{split}"
            self.output_dir = ensure_dir(Path(self.args.output_root) / self.run_id)
            self.recorder = EpisodeRecorder(self.output_dir)
            print(f"\n{'='*60}", flush=True)
            print(f"Starting split: {split}  (run_id: {self.run_id})", flush=True)
            print(f"{'='*60}", flush=True)
            if idx == 0:
                self._setup_sim()
            else:
                self.sim.reset()
                self.adapter.reset_robot(self.robot)
                self.target_object = self.scene[self.target_object_name]
                self.original_object_pos = self.target_object.data.root_state_w[0, :3].clone()
                self.original_object_quat = self.target_object.data.root_state_w[0, 3:7].clone()
                self.scene_runtime.after_setup(scene=self.scene, sim=self.sim, device=self.device)
                self._cache_distractor_positions()
                for _ in range(10):
                    self._step_sim()
            all_results[split] = self._run_episodes_for_current_split()
        return all_results

    def _run_episodes_for_current_split(self) -> dict[str, Any]:
        manifest = self._run_manifest()
        self.recorder.write_run_manifest(manifest)
        self.recorder.write_run_config(self._run_config())
        rows: list[dict[str, Any]] = []
        for index, seed in enumerate(self.seeds):
            episode_id = f"{self.task_spec['task_id']}_{self.args.split}_seed{seed}"
            print(f"\n=== Running {episode_id} ({index + 1}/{len(self.seeds)}) ===", flush=True)
            row = self._run_episode(index, seed, episode_id)
            rows.append(row)
            if not row.get("success"):
                self.recorder.append_failure_case(row)
        result = RunAggregator(self.output_dir).write_all(
            episode_rows=rows,
            run_manifest=manifest,
            task_spec=self.task_spec,
        )
        for line in result.get("summary_lines", []):
            print(line, flush=True)
        print(f"Summary log: {result.get('summary_log')}", flush=True)
        print(f"\nEvaluation complete: {self.output_dir}", flush=True)
        return result

    def _setup_sim(self) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0 / 60.0,
            device=self.device,
            render=sim_utils.RenderCfg(enable_translucency=True),
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
        scene_cfg = self.scene_runtime.build_scene_cfg(
            robot_adapter=self.adapter,
            asset_root=self.args.asset_root,
            num_envs=self.args.num_envs,
        )
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()
        self.resolved_robot = self.adapter.resolve_scene_entities(self.scene)
        self.robot = self.resolved_robot.robot
        self.scene_runtime.after_setup(scene=self.scene, sim=self.sim, device=self.device)
        self.target_object = self.scene[self.target_object_name]
        self.adapter.reset_robot(self.robot)
        controller = self.adapter.create_controller(self.args.num_envs, self.device)
        self.action_manager = ActionManager(
            adapter=self.adapter,
            robot=self.robot,
            controller=controller,
            arm_entity_cfg=self.resolved_robot.arm_entity_cfg,
            gripper_entity_cfg=self.resolved_robot.gripper_entity_cfg,
            num_envs=self.args.num_envs,
            device=self.device,
            max_delta_xyz=self.args.max_delta_xyz,
        )
        self.observation_manager = ObservationManager(
            scene=self.scene,
            resolved_robot=self.resolved_robot,
            instruction=self.args.instruction,
        )
        for _ in range(10):
            self._step_sim()
        self.original_object_pos = self.target_object.data.root_state_w[0, :3].clone()
        self.original_object_quat = self.target_object.data.root_state_w[0, 3:7].clone()
        self._cache_distractor_positions()
        self._dome_light_path = "/World/Light"

    def _cache_distractor_positions(self) -> None:
        """Cache original positions of distractor (non-target) objects for randomization."""
        distractor_names = list(self.scene_runtime.distractor_names)
        self.distractor_objects: dict[str, Any] = {}
        self.distractor_original_pos: dict[str, Any] = {}
        for name in distractor_names:
            if name == self.target_object_name:
                continue
            try:
                obj = self.scene[name]
            except KeyError:
                continue
            self.distractor_objects[name] = obj
            self.distractor_original_pos[name] = obj.data.root_state_w[0, :3].clone()

    def _run_episode(self, episode_index: int, seed: int, episode_id: str) -> dict[str, Any]:
        assert self.action_manager is not None
        self._reset_episode(seed)
        stabilization_info = self._hold_initial_open_gripper()
        self.predicate_manager.reset()
        trajectory: list[dict[str, Any]] = []
        latencies: list[float] = []
        video_frames: list[np.ndarray] = [] if self.args.save_video else []
        termination_reason = "timeout"
        failure_type: str | None = "timeout"
        predicate_result: SuccessEvaluation | None = None
        steps = 0
        start_time = time.time()

        while self.simulation_app.is_running() and steps < self.args.max_steps:
            observation = self.observation_manager.get_observation()

            # Collect a video frame at the start of each control cycle so the
            # first frame always matches the observation that the VLA receives.
            if self.args.save_video:
                rgb = observation["legacy"]["video.main_camera"]
                if isinstance(rgb, np.ndarray) and rgb.ndim >= 3:
                    video_frames.append(rgb[0].copy() if rgb.shape[0] == 1 else rgb.copy())
            try:
                batch = self.policy_client.predict(
                    observation=observation,
                    run_id=self.run_id,
                    episode_id=episode_id,
                    task_id=self.task_spec["task_id"],
                    instruction=self.args.instruction,
                    step_index=steps,
                    control_hz=self.args.control_hz,
                )
                latencies.append(batch.latency_ms)
                self._maybe_log_policy_batch(steps, batch)
            except PolicyOutputInvalid as exc:
                termination_reason = "policy_output_invalid"
                failure_type = "policy_output_invalid"
                print(f"Policy output invalid: {exc}", flush=True)
                break
            except Exception as exc:
                termination_reason = "policy_request_failed"
                failure_type = "policy_request_failed"
                print(f"Policy request failed: {exc}", flush=True)
                break

            horizon = min(self.args.action_horizon, batch.action_horizon)
            for horizon_index in range(horizon):
                if steps >= self.args.max_steps:
                    break
                vla_action = self._execute_policy_action(batch, horizon_index)
                actual_state = self._actual_state()
                snapshot = self._predicate_snapshot(steps)
                predicate_result = self.predicate_manager.update(snapshot)
                trajectory.append(
                    {
                        "step": steps,
                        "seed": seed,
                        "policy_endpoint": batch.endpoint,
                        "vla_action": vla_action,
                        "action_execution": {
                            "mode": "reaction_interpolation",
                            "interpolation_steps": self.args.reaction_interpolation_steps,
                            "sim_steps_per_action": self.args.sim_steps_per_action,
                            "hold_current_orientation": not self.args.use_policy_orientation,
                        },
                        "gripper_cmd_sent": actual_state["gripper_pos"],
                        "actual_state": actual_state,
                        "predicates": asdict(predicate_result),
                    }
                )
                self._maybe_log_predicate_progress(steps, predicate_result)
                steps += 1
                # Capture a frame after the action is executed (horizon iteration)
                if self.args.save_video:
                    rgb = self.observation_manager.get_observation()["legacy"]["video.main_camera"]
                    if isinstance(rgb, np.ndarray) and rgb.ndim >= 3:
                        video_frames.append(rgb[0].copy() if rgb.shape[0] == 1 else rgb.copy())
                if predicate_result.failure_type:
                    termination_reason = "catastrophic_failure"
                    failure_type = predicate_result.failure_type
                    break
                if predicate_result.task_success:
                    termination_reason = "success"
                    failure_type = None
                    print(
                        "Success predicate satisfied: "
                        f"height={predicate_result.object_height_change:.4f}m, "
                        f"distance={predicate_result.hand_object_distance:.4f}m, "
                        f"grasped={predicate_result.grasped}, "
                        f"hold={predicate_result.success_hold_counter}/{self.success_spec.hold_steps}",
                        flush=True,
                    )
                    break
            if termination_reason in {"success", "catastrophic_failure"}:
                break

        if predicate_result is None:
            snapshot = self._predicate_snapshot(steps)
            predicate_result = self.predicate_manager.update(snapshot)
        if termination_reason == "timeout" and predicate_result.failure_type:
            failure_type = predicate_result.failure_type
        success_result = predicate_result
        paths = self.recorder.save_episode(
            episode_id=episode_id,
            episode_index=episode_index,
            instruction=self.args.instruction,
            action_horizon=self.args.action_horizon,
            max_steps=self.args.max_steps,
            success_result=success_result,
            trajectory=trajectory,
        )

        episode_duration = time.time() - start_time
        # Save video if --save_video is enabled
        video_uri: str | None = None
        if self.args.save_video and video_frames:
            # Use actual episode wall-clock duration so the video plays at
            # approximately real-time speed.
            video_fps = max(1.0, len(video_frames) / max(episode_duration, 0.1))
            video_uri = self.recorder.save_video(
                episode_id=episode_id,
                success=bool(success_result.task_success),
                frames=video_frames,
                fps=video_fps,
            )

        traj_metrics = self.metric_manager.trajectory_metrics(trajectory)
        latency_metrics = self.metric_manager.latency_metrics(latencies)
        quality_metrics = success_result.quality_metrics or {}
        failure_classification = classify_failure(
            success=bool(success_result.task_success),
            termination_reason=termination_reason,
            failure_type=failure_type,
            success_result=success_result,
        )
        row = {
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
            "valid_scene": True,
            "success": bool(success_result.task_success),
            "success_logic": success_result.success_logic,
            "goal_pass": success_result.goal_pass,
            "sequence_pass": success_result.sequence_pass,
            "sequence_stage": success_result.sequence_stage,
            "terminal_pass": success_result.terminal_pass,
            "stability_pass": success_result.stability_pass,
            "safety_valid": success_result.safety_valid,
            "hard_failure": success_result.hard_failure,
            "latched_violations": success_result.latched_violations,
            "termination_reason": termination_reason,
            "failure_type": failure_type,
            **failure_classification.to_dict(),
            "completed_stage": success_result.completed_stage,
            "stage_completion": success_result.stage_completion,
            "quality_metrics": success_result.quality_metrics,
            "episode_steps": steps,
            "episode_time_sec": round(time.time() - start_time, 4),
            "initial_stabilized": stabilization_info["stable"],
            "initial_hold_steps": stabilization_info["initial_hold_steps"],
            "initial_stabilization_steps": stabilization_info["adaptive_steps"],
            "initial_stabilization_hold_count": stabilization_info["stable_hold_count"],
            "final_position_error_m": quality_metrics.get("insertion_position_error_m"),
            "final_rotation_error_deg": None,
            "insert_depth_m": quality_metrics.get("insertion_depth_m"),
            "liquid_volume_error_ml": None,
            "spill_ratio": None,
            "collision_count": 0,
            "self_collision_count": 0,
            "max_collision_impulse": None,
            "joint_jerk_rms": traj_metrics.get("joint_jerk_rms"),
            "eef_jerk_rms": traj_metrics.get("eef_jerk_rms"),
            "path_length_m": traj_metrics.get("path_length_m"),
            "bimanual_sync_score": None,
            "safety_response_latency_steps": None,
            "inference_latency_p50_ms": latency_metrics.get("inference_latency_p50_ms"),
            "inference_latency_p95_ms": latency_metrics.get("inference_latency_p95_ms"),
            "video_uri": video_uri,
            "trajectory_uri": paths["trajectory_npz"],
            "legacy_log_uri": paths["legacy_json"],
            "object_lifted": success_result.object_lifted,
            "grasped": success_result.grasped,
            "gripper_closed": success_result.gripper_closed,
            "object_near_hand": success_result.object_near_hand,
            "stable": success_result.stable,
            "hand_object_distance": success_result.hand_object_distance,
            "object_height_change": success_result.object_height_change,
            "predicate_values": asdict(success_result).get("predicate_values", {}),
            "goal_predicates": success_result.goal_predicates,
            "terminal_predicates": success_result.terminal_predicates,
            "stability_predicates": success_result.stability_predicates,
            "safety_predicates": success_result.safety_predicates,
            "grasp_attempts": self.metric_manager.grasp_attempts(trajectory),
        }
        status = "SUCCESS" if row["success"] else "FAILED"
        print(f"{episode_id}: {status}, termination={termination_reason}, steps={steps}", flush=True)
        return row

    def _maybe_log_policy_batch(self, steps: int, batch: Any) -> None:
        interval = max(0, int(getattr(self.args, "policy_log_interval", 10)))
        if interval == 0 or steps % interval != 0:
            return
        first_pos = batch.ee_pos[0].tolist() if batch.ee_pos.shape[0] else []
        first_gripper = float(batch.gripper[0]) if batch.gripper.shape[0] else None
        print(
            "Policy batch "
            f"step={steps}: endpoint={batch.endpoint}, horizon={batch.action_horizon}, "
            f"latency={batch.latency_ms:.1f}ms, first_ee_pos={first_pos}, first_gripper={first_gripper}",
            flush=True,
        )

    def _maybe_log_predicate_progress(self, steps: int, result: SuccessEvaluation) -> None:
        interval = max(0, int(self.args.predicate_log_interval))
        if interval == 0 or steps % interval != 0:
            return
        print(
            "Predicate progress "
            f"step={steps}: grasped={result.grasped}, lifted={result.object_lifted}, near={result.object_near_hand}, "
            f"gripper_closed={result.gripper_closed}, stable={result.stable}, "
            f"height={result.object_height_change:.4f}m, distance={result.hand_object_distance:.4f}m, "
            f"hold={result.success_hold_counter}/{self.success_spec.hold_steps}",
            flush=True,
        )

    def _execute_policy_action(self, batch: Any, horizon_index: int) -> dict[str, Any]:
        """Execute one VLA action using the reaction-style interpolation path.

        The updated standalone client divides each VLA target position into
        several intermediate targets and advances the simulator several physics
        steps per intermediate target.  By default we keep the current
        end-effector orientation, matching that client.
        """
        assert self.action_manager is not None
        self.action_manager.validate(batch)
        idx = min(horizon_index, batch.ee_pos.shape[0] - 1)
        interpolation_steps = max(1, int(self.args.reaction_interpolation_steps))
        sim_steps_per_action = max(1, int(self.args.sim_steps_per_action))
        start_pos, current_quat = self.action_manager.current_ee_pose()
        target_pos = np.asarray(batch.ee_pos[idx], dtype=np.float32)
        policy_quat = np.asarray(batch.ee_quat[idx], dtype=np.float32)
        target_quat = policy_quat if self.args.use_policy_orientation else current_quat
        raw_gripper = float(batch.gripper[idx])
        last_action: dict[str, Any] | None = None

        for interp_index in range(interpolation_steps):
            ratio = float(interp_index + 1) / float(interpolation_steps)
            interp_pos = start_pos + (target_pos - start_pos) * ratio
            last_action = self.action_manager.apply_target(
                target_pos=interp_pos,
                target_quat=target_quat,
                raw_gripper=raw_gripper,
            )
            for _ in range(sim_steps_per_action):
                self._step_sim()

        assert last_action is not None
        last_action["policy_target_ee_pos"] = target_pos.tolist()
        last_action["policy_target_ee_quat"] = policy_quat.tolist()
        return last_action

    def _reset_episode(self, seed: int) -> None:
        assert self.sim is not None and self.robot is not None and self.target_object is not None
        self.scene_runtime.reset_episode(
            scene=self.scene,
            sim=self.sim,
            robot=self.robot,
            adapter=self.adapter,
            target_object=self.target_object,
            original_object_pos=self.original_object_pos,
            original_object_quat=self.original_object_quat,
            seed=seed,
            split=self.args.split,
            device=self.device,
            step_sim=self._step_sim,
            distractor_objects=getattr(self, "distractor_objects", {}),
            distractor_original_pos=getattr(self, "distractor_original_pos", {}),
        )

        # Lighting intensity variation is shared across scene runtimes.
        if self.args.split == "sim_randomized" and hasattr(self, "_dome_light_path"):
            rng = np.random.default_rng(seed)
            intensity_mult = float(rng.uniform(0.8, 1.2))
            try:
                import omni.usd

                stage = omni.usd.get_context().get_stage()
                light_prim = stage.GetPrimAtPath(self._dome_light_path)
                if light_prim.IsValid():
                    attr = light_prim.GetAttribute("inputs:intensity")
                    if attr.IsValid():
                        attr.Set(3000.0 * intensity_mult)
            except Exception:
                pass

        self.initial_object_pos = self.target_object.data.root_state_w[0, :3].clone()

    def _hold_initial_open_gripper(self) -> dict[str, Any]:
        assert self.action_manager is not None and self.robot is not None
        joint_pos_des = self.robot.data.joint_pos[:, self.resolved_robot.arm_entity_cfg.joint_ids].clone()
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.resolved_robot.arm_entity_cfg.joint_ids)
        self.robot.set_joint_position_target(
            self.action_manager.open_tensor,
            joint_ids=self.resolved_robot.gripper_entity_cfg.joint_ids,
        )
        initial_hold_steps = max(0, int(self.args.initial_hold_steps))
        for _ in range(initial_hold_steps):
            self._step_sim()
        stabilization_info = self._wait_for_initial_stability()
        stabilization_info["initial_hold_steps"] = initial_hold_steps
        return stabilization_info

    def _wait_for_initial_stability(self) -> dict[str, Any]:
        max_steps = max(0, int(self.args.stabilization_max_steps))
        required_hold_steps = max(1, int(self.args.stabilization_hold_steps))
        thresholds = {
            "object_lin_vel": float(self.args.stabilization_object_lin_vel_threshold),
            "object_ang_vel": float(self.args.stabilization_object_ang_vel_threshold),
            "joint_vel": float(self.args.stabilization_joint_vel_threshold),
        }
        info: dict[str, Any] = {
            "stable": False,
            "adaptive_steps": 0,
            "stable_hold_count": 0,
            "required_hold_steps": required_hold_steps,
            "thresholds": thresholds,
            "last_observed": {},
        }
        if max_steps == 0:
            info["stable"] = True
            return info

        stable_hold_count = 0
        for step_index in range(max_steps):
            self._step_sim()
            observed = self._initial_stability_observation()
            stable = (
                observed["object_lin_vel_norm"] <= thresholds["object_lin_vel"]
                and observed["object_ang_vel_norm"] <= thresholds["object_ang_vel"]
                and observed["joint_vel_norm"] <= thresholds["joint_vel"]
            )
            stable_hold_count = stable_hold_count + 1 if stable else 0
            info["adaptive_steps"] = step_index + 1
            info["stable_hold_count"] = stable_hold_count
            info["last_observed"] = observed
            if stable_hold_count >= required_hold_steps:
                info["stable"] = True
                return info

        print(
            "Initial stability wait timed out: "
            f"observed={info['last_observed']}, thresholds={thresholds}, "
            f"hold={stable_hold_count}/{required_hold_steps}",
            flush=True,
        )
        return info

    def _initial_stability_observation(self) -> dict[str, float]:
        import torch

        root_state = self.target_object.data.root_state_w[0]
        object_lin_vel = root_state[7:10] if root_state.shape[0] >= 10 else torch.zeros(3, device=self.device)
        object_ang_vel = root_state[10:13] if root_state.shape[0] >= 13 else torch.zeros(3, device=self.device)
        joint_vel = getattr(self.robot.data, "joint_vel", None)
        if joint_vel is None:
            joint_vel_norm = 0.0
        else:
            joint_vel_norm = float(torch.linalg.norm(joint_vel[0]).detach().cpu().item())
        return {
            "object_lin_vel_norm": float(torch.linalg.norm(object_lin_vel).detach().cpu().item()),
            "object_ang_vel_norm": float(torch.linalg.norm(object_ang_vel).detach().cpu().item()),
            "joint_vel_norm": joint_vel_norm,
        }

    def _step_sim(self) -> None:
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim.get_physics_dt())

    def _actual_state(self) -> dict[str, Any]:
        ee_pose = self.robot.data.body_state_w[0, self.resolved_robot.arm_entity_cfg.body_ids[0]]
        gripper_pos = self.robot.data.joint_pos[0, self.resolved_robot.gripper_entity_cfg.joint_ids]
        joint_pos = self.robot.data.joint_pos[0].detach().cpu().numpy().tolist()
        gripper_list = gripper_pos.detach().cpu().numpy().tolist()
        return {
            "ee_pos": ee_pose[:3].detach().cpu().numpy().tolist(),
            "ee_quat": ee_pose[3:7].detach().cpu().numpy().tolist(),
            "joint_pos": joint_pos,
            "gripper_pos": gripper_list,
            "gripper_mapped": 1.0 if gripper_list and gripper_list[0] > 0.0 else 0.0,
        }

    def _predicate_snapshot(self, step_index: int | None = None) -> dict[str, Any]:
        snapshot = self.scene_runtime.build_success_snapshot(
            scene=self.scene,
            target_object=self.target_object,
            initial_object_pos=self.initial_object_pos,
            robot=self.robot,
            resolved_robot=self.resolved_robot,
            step_index=step_index,
            device=self.device,
        )
        quality = self.scene_runtime.build_quality_metrics(snapshot)
        if quality:
            task_state = dict(snapshot.get("task_state") or {})
            task_state.update({key: value for key, value in quality.items() if value is not None})
            snapshot["task_state"] = task_state
        return snapshot

    def _run_manifest(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "eval_group_id": self.eval_group_id,
            "run_type": getattr(self.args, "run_type", "single_task"),
            "suite_id": getattr(self.args, "suite_id", None),
            "benchmark_version": "EAI4S-bench-0.1.0",
            "labbench_code_version": __version__,
            "model_name": self.args.model_name,
            "checkpoint": self.args.checkpoint,
            "policy_provider": getattr(self.args, "policy_provider", None),
            "policy_backend": getattr(self.args, "policy_backend", None),
            "checkpoint_hash": self.args.checkpoint_hash,
            "dataset_version": self.args.dataset_version,
            "eval_config_version": "isaac_eval_v0.1",
            "wrapper_version": "policy_wrapper_v0.1",
            "simulator": "Isaac Sim / Isaac Lab",
            "robot": self.args.robot,
            "task_id": self.task_spec.get("task_id"),
            "task_family": self.task_metadata["task_family"],
            "task_level": self.task_metadata["task_level"],
            "task_tags": self.task_metadata["task_tags"],
            "leaderboard_group": self.task_metadata["leaderboard_group"],
            "failure_modes": self.task_metadata["failure_modes"],
            "control_mode": self.adapter.action_space,
            "control_hz": self.args.control_hz,
            "physics_hz": 60,
            "observation_track": "rgb_ego",
            "action_chunk": self.args.action_horizon,
            "action_execution": {
                "mode": "reaction_interpolation",
                "reaction_interpolation_steps": self.args.reaction_interpolation_steps,
                "sim_steps_per_action": self.args.sim_steps_per_action,
                "use_policy_orientation": self.args.use_policy_orientation,
            },
            "success_predicates": self.success_spec.to_dict(),
            "initial_stability_wait": {
                "initial_hold_steps": self.args.initial_hold_steps,
                "stabilization_max_steps": self.args.stabilization_max_steps,
                "stabilization_hold_steps": self.args.stabilization_hold_steps,
                "object_lin_vel_threshold": self.args.stabilization_object_lin_vel_threshold,
                "object_ang_vel_threshold": self.args.stabilization_object_ang_vel_threshold,
                "joint_vel_threshold": self.args.stabilization_joint_vel_threshold,
            },
            "split": self.args.split,
            "seeds": self.seeds,
            "created_at": datetime.now().astimezone().isoformat(),
            "tuned_on_eval": False,
        }

    def _run_config(self) -> dict[str, Any]:
        return {
            "args": {key: value for key, value in vars(self.args).items() if isinstance(value, (str, int, float, bool, type(None)))},
            "task_spec": self.task_spec,
        }
