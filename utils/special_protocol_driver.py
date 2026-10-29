"""Launch versioned release tasks before main.py creates an Isaac application.

Each episode owns a fresh worker, while successful HDF5s share the historical
dataset/meta/episode.jsonl index consumed by collect_driver.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


PROTOCOLS = {
    "centrifuge_v2": ("eai4s_eval.centrifuge_v2.cli", "hard", "3d828af"),
    "weighing_v1": ("eai4s_eval.weighing_v1.cli", "hard", "3d828af"),
    "rack_recall_v4": ("eai4s_eval.rack_recall_v4.cli", "hard_same_cap", "892f9b5"),
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _publish(dataset: Path, destination: Path, index: int) -> bool:
    """Publish exactly one physically attested demonstration, never a diagnostic file."""
    import h5py
    import numpy as np

    manifest = dataset / "dataset/meta/episode.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()] if manifest.exists() else []
    if len(rows) != 1 or rows[0].get("task_properties", {}).get("physical_success") is not True:
        return False
    source = dataset / "dataset" / f"episode_{rows[0]['episode_index']:04d}.h5"
    if not source.is_file():
        return False
    with h5py.File(source) as h5:
        raw_properties = h5["task_properties"][()] if "task_properties" in h5 else b"{}"
        properties = json.loads(raw_properties.decode() if isinstance(raw_properties, bytes) else str(raw_properties))
        if properties.get("physical_success") is not True:
            return False
        leaves = {}
        h5.visititems(lambda key, value: leaves.update({key.rsplit('/', 1)[-1]: value})
                      if isinstance(value, h5py.Dataset) else None)
        for key in ("actions", "actions_ee", "action_gripper", "camera_1_rgb", "camera_3_rgb"):
            if key not in leaves or len(leaves[key]) == 0:
                return False
        if len({len(leaves[key]) for key in ("actions", "actions_ee", "action_gripper", "camera_1_rgb", "camera_3_rgb")}) != 1:
            return False
        if not all(np.isfinite(leaves[key][:]).all() for key in ("actions", "actions_ee", "action_gripper")):
            return False
    target = destination / "dataset"
    (target / "meta").mkdir(parents=True, exist_ok=True)
    path = target / f"episode_{index:04d}.h5"
    if path.exists():
        raise FileExistsError(path)
    try:
        os.link(source, path)
    except OSError:
        shutil.copy2(source, path)
    rows[0]["episode_index"] = index
    rows[0]["source_episode"] = str(source)
    with (target / "meta/episode.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(rows[0], ensure_ascii=False) + "\n")
    return True


def dispatch(args) -> int | None:
    from omegaconf import OmegaConf

    cfg_path = Path(args.config_dir) / f"{args.config_name}.yaml"
    if not cfg_path.is_file():
        return None
    cfg = OmegaConf.load(cfg_path)
    family = cfg.get("release_protocol")
    if family not in PROTOCOLS:
        return None
    module, variant, revision = PROTOCOLS[family]
    mode = args.mode or cfg.get("mode", "collect")
    mode = "eval" if mode == "infer" else mode
    root = Path(args.run_dir or (
        f"outputs/{'collect' if mode == 'collect' else 'eval_runs'}/"
        f"{datetime.now():%Y.%m.%d/%H.%M.%S}_{cfg.name}")) .resolve()
    if mode == "eval" and args.eval_output_root:
        root = Path(args.eval_output_root).resolve() / f"{datetime.now():%Y%m%dT%H%M%S}_{cfg.name}"
    root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, root / "config.yaml")
    count = args.max_success_episodes or int(cfg.get("max_episodes", 200))
    if mode == "eval":
        count = args.eval_num_episodes or int(cfg.get("evaluation", {}).get("num_episodes", 5))
    if count <= 0:
        raise ValueError("episode count must be positive")
    required_streak = getattr(args, "require_consecutive_successes", 0)
    success = total = failures = streak = 0
    evaluation_rows = []
    started = time.monotonic()
    deadline = started + args.task_timeout_minutes * 60 if args.task_timeout_minutes else float("inf")
    status, code = "completed", 0
    existing = root / "dataset/meta/episode.jsonl"
    first_index = len(existing.read_text().splitlines()) if existing.exists() else 0
    while (total < count if mode == "eval" else success < count or streak < required_streak):
        if time.monotonic() >= deadline:
            status, code = "timeout", 3
            break
        seeds = getattr(args, "episode_seeds", None)
        seed = seeds[total] if mode == "eval" and seeds else (args.seed if args.seed is not None else 101) + total
        episode = root / "attempts" / f"{total:04d}_seed_{seed}"
        episode.mkdir(parents=True, exist_ok=False)
        reference = episode / "reference"
        native = "native_" if mode == "eval" else ""
        config = Path(module.replace(".cli", "").replace(".", "/")) / "configs" / f"{native}{variant}.json"
        payload = _read(config)
        if mode == "eval":
            policy = cfg.get("evaluation", {}).get("policy", {})
            interface = payload["evaluation"]["policy_interface"]
            for key in ("host", "port"):
                if key in policy:
                    interface[key] = policy[key]
                override = getattr(args, "vla_" + key, None)
                if override is not None:
                    interface[key] = override
            if getattr(args, "policy_url", None):
                interface.update(name="http_array", url=args.policy_url)
            from dataclasses import replace
            from eai4s_eval.oracle_runtime.contracts import PolicyProfile
            profile = PolicyProfile.legacy(interface, (
                ("front", "wrist", "display", "weights") if family == "weighing_v1" else ("front", "wrist")))
            profile = replace(profile, action=replace(profile.action, control_hz=20 if family == "rack_recall_v4" else 10))
            action_space = getattr(args, "policy_action_space", None) or policy.get("action_space", "ee_pos")
            if action_space == "ee_pos":
                profile = replace(profile, action=replace(profile.action, representation="eef_pose", frame="world",
                                                          units="meters+quaternion_wxyz+finger_meter"))
            if family == "weighing_v1":
                profile = replace(profile, input_mapping={"front": "camera_1_rgb", "wrist": "camera_3_rgb",
                                                          "display": "display_rgb", "weights": "weights_rgb"})
            payload["evaluation"]["policy_profile"] = profile.public()
            payload["labbench_runtime"] = {
                "max_action_steps": getattr(args, "eval_max_action_steps", None) or int(cfg.get("evaluation", {}).get("max_steps", 500)),
                "execution_horizon": args.execution_horizon,
                "action_space": action_space,
                "policy_profile": getattr(args, "branch_policy_profile", None) or "eai4s_eval/configs/policies/openpi_franka_ee_pos_v1.yaml",
                "backend": getattr(args, "policy_backend", "websocket"),
            }
        payload["source_commit"] = revision
        resolved = episode / "protocol.json"
        resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        command = [sys.executable, "-m", module, "run", "--config", str(resolved),
                   "--output", str(episode / "run"), "--seed", str(seed)]
        if args.headless:
            command.append("--headless")
        if not args.no_video:
            command.append("--save-video")
        if mode == "collect":
            command += ["--dataset-output", str(reference)]
        print(f"SPECIAL_ATTEMPT family={family} seed={seed} log={episode / 'worker.log'}", flush=True)
        with (episode / "worker.log").open("w") as log:
            worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = worker.wait(timeout=max(1, deadline - time.monotonic()) if deadline != float("inf") else None)
            except KeyboardInterrupt:
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
                raise
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
                status, code = "timeout", 3
                break
        total += 1
        scores = _read(episode / "run/scores.json")
        if mode == "eval":
            manifest = _read(episode / "run/run.json")
            manifest.update({key: getattr(args, key) for key in ("model_name", "checkpoint", "split")
                             if getattr(args, key, None) is not None})
            (episode / "run/run.json").write_text(json.dumps(manifest, indent=2))
            valid = scores.get("evaluation_valid") is True and isinstance(scores.get("valid_success"), bool)
            evaluation_rows.append({"seed": seed, "success": scores.get("valid_success") if valid else None,
                                    "evaluation_valid": valid,
                                    "runtime_status": manifest.get("runtime_status"),
                                    "returncode": returncode, "scores": scores,
                                    "run_dir": str(episode / "run")})
            if returncode != 0 or manifest.get("runtime_status") in {"error", "service_error"}:
                status, code = "error", 1
        accepted = returncode == 0 and scores.get("valid_success") is True
        if mode == "collect" and accepted:
            accepted = _publish(reference, root, first_index + success)
        if accepted:
            success += 1
            streak += 1
            failures = 0
        else:
            failures += 1
            streak = 0
        print(f"SPECIAL_PROGRESS success={success}/{count} total={total} streak={streak} failures={failures}", flush=True)
        if mode == "collect" and failures >= args.max_consecutive_failures:
            status, code = "consecutive_failures", 2
            break
    report = dict(config=args.config_name, family=family, source_commit=revision,
                  mode=mode, success_count=success, total_episodes=total,
                  consecutive_failures=failures, success_streak=streak, status=status,
                  duration_seconds=int(time.monotonic()-started), run_dir=str(root))
    (root / "collection_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if mode == "eval":
        summary = {**report, "task_id": args.config_name, "leaderboard_group": "release_protocols_v1",
                   "num_episodes": total, "num_successes": success,
                   "task_success_rate": (sum(row['success'] is True for row in evaluation_rows) /
                                         sum(row['evaluation_valid'] for row in evaluation_rows)
                                         if any(row['evaluation_valid'] for row in evaluation_rows) else None),
                   "score_complete": all(row['evaluation_valid'] for row in evaluation_rows) and total == count,
                   "valid_episodes": sum(row["evaluation_valid"] for row in evaluation_rows),
                   "model_name": getattr(args, "model_name", None), "checkpoint": getattr(args, "checkpoint", None),
                   "episodes": evaluation_rows}
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if mode == "collect":
        from utils.collect_driver import format_collect_result
        print(format_collect_result(**{key: report[key] for key in (
            "config", "success_count", "total_episodes", "consecutive_failures", "status", "duration_seconds", "run_dir")}), flush=True)
    return code


def dispatch_benchmark_cli(args) -> int | None:
    """Keep the external benchmark CLI on the same versioned scenes as main.py."""
    from argparse import Namespace
    from omegaconf import OmegaConf
    from eai4s_eval.eval.runner import parse_seeds
    from eai4s_eval.eval.task_suite import resolve_task_spec_path

    repo = Path(args.benchmark_root).resolve()
    spec_path = resolve_task_spec_path(args.task_spec)
    spec = OmegaConf.load(spec_path)
    name = args.config_name or spec.get("runtime", {}).get("env_config")
    if not name or not (repo / "config" / f"{name}.yaml").exists():
        return None
    cfg = OmegaConf.load(repo / "config" / f"{name}.yaml")
    if cfg.get("release_protocol") not in PROTOCOLS:
        return None
    if args.policy_provider != "openpi":
        raise ValueError("Versioned special protocols currently expose the OpenPI adapter; use a native protocol policy profile for other agents")
    from eai4s_eval.policy import load_openpi_profile
    branch_profile = load_openpi_profile(args.policy_profile)
    seeds = parse_seeds(args.seeds, args.seed_manifest, args.split)[:args.num_episodes]
    if len(seeds) < args.num_episodes:
        raise ValueError("Not enough explicit evaluation seeds")
    options = Namespace(
        config_dir=str(repo / "config"), config_name=name, mode="eval", backend=args.backend,
        headless=args.headless, no_video=not args.save_video, seed=seeds[0], episode_seeds=seeds,
        run_dir=str(Path(args.output_root) / args.run_id) if args.run_id else None,
        eval_output_root=None if args.run_id else args.output_root,
        eval_num_episodes=args.num_episodes, eval_max_action_steps=args.max_steps,
        max_success_episodes=None, task_timeout_minutes=None, max_consecutive_failures=20,
        execution_horizon=args.execution_horizon, require_consecutive_successes=0,
        policy_action_space=branch_profile.action_space, policy_backend=args.policy_backend,
        branch_policy_profile=str(Path(args.policy_profile).resolve()), policy_url=None,
        vla_host=args.vla_host, vla_port=args.vla_port or 8000,
        model_name=args.model_name, checkpoint=args.checkpoint, split=args.split)
    return dispatch(options)
