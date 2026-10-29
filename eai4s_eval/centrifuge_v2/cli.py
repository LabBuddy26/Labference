"""Offline diagnostics for the versioned centrifuge tasks.

Run with ``python -m eai4s_eval.centrifuge_v2.cli`` from the repository root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from eai4s_eval.oracle_runtime import resolve_runtime_contract

from .core import Contract, SLOTS, State, TUBES, legal_actions
from .diagnostics import (ORACLES, PolicyCapabilities, compare_pairs, compare_state_pairs,
                          plan_online_rollouts, plan_state_pairs,
                          score_checkpoint, score_trace, summarize_checkpoint_scores,
                          verify_online_scores)


def read_json(path: str) -> object:
    return json.loads(Path(path).read_text())


def load_evaluation_config(path: str, *, policy_profile: str | None = None,
                           intervention_spec: str | None = None) -> tuple[dict, Contract]:
    data = read_json(path)
    if not isinstance(data, dict) or "evaluation" not in data:
        raise ValueError("configuration needs an evaluation object")
    evaluation = dict(data["evaluation"])
    contract = contract_from_evaluation(evaluation)
    runtime_profile, intervention = resolve_runtime_contract(
        evaluation, rgb_channels=("front", "wrist"),
        policy_profile_path=policy_profile, intervention_path=intervention_spec)
    evaluation["policy_profile"] = runtime_profile.public()
    evaluation["intervention"] = intervention.__dict__
    evaluation["oracle_profile"] = intervention.profile
    evaluation["oracle_transport"] = (intervention.transport if intervention.mode == "assist" else "structured")
    return evaluation, contract


def load_contract(path: str, *, policy_profile: str | None = None,
                  intervention_spec: str | None = None) -> Contract:
    return load_evaluation_config(path, policy_profile=policy_profile,
                                  intervention_spec=intervention_spec)[1]


def contract_from_evaluation(ev: dict) -> Contract:
    if not isinstance(ev, dict):
        raise ValueError("evaluation must be an object")
    if ev.get("task_version") not in {"centrifuge-opposite/v1", "centrifuge-opposite/v2"}:
        raise ValueError("unsupported task_version")
    if ev.get("oracle_profile", "none") not in ORACLES:
        raise ValueError("invalid oracle profile")
    transport = ev.get("oracle_transport", "structured")
    if transport not in {"structured", "text"}:
        raise ValueError("invalid oracle transport")
    interface = ev.get("policy_interface", {})
    if interface:
        capabilities = PolicyCapabilities(
            frozenset(interface.get("supported_reports", [])),
            frozenset(interface.get("oracle_transports", [])),
            interface.get("action_space", "symbolic"),
            bool(interface.get("resettable", False)),
        )
        if ev.get("oracle_profile", "none") != "none" and transport not in capabilities.oracle_transports:
            raise ValueError("configured policy does not support the selected Oracle transport")
        name = interface.get("name")
        if name not in {"diagnostic_rule_expert", "http_symbolic", "http_array", "openpi_array"}:
            raise ValueError("unknown policy interface")
        if (name in {"http_array", "openpi_array"} and capabilities.action_space != "array") or (
                name in {"diagnostic_rule_expert", "http_symbolic"} and capabilities.action_space != "symbolic"):
            raise ValueError("policy interface and action space disagree")
        if interface.get("fixed_subgoal_input", False):
            if (type(interface["fixed_subgoal_input"]) is not bool or
                    capabilities.action_space != "array" or interface.get("coordinate_input") is not True):
                raise ValueError("fixed_subgoal_input requires native array actions and coordinate_input=true")
    if ev.get("evaluation_version") not in {None, "labference/centrifuge-online-v2"}:
        raise ValueError("unknown evaluation version")
    if interface.get("action_space") == "array" and ev.get("evaluation_version") is None:
        raise ValueError("native array runs require evaluation_version")
    visible_slots = ev.get("semantic_visible_slots", [])
    if not isinstance(visible_slots, list) or any(slot not in SLOTS for slot in visible_slots):
        raise ValueError("invalid semantic_visible_slots")
    return Contract(**{k: v for k, v in ev.items() if k not in {
        "oracle_profile", "oracle_transport", "policy_interface", "evaluation_version",
        "semantic_visible_slots", "policy_profile", "intervention"}})


def parse_trace(path: str) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def online_rollout_record(path: str, audit_online: bool = True) -> dict:
    root = Path(path)
    if root.is_file():
        root = root.parent
    manifest = read_json(str(root / "run.json"))
    status = manifest.get("runtime_status")
    score_path = root / "scores.json"
    scored = status in {"finished", "time_limit", "stopped_after_violation"} and score_path.is_file()
    result = {
        "pair_id": manifest.get("pair_id", f"{manifest['difficulty']}-online-{manifest['seed']:04d}"),
        "condition": manifest["oracle_profile"],
        "task_version": manifest["task_version"], "contract_hash": manifest["contract_hash"],
        "scene_seed": manifest.get("scene_seed", manifest["seed"]),
        "decision_seed": manifest.get("decision_seed", manifest["seed"]),
        "max_steps_budget": manifest.get("max_steps_budget"),
        "initial_slot": manifest["initial_slot"],
        "policy_id": manifest.get("policy_id"),
        "policy_revision": manifest.get("policy_revision"),
        "policy_reset_scope": manifest.get("policy_reset_scope"),
        "policy_uses_evaluator_state": manifest.get("policy_uses_evaluator_state"),
        "sensor_condition": manifest.get("sensor_condition"),
        **({"evaluation_version": manifest["evaluation_version"],
            "policy_track": manifest["policy_track"]}
           if manifest.get("evaluation_version") is not None else {}),
        "status": "scored" if scored else ("service_error" if status == "service_error" else "runtime_error"),
    }
    if scored:
        scores = read_json(str(score_path))
        if (scores.get("contract_hash") != manifest["contract_hash"] or
                scores.get("task_version") != manifest["task_version"] or
                not isinstance(scores.get("task_completion", scores.get("valid_success")), bool)):
            raise ValueError(f"inconsistent online scores: {score_path}")
        if audit_online:
            evaluation = manifest.get("evaluation")
            if not isinstance(evaluation, dict):
                raise ValueError(f"online run lacks evaluation contract: {root}")
            contract = contract_from_evaluation(evaluation)
            if contract.hash != manifest["contract_hash"]:
                raise ValueError(f"online run contract hash mismatch: {root}")
            trace = parse_trace(str(root / "events.jsonl"))
            snapshots = parse_trace(str(root / "online_scores.jsonl"))
            reports = set(evaluation.get("policy_interface", {}).get("supported_reports", []))
            audit = verify_online_scores(contract, trace, snapshots, reports,
                                         evaluation.get("evaluation_version"))
            if not audit["matching"] or snapshots[-1]["scores"] != scores:
                raise ValueError(f"online score audit failed: {root}: {audit['mismatches']}")
        if manifest.get("evaluation_version") == "labference/centrifuge-online-v2" and scores.get("evaluation_valid") is not True:
            result["status"] = "invalid_evaluation"
            result["evaluation_valid"] = False
            return result
        result["evaluation_valid"] = scores.get("evaluation_valid", True)
        result["success"] = scores.get("task_completion", scores.get("valid_success"))
        report_scores = scores.get("diagnostic", {}).get("interface_scores", scores.get("module_scores", {}))
        result["decision_accuracy"] = report_scores.get("decision", {}).get("accuracy")
        result["online_legal_next_action_rate"] = scores.get("online_legal_next_action_rate")
        result["evaluation_version"] = manifest.get("evaluation_version")
        result["policy_track"] = manifest.get("policy_track")
    return result


def run_supervised(config: str, output: str, seed: int, max_steps: int | None,
                   save_video: bool, oracle_profile: str | None, policy_url: str | None,
                   close_timeout_s: float = 45.0, track: str | None = None,
                   checkpoint_slot: str | None = None, checkpoint_kind: str = "behavior",
                   policy_profile: str | None = None, intervention_spec: str | None = None,
                   dataset_output: str | None = None, headless: bool = True) -> dict:
    """Isolate Kit shutdown so the parent can record the worker's exit status."""
    if close_timeout_s <= 0:
        raise ValueError("close_timeout_s must be positive")
    out = Path(output).resolve()
    command = [sys.executable, "-m", "eai4s_eval.centrifuge_v2.cli", "run", "--worker",
               "--config", config, "--output", str(out), "--seed", str(seed)]
    if headless:
        command.append("--headless")
    if max_steps is not None:
        command += ["--max-steps", str(max_steps)]
    if save_video:
        command.append("--save-video")
    if oracle_profile is not None:
        command += ["--oracle-profile", oracle_profile]
    if policy_url is not None:
        command += ["--policy-url", policy_url]
    if policy_profile is not None:
        command += ["--policy-profile", str(Path(policy_profile).resolve())]
    if intervention_spec is not None:
        command += ["--intervention-spec", str(Path(intervention_spec).resolve())]
    if dataset_output is not None:
        command += ["--dataset-output", str(Path(dataset_output).resolve())]
    if track is not None:
        command += ["--track", track]
    if checkpoint_slot is not None:
        command += ["--checkpoint-slot", checkpoint_slot]
    if checkpoint_kind != "behavior":
        command += ["--checkpoint-kind", checkpoint_kind]
    worker = subprocess.Popen(command)
    close_started = None
    timed_out = False
    manifest_path = out / "run.json"
    while worker.poll() is None:
        try:
            worker.wait(timeout=.5)
        except subprocess.TimeoutExpired:
            pass
        if manifest_path.exists():
            try:
                manifest = read_json(str(manifest_path))
            except (json.JSONDecodeError, OSError):
                continue
            if manifest.get("simulation_app_close", {}).get("status") == "requested":
                close_started = close_started or time.monotonic()
        if close_started is not None and worker.poll() is None and time.monotonic() - close_started > close_timeout_s:
            worker.kill()
            worker.wait()
            timed_out = True
    if not manifest_path.exists():
        raise RuntimeError(f"simulation worker exited {worker.returncode} without run manifest: {out}")
    manifest = read_json(str(manifest_path))
    close = manifest.get("simulation_app_close", {})
    if close.get("status") == "requested" or timed_out:
        manifest["simulation_app_close"] = {
            "status": "timed_out" if timed_out else ("process_exited" if worker.returncode == 0 else "worker_failed"),
            "worker_exit_code": worker.returncode,
            "duration_s": round(time.monotonic() - close_started, 3) if close_started is not None else None,
        }
    manifest["worker_exit_code"] = worker.returncode
    manifest_path.write_text(json.dumps(manifest, indent=2))
    score_path = out / "scores.json"
    if worker.returncode != 0:
        raise RuntimeError(f"simulation worker exited {worker.returncode}; see {manifest_path}")
    if not score_path.exists():
        raise RuntimeError(f"simulation worker did not write scores: {score_path}")
    return read_json(str(score_path))


def main() -> None:
    p = argparse.ArgumentParser(description="Versioned centrifuge diagnostic evaluation")
    sub = p.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.add_argument("--policy-profile")
    validate.add_argument("--intervention-spec")
    check = sub.add_parser("checkpoints")
    check_source = check.add_mutually_exclusive_group(required=True)
    check_source.add_argument("--input")
    check_source.add_argument("--online-config")
    check.add_argument("--output")
    check.add_argument("--checkpoint-slot", choices=SLOTS)
    check.add_argument("--kind", choices=("behavior", "semantic", "decision", "geometry", "execution"),
                       default="behavior")
    check.add_argument("--seed", type=int, default=101)
    check.add_argument("--max-steps", type=int)
    check.add_argument("--save-video", action="store_true")
    check.add_argument("--oracle-profile", choices=ORACLES)
    check.add_argument("--policy-url")
    check.add_argument("--track", choices=("native", "reported"))
    check.add_argument("--policy-profile")
    check.add_argument("--intervention-spec")
    score = sub.add_parser("score")
    score_source = score.add_mutually_exclusive_group(required=True)
    score_source.add_argument("--config")
    score_source.add_argument("--run")
    score.add_argument("--trace", required=True)
    score.add_argument("--online", help="audit the online_scores.jsonl prefix snapshots")
    score.add_argument("--output")
    compare = sub.add_parser("compare")
    compare.add_argument("--input", required=True)
    compare.add_argument("--condition", choices=ORACLES[1:], required=True)
    compare.add_argument("--plan", required=True, help="pre-registered pair-plan JSON")
    compare.add_argument("--output")
    compare_states = sub.add_parser("compare-states", help="score state-dependent decision pairs")
    compare_states.add_argument("--input", required=True)
    compare_states.add_argument("--plan", required=True)
    compare_states.add_argument("--output")
    compare_state_runs = sub.add_parser("compare-state-runs", help="compare verified online checkpoint runs")
    compare_state_runs.add_argument("--plan", required=True)
    compare_state_runs.add_argument("--runs", nargs="+", required=True)
    compare_state_runs.add_argument("--output")
    rollout_plan = sub.add_parser("rollout-plan", help="pre-register full online episode pairs")
    rollout_plan.add_argument("--config", required=True)
    rollout_plan.add_argument("--seeds", nargs="+", type=int, required=True)
    rollout_plan.add_argument("--max-steps", type=int)
    rollout_plan.add_argument("--output")
    compare_runs = sub.add_parser("compare-runs", help="compare completed online run directories")
    compare_runs.add_argument("--runs", nargs="+", required=True)
    compare_runs.add_argument("--condition", choices=ORACLES[1:], required=True)
    compare_runs.add_argument("--plan", required=True)
    compare_runs.add_argument("--output")
    pair_plan = sub.add_parser("pair-plan", help="pre-register symbolic state-change checkpoint pairs")
    pair_plan.add_argument("--config", required=True)
    pair_plan.add_argument("--seeds", nargs="+", type=int, required=True)
    pair_plan.add_argument("--online", action="store_true",
                           help="register physically verified checkpoint continuations")
    pair_plan.add_argument("--output")
    execute = sub.add_parser("run", help="run one Isaac Sim diagnostic episode")
    execute.add_argument("--config", required=True)
    execute.add_argument("--output", required=True)
    execute.add_argument("--seed", type=int, default=101)
    execute.add_argument("--max-steps", type=int)
    execute.add_argument("--headless", action="store_true",
                         help="run Isaac Sim without opening the GUI")
    execute.add_argument("--save-video", action="store_true")
    execute.add_argument("--oracle-profile", choices=ORACLES)
    execute.add_argument("--policy-url", help="HTTP policy endpoint matching the selected track")
    execute.add_argument("--track", choices=("native", "reported"))
    execute.add_argument("--policy-profile")
    execute.add_argument("--intervention-spec")
    execute.add_argument("--dataset-output", help="current LabBench HDF5 dataset directory")
    execute.add_argument("--checkpoint-slot", choices=SLOTS,
                         help="preload NMR_01 for a verified online state checkpoint")
    execute.add_argument("--checkpoint-kind", choices=("behavior", "execution"), default="behavior",
                         help=argparse.SUPPRESS)
    execute.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.command == "run":
        if args.worker:
            from .runner import run
            result = run(args.config, args.output, args.seed, args.max_steps, args.save_video,
                         args.oracle_profile, args.policy_url, args.track, args.checkpoint_slot,
                         args.checkpoint_kind, args.policy_profile, args.intervention_spec,
                         args.dataset_output, headless=args.headless)
        else:
            result = run_supervised(args.config, args.output, args.seed, args.max_steps,
                                    args.save_video, args.oracle_profile, args.policy_url,
                                    track=args.track, checkpoint_slot=args.checkpoint_slot,
                                    checkpoint_kind=args.checkpoint_kind,
                                    policy_profile=args.policy_profile,
                                    intervention_spec=args.intervention_spec,
                                    dataset_output=args.dataset_output,
                                    headless=args.headless)
    elif args.command == "validate":
        evaluation, contract = load_evaluation_config(
            args.config, policy_profile=args.policy_profile,
            intervention_spec=args.intervention_spec)
        asset = Path(__file__).resolve().parents[2] / contract.source_asset
        if not asset.is_file():
            raise ValueError(f"missing rotor asset: {asset}")
        result = {"valid": True, "task_version": contract.task_version, "contract_hash": contract.hash,
                  "slot_count": len(SLOTS), "difficulty": contract.difficulty,
                  "policy_profile": evaluation["policy_profile"],
                  "intervention": evaluation["intervention"]}
    elif args.command == "checkpoints":
        if args.online_config:
            if not args.output or not args.checkpoint_slot:
                raise ValueError("online checkpoints require --output directory and --checkpoint-slot")
            evaluation = read_json(args.online_config)["evaluation"]
            reports = set(evaluation.get("policy_interface", {}).get("supported_reports", []))
            interface = evaluation.get("policy_interface", {})
            fixed_supported = (interface.get("action_space") == "array" and
                               interface.get("coordinate_input") and interface.get("fixed_subgoal_input"))
            if ((args.kind == "execution" and not fixed_supported) or
                    (args.kind not in {"behavior", "execution"} and args.kind not in reports)):
                result = {"status": "unsupported", "kind": args.kind,
                          "reason": "policy interface lacks a verified online checkpoint output"}
            else:
                scores = run_supervised(args.online_config, args.output, args.seed,
                                        args.max_steps, args.save_video, args.oracle_profile,
                                        args.policy_url, track=args.track,
                                        checkpoint_slot=args.checkpoint_slot,
                                        checkpoint_kind="execution" if args.kind == "execution" else "behavior",
                                        policy_profile=args.policy_profile,
                                        intervention_spec=args.intervention_spec)
                metric = (scores.get("online_legal_next_action_rate") if args.kind == "behavior" else
                          scores.get("diagnostic", {}).get("interface_scores", {}).get(args.kind, {}).get("accuracy"))
                manifest = read_json(str(Path(args.output) / "run.json"))
                valid = bool(manifest.get("checkpoint_verified") and scores.get("evaluation_valid", True))
                result = {"status": "scored" if metric is not None and valid else "unavailable",
                          "kind": args.kind, "accuracy": metric if valid else None,
                          "checkpoint_verified": bool(manifest.get("checkpoint_verified")),
                          "evaluation_valid": bool(scores.get("evaluation_valid", True)),
                          "run_dir": str(Path(args.output).resolve())}
        else:
            payload = read_json(args.input)
            records = payload if isinstance(payload, list) else payload["checkpoints"]
            scored = [score_checkpoint(r) for r in records]
            result = {"checkpoints": scored, "module_scores": summarize_checkpoint_scores(scored)}
    elif args.command == "score":
        if args.run:
            manifest = read_json(args.run)
            contract = contract_from_evaluation(manifest["evaluation"])
            if contract.hash != manifest["contract_hash"]:
                raise ValueError("run manifest contract hash mismatch")
            evaluation = manifest["evaluation"]
        else:
            contract = load_contract(args.config)
            evaluation = read_json(args.config)["evaluation"]
        reports = set(evaluation.get("policy_interface", {}).get("supported_reports", []))
        trace = parse_trace(args.trace)
        if evaluation.get("evaluation_version") == "labference/centrifuge-online-v2":
            from .online_v2 import score_online_v2
            result = score_online_v2(contract, trace, reports)
        else:
            result = score_trace(contract, trace, reports)
        if args.online:
            result["online_replay_audit"] = verify_online_scores(
                contract, trace, parse_trace(args.online), reports,
                evaluation.get("evaluation_version"))
    elif args.command == "pair-plan":
        result = {"checkpoints": plan_state_pairs(load_contract(args.config), args.seeds, args.online)}
    elif args.command == "rollout-plan":
        evaluation = read_json(args.config)["evaluation"]
        result = {"pairs": plan_online_rollouts(
            load_contract(args.config), args.seeds, evaluation.get("evaluation_version"),
            "native" if evaluation.get("policy_interface", {}).get("action_space") == "array" else "reported",
            args.max_steps)}
    elif args.command == "compare-runs":
        plan = read_json(args.plan)
        registered = {item["pair_id"]: {key: item[key] for key in
                      ("task_version", "contract_hash", "scene_seed", "decision_seed",
                       "initial_slot", "max_steps_budget", "evaluation_version", "policy_track")
                      if key in item}
                      for item in plan["pairs"]}
        result = compare_pairs([online_rollout_record(path) for path in args.runs],
                               args.condition, registered)
    elif args.command == "compare-state-runs":
        plan = read_json(args.plan)
        from .online_v2 import online_state_pair_record
        result = compare_state_pairs([online_state_pair_record(path, plan["checkpoints"])
                                      for path in args.runs], plan["checkpoints"])
    elif args.command == "compare":
        payload = read_json(args.input)
        plan = read_json(args.plan)
        registered = {item["oracle_pair_id"]: {
            key: item[key] for key in ("task_version", "contract_hash", "scene_seed",
                                  "decision_seed", "first_actual_slot", "variant")}
            for item in plan["checkpoints"]}
        result = compare_pairs(payload if isinstance(payload, list) else payload["records"],
                               args.condition, registered)
    else:
        payload = read_json(args.input)
        plan = read_json(args.plan)
        result = compare_state_pairs(payload if isinstance(payload, list) else payload["records"],
                                     plan["checkpoints"])
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.command == "checkpoints" and args.online_config and args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / "checkpoint_result.json").write_text(rendered + "\n")
    elif args.command != "run" and getattr(args, "output", None):
        Path(args.output).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
