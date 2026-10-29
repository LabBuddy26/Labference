"""Command line interface for rack-placement recall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from eai4s_eval.oracle_runtime import resolve_runtime_contract, validate_connection_binding

from .core import (CAP_COLORS, Contract, EVALUATION_VERSION, ORACLES, SCORING_TRACKS,
                   SCORING_VERSION, SLOTS,
                   Instance, RackPose, generate_coverage_manifest,
                   generate_instance)
from .online import apply_runtime_validity, compare_pairs, score_trace
from .scene import SCENE_REVISION, validate_scene_geometry


POLICIES = {"diagnostic_rule_expert", "http_symbolic", "http_array", "openpi_array"}
REPORTS = {"semantic", "decision", "geometry", "execution"}


def load_config(path: str | Path, *, policy_profile: str | Path | None = None,
                intervention_spec: str | Path | None = None) -> tuple[dict, Contract]:
    source = json.loads(Path(path).read_text())
    if not isinstance(source, dict) or not isinstance(source.get("evaluation"), dict):
        raise ValueError("configuration requires an evaluation object")
    evaluation = source["evaluation"]
    if evaluation.get("evaluation_version") != EVALUATION_VERSION:
        raise ValueError("unsupported rack-recall evaluation version")
    interface = evaluation.get("policy_interface", {})
    if interface.get("name") not in POLICIES:
        raise ValueError("unknown policy interface")
    if interface.get("action_space") not in {"symbolic", "array"}:
        raise ValueError("policy action_space must be symbolic or array")
    if (interface["name"] in {"http_array", "openpi_array"}) != (
            interface["action_space"] == "array"):
        raise ValueError("policy interface and action space disagree")
    if set(interface.get("supported_reports", [])) - REPORTS:
        raise ValueError("unsupported report kind")
    if set(interface.get("oracle_transports", [])) - {"structured", "text"}:
        raise ValueError("unknown Oracle transport")
    if evaluation.get("oracle_profile", "none") not in ORACLES:
        raise ValueError("unknown Oracle profile")
    if type(evaluation.get("recall_report_supported", False)) is not bool:
        raise ValueError("recall_report_supported must be boolean")
    if evaluation.get("scoring_track") not in SCORING_TRACKS:
        raise ValueError("rack-recall/v4 requires an explicit scoring_track")
    if evaluation.get("history_protocol") != "online_once":
        raise ValueError("rack-recall/v4 requires history_protocol=online_once")
    contract_fields = set(Contract.__dataclass_fields__)
    allowed = contract_fields | {"evaluation_version", "policy_interface", "oracle_profile",
                                 "oracle_transport", "policy_profile", "intervention",
                                 "recall_report_supported", "scoring_track",
                                 "history_protocol"}
    unknown = set(evaluation) - allowed
    if unknown:
        raise ValueError(f"unknown evaluation fields: {sorted(unknown)}")
    contract = Contract(**{key: value for key, value in evaluation.items()
                           if key in contract_fields})
    scene = source.get("scene")
    expected_distractors = 0 if contract.difficulty == "easy" else 3
    expected_motion = contract.difficulty == "hard"
    if (not isinstance(scene, dict) or scene.get("scene_revision") != SCENE_REVISION or
            scene.get("rack_grid") != "3x4" or
            scene.get("distractor_count") != expected_distractors or
            scene.get("environment_moves_rack") != expected_motion or
            scene.get("environment_mode") != contract.environment_mode or
            scene.get("condition_variant", "single") != contract.condition_variant):
        raise ValueError("scene does not match the rack-recall difficulty contract")
    profile, intervention = resolve_runtime_contract(
        evaluation, rgb_channels=("front", "wrist"),
        policy_profile_path=policy_profile, intervention_path=intervention_spec)
    if profile.history_protocol != evaluation["history_protocol"]:
        raise ValueError("policy profile history protocol differs from task protocol")
    validate_connection_binding(profile, interface)
    evaluation["policy_profile"] = profile.public()
    evaluation["intervention"] = intervention.__dict__
    evaluation["oracle_profile"] = intervention.profile
    evaluation["oracle_transport"] = (intervention.transport
                                      if intervention.mode == "assist" else "structured")
    return source, contract


def _dump(value: dict, output: str | None = None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)


def _load_rows(path: str | Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _instance_from_manifest(value: dict) -> Instance:
    return Instance(
        difficulty=value["difficulty"], seed=value["seed"],
        layout_seed=value["layout_seed"], motion_seed=value["motion_seed"],
        policy_seed=value["policy_seed"],
        color_seed=value["color_seed"],
        rack_initial_pose=RackPose(tuple(value["rack_initial_pose"]["position_m"]),
                                   value["rack_initial_pose"]["yaw_deg"]),
        rack_final_pose=RackPose(tuple(value["rack_final_pose"]["position_m"]),
                                 value["rack_final_pose"]["yaw_deg"]),
        target_spawn_xy_m=tuple(value["target_spawn_xy_m"]),
        recovery_center_xy_m=tuple(value["recovery_center_xy_m"]),
        distractor_slots=tuple(value["distractor_slots"]), pair_id=value["pair_id"],
        condition_variant=value.get("condition_variant", "single"),
        cap_colors=tuple(tuple(item) for item in value["cap_colors"]))


def _run_summary(path: str | Path) -> dict:
    root = Path(path)
    if root.is_file():
        root = root.parent
    manifest = json.loads((root / "run.json").read_text())
    scores = apply_runtime_validity(
        json.loads((root / "scores.json").read_text()),
        str(manifest.get("runtime_status", "missing")))
    return {**manifest, **scores}


def _benchmark_pass(row: dict) -> bool:
    """Return the registered slot-reasoning result for a valid run."""
    return bool(row.get("evaluation_valid") is True and
                row.get("benchmark_success") is True)


def _coverage_result_status(expected: dict, row: dict | None) -> str:
    if row is None:
        return "missing"
    expected = {**expected,
                "condition_variant": expected.get("condition_variant",
                                                   expected.get("condition"))}
    fields = ("task_version", "contract_hash", "scoring_version", "scoring_track",
              "history_protocol", "condition_variant", "evaluation_track",
              "environment_mode", "rebuild_revision")
    if any(row.get(field) != expected.get(field) for field in fields):
        return "invalid"
    if row.get("evaluation_valid") is not True:
        return "invalid"
    if row.get("reached_selection_opportunity") is not True:
        return "unreached"
    if row.get("selection_identified") is not True:
        return "unidentified"
    return "correct" if row.get("slot_reasoning_success") is True else "wrong"


def _summarize_autonomous_rollouts(rows: list[dict]) -> dict:
    """Aggregate preregistered autonomous runs without dropping failed episodes."""
    if not rows:
        raise ValueError("at least one autonomous run is required")
    required_equal = ("task_version", "contract_hash", "scoring_version",
                      "scoring_track", "history_protocol", "condition_id",
                      "oracle_profile", "policy_interface", "policy_profile",
                      "environment_mode", "rebuild_revision")
    reference = rows[0]
    for row in rows:
        if row.get("evaluation_track") != "autonomous_rollout":
            raise ValueError("rollout summary only accepts autonomous_rollout results")
        mismatched = [field for field in required_equal
                      if row.get(field) != reference.get(field)]
        if mismatched:
            raise ValueError(f"rollout results mix evaluation conditions: {mismatched}")

    def status(row: dict) -> str:
        if row.get("evaluation_valid") is not True:
            return "invalid"
        if row.get("reached_selection_opportunity") is not True:
            return "unreached"
        if row.get("selection_identified") is not True:
            return "unidentified"
        return "correct" if row.get("slot_reasoning_success") is True else "wrong"

    statuses = [status(row) for row in rows]
    total = len(rows)
    valid = sum(value != "invalid" for value in statuses)
    reached = sum(value in {"unidentified", "wrong", "correct"} for value in statuses)
    identified = sum(value in {"wrong", "correct"} for value in statuses)
    correct = statuses.count("correct")
    return {
        "task_version": reference.get("task_version"),
        "contract_hash": reference.get("contract_hash"),
        "scoring_version": reference.get("scoring_version"),
        "scoring_track": reference.get("scoring_track"),
        "history_protocol": reference.get("history_protocol"),
        "environment_mode": reference.get("environment_mode"),
        "rebuild_revision": reference.get("rebuild_revision"),
        "oracle_profile": reference.get("oracle_profile"),
        "condition_id": reference.get("condition_id"),
        "policy_interface": reference.get("policy_interface"),
        "fixed_denominator": total,
        "valid_run_count": valid,
        "selection_opportunity_reached_count": reached,
        "selection_identified_count": identified,
        "correct_selection_count": correct,
        "physical_completion_count": sum(
            row.get("physical_completion") is True for row in rows),
        "ranking_eligible_count": sum(
            row.get("ranking_eligible") is True for row in rows),
        "opportunity_reach_rate": reached / total,
        "identification_rate_given_reached": identified / reached if reached else None,
        "selection_accuracy_given_identified": correct / identified if identified else None,
        "correct_selection_rate_all_runs": correct / total,
        "physical_completion_rate_all_runs": sum(
            row.get("physical_completion") is True for row in rows) / total,
        "result_class_counts": {
            value: statuses.count(value)
            for value in ("invalid", "unreached", "unidentified", "wrong", "correct")
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Labference rack-placement recall evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.add_argument("--seed", type=int, default=101)
    validate.add_argument("--policy-profile")
    validate.add_argument("--intervention-spec")
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True, help="run directory")
    run.add_argument("--seed", type=int, default=101)
    run.add_argument("--max-steps", type=int)
    run.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--save-video", action="store_true")
    run.add_argument("--oracle-profile", choices=ORACLES)
    run.add_argument("--policy-url")
    run.add_argument("--policy-profile")
    run.add_argument("--intervention-spec")
    run.add_argument("--pair-id")
    run.add_argument("--member", choices=("single", "a", "b"), default="single")
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    check = sub.add_parser("checkpoints")
    check.add_argument("--config", required=True)
    check.add_argument("--output", required=True)
    check.add_argument("--seed", type=int, default=101)
    check.add_argument("--checkpoint-slot", choices=SLOTS, required=True)
    check.add_argument("--max-steps", type=int)
    check.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    check.add_argument("--save-video", action="store_true")
    check.add_argument("--oracle-profile", choices=ORACLES)
    check.add_argument("--policy-url")
    check.add_argument("--policy-profile")
    check.add_argument("--intervention-spec")
    check.add_argument("--pair-id")
    check.add_argument("--member", choices=("single", "a", "b"), default="single")
    check.add_argument("--motion-variant", choices=("base", "alternate"), default="base")
    check.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    for command in (run, check):
        command.add_argument("--layout-seed", type=int, help=argparse.SUPPRESS)
        command.add_argument("--motion-seed", type=int, help=argparse.SUPPRESS)
        command.add_argument("--policy-seed", type=int, help=argparse.SUPPRESS)
        command.add_argument("--color-seed", type=int, help=argparse.SUPPRESS)
        command.add_argument("--initial-yaw-deg", type=float, help=argparse.SUPPRESS)
        command.add_argument("--motion-yaw-deg", type=float, help=argparse.SUPPRESS)
        command.add_argument("--target-cap-color", choices=CAP_COLORS,
                             help=argparse.SUPPRESS)
        command.add_argument("--collection-slot", choices=SLOTS, help=argparse.SUPPRESS)
        command.add_argument("--dataset-output", help=argparse.SUPPRESS)
    score = sub.add_parser("score")
    score.add_argument("--run", required=True)
    score.add_argument("--trace")
    score.add_argument("--output")
    compare = sub.add_parser("compare")
    compare.add_argument("--plan", required=True)
    compare.add_argument("--runs", nargs="+", required=True)
    compare.add_argument("--output")
    plan = sub.add_parser("plan-pairs")
    plan.add_argument("--config", required=True)
    plan.add_argument("--seeds", nargs="+", type=int, default=list(range(101, 106)))
    plan.add_argument("--output", required=True)
    coverage = sub.add_parser("plan-coverage")
    coverage.add_argument("--configs", nargs="+", required=True)
    coverage.add_argument("--seeds", nargs="+", type=int, default=list(range(101, 106)))
    coverage.add_argument("--output", required=True)
    run_coverage = sub.add_parser("run-coverage")
    run_coverage.add_argument("--plan", required=True)
    run_coverage.add_argument("--output-root", required=True)
    run_coverage.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True)
    run_coverage.add_argument("--save-video", action="store_true")
    run_coverage.add_argument("--max-steps", type=int)
    run_coverage.add_argument("--limit", type=int)
    summarize = sub.add_parser("summarize-coverage")
    summarize.add_argument("--plan", required=True)
    summarize.add_argument("--runs", nargs="*", default=[])
    summarize.add_argument("--output")
    summarize_rollouts = sub.add_parser("summarize-rollouts")
    summarize_rollouts.add_argument("--runs", nargs="+", required=True)
    summarize_rollouts.add_argument("--output")
    collection_plan = sub.add_parser("plan-collection")
    collection_plan.add_argument("--configs", nargs=3, required=True)
    collection_plan.add_argument("--split", choices=("train", "validation"), default="train")
    collection_plan.add_argument("--max-attempts", type=int, default=3)
    collection_plan.add_argument("--output", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--plan", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--resume", action="store_true")
    collect.add_argument("--save-video", action="store_true")
    collect.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True)
    collect.add_argument("--max-steps", type=int)
    collect.add_argument("--limit", type=int)
    validate_dataset_parser = sub.add_parser("validate-dataset")
    validate_dataset_parser.add_argument("--dataset", required=True)
    validate_dataset_parser.add_argument("--output")
    export = sub.add_parser("export-lerobot")
    export.add_argument("--dataset", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--repo-id", default="labference/rack_recall_v4")

    args = parser.parse_args(argv)
    if args.command == "validate":
        source, contract = load_config(args.config, policy_profile=args.policy_profile,
                                       intervention_spec=args.intervention_spec)
        instance = generate_instance(contract, args.seed)
        _dump({"valid": True, "task_version": contract.task_version,
               "evaluation_version": EVALUATION_VERSION, "contract_hash": contract.hash,
               "instance_public": instance.public(),
               "scene_geometry": validate_scene_geometry(contract, instance),
               "policy_profile": source["evaluation"]["policy_profile"],
               "intervention": source["evaluation"]["intervention"]})
    elif args.command in {"run", "checkpoints"}:
        from .runner import run_episode, run_supervised
        runner = run_episode if args.worker else run_supervised
        result = runner(args.config, args.output, seed=args.seed,
                        max_steps=args.max_steps, save_video=args.save_video,
                        headless=args.headless,
                        oracle_profile=args.oracle_profile, policy_url=args.policy_url,
                        policy_profile=args.policy_profile,
                        intervention_spec=args.intervention_spec,
                        pair_id=args.pair_id, member=args.member,
                        motion_variant=(args.motion_variant
                                        if args.command == "checkpoints" else "base"),
                        checkpoint_slot=(args.checkpoint_slot
                                         if args.command == "checkpoints" else None),
                        collection_slot=args.collection_slot,
                        dataset_output=args.dataset_output,
                        instance_overrides={key: value for key, value in {
                            "layout_seed": args.layout_seed,
                            "motion_seed": args.motion_seed,
                            "policy_seed": args.policy_seed,
                            "color_seed": args.color_seed,
                            "initial_yaw_deg": args.initial_yaw_deg,
                            "motion_yaw_deg": args.motion_yaw_deg,
                            "target_cap_color": args.target_cap_color,
                        }.items() if value is not None})
        _dump(result)
    elif args.command == "score":
        root = Path(args.run)
        if root.is_file():
            root = root.parent
        manifest = json.loads((root / "run.json").read_text())
        contract = Contract(**{key: value for key, value in manifest["evaluation"].items()
                               if key in Contract.__dataclass_fields__})
        instance = _instance_from_manifest(manifest["instance"])
        result = apply_runtime_validity(
            score_trace(contract, instance,
                        _load_rows(args.trace or root / "events.jsonl")),
            str(manifest.get("runtime_status", "missing")))
        online_path = root / "online_scores.jsonl"
        if online_path.exists() and online_path.stat().st_size:
            online = _load_rows(online_path)[-1]["scores"]
            result["online_offline_matching"] = online == result
        _dump(result, args.output)
    elif args.command == "compare":
        plan_value = json.loads(Path(args.plan).read_text())
        if plan_value.get("scoring_version") != SCORING_VERSION:
            raise ValueError("pair plan uses an incompatible scoring version")
        _dump(compare_pairs(plan_value["pairs"], [_run_summary(path) for path in args.runs]),
              args.output)
    elif args.command == "plan-pairs":
        source, contract = load_config(args.config)
        evaluation = source["evaluation"]
        pairs = []
        for seed in args.seeds:
            instance = generate_instance(contract, seed)
            free = [slot for slot in SLOTS if slot not in instance.distractor_slots]
            prefix = (f"rack-recall-{contract.difficulty}-"
                      f"{contract.condition_variant}-{seed}")
            pair_id = f"{prefix}-state"
            pairs.append({"pair_id": pair_id, "pair_kind": "different_actual_placement",
                          "task_version": contract.task_version, "contract_hash": contract.hash,
                          "scoring_version": SCORING_VERSION,
                          "scoring_track": evaluation["scoring_track"],
                          "history_protocol": evaluation["history_protocol"],
                          "oracle_profile": evaluation["oracle_profile"],
                          "environment_mode": contract.environment_mode,
                          "rebuild_revision": contract.rebuild_revision,
                          "members": [{"member": "a", "seed": seed,
                                       "checkpoint_slot": free[0]},
                                      {"member": "b", "seed": seed,
                                       "checkpoint_slot": free[-1]}]})
            transform_id = f"{prefix}-transform"
            pairs.append({"pair_id": transform_id, "pair_kind": "same_local_slot_different_transform",
                          "task_version": contract.task_version, "contract_hash": contract.hash,
                          "scoring_version": SCORING_VERSION,
                          "scoring_track": evaluation["scoring_track"],
                          "history_protocol": evaluation["history_protocol"],
                          "oracle_profile": evaluation["oracle_profile"],
                          "environment_mode": contract.environment_mode,
                          "rebuild_revision": contract.rebuild_revision,
                          "members": [{"member": "a", "seed": seed,
                                       "checkpoint_slot": free[len(free) // 2], "motion_variant": "base"},
                                      {"member": "b", "seed": seed,
                                       "checkpoint_slot": free[len(free) // 2], "motion_variant": "alternate"}]})
        _dump({"evaluation_version": EVALUATION_VERSION,
               "scoring_version": SCORING_VERSION, "pairs": pairs,
               "fixed_denominator": True}, args.output)
    elif args.command == "plan-coverage":
        members = []
        conditions = []
        for config in args.configs:
            source, contract = load_config(config)
            evaluation = source["evaluation"]
            # Keep the protocol portable across clones and packaged releases.
            # The runner resolves these paths from its invocation directory.
            config_path = Path(config)
            try:
                portable_config = str(config_path.resolve().relative_to(Path.cwd().resolve()))
            except ValueError:
                portable_config = str(config_path)
            condition_plan = generate_coverage_manifest(contract, tuple(args.seeds))
            conditions.append({"condition": contract.condition_variant,
                               "config": portable_config,
                               "member_count": condition_plan["fixed_denominator"]})
            for member_value in condition_plan["members"]:
                members.append({
                    **member_value,
                    "config": portable_config,
                    "scoring_track": evaluation["scoring_track"],
                    "history_protocol": evaluation["history_protocol"],
                    "oracle_profile": evaluation["oracle_profile"],
                    "condition_variant": contract.condition_variant,
                    "evaluation_track": "verified_state_checkpoint",
                })
        if len({member["member_id"] for member in members}) != len(members):
            raise ValueError("coverage configs generated duplicate member ids")
        _dump({"schema_version": "labference/rack-recall-coverage-v4",
               "evaluation_version": EVALUATION_VERSION,
               "scoring_version": SCORING_VERSION,
               "primary_metric": "slot_reasoning_success",
               "placement_skill_gate_applied": False,
               "fixed_denominator": len(members), "conditions": conditions,
               "members": members}, args.output)
    elif args.command == "run-coverage":
        from .runner import run_supervised
        plan_value = json.loads(Path(args.plan).read_text())
        if plan_value.get("scoring_version") != SCORING_VERSION:
            raise ValueError("coverage plan uses an incompatible scoring version")
        members = plan_value.get("members", [])
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("coverage limit must be positive")
            members = members[:args.limit]
        root = Path(args.output_root).resolve()
        results = []
        for member_value in members:
            output = root / member_value["member_id"]
            run_supervised(member_value["config"], str(output),
                           seed=int(member_value["seed"]),
                           checkpoint_slot=member_value["checkpoint_slot"],
                           max_steps=args.max_steps, save_video=args.save_video,
                           headless=args.headless,
                           pair_id=member_value["member_id"],
                           instance_overrides=member_value)
            results.append(_run_summary(output))
        _dump({"planned": len(plan_value.get("members", [])),
               "executed": len(results), "output_root": str(root),
               "primary_metric": "slot_reasoning_success",
               "benchmark_successful": sum(_benchmark_pass(row) for row in results),
               "physical_completion_successful": sum(
                   row.get("valid_success") is True for row in results)})
    elif args.command == "summarize-coverage":
        plan_value = json.loads(Path(args.plan).read_text())
        if plan_value.get("scoring_version") != SCORING_VERSION:
            raise ValueError("coverage plan uses an incompatible scoring version")
        expected = {member["member_id"]: member for member in plan_value.get("members", [])}
        observed = {}
        for run_path in args.runs:
            row = _run_summary(run_path)
            key = row.get("pair_id")
            if key in observed:
                raise ValueError(f"duplicate coverage member result: {key}")
            observed[key] = row
        unknown = sorted(set(observed) - set(expected))
        if unknown:
            raise ValueError(f"results are not in the coverage plan: {unknown}")
        by_slot = {}
        statuses = {key: _coverage_result_status(member_value, observed.get(key))
                    for key, member_value in expected.items()}
        for slot in SLOTS:
            planned = [key for key, member_value in expected.items()
                       if member_value["checkpoint_slot"] == slot]
            completed = [key for key in planned if key in observed]
            passed = [key for key in completed if statuses[key] == "correct"]
            physically_completed = [
                key for key in completed
                if observed[key].get("valid_success") is True
            ]
            by_slot[slot] = {"planned": len(planned), "completed": len(completed),
                             "reasoning_passed": len(passed),
                             "physical_completion_passed": len(physically_completed),
                             "slot_reasoning_rate_fixed_denominator": (
                                 len(passed) / len(planned) if planned else None),
                             "physical_completion_rate_fixed_denominator": (
                                 len(physically_completed) / len(planned)
                                 if planned else None)}
        _dump({"fixed_denominator": len(expected),
               "completed_count": len(observed),
               "primary_metric": "slot_reasoning_success",
               "placement_skill_gate_applied": False,
               "reasoning_passed_count": sum(status == "correct"
                                               for status in statuses.values()),
               "physical_completion_passed_count": sum(
                   row.get("valid_success") is True for row in observed.values()),
               "coverage_rate": len(observed) / len(expected) if expected else None,
               "slot_reasoning_rate_fixed_denominator": (
                   sum(status == "correct" for status in statuses.values()) /
                   len(expected) if expected else None),
               "physical_completion_rate_fixed_denominator": (
                   sum(row.get("valid_success") is True for row in observed.values()) /
                   len(expected) if expected else None),
               "missing_member_ids": sorted(set(expected) - set(observed)),
               "result_class_counts": {
                   status: sum(value == status for value in statuses.values())
                   for status in ("missing", "invalid", "unreached", "unidentified",
                                  "wrong", "correct")
               },
               "per_slot": by_slot}, args.output)
    elif args.command == "summarize-rollouts":
        _dump(_summarize_autonomous_rollouts(
            [_run_summary(path) for path in args.runs]), args.output)
    elif args.command == "plan-collection":
        from .dataset import generate_collection_plan
        configs = []
        for config in args.configs:
            _, contract = load_config(config)
            path = Path(config)
            try:
                portable = str(path.resolve().relative_to(Path.cwd().resolve()))
            except ValueError:
                portable = str(path.resolve())
            configs.append((portable, contract.condition_variant, contract.hash))
        _dump(generate_collection_plan(
            configs, split=args.split, max_attempts=args.max_attempts), args.output)
    elif args.command == "collect":
        from .dataset import COLLECTION_SCHEMA, validate_hdf5_episode
        from .runner import run_supervised
        plan_value = json.loads(Path(args.plan).read_text())
        if plan_value.get("schema_version") != COLLECTION_SCHEMA:
            raise ValueError("unknown collection plan schema")
        units = list(plan_value.get("units", []))
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("collection limit must be positive")
            units = units[:args.limit]
        root = Path(args.output).resolve()
        root.mkdir(parents=True, exist_ok=True)
        index_path = root / "collection_index.jsonl"
        results = []
        for unit in units:
            if unit.get("contract_hash") is not None:
                _, current_contract = load_config(unit["config"])
                if current_contract.hash != unit["contract_hash"]:
                    raise ValueError(
                        f"collection unit {unit['unit_id']} uses a stale contract hash; "
                        "regenerate the collection plan")
            unit_root = root / unit["unit_id"]
            existing = [validate_hdf5_episode(path) for path in unit_root.rglob("*.h5")]
            if args.resume and any(row["valid"] and row["accepted_for_training"]
                                   for row in existing):
                results.append({"unit_id": unit["unit_id"], "status": "skipped_complete"})
                continue
            accepted = False
            for attempt in range(1, int(unit["max_attempts"]) + 1):
                attempt_root = unit_root / f"attempt_{attempt:02d}"
                episode_path = attempt_root / "artifacts/episode.h5"
                if args.resume and episode_path.is_file():
                    report = validate_hdf5_episode(episode_path)
                    if report["valid"] and report["accepted_for_training"]:
                        accepted = True
                        break
                    continue
                record = {"unit_id": unit["unit_id"], "attempt": attempt,
                          "output": str(attempt_root), "status": "running"}
                with index_path.open("a") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                try:
                    scores = run_supervised(
                        unit["config"], str(attempt_root), seed=int(unit["seed"]),
                        max_steps=args.max_steps, save_video=args.save_video,
                        headless=args.headless,
                        pair_id=unit["unit_id"], collection_slot=unit["collection_slot"],
                        dataset_output=str(episode_path),
                        instance_overrides={key: unit[key] for key in (
                            "layout_seed", "motion_seed", "policy_seed", "color_seed",
                            "initial_yaw_deg", "target_cap_color")})
                    report = validate_hdf5_episode(episode_path)
                    accepted = bool(report["valid"] and report["accepted_for_training"])
                    record.update(status=("accepted" if accepted else "completed_not_accepted"),
                                  scores=scores, dataset=report)
                except RuntimeError as exc:
                    record.update(status="runtime_failed", error=str(exc))
                with index_path.open("a") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                results.append(record)
                if accepted:
                    break
        _dump({"planned": len(units), "accepted": sum(
            row.get("status") in {"accepted", "skipped_complete"} for row in results),
               "output": str(root), "results": results})
    elif args.command == "validate-dataset":
        from .dataset import validate_dataset
        _dump(validate_dataset(args.dataset), args.output)
    elif args.command == "export-lerobot":
        from .dataset import export_lerobot
        _dump(export_lerobot(args.dataset, args.output, repo_id=args.repo_id))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"rack-recall evaluation error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
