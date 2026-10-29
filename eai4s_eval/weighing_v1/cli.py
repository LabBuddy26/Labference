"""CLI for feedback-weighing episodes and online diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from eai4s_eval.oracle_runtime import resolve_runtime_contract

from .core import (Contract, EASY_TARGETS, HARD_TARGETS, HARD_WEIGHT_SETS, ORACLES,
                   WEIGHTS, balanced_instances, generate_instance, paired_instances)
from .online import EVALUATION_VERSION, compare_oracle_pairs, compare_pairs, score_trace


POLICIES = {"diagnostic_rule_expert", "http_symbolic", "http_array", "openpi_array"}
REPORTS = {"semantic", "decision", "geometry", "execution"}


def load_config(path: str | Path, *, policy_profile: str | Path | None = None,
                intervention_spec: str | Path | None = None) -> tuple[dict, Contract]:
    source = json.loads(Path(path).read_text())
    if not isinstance(source, dict) or not isinstance(source.get("evaluation"), dict):
        raise ValueError("configuration requires an evaluation object")
    ev = source["evaluation"]
    if ev.get("evaluation_version") != EVALUATION_VERSION:
        raise ValueError("unsupported weighing evaluation version")
    interface = ev.get("policy_interface", {})
    if interface.get("name") not in POLICIES:
        raise ValueError("unknown policy interface")
    if interface.get("action_space") not in {"symbolic", "array"}:
        raise ValueError("policy action_space must be symbolic or array")
    if (interface["name"] in {"http_array", "openpi_array"}) != (interface["action_space"] == "array"):
        raise ValueError("policy interface and action space disagree")
    if set(interface.get("supported_reports", [])) - REPORTS:
        raise ValueError("unsupported report kind")
    if set(interface.get("oracle_transports", [])) - {"structured", "text"}:
        raise ValueError("unknown Oracle transport")
    if ev.get("oracle_profile", "none") not in ORACLES:
        raise ValueError("unknown Oracle profile")
    transport = ev.get("oracle_transport", "structured")
    if transport not in {"structured", "text"}:
        raise ValueError("unknown Oracle transport")
    if ev.get("oracle_profile", "none") != "none" and transport not in interface.get("oracle_transports", []):
        raise ValueError("selected policy does not support Oracle transport")
    if ev.get("oracle_profile") == "geometry" and not interface.get("coordinate_input"):
        raise ValueError("Geometry Oracle requires declared coordinate input")
    if ev.get("oracle_profile") == "skill" and not interface.get("selected_operation_input"):
        raise ValueError("Skill Oracle requires selected-operation input")
    if interface["name"].startswith("http_") and not isinstance(interface.get("url"), str):
        raise ValueError("HTTP policy requires url")
    if interface["name"] == "diagnostic_rule_expert" and interface["action_space"] != "symbolic":
        raise ValueError("rule expert uses symbolic operations")
    contract_fields = set(Contract.__dataclass_fields__)
    allowed = contract_fields | {"evaluation_version", "policy_interface", "oracle_profile", "oracle_transport",
                                 "policy_profile", "intervention"}
    unknown = set(ev) - allowed
    if unknown:
        raise ValueError(f"unknown evaluation fields: {sorted(unknown)}")
    contract = Contract(**{key: value for key, value in ev.items() if key in contract_fields})
    if "scene" in source and not isinstance(source["scene"], dict):
        raise ValueError("scene must be an object")
    scene = source.get("scene", {})
    if scene.get("scale_type") != "physical_load_cell" or not isinstance(
            scene.get("scene_revision"), str) or not scene["scene_revision"].strip():
        raise ValueError("weighing scene needs a versioned physical load cell")
    runtime_profile, intervention = resolve_runtime_contract(
        ev, rgb_channels=("front", "wrist", "display", "weights"),
        policy_profile_path=policy_profile, intervention_path=intervention_spec)
    ev["policy_profile"] = runtime_profile.public()
    ev["intervention"] = intervention.__dict__
    ev["oracle_profile"] = intervention.profile
    ev["oracle_transport"] = (intervention.transport if intervention.mode == "assist" else "structured")
    return source, contract


def load_rows(path: str | Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def planned_seeds(path: str | Path, contract: Contract, scene_revision: str) -> list[int]:
    """Resolve an evaluator-only instance plan into reproducible run seeds."""
    from .scene import generate_scene_layout

    plan = json.loads(Path(path).read_text())
    if (plan.get("evaluator_only") is not True or
            plan.get("evaluation_version") != EVALUATION_VERSION or
            plan.get("task_version") != contract.task_version or
            plan.get("contract_hash") != contract.hash or
            plan.get("scene_revision") != scene_revision or
            not isinstance(plan.get("runs"), list) or not plan["runs"]):
        raise ValueError("instance plan does not match the task contract and scene")
    seeds = []
    strata = Counter()
    for row in plan["runs"]:
        if not isinstance(row, dict):
            raise ValueError("instance plan rows must be objects")
        seed = row.get("seed")
        if type(seed) is not int or seed in seeds:
            raise ValueError("instance plan seeds must be distinct integers")
        instance = generate_instance(contract, seed)
        expected = {"mass_seed": instance.mass_seed, "scene_seed": instance.scene_seed,
                    "target_g": instance.target_g,
                    "desired_weights_g": list(instance.desired_weights_g),
                    "initial_weights_g": list(instance.initial_weights_g),
                    "scene_layout_hash": generate_scene_layout(instance.scene_seed).hash}
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"instance plan row differs from generated seed {seed}")
        seeds.append(seed)
        strata[(instance.target_g, instance.desired_weights_g)] += 1
    repetitions = plan.get("repetitions_per_stratum")
    targets = EASY_TARGETS if contract.difficulty == "easy" else HARD_TARGETS
    answers = tuple((weight,) for weight in WEIGHTS) if contract.difficulty == "easy" else HARD_WEIGHT_SETS
    if (type(repetitions) is not int or repetitions <= 0 or
            strata != Counter({(target, answer): repetitions for target in targets for answer in answers})):
        raise ValueError("instance plan does not cover every target-by-answer stratum equally")
    return seeds


def load_run_summary(path: str | Path) -> dict:
    root = Path(path)
    if root.is_file():
        root = root.parent
    manifest = json.loads((root / "run.json").read_text())
    scores = json.loads((root / "scores.json").read_text())
    return {**manifest, **scores, "condition": manifest.get("oracle_profile"),
            "member": manifest.get("pair_member"),
            "first_choice_correct": scores.get("first_ready_choice_correct")}


def _dump(result: dict, output: str | None = None) -> None:
    formatted = json.dumps(result, indent=2, sort_keys=True)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(formatted + "\n")
    print(formatted)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Labference online feedback-weighing evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.add_argument("--seed", type=int, default=101)
    validate.add_argument("--policy-profile")
    validate.add_argument("--intervention-spec")
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--output", required=True, help="calibration report JSON path")
    calibrate.add_argument("--seed", type=int, default=101)
    calibrate.add_argument("--headless", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True, help="run directory")
    run.add_argument("--seed", type=int, default=101)
    run.add_argument("--mass-seed", type=int)
    run.add_argument("--scene-seed", type=int)
    run.add_argument("--pair-seed", type=int)
    run.add_argument("--pair-member", choices=("a", "b"), default="a")
    run.add_argument("--max-steps", type=int)
    run.add_argument("--headless", action="store_true")
    run.add_argument("--save-video", action="store_true")
    run.add_argument("--oracle-profile", choices=ORACLES)
    run.add_argument("--policy-url")
    run.add_argument("--policy-profile")
    run.add_argument("--intervention-spec")
    run.add_argument("--dataset-output", help="current LabBench HDF5 dataset directory")
    run.add_argument("--checkpoint-kind", choices=("semantic", "decision", "geometry", "execution", "behavior"),
                     default="behavior", help=argparse.SUPPRESS)
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    check = sub.add_parser("checkpoints")
    check.add_argument("--config", required=True)
    check.add_argument("--output", required=True)
    check.add_argument("--seed", type=int, default=101)
    check.add_argument("--mass-seed", type=int)
    check.add_argument("--scene-seed", type=int)
    check.add_argument("--pair-seed", type=int)
    check.add_argument("--pair-member", choices=("a", "b"), default="a")
    check.add_argument("--kind", choices=("semantic", "decision", "geometry", "execution", "behavior"),
                       default="behavior")
    check.add_argument("--oracle-profile", choices=ORACLES)
    check.add_argument("--policy-url")
    check.add_argument("--policy-profile")
    check.add_argument("--intervention-spec")
    check.add_argument("--save-video", action="store_true")
    check.add_argument("--max-steps", type=int)
    score = sub.add_parser("score")
    score.add_argument("--run", required=True, help="run.json or run directory")
    score.add_argument("--trace", help="events.jsonl; defaults to run directory")
    score.add_argument("--output")
    compare = sub.add_parser("compare")
    compare.add_argument("--plan", required=True)
    compare.add_argument("--input", help="JSON list of online run summaries")
    compare.add_argument("--runs", nargs="+", help="read online run directories instead of --input")
    compare.add_argument("--output")
    pairs = sub.add_parser("plan-pairs")
    pairs.add_argument("--config", required=True)
    pairs.add_argument("--seeds", nargs="+", type=int, default=list(range(101, 106)))
    pairs.add_argument("--instance-plan")
    pairs.add_argument("--condition", choices=ORACLES, default="none")
    pairs.add_argument("--output")
    instances = sub.add_parser("plan-instances")
    instances.add_argument("--config", required=True)
    instances.add_argument("--start-seed", type=int, default=101)
    instances.add_argument("--repetitions", type=int, default=1)
    instances.add_argument("--output")
    oracle_plan = sub.add_parser("plan-oracle")
    oracle_plan.add_argument("--config", required=True)
    oracle_plan.add_argument("--seeds", nargs="+", type=int, default=list(range(101, 106)))
    oracle_plan.add_argument("--instance-plan")
    oracle_plan.add_argument("--profiles", nargs="+", choices=ORACLES[1:],
                             default=["semantic", "decision", "geometry"])
    oracle_plan.add_argument("--output")
    oracle_compare = sub.add_parser("compare-oracle")
    oracle_compare.add_argument("--plan", required=True)
    oracle_compare.add_argument("--runs", nargs="+", required=True,
                                help="online run directories or run.json paths")
    oracle_compare.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "validate":
        source, contract = load_config(args.config, policy_profile=args.policy_profile,
                                       intervention_spec=args.intervention_spec)
        from .scene import generate_scene_layout
        instance = generate_instance(contract, args.seed)
        layout = generate_scene_layout(instance.scene_seed)
        _dump({"valid": True, "task_version": contract.task_version,
               "evaluation_version": EVALUATION_VERSION, "contract_hash": contract.hash,
               "difficulty": contract.difficulty, "instance": instance.public(),
               "scene_seed": instance.scene_seed, "mass_seed": instance.mass_seed,
               "scene_layout": layout.public(), "scene_layout_hash": layout.hash,
               "scene": source.get("scene", {}),
               "policy_profile": source["evaluation"]["policy_profile"],
               "intervention": source["evaluation"]["intervention"]})
    elif args.command == "calibrate":
        from .calibration import run_calibration
        _dump(run_calibration(args.config, args.output, seed=args.seed))
    elif args.command == "plan-instances":
        source, contract = load_config(args.config)
        from .scene import generate_scene_layout
        selected = balanced_instances(contract, start_seed=args.start_seed,
                                      repetitions=args.repetitions)
        _dump({"task_version": contract.task_version,
               "evaluation_version": EVALUATION_VERSION,
               "contract_hash": contract.hash,
               "scene_revision": source["scene"]["scene_revision"],
               "evaluator_only": True,
               "stratification": "target_g x desired_weights_g",
               "repetitions_per_stratum": args.repetitions,
               "runs": [{"seed": item.seed, "scene_seed": item.scene_seed,
                         "mass_seed": item.mass_seed, "target_g": item.target_g,
                         "desired_weights_g": list(item.desired_weights_g),
                         "initial_weights_g": list(item.initial_weights_g),
                         "scene_layout_hash": generate_scene_layout(item.scene_seed).hash}
                        for item in selected]}, args.output)
    elif args.command == "plan-pairs":
        source, contract = load_config(args.config)
        from .scene import generate_scene_layout
        seeds = (planned_seeds(args.instance_plan, contract, source["scene"]["scene_revision"])
                 if args.instance_plan else args.seeds)
        plan = []
        for seed in seeds:
            a, b = paired_instances(contract, seed)
            plan.append({"pair_id": a.pair_id, "condition": args.condition,
                         "scene_revision": source["scene"]["scene_revision"],
                         "scene_layout_hash": generate_scene_layout(a.scene_seed).hash,
                         "members": [{"member": "a", "seed": a.seed,
                                      "scene_seed": a.scene_seed, "mass_seed": a.mass_seed},
                                     {"member": "b", "seed": b.seed,
                                      "scene_seed": b.scene_seed, "mass_seed": b.mass_seed}]})
        _dump({"task_version": contract.task_version, "contract_hash": contract.hash,
               "pairs": plan}, args.output)
    elif args.command == "plan-oracle":
        source, contract = load_config(args.config)
        from .scene import generate_scene_layout
        seeds = (planned_seeds(args.instance_plan, contract, source["scene"]["scene_revision"])
                 if args.instance_plan else args.seeds)
        if len(set(seeds)) != len(seeds) or len(set(args.profiles)) != len(args.profiles):
            raise ValueError("Oracle plan seeds and profiles must be unique")
        plan = [{"pair_id": f"{contract.difficulty}-{seed}-{profile}",
                 "task_version": contract.task_version, "contract_hash": contract.hash,
                 "scene_revision": source["scene"]["scene_revision"],
                 "scene_layout_hash": generate_scene_layout(seed + 10_000).hash,
                 "seed": seed, "base_condition": "none", "support_condition": profile}
                for seed in seeds for profile in args.profiles]
        _dump({"evaluation_version": EVALUATION_VERSION, "pairs": plan}, args.output)
    elif args.command in {"run", "checkpoints"}:
        from .runner import run_episode, run_supervised
        kind = args.checkpoint_kind if args.command == "run" else args.kind
        runner = run_episode if args.command == "run" and args.worker else run_supervised
        result = runner(args.config, args.output, seed=args.seed,
                        mass_seed=args.mass_seed, scene_seed=args.scene_seed,
                        pair_seed=args.pair_seed, pair_member=args.pair_member,
                        max_steps=getattr(args, "max_steps", None),
                        save_video=args.save_video,
                        headless=getattr(args, "headless", True),
                        oracle_profile=args.oracle_profile,
                        policy_url=args.policy_url, checkpoint_kind=kind,
                        policy_profile=args.policy_profile,
                        intervention_spec=args.intervention_spec,
                        dataset_output=getattr(args, "dataset_output", None))
        _dump(result)
    elif args.command == "score":
        root = Path(args.run)
        if root.is_file():
            root = root.parent
        manifest = json.loads((root / "run.json").read_text())
        contract = Contract(**{key: value for key, value in manifest["evaluation"].items()
                               if key in Contract.__dataclass_fields__})
        from .core import Instance
        instance = Instance(**{key: tuple(value) if key in {"desired_weights_g", "initial_weights_g"}
                               else value for key, value in manifest["instance"].items()})
        trace = load_rows(args.trace or root / "events.jsonl")
        result = score_trace(contract, instance, trace)
        online_path = root / "online_scores.jsonl"
        if online_path.exists() and online_path.stat().st_size:
            online = load_rows(online_path)[-1]["scores"]
            result["online_offline_matching"] = online == result
        _dump(result, args.output)
    elif args.command == "compare":
        if bool(args.runs) == bool(args.input):
            raise ValueError("compare requires exactly one of --runs or --input")
        plan = json.loads(Path(args.plan).read_text())
        rows = ([load_run_summary(path) for path in args.runs] if args.runs else
                json.loads(Path(args.input).read_text()))
        _dump(compare_pairs(plan["pairs"], rows), args.output)
    elif args.command == "compare-oracle":
        plan = json.loads(Path(args.plan).read_text())
        _dump(compare_oracle_pairs(plan["pairs"],
                                   [load_run_summary(path) for path in args.runs]), args.output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"weighing evaluation error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
