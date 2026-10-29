from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from eai4s_eval.eval.model_identity import resolve_model_name, validate_checkpoint_name
from eai4s_eval.eval.task_suite import default_official_suite_id, load_task_catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an evaluation group and emit a pending official-submission entry."
    )
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/eval_runs"))
    parser.add_argument("--eval-group-id", required=True)
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy-provider", choices=("openpi", "gr00t"), required=True)
    parser.add_argument("--suite-id", default=default_official_suite_id())
    parser.add_argument("--required-splits", default="sim_clean,sim_randomized")
    parser.add_argument("--contributor", default="")
    parser.add_argument("--code-url", default="")
    parser.add_argument("--weights-url", default="")
    return parser


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _manifest_provider(manifest: dict[str, Any]) -> str:
    explicit = str(manifest.get("policy_provider") or "").lower()
    if explicit:
        return explicit
    backend = str(manifest.get("policy_backend") or (manifest.get("policy") or {}).get("backend") or "").lower()
    if "gr00t" in backend or backend == "zmq":
        return "gr00t"
    if "openpi" in backend or backend in {"websocket", "http"}:
        return "openpi"
    return ""


def validate_submission(args: argparse.Namespace) -> dict[str, Any]:
    model_name = resolve_model_name(args.policy_provider, args.model_name)
    checkpoint = validate_checkpoint_name(args.checkpoint, required=True)
    splits = [item.strip() for item in str(args.required_splits).split(",") if item.strip()]
    if not splits or any(item not in {"sim_clean", "sim_dev", "sim_randomized"} for item in splits):
        raise ValueError("required splits must be a comma-separated subset of sim_clean, sim_dev, sim_randomized")

    catalog = load_task_catalog(args.suite_id)
    expected = {
        str(task.get("source_task_id") or task["task_id"])
        for task in catalog["tasks"]
        if task.get("status") == "implemented"
    }
    runs: list[tuple[Path, dict[str, Any]]] = []
    for run_dir in sorted(args.outputs_root.resolve().iterdir()):
        if not run_dir.is_dir():
            continue
        manifest = _read_json(run_dir / "run_manifest.json")
        if not isinstance(manifest, dict) or manifest.get("eval_group_id") != args.eval_group_id:
            continue
        runs.append((run_dir, manifest))
    if not runs:
        raise ValueError(f"no task runs found for eval group {args.eval_group_id}")

    errors: list[str] = []
    selected_run_ids: list[str] = []
    for split in splits:
        split_runs = [(path, manifest) for path, manifest in runs if manifest.get("split") == split]
        observed = {str(manifest.get("task_id") or "") for _, manifest in split_runs}
        missing = sorted(expected - observed)
        if missing:
            errors.append(f"{split}: missing implemented tasks: {', '.join(missing)}")
        for run_dir, manifest in split_runs:
            run_id = str(manifest.get("run_id") or run_dir.name)
            if manifest.get("suite_id") != args.suite_id:
                errors.append(f"{run_id}: suite_id mismatch")
            if manifest.get("model_name") != model_name:
                errors.append(f"{run_id}: model_name mismatch")
            if manifest.get("checkpoint") != checkpoint:
                errors.append(f"{run_id}: checkpoint mismatch")
            if _manifest_provider(manifest) != args.policy_provider:
                errors.append(f"{run_id}: policy_provider mismatch")
            if not isinstance(_read_json(run_dir / "summary.json"), dict):
                errors.append(f"{run_id}: missing or invalid summary.json")
            if not isinstance(_read_json(run_dir / "leaderboard_row.json"), dict):
                errors.append(f"{run_id}: missing or invalid leaderboard_row.json")
            selected_run_ids.append(run_id)
    if errors:
        raise ValueError("official submission validation failed:\n- " + "\n- ".join(errors))

    entry = {
        "submission_id": str(args.submission_id),
        "status": "pending",
        "model_name": model_name,
        "checkpoint": checkpoint,
        "policy_provider": args.policy_provider,
        "suite_id": args.suite_id,
        "splits": splits,
        "run_ids": sorted(set(selected_run_ids)),
    }
    for key in ("contributor", "code_url", "weights_url"):
        value = str(getattr(args, key) or "").strip()
        if value:
            entry[key] = value
    return entry


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        entry = validate_submission(args)
    except (OSError, ValueError) as exc:
        print(str(exc))
        return 1
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
