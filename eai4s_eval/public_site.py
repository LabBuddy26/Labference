from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from eai4s_eval.benchmark_catalog import build_benchmark_dashboard, load_benchmark_catalog


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
SITE_CONFIG_PATH = PACKAGE_ROOT / "configs" / "site" / "public_site_v1.json"
TASK_CONTENT_PATH = PACKAGE_ROOT / "configs" / "site" / "public_task_content_v1.json"
TASK_MEDIA_PATH = PACKAGE_ROOT / "configs" / "site" / "task_media_v1.json"
RELEASE_PROTOCOLS_PATH = PACKAGE_ROOT / "configs" / "site" / "release_protocols_v1.json"
OBJECT_CATALOG_PATH = PACKAGE_ROOT / "configs" / "objects" / "labbench_objects_v1.json"
ASSET_LICENSES_PATH = PACKAGE_ROOT / "configs" / "assets" / "asset_licenses_v1.json"
PUBLIC_IMAGE_AUDIT_PATH = PACKAGE_ROOT / "configs" / "assets" / "public_image_audit_v1.json"
OFFICIAL_SUBMISSIONS_PATH = (
    PACKAGE_ROOT / "configs" / "leaderboards" / "official_submissions_v1.json"
)

PUBLIC_SPLITS = ("sim_randomized", "sim_clean")
RELEASE_SUCCESS_KEYS = (
    "valid_success",
    "ranking_success",
    "benchmark_success",
    "task_completion",
    "clean_completion",
    "success",
)

TRACK_CONTENT_EN = {
    "long_horizon": (
        "Long-Horizon Procedures",
        "Multi-stage laboratory procedures with ordering and state-dependent transitions.",
    ),
    "reasoning_in_action": (
        "Reasoning in Action",
        "Action selection grounded in visual observations and structured laboratory state.",
    ),
}

ATOMIC_ACTION_LABELS_EN = {
    "grasp": "Grasp",
    "place": "Place",
    "press": "Press",
    "pour": "Pour",
    "open": "Open",
    "close": "Close",
    "move": "Move",
    "stir": "Stir",
    "shake": "Shake",
    "lift": "Lift",
    "twist": "Twist",
}

# Maps the raw dataset-instruction strings published in the statistics JSON to
# the official task catalogue (L01-L12 / R01-R12).  Rows whose instruction
# matches a "skill:*" entry are atomic-skill datasets collected outside the
# L/R catalogue.  Rows not present here stay in their raw form.
DATASET_ROW_TASK_MAP_EN = {
    # Long-horizon tasks
    "Clear the bench by placing the vial and tube into their holders": "L12",
    "Insert the NMR object into the sample box": "L08",
    "Open the device, place both beakers, and press the button": "L09",
    "Operate the condition module with the beaker and return it": "L11",
    "Place the three weights onto the scale": "L06",
    "Place the vial into the SamplePlaceBoxB sample box": "L02",
    "Place three vial bottles into separate sample box slots": "L01",
    "Place three vials into the cohort slots": "L10",
    "Pour three beakers into the mixing target and return them": "L04",
    # Reasoning-in-action tasks
    "Insert the NMR tube into rack column 5 row 4": "R09",
    "Insert the tube into the assigned rack socket": "R10",
    "Move the gripper above sample-box socket column 3 row 3": "R11",
    "Read the sample instruction and insert the NMR tube into slot 3": "R08",
    "The centrifuge displays 2000 RPM; press Increase": "R04",
    "The device status is Running; press the Wait button": "R05",
    "The display reads 12.1 degrees Celsius; press the Wait button": "R03",
    "The display reads 30.1 degrees Celsius; press the Open button": "R03",
    "The scale displays 100 g; press the 100 g button": "R01",
    # Atomic-skill datasets (outside the L/R catalogue)
    "Close the laboratory device": "skill:close",
    "Open the laboratory device": "skill:open",
    "Pick up the beaker from the table": "skill:grasp",
    "Pick up the glass rod and stir the liquid": "skill:stir",
    "Pick up the object and place it at the target": "skill:place",
    "Pick up the source container and pour it into the target container": "skill:pour",
}

_BONUS_SUFFIX_RE = re.compile(r"\s*\(\+\d+\)\s*$")


def _strip_bonus_suffix(value: str) -> str:
    """Strip trailing variant markers like "(+162)" from dataset rows."""
    return _BONUS_SUFFIX_RE.sub("", value)

SCORE_METRIC_LABELS_EN = {
    "success": "Completion",
    "geometry_quality_score_100": "Geometry quality",
    "force_quality_score_100": "Force quality",
    "quantity_quality_score_100": "Quantity accuracy",
    "sequence_quality_score_100": "Sequence quality",
    "subgoal_quality_score_100": "Subgoal completion",
    "evidence_quality_score_100": "Evidence quality",
    "efficiency_score_100": "Efficiency",
}


@lru_cache(maxsize=128)
def _file_sha256(path: str, modified_ns: int, size: int) -> str:
    # The stat fingerprint invalidates a cached digest when an asset changes.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matches_audited_file(path: Path, expected_digest: str | None) -> bool:
    if not expected_digest or not path.is_file():
        return False
    stat = path.stat()
    return _file_sha256(str(path), stat.st_mtime_ns, stat.st_size) == expected_digest


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def load_public_site_config(path: str | Path | None = None) -> dict[str, Any]:
    payload = _read_json(Path(path) if path else SITE_CONFIG_PATH)
    if payload.get("schema_version") != "labbench_public_site/v1":
        raise ValueError("unsupported public-site config schema")
    dataset_license = payload.get("dataset_license")
    if not isinstance(dataset_license, dict) or not all(
        isinstance(dataset_license.get(key), str) and dataset_license[key].strip()
        for key in ("name", "spdx_id", "url", "summary", "attribution")
    ):
        raise ValueError("public-site config requires complete dataset_license metadata")
    dataset_access = payload.get("dataset_access")
    if not isinstance(dataset_access, dict) or not isinstance(
        dataset_access.get("provider"), str
    ) or not dataset_access["provider"].strip():
        raise ValueError("public-site config requires complete dataset_access metadata")
    return payload


def load_asset_licenses(path: str | Path | None = None) -> dict[str, Any]:
    payload = _read_json(Path(path) if path else ASSET_LICENSES_PATH)
    if payload.get("schema_version") != "labbench_asset_licenses/v1":
        raise ValueError("unsupported asset-licence registry schema")
    if not isinstance(payload.get("default_record"), dict):
        raise ValueError("asset-licence registry requires default_record")
    if not isinstance(payload.get("objects"), dict):
        raise ValueError("asset-licence registry requires objects")
    if not isinstance(payload.get("media_groups"), dict):
        raise ValueError("asset-licence registry requires media_groups")
    return payload


def load_official_submissions(path: str | Path | None = None) -> dict[str, Any]:
    payload = _read_json(Path(path) if path else OFFICIAL_SUBMISSIONS_PATH)
    if payload.get("schema_version") != "labbench_official_submissions/v1":
        raise ValueError("unsupported official-submission registry schema")
    submissions = payload.get("submissions")
    if not isinstance(submissions, list):
        raise ValueError("official submissions must be a list")
    seen_ids: set[str] = set()
    seen_run_ids: set[str] = set()
    for index, item in enumerate(submissions):
        if not isinstance(item, dict):
            raise ValueError(f"official submission {index} must be an object")
        submission_id = str(item.get("submission_id") or "").strip()
        if not submission_id or submission_id in seen_ids:
            raise ValueError(f"invalid or duplicate submission_id: {submission_id!r}")
        seen_ids.add(submission_id)
        if item.get("status") not in {"verified", "revoked", "pending"}:
            raise ValueError(f"invalid status for official submission {submission_id}")
        run_ids = item.get("run_ids")
        if not isinstance(run_ids, list) or not all(
            isinstance(run_id, str) and run_id.strip() for run_id in run_ids
        ):
            raise ValueError(f"official submission {submission_id} has invalid run_ids")
        duplicate_runs = seen_run_ids.intersection(run_ids)
        if duplicate_runs:
            raise ValueError(
                f"official run ids may belong to only one submission: {sorted(duplicate_runs)}"
            )
        seen_run_ids.update(run_ids)
        if item.get("status") == "verified" and not all(
            isinstance(item.get(key), str) and item[key].strip()
            for key in ("model_name", "checkpoint")
        ):
            raise ValueError(
                f"verified submission {submission_id} requires model_name and checkpoint"
            )
    return payload


def load_release_protocols(path: str | Path | None = None) -> dict[str, Any]:
    """Load the independent release-protocol catalogue used by the public site.

    Release protocols deliberately live outside the canonical 24-task L/R
    catalogue.  Keeping a separate schema prevents a different control rate,
    sensor contract or scoring denominator from silently changing the formal
    benchmark leaderboard.
    """
    payload = _read_json(Path(path) if path else RELEASE_PROTOCOLS_PATH)
    if payload.get("schema_version") != "labbench_release_protocols/v1":
        raise ValueError("unsupported release-protocol schema")
    protocols = payload.get("protocols")
    if not isinstance(protocols, list) or not protocols:
        raise ValueError("release-protocol catalogue requires protocols")
    seen: set[str] = set()
    for index, item in enumerate(protocols):
        if not isinstance(item, dict):
            raise ValueError(f"release protocol {index} must be an object")
        protocol_id = str(item.get("protocol_id") or "").strip()
        if not protocol_id or protocol_id in seen:
            raise ValueError(f"invalid or duplicate release protocol id: {protocol_id!r}")
        seen.add(protocol_id)
        for key in ("family", "family_label_en", "name_en", "summary_en", "task_version", "config_path"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"release protocol {protocol_id} requires {key}")
        if item.get("status") not in {"implemented", "planned"}:
            raise ValueError(f"release protocol {protocol_id} has invalid status")
        for key in ("collection_hz", "physics_hz"):
            if not isinstance(item.get(key), int) or item[key] <= 0:
                raise ValueError(f"release protocol {protocol_id} requires positive {key}")
        for key in ("camera_streams", "action_streams", "metrics"):
            if not isinstance(item.get(key), list) or not item[key] or not all(
                isinstance(value, str) and value.strip() for value in item[key]
            ):
                raise ValueError(f"release protocol {protocol_id} has invalid {key}")
        if not (REPOSITORY_ROOT / item["config_path"]).is_file():
            raise ValueError(f"release protocol config does not exist: {item['config_path']}")
    return payload


def load_public_catalogs(
    *, public_media_enabled: bool | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    benchmark = load_benchmark_catalog()
    content_payload = _read_json(TASK_CONTENT_PATH)
    if content_payload.get("schema_version") != "labbench_public_tasks/v1":
        raise ValueError("unsupported public-task content schema")
    content = content_payload.get("tasks")
    if not isinstance(content, dict):
        raise ValueError("public-task content must contain a tasks object")

    benchmark_ids = {str(task["task_id"]) for task in benchmark["tasks"]}
    if set(content) != benchmark_ids:
        missing = sorted(benchmark_ids - set(content))
        extra = sorted(set(content) - benchmark_ids)
        raise ValueError(f"public-task content mismatch: missing={missing}, extra={extra}")

    object_catalog = _read_json(OBJECT_CATALOG_PATH)
    asset_licenses = load_asset_licenses()
    if object_catalog.get("schema_version") != "labbench_object_catalog/v1":
        raise ValueError("unsupported object-catalog schema")
    objects = object_catalog.get("objects")
    categories = object_catalog.get("categories")
    if not isinstance(objects, list) or not isinstance(categories, list):
        raise ValueError("object catalog requires categories and objects lists")
    category_ids = {str(item.get("category_id")) for item in categories}
    category_by_id = {str(item["category_id"]): item for item in categories}
    object_ids = [str(item.get("object_id") or "") for item in objects]
    if not all(object_ids) or len(object_ids) != len(set(object_ids)):
        raise ValueError("object catalog contains missing or duplicate object ids")
    licence_object_ids = {str(value) for value in asset_licenses["objects"]}
    if licence_object_ids != set(object_ids):
        missing = sorted(set(object_ids) - licence_object_ids)
        extra = sorted(licence_object_ids - set(object_ids))
        raise ValueError(
            f"asset-licence object mismatch: missing={missing}, extra={extra}"
        )

    for item in objects:
        object_id = str(item["object_id"])
        if item.get("category_id") not in category_ids:
            raise ValueError(f"object {object_id} references an unknown category")
        if item.get("status") not in {"available", "planned"}:
            raise ValueError(f"object {object_id} has an invalid status")
        asset_paths = item.get("asset_paths")
        if not isinstance(asset_paths, list):
            raise ValueError(f"object {object_id} asset_paths must be a list")
        if (
            item.get("status") == "available"
            and not asset_paths
            and item.get("asset_distribution") != "not_bundled"
        ):
            raise ValueError(f"available object {object_id} has no asset path")
        for asset_path in asset_paths:
            if not (REPOSITORY_ROOT / str(asset_path)).is_file():
                raise ValueError(f"object {object_id} asset does not exist: {asset_path}")

    object_id_set = set(object_ids)
    public_tracks = []
    for raw_track in benchmark["tracks"]:
        track_id = str(raw_track["track_id"])
        label_en, description_en = TRACK_CONTENT_EN[track_id]
        public_tracks.append(
            {
                "track_id": track_id,
                "label": label_en,
                "prefix": raw_track["prefix"],
                "description": description_en,
                "task_count": raw_track["task_count"],
            }
        )
    public_atomic_actions = [
        {
            "action_id": str(item["action_id"]),
            "label": ATOMIC_ACTION_LABELS_EN[str(item["action_id"])],
        }
        for item in benchmark["atomic_actions"]
    ]
    action_labels = {
        item["action_id"]: item["label"] for item in public_atomic_actions
    }
    track_by_id = {item["track_id"]: item for item in public_tracks}
    object_name_by_id = {
        str(item["object_id"]): str(item["name_en"]) for item in objects
    }
    tasks: list[dict[str, Any]] = []
    object_tasks: dict[str, list[str]] = {object_id: [] for object_id in object_ids}
    if public_media_enabled is None:
        public_media_enabled = bool(
            load_public_site_config().get("public_media_enabled", True)
        )
    media_payload = _read_json(TASK_MEDIA_PATH)
    if media_payload.get("schema_version") != "labbench_task_media/v1":
        raise ValueError("unsupported task-media schema")
    media_by_task: dict[str, list[dict[str, Any]]] = {}
    for item in (media_payload.get("datasets") or []) if public_media_enabled else []:
        media_task_id = item.get("task_id")
        if media_task_id is None:
            continue
        media_task_id = str(media_task_id)
        if media_task_id not in benchmark_ids:
            raise ValueError(f"task media references unknown public task: {media_task_id}")
        variant = str(item["variant"])
        media_by_task.setdefault(media_task_id, []).append(
            {
                "variant": variant,
                "title_en": str(item["title_en"]),
                "scene_description_en": str(item["scene_description_en"]),
                "views": [
                    {
                        "view_id": view,
                        "label_en": "Front camera" if view == "front" else "Wrist camera",
                        "poster": f"web_assets/task_media/{media_task_id.lower()}-{variant}-{view}.webp",
                        "preview": f"web_assets/task_media/{media_task_id.lower()}-{variant}-{view}.mp4",
                    }
                    for view in ("front", "wrist")
                ],
            }
        )
    for raw_task in benchmark["tasks"]:
        task_id = str(raw_task["task_id"])
        public = content[task_id]
        if not isinstance(public, dict):
            raise ValueError(f"public content for {task_id} must be an object")
        referenced_objects = [str(value) for value in public.get("object_ids") or []]
        unknown_objects = set(referenced_objects) - object_id_set
        if unknown_objects:
            raise ValueError(f"task {task_id} references unknown objects: {sorted(unknown_objects)}")
        for object_id in referenced_objects:
            object_tasks[object_id].append(task_id)
        task = {
            key: deepcopy(raw_task[key])
            for key in ("task_id", "track_id", "atomic_actions", "status", "task_spec")
        }
        task.update(public)
        task["availability"] = "available" if raw_task.get("status") == "implemented" else "planned"
        task["track"] = deepcopy(track_by_id[str(raw_task["track_id"])])
        task["asset_names_en"] = [object_name_by_id[value] for value in referenced_objects]
        task["simulation_en"] = (
            "Rigid-body simulation with task-specific geometry, contact, sequence and "
            "stability predicates."
        )
        task["success_evaluation_en"] = (
            "Completion and task-specific quality are evaluated from independent simulator "
            "state rather than controller phase completion."
        )
        task["failure_evaluation_en"] = (
            "Wrong-target placement, object loss, unsafe contact, invalid scene state or "
            "timeout is recorded as a failure when applicable."
        )
        task["score_components"] = []
        for component in raw_task.get("score_components") or []:
            metric = str(component.get("metric") or "")
            task["score_components"].append(
                {
                    "component": SCORE_METRIC_LABELS_EN.get(metric, "Task quality"),
                    "weight": component["weight"],
                    "metric": metric,
                }
            )
        task["atomic_action_labels"] = [
            action_labels.get(str(action), str(action))
            for action in raw_task.get("atomic_actions") or []
        ]
        demos = deepcopy(media_by_task.get(task_id, []))
        task["media"] = {
            "poster": demos[0]["views"][0]["poster"] if demos else None,
            "demos": demos,
        }
        tasks.append(task)

    # Put runnable tasks first in every public catalogue view; roadmap items
    # remain visible at the end for transparency.
    tasks.sort(key=lambda item: (item.get("availability") != "available", str(item.get("task_id") or "")))

    enriched_objects = []
    image_audit = (_read_json(PUBLIC_IMAGE_AUDIT_PATH).get("objects", {})
                   if PUBLIC_IMAGE_AUDIT_PATH.is_file() else {})
    for raw_object in objects:
        item = {
            key: deepcopy(raw_object[key])
            for key in (
                "object_id", "name_en", "category_id", "description_en", "status", "tags"
            )
        }
        category = category_by_id[str(item["category_id"])]
        item["category_label_en"] = category["label_en"]
        item["task_ids"] = sorted(object_tasks[str(item["object_id"])])
        item["image_kind"] = str(raw_object.get("image_kind", "usd_render"))
        source_assets = raw_object.get("asset_paths") or []
        has_source_asset = bool(source_assets and
            (REPOSITORY_ROOT / source_assets[0]).is_file())
        if has_source_asset:
            source_record = (REPOSITORY_ROOT / source_assets[0]).with_suffix(".source.json")
            if source_record.is_file():
                try:
                    record = _read_json(source_record)
                    generator = (REPOSITORY_ROOT / record["generator"]).resolve()
                    has_source_asset = (generator.is_relative_to(REPOSITORY_ROOT)
                        and _matches_audited_file(generator, record.get("generator_sha256"))
                        and _matches_audited_file(REPOSITORY_ROOT / source_assets[0],
                                                record.get("asset_sha256")))
                except (OSError, ValueError, KeyError, TypeError):
                    has_source_asset = False
        preview_path = f"web_assets/object_images/{item['object_id']}.webp"
        audit = image_audit.get(item["object_id"], {})
        provenance_path = (PACKAGE_ROOT / "server/static" / preview_path).with_suffix(".provenance.json")
        if provenance_path.is_file():
            try:
                audit = _read_json(provenance_path)
            except (OSError, ValueError):
                audit = {}
        verified_preview = bool(
            has_source_asset and audit.get("classification") == "usd_render"
            and audit.get("source_asset") == source_assets[0]
            and audit.get("image") == f"eai4s_eval/server/static/{preview_path}"
            and _matches_audited_file(REPOSITORY_ROOT / source_assets[0], audit.get("source_sha256"))
            and _matches_audited_file(PACKAGE_ROOT / "server/static" / preview_path,
                                     audit.get("image_sha256")))
        item["media"] = (
            None
            if (not public_media_enabled or raw_object.get("asset_distribution") == "not_bundled"
                or item["image_kind"] != "usd_render" or not verified_preview)
            else preview_path
        )
        # The public catalogue is a presentation of the implemented laboratory
        # asset set.  Keep planned and robot entries available to the internal
        # task/object validation above, but do not expose them as catalogue
        # objects or generate public media for them.
        if item["status"] != "available" or item["category_id"] == "robots":
            continue
        enriched_objects.append(item)
    visible_category_ids = {str(item["category_id"]) for item in enriched_objects}
    object_catalog = deepcopy(object_catalog)
    object_catalog["categories"] = [
        {"category_id": item["category_id"], "label_en": item["label_en"]}
        for item in categories
        if str(item["category_id"]) in visible_category_ids
    ]
    object_catalog["objects"] = enriched_objects

    public_catalog = {
        "schema_version": "labbench_public_catalog/v1",
        "catalog_id": benchmark["catalog_id"],
        "version": benchmark["version"],
        "tracks": public_tracks,
        "atomic_actions": public_atomic_actions,
        "tasks": tasks,
    }
    return public_catalog, object_catalog


def _verified_submission_by_run_id(
    registry: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(run_id): submission
        for submission in registry.get("submissions") or []
        if submission.get("status") == "verified"
        for run_id in submission.get("run_ids") or []
    }


def _submission_by_model(registry: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for item in registry.get("submissions") or []:
        if item.get("status") != "verified":
            continue
        key = (str(item.get("model_name") or ""), str(item.get("checkpoint") or ""))
        result[key] = item
    return result


def build_public_leaderboard(
    runs: Iterable[dict[str, Any]],
    *,
    split: str,
    registry: dict[str, Any] | None = None,
    public_catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if split not in PUBLIC_SPLITS:
        raise ValueError(f"unsupported public leaderboard split: {split}")
    registry = registry or load_official_submissions()
    public_catalog = public_catalog or load_public_catalogs()[0]
    submission_by_run_id = _verified_submission_by_run_id(registry)
    selected = []
    for run in runs:
        submission = submission_by_run_id.get(str(run.get("run_id") or ""))
        if submission is None or run.get("split") != split:
            continue
        if str(run.get("model_name") or "") != str(submission.get("model_name") or ""):
            continue
        if str(run.get("checkpoint") or "") != str(submission.get("checkpoint") or ""):
            continue
        expected_provider = str(submission.get("policy_provider") or "")
        if expected_provider and str(run.get("policy_provider") or "") != expected_provider:
            continue
        expected_suite = str(submission.get("suite_id") or "")
        if expected_suite and str(run.get("suite_id") or "") != expected_suite:
            continue
        allowed_splits = submission.get("splits") or []
        if allowed_splits and split not in allowed_splits:
            continue
        selected.append(run)
    dashboard = build_benchmark_dashboard(selected)
    public_tracks_by_id = {
        str(item["track_id"]): item for item in public_catalog["tracks"]
    }
    public_tasks_by_id = {
        str(item["task_id"]): item for item in public_catalog["tasks"]
    }
    public_actions_by_id = {
        str(item["action_id"]): item for item in public_catalog["atomic_actions"]
    }
    submission_by_model = _submission_by_model(registry)
    for model in dashboard["models"]:
        submission = submission_by_model.get(
            (str(model.get("model_name") or ""), str(model.get("checkpoint") or "")),
            {},
        )
        model["submission"] = {
            key: submission.get(key)
            for key in (
                "submission_id", "contributor", "verified_at", "code_url", "weights_url",
                "policy_provider", "suite_id", "splits",
            )
            if submission.get(key) not in (None, "")
        }
        model["tracks"] = [
            {
                **deepcopy(public_tracks_by_id[str(item["track_id"])]),
                **{
                    key: deepcopy(value)
                    for key, value in item.items()
                    if key
                    in {
                        "evaluated_task_count",
                        "coverage_rate",
                        "success_rate",
                        "score_100",
                        "score_coverage_rate",
                    }
                },
            }
            for item in model["tracks"]
        ]
        model["tasks"] = [
            {
                **deepcopy(public_tasks_by_id[str(item["task_id"])]),
                **{
                    key: deepcopy(value)
                    for key, value in item.items()
                    if key
                    in {
                        "evaluated",
                        "num_trials",
                        "num_valid_trials",
                        "num_success",
                        "success_rate",
                        "score_100",
                        "score_status",
                        "split",
                        "stage_completion_mean",
                        "inference_latency_p95_ms",
                        "failure_category_distribution",
                        "failure_code_distribution",
                    }
                },
            }
            for item in model["tasks"]
            if str(item["task_id"]) in public_tasks_by_id
            and public_tasks_by_id[str(item["task_id"])].get("availability") == "available"
        ]
        model["atomic_actions"] = [
            {
                **deepcopy(public_actions_by_id[str(item["action_id"])]),
                **{
                    key: deepcopy(value)
                    for key, value in item.items()
                    if key
                    in {
                        "attempts",
                        "successes",
                        "success_rate",
                        "task_count",
                        "task_ids",
                        "sources",
                        "failure_code_distribution",
                    }
                },
            }
            for item in model["atomic_actions"]
        ]
    dashboard["models"].sort(key=lambda item: str(item.get("model_name") or "").lower())
    dashboard["models"].sort(
        key=lambda item: (
            item.get("overall_score_100") is not None,
            float(item["overall_score_100"])
            if item.get("overall_score_100") is not None
            else -1.0,
            float(item["overall_success_rate"])
            if item.get("overall_success_rate") is not None
            else -1.0,
            float((item.get("task_coverage") or {}).get("rate") or 0.0),
        ),
        reverse=True,
    )
    for rank, model in enumerate(dashboard["models"], start=1):
        model["rank"] = rank
    dashboard["schema_version"] = "labbench_public_leaderboard/v1"
    dashboard["split"] = split
    dashboard["ranking_metric"] = "overall_score_100"
    dashboard["verified_submission_count"] = len(dashboard["models"])
    dashboard["tracks"] = deepcopy(public_catalog["tracks"])
    dashboard["atomic_actions"] = deepcopy(public_catalog["atomic_actions"])
    # Planned catalogue entries remain discoverable in the task catalogue but
    # are intentionally absent from the public evaluation leaderboard.
    dashboard["tasks"] = deepcopy([
        task for task in public_catalog["tasks"] if task.get("availability") == "available"
    ])
    return dashboard


def _release_success_value(run: dict[str, Any]) -> bool | None:
    """Return the task-owned success flag without treating missing data as false."""
    scores = run.get("scores") if isinstance(run.get("scores"), dict) else run
    for key in RELEASE_SUCCESS_KEYS:
        value = scores.get(key)
        if isinstance(value, bool):
            return value
    return None


def _release_metric_value(run: dict[str, Any], key: str) -> float | None:
    scores = run.get("scores") if isinstance(run.get("scores"), dict) else run
    value = scores.get(key)
    if not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def build_release_leaderboard(
    release_runs: Iterable[dict[str, Any]],
    *,
    protocols: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate the independent release protocols by model/checkpoint.

    The runner emits task-owned metrics rather than the L/R ``leaderboard_row``
    shape.  This adapter exposes a stable public result contract while keeping
    the original fields (validity, completion, decision and safety metrics) in
    each task row for diagnostics.
    """
    protocols = protocols or load_release_protocols()
    protocol_items = [deepcopy(item) for item in protocols.get("protocols", [])]
    protocol_by_id = {str(item["protocol_id"]): item for item in protocol_items}
    runs = [dict(item) for item in release_runs
            if str(item.get("protocol_id") or "") in protocol_by_id
            and (not item.get("task_version") or item["task_version"] ==
                 protocol_by_id[str(item["protocol_id"])]["task_version"])]
    models: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        protocol = protocol_by_id[str(run["protocol_id"])]
        # Reference executors validate the scene/recorder but are not model
        # submissions and must never receive a public ranking position.
        if run.get("ranking_eligible") is not True:
            continue
        model_name = str(run.get("model_name") or "unknown model")
        checkpoint = str(run.get("checkpoint") or "unknown")
        models.setdefault((model_name, checkpoint), []).append(run)

    model_rows: list[dict[str, Any]] = []
    for (model_name, checkpoint), model_runs in models.items():
        task_rows: list[dict[str, Any]] = []
        for protocol in protocol_items:
            task_runs = [item for item in model_runs if item.get("protocol_id") == protocol["protocol_id"]]
            successes = [_release_success_value(item) for item in task_runs]
            scored = [item for item in successes if item is not None]
            valid_values = [item.get("evaluation_valid") for item in task_runs if isinstance(item.get("evaluation_valid"), bool)]
            success_rate = sum(bool(value) for value in scored) / len(scored) if scored else None
            metric_values: dict[str, float | None] = {}
            for metric in protocol.get("metrics") or []:
                numeric = [_release_metric_value(item, metric) for item in task_runs]
                numeric = [value for value in numeric if value is not None]
                metric_values[metric] = sum(numeric) / len(numeric) if numeric else None
            task_rows.append({
                "protocol_id": protocol["protocol_id"],
                "name_en": protocol["name_en"],
                "family": protocol["family"],
                "difficulty": protocol["difficulty"],
                "evaluated": len(scored),
                "num_trials": len(task_runs),
                "num_success": sum(bool(value) for value in scored),
                "success_rate": success_rate,
                "score_100": None if success_rate is None else round(success_rate * 100.0, 4),
                "evaluation_valid_rate": (
                    sum(bool(value) for value in valid_values) / len(valid_values)
                    if valid_values else None
                ),
                "metrics": metric_values,
                "runs": [
                    {
                        "run_id": item.get("run_id"),
                        "created_at": item.get("created_at"),
                        "status": item.get("runtime_status"),
                        "evaluation_valid": item.get("evaluation_valid"),
                        "success": _release_success_value(item),
                    }
                    for item in task_runs
                ],
            })
        evaluated = [item for item in task_rows if item["success_rate"] is not None]
        overall_score = (
            sum(float(item["score_100"]) for item in evaluated) / len(evaluated)
            if evaluated else None
        )
        trial_successes = sum(item["num_success"] for item in task_rows)
        trial_count = sum(item["evaluated"] for item in task_rows)
        model_rows.append({
            "model_id": "release-" + hashlib.sha1(
                f"{model_name}\0{checkpoint}".encode("utf-8")
            ).hexdigest()[:12],
            "model_name": model_name,
            "checkpoint": checkpoint,
            "overall_score_100": None if overall_score is None else round(overall_score, 4),
            "overall_success_rate": trial_successes / trial_count if trial_count else None,
            "task_coverage": {
                "evaluated": len(evaluated),
                "total": len(protocol_items),
                "rate": len(evaluated) / len(protocol_items) if protocol_items else 0.0,
            },
            "tasks": task_rows,
            "protocol_family_coverage": sorted({item["family"] for item in evaluated}),
        })

    model_rows.sort(
        key=lambda item: (
            item.get("overall_score_100") is not None,
            float(item.get("overall_score_100") or -1.0),
            float(item.get("overall_success_rate") or -1.0),
            item.get("model_name", "").lower(),
        ),
        reverse=True,
    )
    for rank, model in enumerate(model_rows, start=1):
        model["rank"] = rank
    return {
        "schema_version": "labbench_release_leaderboard/v1",
        "leaderboard_id": str(protocols.get("leaderboard_id") or "labference_release_protocols_v1"),
        "title_en": str(protocols.get("title_en") or "Independent release protocols"),
        "description_en": str(protocols.get("description_en") or ""),
        "protocol_count": len(protocol_items),
        "protocols": protocol_items,
        "models": model_rows,
        "verified_submission_count": len(model_rows),
        "run_count": len(runs),
    }


def build_featured_leaderboard(release_runs, *, protocols=None):
    """Versioned four-task scores; retain the legacy eight-protocol builder."""
    from eai4s_eval.eval.batch_results import FEATURED, SCORE_VERSION
    protocols = deepcopy(protocols or load_release_protocols())
    protocols['protocols'] = [p for p in protocols['protocols'] if p['protocol_id'] in FEATURED]
    groups = {}
    for original in release_runs:
        if original.get('protocol_id') not in FEATURED:
            continue
        if original.get('execution_mode') == 'reference':
            continue
        if original.get('ranking_eligible') is not True and original.get('execution_mode') != 'native':
            continue
        item = deepcopy(original)
        item['source_ranking_eligible'] = original.get('ranking_eligible') is True
        if item.get('evaluation_valid') is not True:
            item['scores'] = {}
        # Keep invalid attempts visible in the coverage, but never scored.
        item['ranking_eligible'] = True
        key = (item.get('model_name'), item.get('checkpoint'), item.get('split', 'unknown'),
               item.get('eval_group_id', 'legacy-unverified'))
        groups.setdefault(key, []).append(item)
    models = []
    total_runs = 0
    for key, items in groups.items():
        unique = {}
        duplicate = False
        for item in items:
            identity = (item['protocol_id'], item.get('seed', item.get('run_id')))
            if identity in unique:
                duplicate = True
            unique[identity] = item
        items = list(unique.values())
        total_runs += len(items)
        legacy = build_release_leaderboard(items, protocols=protocols)
        if not legacy['models']:
            continue
        model = legacy['models'][0]
        complete = not duplicate
        for task in model['tasks']:
            task_items = [r for r in items if r['protocol_id'] == task['protocol_id']]
            declarations = [r.get('planned_seeds') for r in task_items]
            expected = declarations[0] if declarations and isinstance(declarations[0], list) else None
            valid_items = [r for r in task_items if r.get('evaluation_valid') is True
                           and isinstance(_release_success_value(r), bool)]
            task_complete = (bool(expected) and len(expected) == len(set(expected))
                             and all(d == expected for d in declarations)
                             and {r.get('seed') for r in valid_items} == set(expected)
                             and all(r['source_ranking_eligible'] for r in valid_items)
                             and len(valid_items) == len(expected) == len(task_items))
            task['complete'] = bool(task_complete)
            task['planned'] = len(expected) if expected else None
            complete = complete and task_complete
        model.update(provisional_score_100=model['overall_score_100'], complete=bool(complete),
                     score_version=SCORE_VERSION, split=key[2], eval_group_id=key[3])
        if not complete:
            model['overall_score_100'] = None
        model['model_id'] += '-' + hashlib.sha1(str(key[2:]).encode()).hexdigest()[:8]
        model['family_coverage'] = {'evaluated': len(model['protocol_family_coverage']), 'total': 4}
        models.append(model)
    models.sort(key=lambda m: (m['overall_score_100'] is None,
                              -float(m['overall_score_100'] or 0), str(m.get('model_name', '')).lower(),
                              str(m.get('checkpoint', '')), str(m['split']), str(m['eval_group_id'])))
    rank = 0
    for model in models:
        if model['complete']:
            rank += 1
            model['rank'] = rank
        else:
            model['rank'] = None
    return dict(schema_version='labference_featured_leaderboard/v2', score_version=SCORE_VERSION,
                leaderboard_id='labference_special_featured_v1', title_en='Four featured workflows',
                description_en='Equal-weight task success rates. Partial results are not ranked.',
                protocol_count=4, protocols=protocols['protocols'], models=models,
                verified_submission_count=sum(m['complete'] for m in models), run_count=total_runs)


def build_special_task_families(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Public release state is curated separately from runtime implementation."""
    protocols = {item["protocol_id"]: item for item in config["protocols"]}
    families = []
    seen = set()
    for raw in config.get("families", []):
        family = deepcopy(raw)
        key = family["family"]
        featured = protocols[family["featured_protocol_id"]]
        if key in seen or featured["family"] != key or family["public_status"] not in {"implemented", "pending"}:
            raise ValueError("invalid special task family presentation")
        seen.add(key)
        family["name_en"] = featured["family_label_en"]
        family["summary_en"] = featured["summary_en"]
        family["protocols"] = [deepcopy(item) for item in protocols.values() if item["family"] == key]
        family["status_label"] = "Implemented" if family["public_status"] == "implemented" else "Pending implementation"
        if family["public_status"] != "implemented":
            family.pop("demo", None)
        families.append(family)
    return families


def build_public_snapshot(
    runs: Iterable[dict[str, Any]],
    *,
    dataset_statistics: dict[str, Any] | None = None,
    public_media_enabled: bool | None = None,
    release_runs: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    runs = list(runs)
    site = load_public_site_config()
    if public_media_enabled is not None:
        site["public_media_enabled"] = bool(public_media_enabled)
    catalog, objects = load_public_catalogs(
        public_media_enabled=bool(site.get("public_media_enabled", True))
    )
    registry = load_official_submissions()
    public_statistics = deepcopy(dataset_statistics)
    if isinstance(public_statistics, dict):
        task_name_by_id = {
            str(task["task_id"]): str(task["name_en"])
            for task in catalog["tasks"]
        }
        for section_name in ("real_robot_teleoperation", "umi", "simulation"):
            section = public_statistics.get(section_name)
            rows = section.get("rows") if isinstance(section, dict) else None
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                row["mapped"] = False
                task_id = str(row.get("task_id") or "")
                if task_id in task_name_by_id:
                    row["task_name"] = task_name_by_id[task_id]
                    row["mapped"] = True
                    continue
                # Raw collection rows are keyed by their lerobot_* slug; map
                # the human-readable instruction to the official task id.
                instruction = _strip_bonus_suffix(str(row.get("task_name") or ""))
                catalog_id = DATASET_ROW_TASK_MAP_EN.get(instruction)
                if catalog_id:
                    row["task_id"] = catalog_id
                    row["mapped"] = True
                    if catalog_id in task_name_by_id:
                        row["task_name"] = task_name_by_id[catalog_id]
                    # skill:* rows keep their descriptive instruction text.
                elif any("\u3400" <= character <= "\u9fff" for character in str(row.get("task_name") or "")):
                    row["task_name"] = task_id or "Unmapped task"
    # Ordinary results belong only to internal reports/console, even when
    # this snapshot is served by an application with the console enabled.
    leaderboards = {}
    release_protocol_catalog = load_release_protocols()
    special_families = build_special_task_families(release_protocol_catalog)
    release_leaderboard = build_featured_leaderboard(
        list(release_runs or []), protocols=release_protocol_catalog
    )
    release_leaderboard["family_count"] = len(special_families)
    for model in release_leaderboard["models"]:
        model["family_coverage"] = {"evaluated": len(model["protocol_family_coverage"]),
                                    "total": len(special_families)}
    for family in special_families:
        for item in objects["objects"]:
            if item["object_id"] in family.get("object_ids", []):
                item.setdefault("special_families", []).append({
                    "family": family["family"], "name_en": family["name_en"],
                    "protocol_id": family["featured_protocol_id"]})
    available_tasks = sum(task["availability"] == "available" for task in catalog["tasks"])
    available_objects = sum(item["status"] == "available" for item in objects["objects"])
    return {
        "schema_version": "labbench_public_snapshot/v1",
        "site": site,
        "catalog": catalog,
        "release_protocols": release_protocol_catalog["protocols"],
        "special_families": special_families,
        "objects": objects,
        "dataset_statistics": public_statistics,
        "leaderboards": leaderboards,
        "release_leaderboard": release_leaderboard,
        "summary": {
            "task_count": len(catalog["tasks"]),
            "special_task_count": len(special_families),
            "implemented_special_task_count": sum(item["public_status"] == "implemented" for item in special_families),
            "available_task_count": available_tasks,
            "planned_task_count": len(catalog["tasks"]) - available_tasks,
            "object_count": len(objects["objects"]),
            "available_object_count": available_objects,
            "track_count": len(catalog["tracks"]),
            "atomic_action_count": len(catalog["atomic_actions"]),
            "release_protocol_count": len(release_leaderboard["protocols"]),
            "release_run_count": release_leaderboard["run_count"],
        },
    }


def public_task(snapshot: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    normalized = task_id.upper()
    return next(
        (task for task in snapshot["catalog"]["tasks"] if task["task_id"] == normalized),
        None,
    )


def public_release_protocol(snapshot: dict[str, Any], protocol_id: str) -> dict[str, Any] | None:
    normalized = str(protocol_id or "").strip().lower()
    return next(
        (item for item in snapshot.get("release_protocols", [])
         if str(item.get("protocol_id") or "").lower() == normalized),
        None,
    )


def public_object(snapshot: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    return next(
        (item for item in snapshot["objects"]["objects"] if item["object_id"] == object_id),
        None,
    )
