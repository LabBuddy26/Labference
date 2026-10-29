from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from eai4s_eval.server.app import create_app


_PUBLIC_MEDIA_PREFIXES = (
    "web_assets/brand/",
    "web_assets/hero/",
    "web_assets/object_images/",
    "web_assets/task_media/",
)
_PUBLIC_MEDIA_SUFFIXES = {".webp", ".mp4", ".jpg", ".jpeg", ".png"}


def _normalize_base_url(value: str) -> str:
    value = str(value or "").strip()
    if not value or value == "/":
        return ""
    if "://" in value or "?" in value or "#" in value:
        raise ValueError("base_url must be a path prefix such as /LabBench")
    return "/" + value.strip("/")


def _write_response(client, route: str, destination: Path) -> None:
    response = client.get(route)
    if response.status_code != 200:
        raise RuntimeError(f"failed to export {route}: HTTP {response.status_code}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.data)


def _referenced_public_media(snapshot: dict) -> set[str]:
    paths = {
        "web_assets/brand/eai4s-labbench-mark.webp",
    }
    if snapshot["site"].get("show_hero_image", False):
        paths.add("web_assets/hero/labbench-family-assets-v2.webp")
    for family in snapshot.get("special_families", []):
        if family.get("scene_image"):
            paths.add(str(family["scene_image"]))
        if family.get("public_status") == "implemented":
            for value in (family.get("demo") or {}).values():
                paths.add(str(value))
    for partner in snapshot["site"].get("partners", []):
        if partner.get("logo"):
            paths.add(str(partner["logo"]))
    for item in snapshot["objects"]["objects"]:
        if item.get("media"):
            paths.add(str(item["media"]))
    for task in snapshot["catalog"]["tasks"]:
        media = task.get("media") or {}
        if media.get("poster"):
            paths.add(str(media["poster"]))
        for demo in media.get("demos") or []:
            for view in demo.get("views") or []:
                for key in ("poster", "preview"):
                    if view.get(key):
                        paths.add(str(view[key]))
    return paths


def _copy_public_media(
    *, source_static: Path, destination_static: Path, snapshot: dict
) -> tuple[int, int]:
    total_bytes = 0
    paths = _referenced_public_media(snapshot)
    for relative in sorted(paths):
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.suffix.lower() not in _PUBLIC_MEDIA_SUFFIXES
            or not relative.startswith(_PUBLIC_MEDIA_PREFIXES)
        ):
            raise RuntimeError(f"unsafe public media reference: {relative}")
        source = source_static / path
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"referenced public media is missing: {relative}")
        destination = destination_static / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        total_bytes += source.stat().st_size
    return len(paths), total_bytes


def export_public_site(
    *,
    output_dir: Path,
    outputs_root: Path,
    release_outputs_root: Path | None = None,
    dataset_statistics_path: Path | None = None,
    base_url: str = "",
    include_public_media: bool = False,
) -> Path:
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    base_url = _normalize_base_url(base_url)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.build-", dir=output_dir.parent)
    )
    try:
        app = create_app(
            outputs_root=outputs_root,
            release_outputs_root=release_outputs_root,
            dataset_statistics_path=dataset_statistics_path,
            console_enabled=False,
            public_base_url=base_url,
            public_media_enabled=include_public_media,
        )
        client = app.test_client()
        routes = {
            "/": temporary_root / "index.html",
            "/tasks/": temporary_root / "tasks" / "index.html",
            "/objects/": temporary_root / "objects" / "index.html",
            "/dataset/": temporary_root / "dataset" / "index.html",
            "/leaderboard/": temporary_root / "leaderboard" / "index.html",
            "/docs/": temporary_root / "docs" / "index.html",
        }
        snapshot_response = client.get("/api/public/site")
        if snapshot_response.status_code != 200:
            raise RuntimeError("failed to build public snapshot")
        snapshot = snapshot_response.get_json()
        for task in snapshot["catalog"]["tasks"]:
            routes[f"/tasks/{task['task_id']}/"] = (
                temporary_root / "tasks" / task["task_id"] / "index.html"
            )
        for item in snapshot["objects"]["objects"]:
            routes[f"/objects/{item['object_id']}/"] = (
                temporary_root / "objects" / item["object_id"] / "index.html"
            )
        for route, destination in routes.items():
            _write_response(client, route, destination)

        source_static = Path(app.static_folder or "")
        if source_static.is_dir():
            shutil.copytree(
                source_static,
                temporary_root / "static",
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("web_assets"),
            )
        media_file_count = 0
        media_bytes = 0
        if include_public_media:
            media_file_count, media_bytes = _copy_public_media(
                source_static=source_static,
                destination_static=temporary_root / "static",
                snapshot=snapshot,
            )

        data_dir = temporary_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        task_payload = dict(snapshot["catalog"])
        task_payload["release_protocols"] = snapshot.get("release_protocols", [])
        payloads = {
            "public-site.json": snapshot,
            "tasks.json": task_payload,
            "objects.json": snapshot["objects"],
            "release-protocols.json": {
                "schema_version": "labbench_release_protocols_public/v1",
                "protocols": snapshot.get("release_protocols", []),
            },
            "release-leaderboard.json": snapshot.get("release_leaderboard", {}),
        }
        for filename, payload in payloads.items():
            (data_dir / filename).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        (temporary_root / "export-manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "labbench_static_export/v1",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "base_url": base_url,
                    "page_count": len(routes),
                    "console_included": False,
                    "public_media_included": bool(include_public_media),
                    "media_file_count": media_file_count,
                    "media_bytes": media_bytes,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary_root / ".nojekyll").touch()

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary_root, output_dir)
        return output_dir
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the read-only Labference public portal.")
    parser.add_argument("--output-dir", type=Path, default=Path("site-dist"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/eval_runs"))
    parser.add_argument("--release-outputs-root", type=Path, default=None)
    parser.add_argument("--dataset-statistics", type=Path, default=None)
    parser.add_argument("--base-url", type=str, default="", help="GitHub Pages path prefix, e.g. /LabBench")
    parser.add_argument(
        "--include-public-media",
        action="store_true",
        help="Include only reviewed WebP/MP4 files referenced by public pages.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = export_public_site(
        output_dir=args.output_dir,
        outputs_root=args.outputs_root,
        release_outputs_root=args.release_outputs_root,
        dataset_statistics_path=args.dataset_statistics,
        base_url=args.base_url,
        include_public_media=args.include_public_media,
    )
    print(f"Exported public site to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
