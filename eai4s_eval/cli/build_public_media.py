from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from eai4s_eval.public_site import load_public_catalogs


STATIC_ROOT = Path(__file__).resolve().parents[1] / "server" / "static"
WEB_ASSETS_ROOT = STATIC_ROOT / "web_assets"
DEFAULT_MEDIA_MANIFEST = (
    Path(__file__).resolve().parents[1] / "configs" / "site" / "task_media_v1.json"
)


def _parse_task_video(value: str) -> tuple[str, Path]:
    task_id, separator, raw_path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("expected TASK_ID=/path/to/video.mp4")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"video does not exist: {path}")
    return task_id.upper(), path


def _run_ffmpeg(arguments: list[str]) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is required to build public task media")
    completed = subprocess.run(
        [executable, "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {completed.returncode}")


def _video_duration(source: Path) -> float:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise RuntimeError("ffprobe is required to select task poster frames")
    completed = subprocess.run(
        [
            executable, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {source}")
    return max(0.0, float(completed.stdout.strip()))


def _build_task_media(media_stem: str, source: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    poster = output_root / f"{media_stem.lower()}.webp"
    preview = output_root / f"{media_stem.lower()}.mp4"
    duration = _video_duration(source)
    poster_time = min(max(duration * 0.55, 1.0), max(duration - 0.2, 0.0))
    # Keep the temporary files on the same filesystem as the destination so
    # the final replacements remain atomic (the repository may not share /tmp).
    with tempfile.TemporaryDirectory(prefix=".labbench-media-", dir=output_root) as directory:
        temporary = Path(directory)
        temporary_poster = temporary / poster.name
        temporary_preview = temporary / preview.name
        _run_ffmpeg(
            [
                "-ss", f"{poster_time:.3f}", "-i", str(source), "-frames:v", "1",
                "-vf", "scale=640:480:flags=lanczos", "-quality", "85", str(temporary_poster),
            ]
        )
        _run_ffmpeg(
            [
                "-i", str(source), "-an", "-vf", "fps=10,scale=640:480:flags=lanczos",
                "-c:v", "libx264", "-preset", "medium", "-crf", "27",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary_preview),
            ]
        )
        os.replace(temporary_poster, poster)
        os.replace(temporary_preview, preview)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build task posters, previews and object renders for the public portal.")
    parser.add_argument(
        "--task-video", action="append", default=[], type=_parse_task_video,
        metavar="TASK_ID=VIDEO", help="May be repeated for multiple public benchmark tasks.",
    )
    parser.add_argument(
        "--extract-first-episodes", action="store_true",
        help="Copy both camera streams from the middle episode and build mapped public previews.",
    )
    parser.add_argument("--media-manifest", type=Path, default=DEFAULT_MEDIA_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--raw-video-root",
        type=Path,
        default=WEB_ASSETS_ROOT / "source_videos",
    )
    parser.add_argument(
        "--render-objects", action="store_true",
        help="Render every available USD object through the Isaac thumbnail entry point.",
    )
    parser.add_argument(
        "--isaac-python", type=Path, default=None,
        help="Isaac environment Python used with --render-objects.",
    )
    parser.add_argument("--object-resolution", type=int, default=640)
    parser.add_argument("--object-settle-steps", type=int, default=30)
    parser.add_argument("--static-root", type=Path, default=STATIC_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalog, objects = load_public_catalogs()
    task_ids = {task["task_id"] for task in catalog["tasks"]}
    for task_id, video in args.task_video:
        if task_id not in task_ids:
            raise ValueError(f"unknown public task id: {task_id}")
        _build_task_media(task_id, video, args.static_root / "web_assets" / "task_media")
        print(f"Built poster and preview for {task_id}")

    if args.extract_first_episodes:
        manifest = json.loads(args.media_manifest.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "labbench_task_media/v1":
            raise ValueError("unsupported task-media manifest")
        dataset_root = (args.dataset_root or Path(manifest["dataset_root"])).expanduser().resolve()
        raw_root = args.raw_video_root.expanduser().resolve()
        for item in manifest.get("datasets") or []:
            dataset_name = str(item["dataset"])
            copied: dict[str, Path] = {}
            for view in ("front", "wrist"):
                view_root = dataset_root / dataset_name / "videos"
                sources = sorted(view_root.glob(f"chunk-*/observation.images.{view}_camera/episode_*.mp4"))
                if not sources:
                    raise FileNotFoundError(f"No {view} camera videos found for {dataset_name}")
                source = sources[len(sources) // 2]
                if not source.is_file():
                    raise FileNotFoundError(source)
                destination = raw_root / dataset_name / f"{view}.mp4"
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".mp4.tmp")
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
                copied[view] = destination
            task_id = item.get("task_id")
            if task_id is None:
                print(f"Copied local-only dual-view episode for {dataset_name}")
                continue
            task_id = str(task_id).upper()
            if task_id not in task_ids:
                raise ValueError(f"unknown mapped public task id: {task_id}")
            variant = str(item["variant"])
            for view, source in copied.items():
                stem = f"{task_id.lower()}-{variant}-{view}"
                _build_task_media(stem, source, args.static_root / "web_assets" / "task_media")
            print(f"Built dual-view public media for {task_id} ({variant})")

    if args.render_objects:
        if args.isaac_python is None or not args.isaac_python.is_file():
            raise ValueError("--render-objects requires an existing --isaac-python executable")
        for item in objects["objects"]:
            if item["status"] != "available" or not item.get("media"):
                continue
            completed = subprocess.run(
                [
                    str(args.isaac_python), "-m", "eai4s_eval.cli.render_object_thumbnail",
                    "--object-id", item["object_id"],
                    "--output", str(args.static_root / item["media"]),
                    "--resolution", str(args.object_resolution),
                    "--settle-steps", str(args.object_settle_steps),
                ],
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"object rendering failed for {item['object_id']}")
            print(f"Rendered {item['object_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
