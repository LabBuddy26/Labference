from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eai4s_eval.utils.io import append_jsonl, ensure_dir, write_json

_logger = logging.getLogger(__name__)


def _write_video_mp4(output_path: Path, frames: list[np.ndarray], fps: float) -> bool:
    """Write a list of RGB frames to an MP4 file.

    Returns ``True`` on success, ``False`` if no suitable encoder is available.
    Tries **cv2** (OpenCV) first, then **imageio**, then falls back to saving
    the raw frames as a compressed ``.npz`` beside *output_path*.
    """
    if not frames:
        return False

    # -- OpenCV path (most common in Isaac Sim environments) ------------------
    try:
        import cv2

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        for frame in frames:
            # Isaac Sim cameras produce RGB; cv2 expects BGR
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return True
    except Exception:
        _logger.debug("cv2 video writer unavailable, trying imageio", exc_info=True)

    # -- imageio fallback -----------------------------------------------------
    try:
        import imageio  # type: ignore

        imageio.mimsave(str(output_path), frames, fps=fps, macro_block_size=1)  # type: ignore[call-arg]
        return True
    except Exception:
        _logger.debug("imageio video writer unavailable", exc_info=True)

    # -- Raw .npz fallback ----------------------------------------------------
    npz_path = output_path.with_suffix(".npz")
    try:
        np.savez_compressed(
            npz_path,
            frames=np.stack([f.astype(np.uint8) for f in frames], axis=0),
            fps=np.float32(fps),
        )
        _logger.warning("No video encoder available; saved raw frames to %s", npz_path)
    except Exception:
        _logger.debug("npz fallback also failed", exc_info=True)

    return False


class EpisodeRecorder:
    def __init__(self, run_dir: str | Path):
        self.run_dir = ensure_dir(run_dir)
        self.trajectory_dir = ensure_dir(self.run_dir / "trajectories")
        self.log_dir = ensure_dir(self.run_dir / "logs")
        self.video_success_dir = ensure_dir(self.run_dir / "videos" / "success")
        self.video_failure_dir = ensure_dir(self.run_dir / "videos" / "failure")

    def write_run_manifest(self, manifest: dict[str, Any]) -> Path:
        return write_json(self.run_dir / "run_manifest.json", manifest)

    def write_run_config(self, config: dict[str, Any]) -> Path:
        return write_json(self.run_dir / "run_config.json", config)

    def save_episode(
        self,
        *,
        episode_id: str,
        episode_index: int,
        instruction: str,
        action_horizon: int,
        max_steps: int,
        success_result: Any,
        trajectory: list[dict[str, Any]],
        episode_steps: int | None = None,
        sim_steps: int | None = None,
    ) -> dict[str, str]:
        serializable_success = asdict(success_result) if is_dataclass(success_result) else success_result
        status_tag = "SUCCESS" if serializable_success and serializable_success.get("task_success") else "FAILED"
        json_path = self.log_dir / f"episode_{episode_index:03d}_{status_tag}_{episode_id}.json"
        write_json(
            json_path,
            {
                "episode": episode_index,
                "episode_id": episode_id,
                "instruction": instruction,
                "max_steps": max_steps,
                "action_horizon": action_horizon,
                "total_steps": len(trajectory),
                "episode_steps": len(trajectory) if episode_steps is None else int(episode_steps),
                "sim_steps": len(trajectory) if sim_steps is None else int(sim_steps),
                "success": serializable_success,
                "trajectory": trajectory,
            },
        )

        npz_path = self.trajectory_dir / f"{episode_id}.npz"
        if trajectory:
            payload: dict[str, Any] = {
                "steps": np.asarray([t["step"] for t in trajectory], dtype=np.int32),
                "actual_ee_pos": np.asarray([t["actual_state"]["ee_pos"] for t in trajectory], dtype=np.float32),
                "actual_ee_quat": np.asarray([t["actual_state"]["ee_quat"] for t in trajectory], dtype=np.float32),
                "actual_gripper": np.asarray([t["actual_state"]["gripper_pos"] for t in trajectory], dtype=np.float32),
            }
            first_action = trajectory[0].get("vla_action", {})
            if "joint_positions" in first_action:
                payload["vla_joint_positions"] = np.asarray(
                    [t.get("vla_action", {}).get("joint_positions", []) for t in trajectory],
                    dtype=object,
                )
                payload["actual_joint_positions"] = np.asarray(
                    [t.get("actual_state", {}).get("joint_pos", []) for t in trajectory],
                    dtype=np.float32,
                )
                # Pre-anchor chunk[0] misalignment per step (7-D), NaN when no chunk
                # was anchored yet. Constant within each chunk; changes at boundaries.
                payload["vla_anchor_offsets"] = np.asarray(
                    [
                        np.asarray(t.get("vla_anchor_offset", []) or np.full(7, np.nan), dtype=np.float32)
                        for t in trajectory
                    ],
                    dtype=np.float32,
                )
            else:
                payload["vla_ee_pos"] = np.asarray([t["vla_action"]["ee_pos"] for t in trajectory], dtype=np.float32)
                payload["vla_ee_quat"] = np.asarray([t["vla_action"]["ee_quat"] for t in trajectory], dtype=np.float32)
                payload["vla_gripper"] = np.asarray(
                    [t["vla_action"].get("gripper_raw", np.nan) for t in trajectory], dtype=np.float32
                )
            np.savez_compressed(npz_path, **payload)
        else:
            np.savez_compressed(npz_path, steps=np.asarray([], dtype=np.int32))

        return {"legacy_json": str(json_path), "trajectory_npz": str(npz_path)}

    def append_failure_case(self, row: dict[str, Any]) -> None:
        append_jsonl(self.run_dir / "failure_cases.jsonl", row)

    def save_video(
        self,
        *,
        episode_id: str,
        success: bool,
        frames: list[np.ndarray],
        fps: float = 10.0,
    ) -> str | None:
        """Save collected RGB frames as an MP4 video.

        Parameters
        ----------
        episode_id:
            Used as the base filename (``{episode_id}.mp4``).
        success:
            If ``True`` the video is saved under ``videos/success/``, otherwise
            under ``videos/failure/``.
        frames:
            List of RGB uint8 numpy arrays, each with shape ``(H, W, 3)``.
        fps:
            Frame rate written into the video container.

        Returns
        -------
        str or None
            Absolute path to the saved ``.mp4`` (or ``.npz`` fallback) on
            success, or ``None`` if no frames were provided or all encoders
            failed.
        """
        if not frames:
            return None

        target_dir = self.video_success_dir if success else self.video_failure_dir
        output_path = target_dir / f"{episode_id}.mp4"

        ok = _write_video_mp4(output_path, frames, fps)
        if ok:
            return str(output_path.resolve())

        # If MP4 encoding failed, _write_video_mp4 may have saved a .npz
        npz_path = output_path.with_suffix(".npz")
        if npz_path.exists():
            return str(npz_path.resolve())

        return None
