from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

import numpy as np


class BenchmarkConfigError(ValueError):
    """Raised when a benchmark configuration cannot produce valid observations."""


def _camera_names(cfg: Mapping[str, Any] | Any) -> set[str]:
    cameras = cfg.get("cameras", []) if hasattr(cfg, "get") else getattr(cfg, "cameras", [])
    names: set[str] = set()
    for camera in cameras or []:
        name = camera.get("name") if hasattr(camera, "get") else getattr(camera, "name", None)
        image_type = camera.get("image_type", "rgb") if hasattr(camera, "get") else getattr(camera, "image_type", "rgb")
        if not name:
            continue
        for kind in str(image_type).split("+"):
            names.add(f"{name}_{kind}")
    return names


def validate_camera_contract(cfg: Mapping[str, Any] | Any) -> dict[str, str]:
    """Validate configured inference sources against cameras emitted by the task.

    ``infer.obs_names`` maps a camera dataset key to the key expected by the
    policy.  The old runner waited forever when a source camera was misspelled;
    evaluation now fails before Isaac is launched.
    """

    infer = cfg.get("infer", {}) if hasattr(cfg, "get") else getattr(cfg, "infer", {})
    obs_names = infer.get("obs_names", {}) if hasattr(infer, "get") else getattr(infer, "obs_names", {})
    mapping = {str(source): str(target) for source, target in dict(obs_names or {}).items()}
    available = _camera_names(cfg)
    missing = sorted(source for source in mapping if source not in available)
    if missing:
        raise BenchmarkConfigError(
            "infer.obs_names references camera streams that are not configured: "
            f"{', '.join(missing)}; available streams: {', '.join(sorted(available)) or '(none)'}"
        )
    return mapping


def chw_to_hwc_uint8(image: Any) -> np.ndarray:
    """Convert one dataset image to the HWC uint8 contract used by OpenPI."""

    array = np.asarray(image)
    if array.ndim == 4:
        array = array[-1]
    if array.ndim != 3:
        raise BenchmarkConfigError(f"camera image must have 3 dimensions, got {array.shape}")
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    elif array.shape[-1] != 3:
        raise BenchmarkConfigError(f"camera image must have 1, 3, or 4 channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        finite = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        if finite.size and float(np.max(finite)) <= 1.0:
            finite = finite * 255.0
        array = np.clip(finite, 0.0, 255.0)
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))


def split_interval(bounds: Sequence[float], split: str) -> tuple[float, float]:
    """Return the deterministic interval for one benchmark split."""

    if len(bounds) != 2:
        raise BenchmarkConfigError(f"position range must contain two values, got {bounds}")
    low, high = float(bounds[0]), float(bounds[1])
    if low > high:
        raise BenchmarkConfigError(f"position range lower bound exceeds upper bound: {bounds}")
    midpoint = (low + high) / 2.0
    if split == "sim_clean":
        return midpoint, midpoint
    if split == "sim_dev":
        return low, high
    if split == "sim_randomized":
        half_width = (high - low) / 2.0 * 1.25
        return midpoint - half_width, midpoint + half_width
    raise BenchmarkConfigError(f"unknown evaluation split: {split}")


def seed_everything(seed: int) -> None:
    """Seed every RNG available without requiring torch in lightweight tests."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass
