from __future__ import annotations

import re


DEFAULT_MODEL_NAMES = {"openpi": "OpenPI", "gr00t": "GR00T N1.5"}
CHECKPOINT_RE = re.compile(r"^[A-Za-z0-9_.=-]+$")


def resolve_model_name(policy_provider: str, model_name: str | None = None) -> str:
    provider = str(policy_provider or "openpi").strip().lower()
    if provider not in DEFAULT_MODEL_NAMES:
        raise ValueError(f"unsupported policy provider: {provider}")
    resolved = str(model_name or DEFAULT_MODEL_NAMES[provider]).strip()
    lowered = resolved.lower()
    if provider == "gr00t" and "openpi" in lowered:
        raise ValueError("GR00T evaluations cannot use an OpenPI model name")
    if provider == "openpi" and "gr00t" in lowered:
        raise ValueError("OpenPI evaluations cannot use a GR00T model name")
    if not resolved:
        raise ValueError("model name must not be empty")
    return resolved


def validate_checkpoint_name(checkpoint: str | None, *, required: bool = False) -> str | None:
    value = str(checkpoint or "").strip()
    if not value:
        if required:
            raise ValueError("checkpoint is required for an official suite evaluation")
        return None
    if len(value) > 128 or not CHECKPOINT_RE.fullmatch(value):
        raise ValueError("checkpoint must use at most 128 letters, digits, '.', '_', '=' or '-'")
    return value
