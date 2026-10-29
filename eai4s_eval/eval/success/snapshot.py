from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class MissingSnapshotValue(KeyError):
    pass


@dataclass
class EvaluationSnapshot:
    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | "EvaluationSnapshot") -> "EvaluationSnapshot":
        if isinstance(values, EvaluationSnapshot):
            return values
        return cls(dict(values))

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.values:
            return self.values[key]
        current: Any = self.values
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def require(self, key: str) -> Any:
        marker = object()
        value = self.get(key, marker)
        if value is marker:
            raise MissingSnapshotValue(key)
        return value

    def first_available(self, keys: list[str] | tuple[str, ...]) -> tuple[str, Any]:
        marker = object()
        for key in keys:
            value = self.get(key, marker)
            if value is not marker:
                return key, value
        raise MissingSnapshotValue(" or ".join(keys))

    def array(self, key: str) -> np.ndarray:
        return np.asarray(self.require(key), dtype=np.float64)

    def scalar(self, key: str) -> float:
        value = self.require(key)
        return float(np.asarray(value, dtype=np.float64))
