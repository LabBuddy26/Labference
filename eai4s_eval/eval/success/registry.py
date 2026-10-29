from __future__ import annotations

from typing import Any, Callable

from eai4s_eval.eval.success.schemas import PredicateValue
from eai4s_eval.eval.success.snapshot import EvaluationSnapshot

PredicateFn = Callable[[EvaluationSnapshot, dict[str, Any]], PredicateValue]


class PredicateRegistry:
    def __init__(self) -> None:
        self._predicates: dict[str, PredicateFn] = {}

    def register(self, name: str, fn: PredicateFn) -> None:
        if not name:
            raise ValueError("Predicate name cannot be empty")
        self._predicates[name] = fn

    def get(self, name: str) -> PredicateFn:
        try:
            return self._predicates[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._predicates))
            raise ValueError(f"Unknown success predicate {name!r}. Available predicates: {available}") from exc

    def evaluate(self, name: str, snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
        return self.get(name)(snapshot, args)

    def names(self) -> list[str]:
        return sorted(self._predicates)
