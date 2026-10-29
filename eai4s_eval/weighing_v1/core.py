"""Simulator-independent weighing contract, instances, and physical state rules."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import random
from typing import Any


TASK_VERSION = "weighing-feedback/v1"
WEIGHTS = (10, 20, 50, 100)
EASY_TARGETS = (160, 180, 200)
HARD_TARGETS = (200, 220, 240)
HARD_WEIGHT_SETS = ((10, 20), (10, 50), (10, 100), (20, 50), (20, 100),
                    (50, 100), (10, 20, 50), (10, 20, 100),
                    (10, 50, 100), (20, 50, 100))
ORACLES = ("none", "semantic", "decision", "geometry", "skill")
OPERATION_TYPES = ("add", "remove", "wait", "stop")


def _finite(value: float) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class Contract:
    difficulty: str
    physics_hz: int = 120
    readout_tau_s: float = 0.25
    display_quantum_g: float = 0.1
    stable_window_s: float = 1.0
    final_window_s: float = 3.0
    stable_range_g: float = 0.5
    target_tolerance_g: float = 1.0
    max_linear_speed_m_s: float = 0.005
    max_angular_speed_rad_s: float = 0.05
    max_steps: int = 7200
    task_version: str = TASK_VERSION

    def __post_init__(self) -> None:
        if self.difficulty not in {"easy", "hard"} or self.task_version != TASK_VERSION:
            raise ValueError("unsupported weighing task version or difficulty")
        numeric = (self.readout_tau_s, self.display_quantum_g, self.stable_window_s,
                   self.final_window_s, self.stable_range_g, self.target_tolerance_g,
                   self.max_linear_speed_m_s, self.max_angular_speed_rad_s)
        if self.physics_hz <= 0 or self.max_steps <= 0 or not all(_finite(v) and v > 0 for v in numeric):
            raise ValueError("weighing thresholds must be finite and positive")
        if self.final_window_s < self.stable_window_s:
            raise ValueError("final stability window must include the reading window")

    @property
    def stable_steps(self) -> int:
        return math.ceil(self.physics_hz * self.stable_window_s)

    @property
    def final_steps(self) -> int:
        return math.ceil(self.physics_hz * self.final_window_s)

    def canonical(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def hash(self) -> str:
        value = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Instance:
    difficulty: str
    seed: int
    scene_seed: int
    mass_seed: int
    target_g: int
    sample_mass_g: int
    desired_weights_g: tuple[int, ...]
    initial_weights_g: tuple[int, ...]
    pair_id: str

    def __post_init__(self) -> None:
        if self.difficulty not in {"easy", "hard"}:
            raise ValueError("unknown weighing difficulty")
        desired = set(self.desired_weights_g)
        initial = set(self.initial_weights_g)
        if (not desired <= set(WEIGHTS) or not initial <= set(WEIGHTS) or
                len(desired) != len(self.desired_weights_g) or len(initial) != len(self.initial_weights_g) or
                self.target_g != self.sample_mass_g + sum(desired) or self.sample_mass_g <= 0):
            raise ValueError("invalid weighing instance masses")
        if self.difficulty == "easy" and (len(desired) != 1 or initial):
            raise ValueError("easy instance requires one missing standard weight")
        if self.difficulty == "hard" and (len(desired) not in (2, 3) or
                len(desired ^ initial) < 2):
            raise ValueError("hard instance requires at least two corrections")

    def public(self) -> dict[str, Any]:
        return {"task_version": TASK_VERSION, "difficulty": self.difficulty,
                "target_g": self.target_g, "weight_labels_g": list(WEIGHTS),
                "rule": f"Reach {self.target_g:.1f} g using the live stable scale reading and actual supported weights. "
                        "Add or remove standards to reach the target. Wait for a stable reading "
                        "before the next choice; finish only after the load remains stable."}


def generate_instance(contract: Contract, seed: int, *, mass_seed: int | None = None,
                      scene_seed: int | None = None, pair_id: str | None = None) -> Instance:
    rng = random.Random(seed if mass_seed is None else mass_seed)
    mass_seed = seed if mass_seed is None else mass_seed
    scene_seed = seed + 10_000 if scene_seed is None else scene_seed
    if contract.difficulty == "easy":
        target = rng.choice(EASY_TARGETS)
        desired = (rng.choice(WEIGHTS),)
        initial: tuple[int, ...] = ()
    else:
        target = rng.choice(HARD_TARGETS)
        desired = rng.choice(HARD_WEIGHT_SETS)
        possible = [tuple(v for j, v in enumerate(WEIGHTS) if mask & (1 << j))
                    for mask in range(16) if
                    len(set(v for j, v in enumerate(WEIGHTS) if mask & (1 << j)) ^ set(desired)) >= 2]
        initial = rng.choice(possible)
    return Instance(contract.difficulty, seed, scene_seed, mass_seed, target,
                    target - sum(desired), desired, initial,
                    pair_id or f"weighing-{contract.difficulty}-{scene_seed}-{seed}")


def paired_instances(contract: Contract, seed: int) -> tuple[Instance, Instance]:
    """Same target/layout and different hidden preload, requiring a changed choice."""
    first = generate_instance(contract, seed, pair_id=f"{contract.difficulty}-{seed}")
    if contract.difficulty == "easy":
        replacement = next(weight for weight in WEIGHTS if weight != first.desired_weights_g[0])
        second = Instance("easy", seed + 1, first.scene_seed, seed + 1, first.target_g,
                          first.target_g - replacement, (replacement,), (), first.pair_id)
    else:
        options = [option for option in HARD_WEIGHT_SETS
                   if option != first.desired_weights_g and
                   len(set(option) ^ set(first.initial_weights_g)) >= 2]
        alternate = random.Random(seed ^ 0x5EED).choice(options)
        second = Instance("hard", seed + 1, first.scene_seed, seed + 1, first.target_g,
                          first.target_g - sum(alternate), alternate, first.initial_weights_g,
                          first.pair_id)
    return first, second


def balanced_instances(contract: Contract, *, start_seed: int = 101,
                       repetitions: int = 1) -> tuple[Instance, ...]:
    """Pre-register equal target-by-answer strata without changing the v1 sampler."""
    if type(start_seed) is not int or type(repetitions) is not int or repetitions <= 0:
        raise ValueError("start_seed must be an integer and repetitions must be positive")
    targets = EASY_TARGETS if contract.difficulty == "easy" else HARD_TARGETS
    answers = tuple((weight,) for weight in WEIGHTS) if contract.difficulty == "easy" else HARD_WEIGHT_SETS
    cells = {(target, answer): [] for target in targets for answer in answers}
    for seed in range(start_seed, start_seed + 10000 * repetitions):
        instance = generate_instance(contract, seed)
        bucket = cells[(instance.target_g, instance.desired_weights_g)]
        if len(bucket) < repetitions:
            bucket.append(instance)
        if all(len(bucket) == repetitions for bucket in cells.values()):
            return tuple(instance for bucket in cells.values() for instance in bucket)
    raise RuntimeError("could not fill all weighing strata within the seed budget")


@dataclass(frozen=True)
class Measurement:
    step: int
    sensor_valid: bool
    measured_g: float
    displayed_g: float
    supported_weights_g: tuple[int, ...]
    sample_supported: bool
    sample_untouched: bool
    released: bool
    no_external_support: bool
    no_hand_press: bool
    max_linear_speed_m_s: float
    max_angular_speed_rad_s: float
    overloaded: bool = False

    def __post_init__(self) -> None:
        if (self.step < 0 or len(set(self.supported_weights_g)) != len(self.supported_weights_g) or
                any(weight not in WEIGHTS for weight in self.supported_weights_g)):
            raise ValueError("invalid measured weights or step")
        if self.sensor_valid and not all(_finite(v) for v in
                                         (self.measured_g, self.displayed_g,
                                          self.max_linear_speed_m_s,
                                          self.max_angular_speed_rad_s)):
            raise ValueError("valid sensor sample must be finite")


@dataclass
class State:
    supported_weights_g: tuple[int, ...] = ()
    displayed_g: float = 0.0
    sensor_valid: bool = False
    reading_stable: bool = False
    stable_steps: int = 0
    final_steps: int = 0
    sample_supported: bool = False
    sample_untouched: bool = True
    released: bool = False
    no_external_support: bool = True
    no_hand_press: bool = True
    last_step: int = -1
    _window: deque[float] = field(default_factory=deque, repr=False)

    def public(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if key != "_window"}


def advance(contract: Contract, instance: Instance, state: State, measured: Measurement) -> State:
    if measured.step <= state.last_step:
        raise ValueError("measurements must have increasing step numbers")
    changed = state.supported_weights_g != tuple(sorted(measured.supported_weights_g))
    state.last_step = measured.step
    state.supported_weights_g = tuple(sorted(measured.supported_weights_g))
    state.displayed_g = measured.displayed_g
    state.sensor_valid = measured.sensor_valid
    state.sample_supported = measured.sample_supported
    state.sample_untouched = measured.sample_untouched
    state.released = measured.released
    state.no_external_support = measured.no_external_support
    state.no_hand_press = measured.no_hand_press
    eligible = (measured.sensor_valid and not measured.overloaded and measured.sample_supported and measured.sample_untouched and
                measured.released and measured.no_external_support and measured.no_hand_press and
                measured.max_linear_speed_m_s <= contract.max_linear_speed_m_s and
                measured.max_angular_speed_rad_s <= contract.max_angular_speed_rad_s)
    if changed or not eligible:
        state._window.clear()
        state.stable_steps = 0
        state.final_steps = 0
        state.reading_stable = False
        return state
    state._window.append(measured.displayed_g)
    while len(state._window) > contract.stable_steps:
        state._window.popleft()
    if (len(state._window) >= contract.stable_steps and
            max(state._window) - min(state._window) <= contract.stable_range_g):
        state.stable_steps += 1
    else:
        state.stable_steps = 0
    state.reading_stable = state.stable_steps >= 1
    target_ok = (abs(measured.displayed_g - instance.target_g) <= contract.target_tolerance_g and
                 state.supported_weights_g == instance.desired_weights_g)
    state.final_steps = state.final_steps + 1 if target_ok and (
        max(state._window) - min(state._window) <= contract.stable_range_g) else 0
    return state


def legal_actions(contract: Contract, instance: Instance, state: State) -> list[dict[str, Any]]:
    if contract.difficulty != instance.difficulty:
        raise ValueError("difficulty mismatch")
    if not state.sensor_valid or not state.reading_stable or not state.sample_supported:
        return [{"type": "wait"}]
    if state.final_steps >= contract.final_steps:
        return [{"type": "stop"}]
    present, desired = set(state.supported_weights_g), set(instance.desired_weights_g)
    if present == desired:
        return [{"type": "wait"}]
    return ([{"type": "remove", "weight_g": weight} for weight in WEIGHTS if weight in present - desired] +
            [{"type": "add", "weight_g": weight} for weight in WEIGHTS if weight in desired - present])


def normalize_action(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("type") not in OPERATION_TYPES:
        return None
    kind = value["type"]
    if kind in {"wait", "stop"} and set(value) == {"type"}:
        return {"type": kind}
    weight = value.get("weight_g")
    if (kind in {"add", "remove"} and set(value) == {"type", "weight_g"} and
            type(weight) is int and weight in WEIGHTS):
        return {"type": kind, "weight_g": weight}
    return None


def task_complete(contract: Contract, instance: Instance, state: State) -> bool:
    return legal_actions(contract, instance, state) == [{"type": "stop"}]
