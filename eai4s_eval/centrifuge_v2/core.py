"""Simulator-independent task contract and physical evidence rules."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any


TASK_VERSION = "centrifuge-opposite/v2"
SLOTS = tuple(f"S{i:02d}" for i in range(12))
TUBES = ("NMR_01", "NMR_02")


def insertion_matches_commit(contract: "Contract", tube: str,
                             committed_slot: str | None, actual_slot: str) -> bool:
    if contract.difficulty == "easy" and tube == "NMR_01":
        return committed_slot is None or committed_slot == actual_slot
    return committed_slot == actual_slot


def slew_lift_height(current: float, target: float, step_m: float = 0.0005) -> float:
    if not all(map(math.isfinite, (current, target, step_m))) or step_m <= 0:
        raise ValueError("lift heights and step must be finite and positive")
    return min(target, current + step_m)


def gripper_target_for_insertion(mouth_world: tuple[float, float, float],
                                 insertion_axis_world: tuple[float, float, float],
                                 depth_m: float, gripper_minus_tube_world: tuple[float, float, float],
                                 place_controller_offset_m: float = 0.02) -> tuple[float, float, float]:
    """Convert a measured tube-root insertion goal to PlaceController's EEF goal."""
    values = (*mouth_world, *insertion_axis_world, depth_m, *gripper_minus_tube_world)
    if not all(map(math.isfinite, values)) or depth_m < 0:
        raise ValueError("non-finite or negative insertion geometry")
    axis = unit(insertion_axis_world)
    if axis is None:
        raise ValueError("zero insertion axis")
    return tuple(mouth_world[i] + depth_m * axis[i] + gripper_minus_tube_world[i]
                 - (place_controller_offset_m if i == 2 else 0.0) for i in range(3))


def insertion_clearance_target_z(mouth_z: float, asset_clearance_z: float,
                                 gripper_minus_tube_z: float,
                                 max_reachable_tube_clearance_m: float = .24) -> float:
    """Return an EEF height that puts the held tube at the safe clearance.

    The controller commands the gripper frame, while the clearance is defined
    for the tube. Include the measured vertical grasp offset before capping the
    target to the arm's reachable tube height.
    """
    values = (mouth_z, asset_clearance_z, gripper_minus_tube_z,
              max_reachable_tube_clearance_m)
    if not all(map(math.isfinite, values)) or max_reachable_tube_clearance_m <= 0:
        raise ValueError("clearance geometry must be finite and positive")
    grasp_offset = max(0.0, float(gripper_minus_tube_z))
    rim_clearance = float(asset_clearance_z) + grasp_offset
    reachable_clearance = (float(mouth_z) + max_reachable_tube_clearance_m +
                           grasp_offset)
    minimum_clearance = float(mouth_z) + max(.14, grasp_offset + .06)
    return max(minimum_clearance, min(rim_clearance, reachable_clearance))


def inspect_joint_targets(current: list[float], targets: list[float | None],
                          joint_indices: list[int] | None = None) -> float:
    indices = list(range(len(targets))) if joint_indices is None else list(joint_indices)
    if len(indices) != len(targets):
        raise ValueError("joint target and index lengths differ")
    raw_max = 0.0
    for index, target in zip(indices, targets):
        if target is None:
            continue
        if index < 0 or index >= len(current) or not all(map(math.isfinite, (float(target), float(current[index])))):
            raise ValueError("non-finite or out-of-range joint target")
        delta = float(target) - float(current[index])
        raw_max = max(raw_max, abs(delta))
    return raw_max


def limit_joint_targets(current: list[float], targets: list[float | None],
                        joint_indices: list[int] | None = None,
                        arm_step_rad: float = .08) -> list[float | None]:
    inspect_joint_targets(current, targets, joint_indices)
    indices = list(range(len(targets))) if joint_indices is None else list(joint_indices)
    limited = []
    for index, target in zip(indices, targets):
        if target is None:
            limited.append(None)
            continue
        if arm_step_rad <= 0 or not math.isfinite(arm_step_rad):
            raise ValueError("joint step limit must be finite and positive")
        if index < 7:
            current_value = float(current[index])
            # Lula exposes some Franka wrist joints as unbounded even though
            # repeated revolutions are not useful for these Cartesian tasks.
            # Keep all arm targets in a canonical +/-2pi band and actively
            # unwind a state that has already accumulated extra turns.
            if abs(current_value) > 2.0 * math.pi:
                delta = -math.copysign(arm_step_rad, current_value)
            else:
                bounded_target = max(-2.0 * math.pi,
                                     min(2.0 * math.pi, float(target)))
                delta = max(-arm_step_rad,
                            min(arm_step_rad, bounded_target - current_value))
            limited.append(current_value + delta)
        else:
            limited.append(float(target))
    return limited


def limit_joint_velocities(current: list[float], velocities: list[float | None],
                           joint_indices: list[int] | None = None,
                           max_arm_speed_rad_s: float = 2.0) -> list[float | None]:
    """Bound Franka velocity targets and prevent further continuous-joint windup."""
    indices = list(range(len(velocities))) if joint_indices is None else list(joint_indices)
    if len(indices) != len(velocities) or max_arm_speed_rad_s <= 0 or not math.isfinite(max_arm_speed_rad_s):
        raise ValueError("joint velocity indices and speed limit are invalid")
    limited: list[float | None] = []
    for index, velocity in zip(indices, velocities):
        if velocity is None:
            limited.append(None)
            continue
        if index < 0 or index >= len(current) or not all(
            map(math.isfinite, (float(current[index]), float(velocity)))
        ):
            raise ValueError("non-finite or out-of-range joint velocity")
        value = float(velocity)
        if index < 7:
            value = max(-max_arm_speed_rad_s, min(max_arm_speed_rad_s, value))
            position = float(current[index])
            if abs(position) > 2.0 * math.pi:
                value = -math.copysign(max_arm_speed_rad_s, position)
            elif abs(position) >= 2.0 * math.pi - .05 and value * position > 0:
                value = 0.0
        limited.append(value)
    return limited


def joint_limit_avoidance_delta(current: list[float], jacobian: Any,
                                limits: Any, gain: float = 0.04) -> list[float]:
    """Return a small null-space posture correction away from arm joint limits.

    This is intended for position-only differential IK, where redundant arm
    joints can otherwise drift into a limit while the end-effector target is
    still reachable.  It never replaces the Cartesian task-space correction.
    """
    import numpy as np

    q = np.asarray(current, dtype=float)
    j = np.asarray(jacobian, dtype=float)
    bounds = np.asarray(limits, dtype=float)
    if (q.ndim != 1 or j.ndim != 2 or j.shape[1] != len(q) or
            bounds.shape != (len(q), 2) or not np.all(np.isfinite(q)) or
            not np.all(np.isfinite(j)) or not np.all(np.isfinite(bounds)) or
            not math.isfinite(gain) or gain < 0 or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError("invalid joint-limit avoidance inputs")
    if gain == 0:
        return np.zeros_like(q).tolist()
    center = bounds.mean(axis=1)
    half_range = np.maximum((bounds[:, 1] - bounds[:, 0]) * 0.5, 0.1)
    posture_gradient = gain * (center - q) / half_range
    damping = 0.015
    inverse = j.T @ np.linalg.solve(
        j @ j.T + damping**2 * np.eye(j.shape[0]), np.eye(j.shape[0]))
    nullspace = np.eye(len(q)) - inverse @ j
    correction = nullspace @ posture_gradient
    return correction.tolist()


def wrap_joint_targets_near_current(current: list[float], targets: list[float | None],
                                    lower: list[float], upper: list[float],
                                    joint_indices: list[int] | None = None
                                    ) -> list[float | None]:
    """Canonicalize continuous revolute targets to the nearest equivalent angle."""
    if len(lower) != len(current) or len(upper) != len(current):
        raise ValueError("joint limit arrays must match the current joint state")
    inspect_joint_targets(current, targets, joint_indices)
    indices = list(range(len(targets))) if joint_indices is None else list(joint_indices)
    result: list[float | None] = []
    for index, target in zip(indices, targets):
        if target is None or index >= 7:
            result.append(None if target is None else float(target))
            continue
        lo, hi = float(lower[index]), float(upper[index])
        width = hi - lo
        if not math.isfinite(width) or width >= 2.0 * math.pi - 1e-3:
            delta = float(target) - float(current[index])
            delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
            result.append(float(current[index]) + delta)
        else:
            result.append(min(hi, max(lo, float(target))))
    return result


def operation_step_limit(kind: str, tube: str, *, collection_mode: bool) -> int:
    """Return per-action budget; long physical demonstrations need extra transit time."""
    if kind == "pick":
        return 3000 if tube == "NMR_02" else 1300
    return 30000 if collection_mode else 14000


def cartesian_waypoint(current: list[float], target: list[float],
                       max_distance_m: float = 0.02) -> tuple[float, float, float]:
    """Bound a free-space IK request to a short Cartesian segment."""
    values = (*current, *target, max_distance_m)
    if (len(current) != 3 or len(target) != 3 or
            not all(map(math.isfinite, values)) or max_distance_m <= 0):
        raise ValueError("Cartesian waypoint inputs must be finite 3-vectors and a positive step")
    delta = [float(target[i]) - float(current[i]) for i in range(3)]
    distance = math.sqrt(sum(value * value for value in delta))
    scale = min(1.0, max_distance_m / distance) if distance > 1e-12 else 0.0
    return tuple(float(current[i]) + scale * delta[i] for i in range(3))


def cartesian_waypoint_reached(current: list[float], waypoint: list[float],
                               tolerance_m: float = 0.02) -> bool:
    """Treat small residual FK error as progress so free-space IK can replan."""
    values = (*current, *waypoint, tolerance_m)
    if (len(current) != 3 or len(waypoint) != 3 or
            not all(map(math.isfinite, values)) or tolerance_m <= 0):
        return False
    return math.sqrt(sum((float(current[i]) - float(waypoint[i])) ** 2
                         for i in range(3))) <= tolerance_m


def choose_accessible_first_slot(slots_world: dict[str, tuple[float, float, float]],
                                 base_xy: list[float],
                                 source_xy: list[float]) -> str:
    """Choose the most robot-facing socket, with a pickup-distance tie break.

    Do not shift the result by an arbitrary slot index: that shift moved the
    selected opening toward the lid even for an unrotated machine.
    """
    if (set(slots_world) != set(SLOTS) or len(base_xy) != 2 or len(source_xy) != 2 or
            not all(math.isfinite(float(value)) for value in (*base_xy, *source_xy))):
        raise ValueError("slot map and approach coordinates are invalid")
    points = {slot: tuple(map(float, slots_world[slot][:2])) for slot in SLOTS}
    if not all(math.isfinite(value) for point in points.values() for value in point):
        raise ValueError("slot coordinates must be finite")
    center = tuple(sum(points[slot][axis] for slot in SLOTS) / len(SLOTS)
                   for axis in range(2))
    toward_robot = (float(base_xy[0]) - center[0], float(base_xy[1]) - center[1])
    length = math.hypot(*toward_robot)
    if length <= 1e-9:
        raise ValueError("robot base must be outside the rotor center")
    toward_robot = (toward_robot[0] / length, toward_robot[1] / length)
    candidates = []
    for index in range(len(SLOTS) // 2):
        slot = SLOTS[index]
        point = points[slot]
        frontage = ((point[0] - center[0]) * toward_robot[0] +
                    (point[1] - center[1]) * toward_robot[1])
        source_distance = math.hypot(point[0] - float(source_xy[0]),
                                     point[1] - float(source_xy[1]))
        # Prefer sockets on the approach-facing rim, while keeping the first
        # pickup-to-hole transfer short. This avoids selecting a back-side
        # hole whose route intersects the centrifuge housing.
        candidates.append((frontage - .05 * source_distance, -source_distance, slot))
    return max(candidates)[2]


def joint_waypoint_is_local(current: list[float], target: list[float],
                            max_delta_rad: float = .10) -> bool:
    """Reject IK branches whose interpolation is too large for one local segment."""
    if (len(current) == 0 or len(current) != len(target) or
            not math.isfinite(max_delta_rad) or max_delta_rad <= 0):
        return False
    values = [*map(float, current), *map(float, target)]
    return (all(map(math.isfinite, values)) and
            max(abs(float(target[i]) - float(current[i]))
                for i in range(len(current))) <= max_delta_rad)


def classify_contacts(contacts: list[dict]) -> dict[str, bool]:
    names = [str(c.get(k, "")) for c in contacts for k in ("body0", "body1")]
    return {
        "rotor": any("/World/Centrifugal" in name for name in names),
        "gripper": any("/World/Franka/" in name and ("finger" in name or "hand" in name)
                       for name in names),
        "left_finger": any("/World/Franka/panda_leftfinger" in name for name in names),
        "right_finger": any("/World/Franka/panda_rightfinger" in name for name in names),
    }


@dataclass(frozen=True)
class Contract:
    difficulty: str
    slot_count: int = 12
    rotor_radius_m: float = 0.0693
    rotor_center_offset_m: tuple[float, float, float] = (0.0, -0.11, 0.147)
    slot0_angle_deg: float = 199.438
    socket_depth_m: float = 0.038
    tube_radius_m: float = 0.006
    socket_radius_m: float = 0.011
    radial_tolerance_m: float = 0.005
    axis_tolerance_deg: float = 10.0
    insertion_depth_fraction: float = 0.60
    stable_steps: int = 30
    final_stable_steps: int = 180
    max_steps: int = 2000
    source_asset: str = "assets/chemistry_lab/assets/CentrifugalMachine_mesh.usd"
    rotor_spawn_position_m: tuple[float, float, float] = (0.05, 0.41, 0.72)
    franka_base_position_m: tuple[float, float, float] = (-0.4, 0.0, 0.71)
    front_camera_position_m: tuple[float, float, float] = (0.55, -0.35, 1.40)
    front_camera_orientation_wxyz: tuple[float, float, float, float] = (
        0.79570278, 0.44999645, 0.19957126, 0.35289035)
    front_camera_focal_length_mm: float = 12.0
    task_version: str = TASK_VERSION

    def __post_init__(self) -> None:
        if self.difficulty not in {"easy", "hard"}:
            raise ValueError("difficulty must be easy or hard")
        if self.task_version not in {"centrifuge-opposite/v1", "centrifuge-opposite/v2"}:
            raise ValueError("unsupported centrifuge task version")
        if len(self.rotor_spawn_position_m) != 3 or not all(map(math.isfinite, self.rotor_spawn_position_m)):
            raise ValueError("invalid rotor spawn position")
        if len(self.franka_base_position_m) != 3 or not all(map(math.isfinite, self.franka_base_position_m)):
            raise ValueError("invalid Franka base position")
        if (len(self.front_camera_position_m) != 3 or
                not all(map(math.isfinite, self.front_camera_position_m)) or
                len(self.front_camera_orientation_wxyz) != 4 or
                not all(map(math.isfinite, self.front_camera_orientation_wxyz)) or
                self.front_camera_focal_length_mm <= 0):
            raise ValueError("invalid front camera configuration")
        if self.slot_count != 12 or self.slot_count % 2:
            raise ValueError("v1 is calibrated for exactly 12 opposite slots")
        if self.radial_tolerance_m <= 0 or self.radial_tolerance_m > self.socket_radius_m - self.tube_radius_m + 1e-9:
            raise ValueError("radial tolerance exceeds the socket clearance")
        if not 0 < self.insertion_depth_fraction <= 1 or self.stable_steps <= 0 or self.final_stable_steps < self.stable_steps:
            raise ValueError("invalid insertion or stability threshold")

    def opposite(self, slot: str) -> str:
        if slot not in SLOTS:
            raise ValueError(f"unknown slot: {slot}")
        return SLOTS[(SLOTS.index(slot) + self.slot_count // 2) % self.slot_count]

    def slot_local_center(self, slot: str) -> tuple[float, float, float]:
        """Position in rotor USD coordinates at the registered nominal transform.

        The XY ring fits the legacy centers. The Z coordinate is the mouth
        plane measured from the rotor mesh, not the legacy insertion target.
        """
        if slot not in SLOTS:
            raise ValueError(f"unknown slot: {slot}")
        theta = math.radians(self.slot0_angle_deg - 30 * SLOTS.index(slot))
        x, y, z = self.rotor_center_offset_m
        # The asset is used at scale 0.5 and rotated 180 degrees about X.
        return (x + 2 * self.rotor_radius_m * math.cos(theta),
                -2 * y - 2 * self.rotor_radius_m * math.sin(theta),
                -2 * z)

    def canonical(self) -> dict[str, Any]:
        data = {
            "task_version": self.task_version,
            "difficulty": self.difficulty,
            "slot_count": self.slot_count,
            "rotor_radius_m": self.rotor_radius_m,
            "rotor_center_offset_m": list(self.rotor_center_offset_m),
            "slot0_angle_deg": self.slot0_angle_deg,
            "socket_depth_m": self.socket_depth_m,
            "tube_radius_m": self.tube_radius_m,
            "socket_radius_m": self.socket_radius_m,
            "radial_tolerance_m": self.radial_tolerance_m,
            "axis_tolerance_deg": self.axis_tolerance_deg,
            "insertion_depth_fraction": self.insertion_depth_fraction,
            "stable_steps": self.stable_steps,
            "final_stable_steps": self.final_stable_steps,
            "max_steps": self.max_steps,
            "source_asset": self.source_asset,
        }
        if self.task_version == "centrifuge-opposite/v2":
            data["rotor_spawn_position_m"] = list(self.rotor_spawn_position_m)
            data["franka_base_position_m"] = list(self.franka_base_position_m)
            data["front_camera_position_m"] = list(self.front_camera_position_m)
            data["front_camera_orientation_wxyz"] = list(self.front_camera_orientation_wxyz)
            data["front_camera_focal_length_mm"] = self.front_camera_focal_length_mm
        return data

    @property
    def hash(self) -> str:
        raw = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Observation:
    tube: str
    tip_world: tuple[float, float, float]
    axis_world: tuple[float, float, float]
    linear_speed_m_s: float
    gripper_open: bool
    supported: bool
    position_delta_m: float | None = None


@dataclass
class State:
    difficulty: str
    tube_status: dict[str, str] = field(default_factory=lambda: {t: "table" for t in TUBES})
    occupied: dict[str, str] = field(default_factory=dict)
    selected_slot: dict[str, str] = field(default_factory=dict)
    stable_counts: dict[str, int] = field(default_factory=dict)
    first_actual_slot: str | None = None
    terminal_steps: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "difficulty": self.difficulty,
            "tube_status": dict(self.tube_status),
            "occupied": dict(self.occupied),
            "selected_slot": dict(self.selected_slot),
            "first_actual_slot": self.first_actual_slot,
            "terminal_steps": self.terminal_steps,
        }


def legal_actions(contract: Contract, state: State) -> list[dict[str, str]]:
    if state.difficulty != contract.difficulty:
        raise ValueError("state and contract difficulty mismatch")
    if contract.difficulty == "easy":
        if state.tube_status["NMR_02"] == "inserted":
            return [{"type": "stop"}] if state.terminal_steps >= contract.final_stable_steps else [{"type": "wait"}]
        target = state.first_actual_slot
        if target is None:
            return [{"type": "wait"}]
        tube = "NMR_02"
        if state.tube_status[tube] == "held":
            return [{"type": "insert", "tube": tube, "slot": contract.opposite(target)}]
        return [{"type": "pick", "tube": tube}]

    if state.first_actual_slot is None:
        tube = "NMR_01"
        if state.tube_status[tube] == "held":
            return [{"type": "insert", "tube": tube, "slot": s}
                    for s in SLOTS if s not in state.occupied and contract.opposite(s) not in state.occupied]
        return [{"type": "pick", "tube": tube}]
    if state.tube_status["NMR_02"] == "inserted":
        return [{"type": "stop"}] if state.terminal_steps >= contract.final_stable_steps else [{"type": "wait"}]
    if state.tube_status["NMR_02"] == "held":
        target = contract.opposite(state.first_actual_slot)
        return [{"type": "insert", "tube": "NMR_02", "slot": target}] if target not in state.occupied else [{"type": "wait"}]
    return [{"type": "pick", "tube": "NMR_02"}]


def unit(v: tuple[float, float, float]) -> tuple[float, float, float] | None:
    if not all(math.isfinite(x) for x in v):
        return None
    norm = math.sqrt(sum(x * x for x in v))
    return tuple(x / norm for x in v) if norm > 1e-9 else None


def evaluate_insertion(
    contract: Contract,
    observation: Observation,
    slots_world: dict[str, tuple[float, float, float]],
    insertion_axis_world: tuple[float, float, float],
) -> dict[str, Any]:
    """Classify a measured tube tip against every socket, never controller target."""
    axis = unit(insertion_axis_world)
    tube_axis = unit(observation.axis_world)
    if axis is None or tube_axis is None:
        raise ValueError("invalid insertion axis")
    if not all(math.isfinite(x) for x in (*observation.tip_world, observation.linear_speed_m_s)):
        raise ValueError("non-finite physics observation")
    candidates = []
    for slot, mouth in slots_world.items():
        delta = tuple(observation.tip_world[i] - mouth[i] for i in range(3))
        depth = sum(delta[i] * axis[i] for i in range(3))
        radial = math.sqrt(max(0.0, sum(x*x for x in delta) - depth*depth))
        candidates.append((radial, slot, depth))
    radial, slot, depth = min(candidates) if candidates else (math.inf, None, -math.inf)
    axis_error = math.degrees(math.acos(max(-1.0, min(1.0, abs(sum(axis[i]*tube_axis[i] for i in range(3)))))))
    pass_geometry = (
        slot is not None and radial <= contract.radial_tolerance_m
        and contract.socket_depth_m * contract.insertion_depth_fraction <= depth <= contract.socket_depth_m
        and axis_error <= contract.axis_tolerance_deg
    )
    stationary = (observation.position_delta_m <= 0.0003 if observation.position_delta_m is not None
                  else observation.linear_speed_m_s <= 0.01)
    return {
        "tube": observation.tube,
        "actual_slot": slot if pass_geometry else None,
        "nearest_slot": slot,
        "radial_error_m": radial,
        "insertion_depth_m": depth,
        "axis_error_deg": axis_error,
        "geometry_pass": pass_geometry,
        "released": observation.gripper_open,
        "supported": observation.supported,
        "linear_speed_m_s": observation.linear_speed_m_s,
        "stability_speed_limit_m_s": 0.01,
        "position_delta_m": observation.position_delta_m,
        "stability_displacement_limit_m": 0.0003,
        "stationarity_source": "pose_delta" if observation.position_delta_m is not None else "velocity",
        "stable_sample": pass_geometry and observation.gripper_open and observation.supported
                         and stationary,
    }


def advance_state(contract: Contract, state: State, measured: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Update only from measured, continuously stable physical occupancy."""
    events: list[dict[str, str]] = []
    for tube in TUBES:
        evidence = measured.get(tube, {})
        slot = evidence.get("actual_slot")
        if not evidence.get("stable_sample") or slot is None:
            state.stable_counts[tube] = 0
            if state.tube_status[tube] == "inserted":
                old = next((s for s, t in state.occupied.items() if t == tube), None)
                if old:
                    state.occupied.pop(old)
                    events.append({"type": "insertion_lost", "tube": tube, "slot": old})
                if tube == "NMR_01":
                    state.first_actual_slot = None
                state.tube_status[tube] = "table"
            continue
        if slot in state.occupied and state.occupied[slot] != tube:
            state.stable_counts[tube] = 0
            continue
        state.stable_counts[tube] = state.stable_counts.get(tube, 0) + 1
        if state.stable_counts[tube] < contract.stable_steps or state.occupied.get(slot) == tube:
            continue
        for old, assigned in list(state.occupied.items()):
            if assigned == tube:
                state.occupied.pop(old)
        state.occupied[slot] = tube
        state.tube_status[tube] = "inserted"
        if tube == "NMR_01":
            state.first_actual_slot = slot
        events.append({"type": "inserted", "tube": tube, "slot": slot})
    complete = (state.first_actual_slot is not None and
                state.occupied.get(state.first_actual_slot) == "NMR_01" and
                state.occupied.get(contract.opposite(state.first_actual_slot)) == "NMR_02")
    state.terminal_steps = state.terminal_steps + 1 if complete else 0
    return events
