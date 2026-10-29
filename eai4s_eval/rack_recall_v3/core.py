"""Simulator-independent contract for rectangular-rack placement recall."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import random
from typing import Any, Mapping


TASK_VERSION = "rack-placement-recall/v3"
EVALUATION_VERSION = "labference/rack-recall-online-v3.1"
SCORING_VERSION = "labference/rack-recall-memory-selection-v2"
SCORING_TRACKS = (
    "explicit_slot_report",
    "symbolic_slot_selection",
    "native_behavior_selection",
)
ROWS = ("A", "B", "C")
COLUMNS = (1, 2, 3, 4)
SLOTS = tuple(f"{row}{column}" for row in ROWS for column in COLUMNS)
TARGET_TUBE = "target_tube"
DISTRACTOR_TUBES = ("distractor_01", "distractor_02", "distractor_03")
ORACLES = ("none", "semantic", "decision", "geometry", "skill")
MOTION_YAWS_DEG = (-90.0, 90.0, 180.0)
INITIAL_YAWS_DEG = (0.0, 90.0, 180.0, 270.0)
APPEARANCE_CONDITIONS = ("single", "same_cap", "distinct_cap")
CAP_COLORS = ("blue", "red", "green", "yellow")


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _vec(values: tuple[float, ...], size: int, label: str) -> None:
    if len(values) != size or not all(_finite(value) for value in values):
        raise ValueError(f"{label} must contain {size} finite numbers")


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RackPose:
    position_m: tuple[float, float, float]
    yaw_deg: float

    def __post_init__(self) -> None:
        _vec(self.position_m, 3, "rack position")
        if not _finite(self.yaw_deg):
            raise ValueError("rack yaw must be finite")

    def local_to_world(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        _vec(point, 3, "local point")
        angle = math.radians(self.yaw_deg)
        c, s = math.cos(angle), math.sin(angle)
        x, y, z = point
        px, py, pz = self.position_m
        return (px + c * x - s * y, py + s * x + c * y, pz + z)

    def world_to_local(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        _vec(point, 3, "world point")
        angle = math.radians(-self.yaw_deg)
        c, s = math.cos(angle), math.sin(angle)
        x, y, z = (point[i] - self.position_m[i] for i in range(3))
        return (c * x - s * y, s * x + c * y, z)

    def public(self) -> dict[str, Any]:
        return {"position_m": list(self.position_m), "yaw_deg": self.yaw_deg}


@dataclass(frozen=True)
class Contract:
    difficulty: str
    condition_variant: str = "single"
    row_count: int = 3
    column_count: int = 4
    row_pitch_m: float = 0.050
    column_pitch_m: float = 0.065
    table_surface_z_m: float = 0.707
    rack_height_m: float = 0.180
    rack_mouth_height_m: float = 0.170
    rack_lower_guide_height_m: float = 0.100
    rack_support_top_height_m: float = 0.070
    rack_support_thickness_m: float = 0.008
    socket_depth_m: float = 0.070
    rack_bottom_support_depth_m: float = 0.100
    bottom_support_tolerance_m: float = 0.006
    recovery_region_size_m: float = 0.160
    recovery_tip_clearance_m: float = 0.005
    tube_scale: float = 1.20
    tube_cap_scale: float = 1.25
    tube_radius_m: float = 0.016932
    tube_length_m: float = 0.14040
    tube_cap_radius_m: float = 0.0215625
    tube_cap_height_m: float = 0.01560
    tube_mass_kg: float = 0.0190
    tube_center_of_mass_z_m: float = -0.016614
    tube_angular_damping: float = 12.0
    grasp_point_from_center_m: float = 0.0678
    tcp_to_finger_pad_z_m: float = 0.0065
    finger_clearance_above_rack_m: float = 0.003
    socket_radius_m: float = 0.021
    radial_tolerance_m: float = 0.0023
    axis_tolerance_deg: float = 10.0
    insertion_depth_fraction: float = 0.60
    stable_steps: int = 60
    final_stable_steps: int = 360
    occlusion_steps: int = 90
    rack_motion_steps: int = 1440
    max_rack_translation_m: float = 0.060
    max_rack_linear_speed_m_s: float = 0.015
    max_rack_angular_speed_deg_s: float = 25.0
    max_tube_speed_m_s: float = 0.015
    max_position_delta_m: float = 0.00015
    max_steps: int = 52000
    physics_hz: int = 120
    collection_hz: int = 20
    free_motion_speed_m_s: float = 0.040
    approach_speed_m_s: float = 0.010
    insertion_speed_m_s: float = 0.003
    insertion_axial_force_limit_n: float = 2.0
    insertion_lateral_force_limit_n: float = 1.0
    insertion_force_hold_s: float = 0.10
    insertion_stall_window_s: float = 1.0
    insertion_stall_progress_m: float = 0.0005
    insertion_retract_m: float = 0.030
    insertion_max_recoveries: int = 3
    insertion_mode: str = "guided_gravity_release"
    behavior_selection_radius_m: float = 0.0125
    behavior_selection_min_height_m: float = 0.015
    behavior_selection_max_height_m: float = 0.045
    behavior_selection_dwell_s: float = 0.20
    behavior_selection_descent_m: float = 0.005
    # Keep the complete 3x4 rack inside the dexterous workspace. At x=-0.4
    # the cap-side grasp drives panda_joint6 to its upper limit before the
    # tube tip clears the upper guide plate.
    franka_base_position_m: tuple[float, float, float] = (-0.3, 0.0, 0.697)
    franka_initial_joint_positions_rad: tuple[float, ...] = (
        0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04,
    )
    robot_warmup_steps: int = 60
    source_rack_asset: str = "assets/converted/rack_3x4.usd"
    source_tube_asset: str = "assets/converted/centrifuge_tube.usd"
    task_version: str = TASK_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "franka_base_position_m",
                           tuple(self.franka_base_position_m))
        object.__setattr__(self, "franka_initial_joint_positions_rad",
                           tuple(self.franka_initial_joint_positions_rad))
        if self.difficulty not in {"easy", "hard"} or self.task_version != TASK_VERSION:
            raise ValueError("unsupported rack-recall difficulty or task version")
        if self.condition_variant not in APPEARANCE_CONDITIONS:
            raise ValueError("unknown rack-recall condition variant")
        if self.difficulty == "easy" and self.condition_variant != "single":
            raise ValueError("easy rack-recall requires the single-tube condition")
        if self.difficulty == "hard" and self.condition_variant == "single":
            raise ValueError("hard rack-recall requires same_cap or distinct_cap")
        if (self.row_count, self.column_count) != (3, 4):
            raise ValueError("rack-placement-recall/v3 requires a 3x4 rack")
        if ((self.column_count - 1) * self.column_pitch_m <=
                (self.row_count - 1) * self.row_pitch_m):
            raise ValueError("rack-placement-recall/v3 requires a rectangular long axis")
        positive = (self.row_pitch_m, self.column_pitch_m, self.table_surface_z_m,
                    self.rack_height_m, self.rack_mouth_height_m,
                    self.rack_lower_guide_height_m,
                    self.rack_support_top_height_m,
                    self.rack_support_thickness_m,
                    self.socket_depth_m, self.rack_bottom_support_depth_m,
                    self.bottom_support_tolerance_m, self.recovery_region_size_m,
                    self.recovery_tip_clearance_m,
                    self.tube_scale, self.tube_cap_scale,
                    self.tube_radius_m, self.tube_length_m,
                    self.tube_cap_radius_m, self.tube_cap_height_m,
                    self.tube_mass_kg, self.tube_angular_damping,
                    self.grasp_point_from_center_m,
                    self.tcp_to_finger_pad_z_m,
                    self.finger_clearance_above_rack_m,
                    self.socket_radius_m, self.radial_tolerance_m,
                    self.axis_tolerance_deg, self.max_rack_translation_m,
                    self.max_rack_linear_speed_m_s, self.max_rack_angular_speed_deg_s,
                    self.max_tube_speed_m_s, self.max_position_delta_m,
                    self.free_motion_speed_m_s, self.approach_speed_m_s,
                    self.insertion_speed_m_s, self.insertion_axial_force_limit_n,
                    self.insertion_lateral_force_limit_n, self.insertion_force_hold_s,
                    self.insertion_stall_window_s, self.insertion_stall_progress_m,
                    self.insertion_retract_m, self.behavior_selection_radius_m,
                    self.behavior_selection_min_height_m,
                    self.behavior_selection_max_height_m,
                    self.behavior_selection_dwell_s,
                    self.behavior_selection_descent_m)
        if not all(_finite(value) and value > 0 for value in positive):
            raise ValueError("rack geometry and thresholds must be finite and positive")
        if self.socket_radius_m <= self.tube_radius_m:
            raise ValueError("socket must be wider than the tube")
        if self.tube_cap_radius_m <= self.tube_radius_m:
            raise ValueError("tube cap must provide a wider physical flange")
        if self.tube_cap_radius_m <= self.socket_radius_m:
            raise ValueError("tube cap must remain wider than the rack opening")
        if self.rack_bottom_support_depth_m <= self.socket_depth_m:
            raise ValueError("rack bottom support must be below the lower guide")
        if not (self.rack_height_m > self.rack_mouth_height_m >
                self.rack_lower_guide_height_m > self.rack_support_top_height_m):
            raise ValueError("rack plate and support heights must be strictly ordered")
        if not math.isclose(
                self.socket_depth_m,
                self.rack_mouth_height_m - self.rack_lower_guide_height_m,
                abs_tol=1e-9):
            raise ValueError("socket depth must match the two CAD guide surfaces")
        if not math.isclose(
                self.rack_bottom_support_depth_m,
                self.rack_mouth_height_m - self.rack_support_top_height_m,
                abs_tol=1e-9):
            raise ValueError("bottom support depth must match the visible support plate")
        if self.tube_cap_height_m >= self.tube_length_m / 4:
            raise ValueError("tube cap height is inconsistent with tube length")
        if not (self.tube_length_m / 2 - self.tube_cap_height_m / 2 <=
                self.grasp_point_from_center_m < self.tube_length_m / 2):
            raise ValueError("grasp point must lie in the upper half of the tube cap")
        estimated_pad_half_height_m = .010
        seated_grasp_height = (self.grasp_point_from_center_m -
                               (self.tube_length_m / 2 - self.tube_cap_height_m))
        if seated_grasp_height - estimated_pad_half_height_m < self.finger_clearance_above_rack_m:
            raise ValueError("cap grasp does not preserve finger clearance above the rack")
        if (not _finite(self.tube_center_of_mass_z_m) or
                abs(self.tube_center_of_mass_z_m) >= self.tube_length_m / 2):
            raise ValueError("tube center of mass must lie inside the tube body")
        if self.radial_tolerance_m > self.socket_radius_m - self.tube_radius_m + 1e-9:
            raise ValueError("radial tolerance exceeds socket clearance")
        if not 0 < self.insertion_depth_fraction <= 1:
            raise ValueError("insertion depth fraction must be in (0, 1]")
        if min(self.stable_steps, self.final_stable_steps, self.occlusion_steps,
               self.rack_motion_steps, self.max_steps, self.robot_warmup_steps) <= 0:
            raise ValueError("step budgets must be positive")
        if (type(self.physics_hz) is not int or type(self.collection_hz) is not int or
                self.physics_hz <= 0 or self.collection_hz <= 0 or
                self.physics_hz % self.collection_hz):
            raise ValueError("physics_hz must be an integer multiple of collection_hz")
        if type(self.insertion_max_recoveries) is not int or self.insertion_max_recoveries < 0:
            raise ValueError("insertion_max_recoveries must be a non-negative integer")
        if self.insertion_mode not in {"guided_gravity_release", "controlled_descent"}:
            raise ValueError("unknown insertion mode")
        if self.behavior_selection_max_height_m <= self.behavior_selection_min_height_m:
            raise ValueError("behavior selection height interval is invalid")
        if self.final_stable_steps < self.stable_steps:
            raise ValueError("final stability must include insertion stability")
        _vec(self.franka_base_position_m, 3, "Franka base position")
        _vec(self.franka_initial_joint_positions_rad, 9,
             "Franka initial joint positions")

    def slot_local_center(self, slot: str) -> tuple[float, float, float]:
        if slot not in SLOTS:
            raise ValueError(f"unknown rack slot: {slot}")
        row = ROWS.index(slot[0])
        column = COLUMNS.index(int(slot[1:]))
        x = (column - (self.column_count - 1) / 2.0) * self.column_pitch_m
        y = (row - (self.row_count - 1) / 2.0) * self.row_pitch_m
        return (x, y, self.rack_mouth_height_m)

    def slot_world_center(self, slot: str, pose: RackPose) -> tuple[float, float, float]:
        return pose.local_to_world(self.slot_local_center(slot))

    def canonical(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def hash(self) -> str:
        return _digest(self.canonical())


@dataclass(frozen=True)
class Instance:
    difficulty: str
    seed: int
    layout_seed: int
    motion_seed: int
    policy_seed: int
    color_seed: int
    rack_initial_pose: RackPose
    rack_final_pose: RackPose
    target_spawn_xy_m: tuple[float, float]
    recovery_center_xy_m: tuple[float, float]
    distractor_slots: tuple[str, ...]
    pair_id: str
    condition_variant: str = "single"
    cap_colors: tuple[tuple[str, str], ...] = ((TARGET_TUBE, "blue"),)

    def __post_init__(self) -> None:
        if self.difficulty not in {"easy", "hard"}:
            raise ValueError("unknown rack-recall difficulty")
        if self.condition_variant not in APPEARANCE_CONDITIONS:
            raise ValueError("unknown rack-recall condition variant")
        if self.difficulty == "easy" and self.condition_variant != "single":
            raise ValueError("easy rack-recall requires the single-tube condition")
        if self.difficulty == "hard" and self.condition_variant == "single":
            raise ValueError("hard rack-recall requires same_cap or distinct_cap")
        expected = 0 if self.difficulty == "easy" else 3
        if len(self.distractor_slots) != expected or len(set(self.distractor_slots)) != expected:
            raise ValueError(f"{self.difficulty} requires {expected} distinct distractors")
        if any(slot not in SLOTS for slot in self.distractor_slots):
            raise ValueError("unknown distractor slot")
        _vec(self.target_spawn_xy_m, 2, "target spawn")
        _vec(self.recovery_center_xy_m, 2, "recovery center")
        translation = math.dist(self.rack_initial_pose.position_m[:2],
                                self.rack_final_pose.position_m[:2])
        if self.difficulty == "easy" and (translation > 1e-9 or
                abs(self.rack_initial_pose.yaw_deg - self.rack_final_pose.yaw_deg) > 1e-9):
            raise ValueError("easy rack must remain stationary")
        colors = dict(self.cap_colors)
        expected_tubes = {TARGET_TUBE, *DISTRACTOR_TUBES[:expected]}
        if set(colors) != expected_tubes or any(color not in CAP_COLORS for color in colors.values()):
            raise ValueError("cap color assignment does not match the scene tubes")
        if self.condition_variant == "same_cap" and len(set(colors.values())) != 1:
            raise ValueError("same_cap requires visually identical cap colors")
        if self.condition_variant == "distinct_cap" and len(set(colors.values())) != 4:
            raise ValueError("distinct_cap requires four different cap colors")

    @property
    def layout_hash(self) -> str:
        return _digest({"rack_initial_pose": self.rack_initial_pose.public(),
                        "target_spawn_xy_m": self.target_spawn_xy_m,
                        "recovery_center_xy_m": self.recovery_center_xy_m,
                        "distractor_slots": self.distractor_slots,
                        "condition_variant": self.condition_variant,
                        "cap_colors": self.cap_colors})

    @property
    def motion_hash(self) -> str:
        return _digest({"rack_final_pose": self.rack_final_pose.public(),
                        "motion_seed": self.motion_seed})

    def public(self) -> dict[str, Any]:
        return {
            "task_version": TASK_VERSION,
            "difficulty": self.difficulty,
            "condition_variant": self.condition_variant,
            "appearance_condition": self.condition_variant,
            "rack_grid": "3x4",
            "slot_labels": list(SLOTS),
            "rule": (
                "Pick the loose centrifuge tube and place it in any empty rack slot. "
                "After the occlusion, use the visible asymmetric rack orientation and the "
                "earlier placement outcome to retrieve that same tube, place it in the "
                "recovery area, and stop only after it is stable. Cap color may be an "
                "authorized appearance cue, but it never directly marks the answer."
            ),
        }

    def cap_color(self, tube: str) -> str:
        try:
            return dict(self.cap_colors)[tube]
        except KeyError as exc:
            raise ValueError(f"unknown tube color request: {tube}") from exc


def generate_instance(contract: Contract, seed: int, *, layout_seed: int | None = None,
                      motion_seed: int | None = None, policy_seed: int | None = None,
                      color_seed: int | None = None, pair_id: str | None = None,
                      reserved_slot: str | None = None,
                      initial_yaw_deg: float | None = None,
                      motion_yaw_deg: float | None = None,
                      target_cap_color: str | None = None) -> Instance:
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    layout_seed = seed + 10_000 if layout_seed is None else layout_seed
    motion_seed = seed + 20_000 if motion_seed is None else motion_seed
    policy_seed = seed + 30_000 if policy_seed is None else policy_seed
    color_seed = seed + 40_000 if color_seed is None else color_seed
    if reserved_slot is not None and reserved_slot not in SLOTS:
        raise ValueError("reserved slot must be one of A1-C4")
    layout_rng = random.Random(layout_seed)
    motion_rng = random.Random(motion_seed)
    color_rng = random.Random(color_seed)
    table_z = contract.table_surface_z_m
    initial = RackPose((round(0.19 + layout_rng.uniform(-.012, .012), 4),
                        round(0.17 + layout_rng.uniform(-.012, .012), 4), table_z),
                       float(layout_rng.choice(INITIAL_YAWS_DEG)
                             if initial_yaw_deg is None else initial_yaw_deg))
    distractor_candidates = [slot for slot in SLOTS if slot != reserved_slot]
    distractors = (tuple(sorted(layout_rng.sample(distractor_candidates, 3), key=SLOTS.index))
                   if contract.difficulty == "hard" else ())
    target_xy = (round(0.19 + layout_rng.uniform(-.010, .010), 4),
                 round(-.18 + layout_rng.uniform(-.010, .010), 4))
    # Keep the terminal tray in Franka's comfortable high-clearance workspace.
    # The previous (+.34, +.01) placement approached a wrist singularity for
    # the fixed cap-grasp orientation and made an otherwise irrelevant terminal
    # transport dominate the recall task.
    recovery_xy = (round(.315 + layout_rng.uniform(-.008, .008), 4),
                   round(-.055 + layout_rng.uniform(-.008, .008), 4))
    if contract.difficulty == "hard":
        distance = motion_rng.uniform(.025, contract.max_rack_translation_m)
        angle = motion_rng.uniform(-math.pi, math.pi)
        dx, dy = distance * math.cos(angle), distance * math.sin(angle)
        final = RackPose((round(initial.position_m[0] + dx, 4),
                          round(initial.position_m[1] + dy, 4), table_z),
                         initial.yaw_deg + (motion_rng.choice(MOTION_YAWS_DEG)
                                            if motion_yaw_deg is None else motion_yaw_deg))
    else:
        final = initial
    target_color = (target_cap_color or color_rng.choice(CAP_COLORS))
    if target_color not in CAP_COLORS:
        raise ValueError("target cap color must be in the registered palette")
    if contract.condition_variant == "distinct_cap":
        others = [color for color in CAP_COLORS if color != target_color]
        color_rng.shuffle(others)
        cap_colors = ((TARGET_TUBE, target_color), *zip(DISTRACTOR_TUBES, others))
    elif contract.condition_variant == "same_cap":
        cap_colors = tuple((name, target_color)
                           for name in (TARGET_TUBE, *DISTRACTOR_TUBES))
    else:
        cap_colors = ((TARGET_TUBE, target_color),)
    return Instance(contract.difficulty, seed, layout_seed, motion_seed, policy_seed,
                    color_seed,
                    initial, final, target_xy, recovery_xy, distractors,
                    pair_id or (
                        f"rack-recall-{contract.difficulty}-"
                        f"{contract.condition_variant}-{seed}"),
                    contract.condition_variant, tuple(cap_colors))


def generate_coverage_manifest(contract: Contract,
                               seeds: tuple[int, ...] = (101, 102, 103, 104, 105)) -> dict[str, Any]:
    """Create the fixed-denominator, slot-balanced checkpoint protocol."""
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("coverage seeds must be a non-empty integer sequence")
    members = []
    for seed_index, seed in enumerate(seeds):
        for slot_index, slot in enumerate(SLOTS):
            initial_yaw = INITIAL_YAWS_DEG[(slot_index + seed_index) % 4]
            motion_yaw = (0.0 if contract.difficulty == "easy" else
                          MOTION_YAWS_DEG[(slot_index + seed_index) % len(MOTION_YAWS_DEG)])
            target_color = CAP_COLORS[(slot_index + seed_index) % len(CAP_COLORS)]
            member_id = f"{contract.condition_variant}-{slot}-s{seed}"
            members.append({
                "member_id": member_id,
                "task_version": contract.task_version,
                "contract_hash": contract.hash,
                "scoring_version": SCORING_VERSION,
                "condition": contract.condition_variant,
                "seed": seed,
                "layout_seed": seed + 10_000,
                "motion_seed": seed + 20_000,
                "policy_seed": seed + 30_000,
                "color_seed": seed + 40_000,
                "checkpoint_slot": slot,
                "initial_yaw_deg": initial_yaw,
                "motion_yaw_deg": motion_yaw,
                "target_cap_color": target_color,
                "evaluation_track": "verified_state_checkpoint",
                "history_protocol": "online_once",
            })
    return {
        "schema_version": "labference/rack-recall-coverage-v3",
        "task_version": contract.task_version,
        "evaluation_version": EVALUATION_VERSION,
        "scoring_version": SCORING_VERSION,
        "primary_metric": "slot_reasoning_success",
        "placement_skill_gate_applied": False,
        "condition": contract.condition_variant,
        "seeds": list(seeds),
        "slots": list(SLOTS),
        "fixed_denominator": len(members),
        "members": members,
    }


@dataclass(frozen=True)
class TubeObservation:
    tube: str
    center_world_m: tuple[float, float, float]
    axis_world: tuple[float, float, float]
    linear_speed_m_s: float
    released: bool
    rack_supported: bool
    left_finger_contact: bool = False
    right_finger_contact: bool = False
    in_recovery_region: bool = False
    position_delta_m: float | None = None
    rack_contact_force_world_n: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rack_axial_force_n: float = 0.0
    rack_lateral_force_n: float = 0.0
    above_recovery_region: bool = False
    recovery_table_supported: bool = False

    def __post_init__(self) -> None:
        _vec(self.center_world_m, 3, "tube center")
        _vec(self.axis_world, 3, "tube axis")
        _vec(self.rack_contact_force_world_n, 3, "rack contact force")
        if not _finite(self.linear_speed_m_s) or self.linear_speed_m_s < 0:
            raise ValueError("tube speed must be finite and non-negative")
        if (not _finite(self.rack_axial_force_n) or self.rack_axial_force_n < 0 or
                not _finite(self.rack_lateral_force_n) or self.rack_lateral_force_n < 0):
            raise ValueError("rack contact forces must be finite and non-negative")


@dataclass(frozen=True)
class Measurement:
    step: int
    rack_pose: RackPose
    environment_phase: str
    occluder_visible: bool
    tubes: Mapping[str, TubeObservation]
    rack_motion_complete: bool
    evidence_delivered: bool

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("measurement step must be a non-negative integer")
        if self.environment_phase not in {"placement", "occluded", "recall", "recovery"}:
            raise ValueError("unknown environment phase")
        if TARGET_TUBE not in self.tubes:
            raise ValueError("measurement is missing the target tube")


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float] | None:
    if not all(_finite(value) for value in vector):
        return None
    norm = math.sqrt(sum(value * value for value in vector))
    return tuple(value / norm for value in vector) if norm > 1e-9 else None


def evaluate_tube(contract: Contract, observation: TubeObservation,
                  rack_pose: RackPose) -> dict[str, Any]:
    tip = tuple(observation.center_world_m[i] + observation.axis_world[i] *
                contract.tube_length_m / 2.0 for i in range(3))
    distances = {slot: math.dist(tip[:2], contract.slot_world_center(slot, rack_pose)[:2])
                 for slot in SLOTS}
    nearest = min(distances, key=distances.get)
    mouth = contract.slot_world_center(nearest, rack_pose)
    insertion_axis = (0.0, 0.0, -1.0)
    axis = _unit(observation.axis_world)
    dot = None if axis is None else max(-1.0, min(1.0, sum(
        axis[i] * insertion_axis[i] for i in range(3))))
    axis_error = math.inf if dot is None else math.degrees(math.acos(dot))
    depth = mouth[2] - tip[2]
    radial = distances[nearest]
    # The rack guides the tube through two apertures, while the rack bottom
    # plate carries its weight. The cap remains above the upper guide and is
    # not used as a support predicate.
    cap_base = tuple(
        observation.center_world_m[i] - observation.axis_world[i] *
        (contract.tube_length_m / 2.0 - contract.tube_cap_height_m)
        for i in range(3))
    cap_clearance = cap_base[2] - mouth[2]
    bottom_support_z = mouth[2] - contract.rack_bottom_support_depth_m
    bottom_support_error = abs(tip[2] - bottom_support_z)
    geometry = (radial <= contract.radial_tolerance_m and
                depth >= contract.socket_depth_m * contract.insertion_depth_fraction and
                bottom_support_error <= contract.bottom_support_tolerance_m and
                axis_error <= contract.axis_tolerance_deg)
    stable = (geometry and observation.released and observation.rack_supported and
              observation.linear_speed_m_s <= contract.max_tube_speed_m_s and
              (observation.position_delta_m is None or
               observation.position_delta_m <= contract.max_position_delta_m))
    return {"tube": observation.tube, "tip_world_m": list(tip),
            "nearest_slot": nearest, "actual_slot": nearest if geometry else None,
            "radial_error_m": radial, "insertion_depth_m": depth,
            "cap_base_world_m": list(cap_base),
            "cap_base_clearance_above_upper_plate_m": cap_clearance,
            "bottom_support_world_z_m": bottom_support_z,
            "bottom_support_error_m": bottom_support_error,
            "axis_error_deg": axis_error, "geometry_pass": geometry,
            "stable_sample": stable, "released": observation.released,
            "rack_supported": observation.rack_supported,
            "rack_contact_force_world_n": list(observation.rack_contact_force_world_n),
            "rack_axial_force_n": observation.rack_axial_force_n,
            "rack_lateral_force_n": observation.rack_lateral_force_n,
            "both_fingers_contact": observation.left_finger_contact and
                                    observation.right_finger_contact,
            "above_recovery_region": observation.above_recovery_region,
            "recovery_table_supported": observation.recovery_table_supported,
            "in_recovery_region": observation.in_recovery_region}


@dataclass
class State:
    difficulty: str
    phase: str = "placement"
    target_status: str = "table"
    actual_placement_slot: str | None = None
    occupied: dict[str, str] = field(default_factory=dict)
    insertion_stable_count: int = 0
    recovery_stable_count: int = 0
    environment_ready: bool = False
    failure_seen: bool = False
    violations: list[str] = field(default_factory=list)
    grasp_stable_count: int = 0
    released_invalid_count: int = 0
    occlusion_observed: bool = False
    retrieval_stable_count: int = 0
    distractor_invalid_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def initial(cls, instance: Instance) -> "State":
        return cls(instance.difficulty,
                   occupied={slot: tube for slot, tube in
                             zip(instance.distractor_slots, DISTRACTOR_TUBES)})

    def public(self) -> dict[str, Any]:
        return {"difficulty": self.difficulty, "phase": self.phase,
                "target_status": self.target_status,
                "occupied": dict(self.occupied),
                "environment_ready": self.environment_ready}

    def evaluator(self) -> dict[str, Any]:
        return {**self.public(), "actual_placement_slot": self.actual_placement_slot,
                "insertion_stable_count": self.insertion_stable_count,
                "recovery_stable_count": self.recovery_stable_count,
                "failure_seen": self.failure_seen, "violations": list(self.violations),
                "grasp_stable_count": self.grasp_stable_count,
                "released_invalid_count": self.released_invalid_count,
                "occlusion_observed": self.occlusion_observed,
                "retrieval_stable_count": self.retrieval_stable_count,
                "distractor_invalid_counts": dict(self.distractor_invalid_counts)}


def legal_actions(contract: Contract, state: State) -> list[dict[str, Any]]:
    if state.difficulty != contract.difficulty:
        raise ValueError("state and contract difficulty mismatch")
    if state.phase == "placement":
        if state.target_status == "held":
            return [{"type": "insert", "tube": TARGET_TUBE, "slot": slot}
                    for slot in SLOTS if slot not in state.occupied]
        return [{"type": "pick", "tube": TARGET_TUBE}]
    if state.phase == "transition":
        return [{"type": "wait"}]
    if state.phase == "recall":
        if state.actual_placement_slot not in SLOTS:
            return [{"type": "wait"}]
        return [{"type": "retrieve", "slot": state.actual_placement_slot}]
    if state.phase == "recovery":
        if state.target_status == "held":
            return [{"type": "place_recovery", "tube": TARGET_TUBE}]
        if state.target_status == "recovering":
            return [{"type": "wait"}]
        return [{"type": "pick", "tube": TARGET_TUBE}]
    if state.phase == "complete":
        return ([{"type": "stop"}] if state.recovery_stable_count >= contract.final_stable_steps
                else [{"type": "wait"}])
    raise ValueError(f"unknown task phase: {state.phase}")


def normalize_action(action: Any) -> dict[str, Any] | None:
    if not isinstance(action, Mapping) or not isinstance(action.get("type"), str):
        return None
    kind = action["type"]
    if kind in {"wait", "stop"}:
        return {"type": kind}
    if kind == "pick" and action.get("tube") == TARGET_TUBE:
        return {"type": kind, "tube": TARGET_TUBE}
    if kind == "insert" and action.get("tube") == TARGET_TUBE and action.get("slot") in SLOTS:
        return {"type": kind, "tube": TARGET_TUBE, "slot": action["slot"]}
    if kind == "retrieve" and action.get("slot") in SLOTS:
        return {"type": kind, "slot": action["slot"]}
    if kind == "place_recovery" and action.get("tube") == TARGET_TUBE:
        return {"type": kind, "tube": TARGET_TUBE}
    return None


def advance_state(contract: Contract, instance: Instance, state: State,
                  measurement: Measurement) -> list[dict[str, Any]]:
    evidence = {tube: evaluate_tube(contract, value, measurement.rack_pose)
                for tube, value in measurement.tubes.items()}
    events: list[dict[str, Any]] = []
    target_raw = measurement.tubes[TARGET_TUBE]
    target = evidence[TARGET_TUBE]

    for slot, tube in list(state.occupied.items()):
        if tube == TARGET_TUBE:
            continue
        sample = evidence.get(tube)
        if state.phase != "transition":
            invalid = sample is None or sample["actual_slot"] != slot
            state.distractor_invalid_counts[tube] = (
                state.distractor_invalid_counts.get(tube, 0) + 1 if invalid else 0)
            if (state.distractor_invalid_counts[tube] >= contract.stable_steps and
                    "distractor_displaced" not in state.violations):
                state.violations.append("distractor_displaced")
                events.append({"type": "violation", "kind": "distractor_displaced",
                               "tube": tube, "expected_slot": slot})
                state.failure_seen = True

    if state.phase == "placement":
        lifted_from_table = target["tip_world_m"][2] >= (
            contract.table_surface_z_m + .010)
        if (state.target_status == "table" and target["both_fingers_contact"] and
                lifted_from_table):
            state.grasp_stable_count += 1
            if state.grasp_stable_count == 3:
                state.target_status = "held"
                events.append({"type": "physical", "event": "target_grasped",
                               "evidence": "bilateral_contact_and_lift"})
        elif state.target_status == "table":
            state.grasp_stable_count = 0
        elif (state.target_status == "held" and not target["both_fingers_contact"] and
              target["actual_slot"] is None and
              target["tip_world_m"][2] <= contract.table_surface_z_m + .005):
            state.target_status = "table"
            state.failure_seen = True
            state.grasp_stable_count = 0
            events.append({"type": "physical", "event": "target_dropped"})
        elif (state.target_status == "held" and target["released"] and
              target["actual_slot"] is None):
            state.released_invalid_count += 1
            if state.released_invalid_count >= contract.stable_steps:
                state.target_status = "table"
                state.failure_seen = True
                state.grasp_stable_count = 0
                events.append({"type": "physical",
                               "event": "target_released_without_valid_insertion"})
        else:
            state.released_invalid_count = 0
        if (state.target_status == "held" and target["stable_sample"] and
                target["actual_slot"] not in instance.distractor_slots):
            state.insertion_stable_count += 1
            if state.insertion_stable_count == contract.stable_steps:
                state.actual_placement_slot = target["actual_slot"]
                state.occupied[state.actual_placement_slot] = TARGET_TUBE
                state.target_status = "inserted"
                state.phase = "transition"
                events.append({"type": "physical", "event": "placement_confirmed",
                               "actual_slot": state.actual_placement_slot})
        else:
            if state.insertion_stable_count:
                events.append({"type": "physical", "event": "placement_stability_reset"})
            state.insertion_stable_count = 0
    elif state.phase == "transition":
        if measurement.environment_phase == "occluded" and measurement.occluder_visible:
            state.occlusion_observed = True
        if (state.occlusion_observed and measurement.environment_phase == "recall" and
                measurement.rack_motion_complete and
                measurement.evidence_delivered and not measurement.occluder_visible):
            state.phase = "recall"
            state.environment_ready = True
            events.append({"type": "environment", "event": "recall_evidence_delivered"})
    elif state.phase == "recall":
        placed_mouth_z = (None if state.actual_placement_slot is None else
                          contract.slot_world_center(
                              state.actual_placement_slot, measurement.rack_pose)[2])
        extracted = (placed_mouth_z is not None and
                     target["tip_world_m"][2] >= placed_mouth_z + .005)
        if target["both_fingers_contact"] and target["actual_slot"] is None and extracted:
            state.retrieval_stable_count += 1
            if state.retrieval_stable_count == contract.stable_steps:
                old_slot = state.actual_placement_slot
                if old_slot is not None:
                    state.occupied.pop(old_slot, None)
                state.target_status = "held"
                state.phase = "recovery"
                events.append({"type": "physical", "event": "target_retrieved",
                               "from_slot": old_slot})
        else:
            state.retrieval_stable_count = 0
        for slot, tube in state.occupied.items():
            if tube != TARGET_TUBE and evidence.get(tube, {}).get("both_fingers_contact"):
                state.failure_seen = True
                if "distractor_touched" not in state.violations:
                    state.violations.append("distractor_touched")
                    events.append({"type": "violation", "kind": "distractor_touched",
                                   "tube": tube, "slot": slot})
    elif state.phase in {"recovery", "complete"}:
        lifted_from_table = target["tip_world_m"][2] >= contract.table_surface_z_m + .010
        if (state.phase == "recovery" and state.target_status != "held" and
                target["both_fingers_contact"] and lifted_from_table):
            state.grasp_stable_count += 1
            if state.grasp_stable_count == 3:
                state.target_status = "held"
                events.append({"type": "physical", "event": "target_regrasped"})
        elif state.phase == "recovery" and state.target_status != "held":
            state.grasp_stable_count = 0
        if (state.phase == "recovery" and state.target_status in {"held", "recovering"} and
                target_raw.released and not target_raw.above_recovery_region):
            state.target_status = "table"
            state.failure_seen = True
            state.recovery_stable_count = 0
            events.append({"type": "physical", "event": "target_dropped_during_recovery"})
        if (target_raw.in_recovery_region and target_raw.recovery_table_supported and
                target_raw.released and (
                target_raw.linear_speed_m_s <= contract.max_tube_speed_m_s and
                (target_raw.position_delta_m is None or
                 target_raw.position_delta_m <= contract.max_position_delta_m))):
            state.recovery_stable_count += 1
            if state.recovery_stable_count == 1:
                state.target_status = "recovering"
                events.append({"type": "physical", "event": "recovery_stabilizing"})
            if state.recovery_stable_count == contract.final_stable_steps:
                state.target_status = "recovered"
                state.phase = "complete"
                events.append({"type": "physical", "event": "recovery_confirmed"})
        else:
            if state.recovery_stable_count:
                events.append({"type": "physical", "event": "recovery_stability_reset"})
            state.recovery_stable_count = 0
    return events


def task_complete(contract: Contract, state: State) -> bool:
    return (state.phase == "complete" and state.target_status == "recovered" and
            state.recovery_stable_count >= contract.final_stable_steps and
            "distractor_displaced" not in state.violations)
