from __future__ import annotations

from typing import Any

import numpy as np

from eai4s_eval.eval.success.registry import PredicateRegistry
from eai4s_eval.eval.success.schemas import PredicateValue
from eai4s_eval.eval.success.snapshot import EvaluationSnapshot, MissingSnapshotValue


class MissingPredicateState(ValueError):
    pass


def _ok(name: str, passed: bool, value: Any = None, **details: Any) -> PredicateValue:
    return PredicateValue(name=name, passed=bool(passed), value=value, details=details)


def _missing(name: str, exc: MissingSnapshotValue) -> MissingPredicateState:
    return MissingPredicateState(f"Predicate {name!r} requires snapshot field {exc.args[0]!r}")


def _array(snapshot: EvaluationSnapshot, key: str, name: str) -> np.ndarray:
    try:
        return snapshot.array(key)
    except MissingSnapshotValue as exc:
        raise _missing(name, exc) from exc


def _scalar_any(snapshot: EvaluationSnapshot, keys: list[str], name: str) -> tuple[str, float]:
    try:
        key, value = snapshot.first_available(keys)
    except MissingSnapshotValue as exc:
        raise _missing(name, exc) from exc
    return key, float(np.asarray(value, dtype=np.float64))


def _bool_any(snapshot: EvaluationSnapshot, keys: list[str], name: str) -> tuple[str, bool]:
    try:
        key, value = snapshot.first_available(keys)
    except MissingSnapshotValue as exc:
        raise _missing(name, exc) from exc
    return key, bool(value)


def gripper_closed(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "gripper_closed"
    # Franka finger joints are 0.0 when fully closed and about 0.04 when open.
    # A grasped object may stop the fingers between those endpoints, so the
    # configured threshold separates an occupied/closed grasp from fully open.
    threshold = float(args.get("closed_threshold", args.get("grip_threshold", 0.035)))
    gripper = _array(snapshot, str(args.get("gripper_key", "gripper_pos")), name)
    mean_pos = float(np.mean(gripper))
    return _ok(name, mean_pos < threshold, round(mean_pos, 6), threshold=threshold)


def object_near_hand(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "object_near_hand"
    threshold = float(args.get("distance_threshold_m", args.get("threshold_m", 0.15)))
    object_key = str(args.get("object_key", "object_pos"))
    hand_key = str(args.get("hand_key", "hand_pos"))
    obj = _array(snapshot, object_key, name)
    hand = _array(snapshot, hand_key, name)
    distance = float(np.linalg.norm(obj[:3] - hand[:3]))
    return _ok(name, distance < threshold, round(distance, 6), distance_m=distance, threshold_m=threshold)


def near(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "near"
    threshold = float(args.get("distance_threshold_m", args.get("threshold_m", 0.05)))
    a_key = str(args.get("a_key", "object_pos"))
    b_key = str(args.get("b_key", "target_pos"))
    axes = str(args.get("axes", args.get("distance_axes", "xyz"))).lower()
    a = _array(snapshot, a_key, name)
    b = _array(snapshot, b_key, name)
    if axes in {"xy", "planar", "horizontal"}:
        delta = a[:2] - b[:2]
    elif axes in {"z", "vertical"}:
        delta = np.asarray([a[2] - b[2]], dtype=np.float64)
    else:
        axes = "xyz"
        delta = a[:3] - b[:3]
    distance = float(np.linalg.norm(delta))
    return _ok(name, distance <= threshold, round(distance, 6), distance_m=distance, threshold_m=threshold, axes=axes)


def grasped(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "grasped"
    grip = gripper_closed(snapshot, args)
    near_hand = object_near_hand(snapshot, args)
    passed = grip.passed and near_hand.passed
    return _ok(
        name,
        passed,
        passed,
        gripper_closed=grip.passed,
        object_near_hand=near_hand.passed,
        gripper_mean=grip.value,
        grip_threshold=grip.details["threshold"],
        hand_object_distance_m=near_hand.details["distance_m"],
        distance_threshold_m=near_hand.details["threshold_m"],
    )


def lifted(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "lifted"
    threshold = float(args.get("height_threshold_m", 0.05))
    object_key = str(args.get("object_key", "object_pos"))
    initial_key = str(args.get("initial_object_key", "initial_object_pos"))
    obj = _array(snapshot, object_key, name)
    initial = _array(snapshot, initial_key, name)
    height_change = float(obj[2] - initial[2])
    return _ok(name, height_change >= threshold, round(height_change, 6), height_change_m=height_change, threshold_m=threshold)


def object_velocity_below(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = str(args.get("id") or "object_velocity_below")
    lin_threshold = float(args.get("linear_velocity_threshold_mps", args.get("lin_threshold_mps", 0.05)))
    ang_threshold = float(args.get("angular_velocity_threshold_radps", args.get("ang_threshold_radps", 0.50)))
    lin = np.asarray(snapshot.get(str(args.get("lin_vel_key", "object_lin_vel")), [0.0, 0.0, 0.0]), dtype=np.float64)
    ang = np.asarray(snapshot.get(str(args.get("ang_vel_key", "object_ang_vel")), [0.0, 0.0, 0.0]), dtype=np.float64)
    lin_norm = float(np.linalg.norm(lin))
    ang_norm = float(np.linalg.norm(ang))
    return _ok(
        name,
        lin_norm < lin_threshold and ang_norm < ang_threshold,
        {"linear": round(lin_norm, 6), "angular": round(ang_norm, 6)},
        linear_velocity_mps=lin_norm,
        angular_velocity_radps=ang_norm,
        linear_threshold_mps=lin_threshold,
        angular_threshold_radps=ang_threshold,
    )


def object_out_of_workspace(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "object_out_of_workspace"
    radius = float(args.get("workspace_radius_m", args.get("radius_m", 2.0)))
    obj = _array(snapshot, str(args.get("object_key", "object_pos")), name)
    reference = np.asarray(snapshot.get(str(args.get("reference_key", "initial_object_pos")), [0.0, 0.0, 0.0]), dtype=np.float64)
    distance = float(np.linalg.norm(obj[:2] - reference[:2]))
    return _ok(name, distance > radius, round(distance, 6), distance_xy_m=distance, radius_m=radius)


def object_dropped(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "object_dropped"
    min_height = float(args.get("min_height_m", args.get("height_threshold_m", 0.02)))
    if snapshot.get("object_height", None) is not None or snapshot.get("task_state.object_height", None) is not None:
        _, height = _scalar_any(snapshot, ["object_height", "task_state.object_height"], name)
    else:
        obj = _array(snapshot, str(args.get("object_key", "object_pos")), name)
        height = float(obj[2])
    return _ok(name, height < min_height, round(height, 6), object_height_m=height, min_height_m=min_height)


def wrong_target_reached(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "wrong_target_reached"
    if snapshot.get("wrong_target_reached", None) is not None or snapshot.get("task_state.wrong_target_reached", None) is not None:
        _, value = _bool_any(snapshot, ["wrong_target_reached", "task_state.wrong_target_reached"], name)
        return _ok(name, value, value)
    expected = args.get("expected_target_region_id", args.get("target_region_id"))
    try:
        _, reached = snapshot.first_available(["target_region_id", "reached_region_id", "task_state.target_region_id", "task_state.reached_region_id"])
    except MissingSnapshotValue as exc:
        raise _missing(name, exc) from exc
    return _ok(name, expected is not None and reached != expected, reached, expected_target_region_id=expected)


def wrong_object_selected(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "wrong_object_selected"
    if snapshot.get("wrong_object_selected", None) is not None or snapshot.get("task_state.wrong_object_selected", None) is not None:
        _, value = _bool_any(snapshot, ["wrong_object_selected", "task_state.wrong_object_selected"], name)
        return _ok(name, value, value)
    expected = args.get("expected_object_id", args.get("object_id"))
    try:
        _, selected = snapshot.first_available(["selected_object_id", "grasped_object_id", "task_state.selected_object_id", "task_state.grasped_object_id"])
    except MissingSnapshotValue as exc:
        raise _missing(name, exc) from exc
    return _ok(name, expected is not None and selected != expected, selected, expected_object_id=expected)


def contact_force_exceeded(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "contact_force_exceeded"
    threshold = float(args.get("max_force_n", args.get("threshold_n", 50.0)))
    _, force = _scalar_any(snapshot, ["max_contact_force_n", "contact_force_n", "task_state.max_contact_force_n"], name)
    return _ok(name, force > threshold, round(force, 6), max_contact_force_n=force, threshold_n=threshold)


def object_tipped_over(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "object_tipped_over"
    min_cosine = float(args.get("min_upright_cosine", args.get("min_cosine", 0.70)))
    _, cosine = _scalar_any(snapshot, ["object_upright_cosine", "object_up_dot", "task_state.object_upright_cosine"], name)
    return _ok(name, cosine < min_cosine, round(cosine, 6), upright_cosine=cosine, min_cosine=min_cosine)


def released(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "released"
    # Franka finger joints increase as the gripper opens (about 0.04 when
    # fully open).  A cup held between the fingers commonly leaves each joint
    # near 0.027, which must not be mistaken for a release.
    threshold = float(args.get("open_threshold", 0.035))
    gripper = _array(snapshot, str(args.get("gripper_key", "gripper_pos")), name)
    mean_pos = float(np.mean(gripper))
    return _ok(name, mean_pos >= threshold, round(mean_pos, 6), open_threshold=threshold)


def tcp_far(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "tcp_far"
    threshold = float(args.get("distance_threshold_m", args.get("threshold_m", 0.08)))
    obj = _array(snapshot, str(args.get("object_key", "object_pos")), name)
    hand = _array(snapshot, str(args.get("hand_key", "hand_pos")), name)
    distance = float(np.linalg.norm(obj[:3] - hand[:3]))
    return _ok(name, distance > threshold, round(distance, 6), distance_m=distance, threshold_m=threshold)


def inserted(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "inserted"
    min_depth = float(args.get("min_depth_m", args.get("depth_threshold_m", 0.02)))
    _, depth = _scalar_any(snapshot, ["insert_depth_m", "task_state.insert_depth_m"], name)
    return _ok(name, depth >= min_depth, round(depth, 6), insert_depth_m=depth, min_depth_m=min_depth)


def aligned(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "aligned"
    min_cosine = float(args.get("min_cosine", 0.95))
    _, cosine = _scalar_any(snapshot, ["alignment_cosine", "axis_dot", "task_state.alignment_cosine"], name)
    return _ok(name, cosine >= min_cosine, round(cosine, 6), alignment_cosine=cosine, min_cosine=min_cosine)


def insert_depth_reached(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    value = inserted(snapshot, args)
    value.name = "insert_depth_reached"
    return value


def volume_transferred(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "volume_transferred"
    target = float(args.get("target_ml", args.get("min_volume_ml", 0.0)))
    tolerance = float(args.get("tolerance_ml", 0.5))
    _, volume = _scalar_any(snapshot, ["volume_transferred_ml", "task_state.volume_transferred_ml"], name)
    if "target_ml" in args:
        passed = abs(volume - target) <= tolerance
    else:
        passed = volume >= target
    return _ok(
        name,
        passed,
        round(volume, 6),
        volume_ml=volume,
        target_ml=target,
        tolerance_ml=tolerance,
        exact_target="target_ml" in args,
    )


def spill_below(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "spill_below"
    max_spill = float(args.get("max_spill_ml", args.get("threshold_ml", 0.1)))
    _, spill = _scalar_any(snapshot, ["spill_ml", "task_state.spill_ml"], name)
    return _ok(name, spill <= max_spill, round(spill, 6), spill_ml=spill, max_spill_ml=max_spill)


def target_container_correct(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "target_container_correct"
    _, value = _bool_any(snapshot, ["target_container_correct", "task_state.target_container_correct"], name)
    return _ok(name, value, value)


def container_upright(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "container_upright"
    min_cosine = float(args.get("min_cosine", 0.90))
    _, cosine = _scalar_any(snapshot, ["container_upright_cosine", "object_up_dot", "task_state.container_upright_cosine"], name)
    return _ok(name, cosine >= min_cosine, round(cosine, 6), upright_cosine=cosine, min_cosine=min_cosine)


def pipette_aspirated(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "pipette_aspirated"
    target = float(args.get("target_ml", args.get("min_volume_ml", 0.0)))
    tolerance = float(args.get("tolerance_ml", 0.05))
    _, volume = _scalar_any(snapshot, ["pipette_aspirated_ml", "task_state.pipette_aspirated_ml"], name)
    passed = abs(volume - target) <= tolerance if "target_ml" in args else volume >= target
    return _ok(
        name,
        passed,
        round(volume, 6),
        volume_ml=volume,
        target_ml=target,
        tolerance_ml=tolerance,
        exact_target="target_ml" in args,
    )


def pipette_dispensed(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "pipette_dispensed"
    target = float(args.get("target_ml", args.get("min_volume_ml", 0.0)))
    tolerance = float(args.get("tolerance_ml", 0.05))
    _, volume = _scalar_any(snapshot, ["pipette_dispensed_ml", "task_state.pipette_dispensed_ml"], name)
    passed = abs(volume - target) <= tolerance if "target_ml" in args else volume >= target
    return _ok(
        name,
        passed,
        round(volume, 6),
        volume_ml=volume,
        target_ml=target,
        tolerance_ml=tolerance,
        exact_target="target_ml" in args,
    )


def tip_in_liquid(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "tip_in_liquid"
    if snapshot.get("tip_in_liquid", None) is not None or snapshot.get("task_state.tip_in_liquid", None) is not None:
        _, value = _bool_any(snapshot, ["tip_in_liquid", "task_state.tip_in_liquid"], name)
        return _ok(name, value, value)
    min_depth = float(args.get("min_tip_depth_m", 0.001))
    _, depth = _scalar_any(snapshot, ["tip_depth_m", "task_state.tip_depth_m"], name)
    return _ok(name, depth >= min_depth, round(depth, 6), tip_depth_m=depth, min_tip_depth_m=min_depth)


def no_cross_contamination(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "no_cross_contamination"
    if snapshot.get("cross_contamination", None) is not None or snapshot.get("task_state.cross_contamination", None) is not None:
        _, contaminated = _bool_any(snapshot, ["cross_contamination", "task_state.cross_contamination"], name)
        return _ok(name, not contaminated, not contaminated)
    _, count = _scalar_any(snapshot, ["contamination_events", "task_state.contamination_events"], name)
    return _ok(name, count <= 0, int(count), contamination_events=int(count))


def button_pressed(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "button_pressed"
    _, value = _bool_any(snapshot, ["button_pressed", "task_state.button_pressed"], name)
    criteria = snapshot.get("task_state.predicate_diagnostics.button_pressed", None)
    if isinstance(criteria, dict):
        criteria = criteria.get("criteria")
    return _ok(name, value, value, criteria=criteria)


def knob_turned(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "knob_turned"
    target = float(args.get("target_rad", args.get("min_delta_rad", 0.0)))
    tolerance = float(args.get("tolerance_rad", 0.05))
    _, angle = _scalar_any(snapshot, ["knob_angle_rad", "task_state.knob_angle_rad"], name)
    passed = abs(angle - target) <= tolerance if "target_rad" in args else angle >= target
    return _ok(
        name,
        passed,
        round(angle, 6),
        angle_rad=angle,
        target_rad=target,
        tolerance_rad=tolerance,
        exact_target="target_rad" in args,
    )


def joint_in_range(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "joint_in_range"
    low = float(args.get("min", args.get("lower", -np.inf)))
    high = float(args.get("max", args.get("upper", np.inf)))
    key = str(args.get("joint_key", "task_state.joint_value"))
    _, value = _scalar_any(snapshot, [key], name)
    return _ok(name, low <= value <= high, round(value, 6), min=low, max=high)


def device_state(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "device_state"
    expected = args.get("expected", args.get("state"))
    try:
        _, value = snapshot.first_available(["device_state", "task_state.device_state"])
    except MissingSnapshotValue as exc:
        raise _missing(name, exc) from exc
    return _ok(name, value == expected, value, expected=expected)


def task_state_true(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = str(args.get("id") or "task_state_true")
    key = str(args.get("key") or "task_state.value")
    try:
        _, value = snapshot.first_available([key])
    except MissingSnapshotValue as exc:
        raise _missing(name, exc) from exc
    diagnostic_key = str(args.get("id") or key.rsplit(".", 1)[-1])
    criteria = snapshot.get(f"task_state.predicate_diagnostics.{diagnostic_key}", None)
    if isinstance(criteria, dict):
        criteria = criteria.get("criteria")
    return _ok(name, bool(value), bool(value), key=key, criteria=criteria)


def forbidden_collision(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "forbidden_collision"
    try:
        _, value = _bool_any(snapshot, ["forbidden_collision", "task_state.forbidden_collision"], name)
    except MissingPredicateState:
        collisions = snapshot.get("collisions", [])
        forbidden_pairs = set(tuple(pair) for pair in args.get("pairs", []))
        if not collisions or not forbidden_pairs:
            return _ok(name, False, False, checked_pairs=len(forbidden_pairs))
        observed = {tuple(pair) for pair in collisions}
        value = bool(observed.intersection(forbidden_pairs))
    return _ok(name, value, value)


def wrong_object_moved(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "wrong_object_moved"
    threshold = float(args.get("distance_threshold_m", 0.02))
    moved = snapshot.get("wrong_object_moved", None)
    if moved is not None:
        return _ok(name, bool(moved), bool(moved))
    motions = snapshot.get("distractor_motion_m", None)
    if motions is None:
        raise MissingPredicateState("Predicate 'wrong_object_moved' requires wrong_object_moved or distractor_motion_m")
    max_motion = max((float(v) for v in motions.values()), default=0.0) if isinstance(motions, dict) else float(motions)
    return _ok(name, max_motion > threshold, round(max_motion, 6), max_motion_m=max_motion, threshold_m=threshold)


def timeout(snapshot: EvaluationSnapshot, args: dict[str, Any]) -> PredicateValue:
    name = "timeout"
    step = snapshot.get("step_index", None)
    max_steps = args.get("max_steps")
    if step is None or max_steps is None:
        return _ok(name, False, False)
    return _ok(name, int(step) >= int(max_steps), int(step), max_steps=int(max_steps))


def default_predicate_registry() -> PredicateRegistry:
    registry = PredicateRegistry()
    for name, fn in {
        "gripper_closed": gripper_closed,
        "object_near_hand": object_near_hand,
        "near": near,
        "grasped": grasped,
        "lifted": lifted,
        "object_lifted": lifted,
        "object_velocity_below": object_velocity_below,
        "stable": object_velocity_below,
        "stable_pose": object_velocity_below,
        "object_out_of_workspace": object_out_of_workspace,
        "object_dropped": object_dropped,
        "wrong_target_reached": wrong_target_reached,
        "wrong_object_selected": wrong_object_selected,
        "contact_force_exceeded": contact_force_exceeded,
        "object_tipped_over": object_tipped_over,
        "released": released,
        "tcp_far": tcp_far,
        "inserted": inserted,
        "aligned": aligned,
        "insert_depth_reached": insert_depth_reached,
        "volume_transferred": volume_transferred,
        "spill_below": spill_below,
        "target_container_correct": target_container_correct,
        "container_upright": container_upright,
        "pipette_aspirated": pipette_aspirated,
        "pipette_dispensed": pipette_dispensed,
        "tip_in_liquid": tip_in_liquid,
        "no_cross_contamination": no_cross_contamination,
        "button_pressed": button_pressed,
        "knob_turned": knob_turned,
        "joint_in_range": joint_in_range,
        "device_state": device_state,
        "task_state_true": task_state_true,
        "forbidden_collision": forbidden_collision,
        "wrong_object_moved": wrong_object_moved,
        "timeout": timeout,
    }.items():
        registry.register(name, fn)
    return registry
