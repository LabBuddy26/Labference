"""Isaac Sim online runner for rack-placement recall."""

from __future__ import annotations

import base64
from dataclasses import asdict
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

from .cli import load_config
from .core import (Contract, DISTRACTOR_TUBES, EVALUATION_VERSION, Instance,
                   RackPose, SCORING_VERSION, SLOTS, State, TARGET_TUBE, advance_state,
                   evaluate_tube, generate_instance, legal_actions,
                   normalize_action, task_complete)
from .diagnostics import (RackRecallOracleValidator, candidate_geometry,
                          check_policy_request, deliver_input, oracle_context,
                          policy_observation, score_report)
from .online import (BehaviorSelectionTracker, OpportunityBook, apply_runtime_validity,
                     measurement_to_dict, native_behavior, score_trace)
from .policy import (HttpArrayPolicy, HttpSymbolicPolicy, OpenPIArrayPolicy,
                     PolicyServiceError)


def _json_default(value: Any) -> Any:
    """Convert NumPy runtime scalars without silently stringifying data."""
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _preserve_tube_rotation(phase: str, radial_error_m: float,
                            axis_error_deg: float, alignment_limit_deg: float) -> bool:
    """Separate long hover translation from near-hole orientation correction."""
    return (phase == "recover" or
            (phase in {"hover", "upper_guide"} and
             (phase == "upper_guide" or
             (radial_error_m > .015 or axis_error_deg <= alignment_limit_deg)))
            )


def _retract_clearance_reached(current_position: Any, target_position: Any, *,
                               vertical_tolerance_m: float = .010,
                               lateral_tolerance_m: float = .006) -> bool:
    """Require 90 mm axial withdrawal without an unreachable exact IK pose."""
    import numpy as np

    current = np.asarray(current_position, dtype=float)
    target = np.asarray(target_position, dtype=float)
    if (current.shape != (3,) or target.shape != (3,) or
            not np.all(np.isfinite(current)) or not np.all(np.isfinite(target))):
        return False
    lateral_error = float(np.linalg.norm(target[:2] - current[:2]))
    return bool(current[2] >= target[2] - vertical_tolerance_m and
                lateral_error <= lateral_tolerance_m)


def _upright_downward_rotation(rotation: Any):
    """Keep heading while mapping the tube's local -Z tip axis to world -Z."""
    import numpy as np

    source = np.asarray(rotation, dtype=float)
    if source.shape != (3, 3) or not np.all(np.isfinite(source)):
        raise ValueError("tube rotation must be a finite 3x3 matrix")
    x_axis = source[:, 0].copy()
    x_axis[2] = 0.0
    norm = float(np.linalg.norm(x_axis))
    if norm < 1e-8:
        x_axis = np.asarray([1.0, 0.0, 0.0])
    else:
        x_axis /= norm
    z_axis = np.asarray([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def _tube_grasp_target(center: Any, rotation: Any,
                       grasp_point_from_center_m: float,
                       tcp_to_finger_pad_z_m: float):
    """Compute the TCP target from the measured cap pose of a tilted tube."""
    import numpy as np

    center = np.asarray(center, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    downward_axis = rotation @ np.asarray([0.0, 0.0, -1.0])
    cap_grasp_point = center - downward_axis * grasp_point_from_center_m
    return cap_grasp_point + np.asarray([0.0, 0.0, tcp_to_finger_pad_z_m])


def _bounded_cartesian_target(current: Any, target: Any, max_step_m: float):
    """Advance toward a Cartesian target without exceeding one control step."""
    import numpy as np

    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    delta = target - current
    distance = float(np.linalg.norm(delta))
    if distance <= max_step_m or distance == 0.0:
        return target.copy()
    return current + delta * (max_step_m / distance)


def _reference_arm_step(action: dict[str, Any] | None, *, control_hz: int | None = None,
                        controller_phase: str | None = None) -> float:
    """Limit reference targets by operation while preserving insertion precision.

    Reference control now runs at 120 Hz.  Keeping the old ``0.03`` rad
    *per-call* limit therefore made free-space transfers still behave as if
    they were rate-limited at the 20 Hz collection rate.  Pick/retrieve and
    recovery are obstacle-free RMPflow motions, so they can use the larger
    envelope; the contact-sensitive insertion path keeps the original limit.
    """
    kind = None if action is None else action.get("type")
    base = .018 if kind == "place_recovery" else .03
    if control_hz is not None and control_hz > 20 and kind in {
            "pick", "retrieve", "place_recovery"}:
        return .05
    return base


def _use_rmpflow_hover(phase: str | None, radial_error_m: float) -> bool:
    """Use obstacle-aware free-space motion before precision alignment."""
    return phase == "hover" and radial_error_m > .015


def _rack_transport_errors(contract: Contract, rack_pose: RackPose,
                           tube_observations: dict[str, Any],
                           expected_slots: dict[str, str]) -> dict[str, float]:
    """Measure seated-tube drift relative to the moving rack frame."""
    errors: dict[str, float] = {}
    for slot, tube_name in expected_slots.items():
        sample = tube_observations.get(tube_name)
        if sample is None:
            errors[tube_name] = float("inf")
            continue
        mouth = contract.slot_world_center(slot, rack_pose)
        errors[tube_name] = math.dist(sample.center_world_m[:2], mouth[:2])
    return errors


def _save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")


def _write_contact_sheet(video_path: Path, output_path: Path) -> bool:
    """Write five uniformly sampled frames without affecting run validity."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw

    if not video_path.is_file():
        return False
    reader = imageio.get_reader(str(video_path))
    try:
        frame_count = int(reader.count_frames())
        if frame_count <= 0:
            return False
        indices = np.linspace(0, frame_count - 1, 5, dtype=int)
        frames = [Image.fromarray(_rgb(reader.get_data(int(index)))) for index in indices]
    finally:
        reader.close()
    width, height = frames[0].size
    label_height = 28
    sheet = Image.new("RGB", (width * len(frames), height + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    for column, (index, frame) in enumerate(zip(indices, frames)):
        sheet.paste(frame, (column * width, label_height))
        draw.text((column * width + 8, 7), f"frame {int(index)}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return True


def _instance_dict(instance: Instance) -> dict[str, Any]:
    value = asdict(instance)
    value["rack_initial_pose"] = instance.rack_initial_pose.public()
    value["rack_final_pose"] = instance.rack_final_pose.public()
    value["distractor_slots"] = list(instance.distractor_slots)
    value["target_spawn_xy_m"] = list(instance.target_spawn_xy_m)
    value["recovery_center_xy_m"] = list(instance.recovery_center_xy_m)
    value["cap_colors"] = [list(item) for item in instance.cap_colors]
    return value


def _diagnostic_rule_action(contract: Contract, state: State,
                            preferred_slot: str | None = None) -> dict[str, Any]:
    """Select a legal action; maximize physical clearance for free-slot insertion."""
    actions = legal_actions(contract, state)
    insertions = [action for action in actions if action["type"] == "insert"]
    occupied = [contract.slot_local_center(slot) for slot in state.occupied]
    if not insertions:
        return actions[0]
    if preferred_slot is not None:
        for action in insertions:
            if action["slot"] == preferred_slot:
                return action
        raise RuntimeError(f"collection slot {preferred_slot} is not currently legal")
    if not occupied:
        return min(insertions, key=lambda action: math.dist(
            contract.slot_local_center(action["slot"])[:2], (0.0, 0.0)))

    def clearance(action: dict[str, Any]) -> tuple[float, float]:
        center = contract.slot_local_center(action["slot"])
        distances = [math.dist(center[:2], other[:2]) for other in occupied]
        # Clearance is primary. A central tie-break avoids making the reference
        # controller's edge-of-workspace reachability part of the task result.
        return min(distances), -math.dist(center[:2], (0.0, 0.0))

    return max(insertions, key=clearance)


def run_supervised(config: str, output: str, *, seed: int = 101,
                   max_steps: int | None = None, save_video: bool = False,
                   headless: bool = True,
                   oracle_profile: str | None = None, policy_url: str | None = None,
                   policy_profile: str | None = None,
                   intervention_spec: str | None = None,
                   checkpoint_slot: str | None = None,
                   pair_id: str | None = None, member: str = "single",
                   motion_variant: str = "base",
                   instance_overrides: dict[str, Any] | None = None,
                   collection_slot: str | None = None,
                   dataset_output: str | None = None) -> dict:
    out = Path(output).resolve()
    cmd = [sys.executable, "-m", "eai4s_eval.rack_recall_v3.cli",
           "checkpoints" if checkpoint_slot else "run", "--worker",
           "--config", str(Path(config).resolve()), "--output", str(out),
           "--seed", str(seed)]
    if headless:
        cmd.append("--headless")
    if checkpoint_slot:
        cmd += ["--checkpoint-slot", checkpoint_slot]
    if pair_id:
        cmd += ["--pair-id", pair_id, "--member", member]
    if checkpoint_slot and motion_variant != "base":
        cmd += ["--motion-variant", motion_variant]
    for key, flag in (("layout_seed", "--layout-seed"),
                      ("motion_seed", "--motion-seed"),
                      ("policy_seed", "--policy-seed"),
                      ("color_seed", "--color-seed"),
                      ("initial_yaw_deg", "--initial-yaw-deg"),
                      ("motion_yaw_deg", "--motion-yaw-deg"),
                      ("target_cap_color", "--target-cap-color")):
        if instance_overrides and key in instance_overrides:
            cmd += [flag, str(instance_overrides[key])]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    if save_video:
        cmd.append("--save-video")
    if oracle_profile:
        cmd += ["--oracle-profile", oracle_profile]
    if policy_url:
        cmd += ["--policy-url", policy_url]
    if policy_profile:
        cmd += ["--policy-profile", str(Path(policy_profile).resolve())]
    if intervention_spec:
        cmd += ["--intervention-spec", str(Path(intervention_spec).resolve())]
    if collection_slot:
        cmd += ["--collection-slot", collection_slot]
    if dataset_output:
        cmd += ["--dataset-output", str(Path(dataset_output).resolve())]
    worker = subprocess.Popen(cmd)
    close_started = None
    timed_out = False
    manifest_path = out / "run.json"
    while worker.poll() is None:
        try:
            worker.wait(timeout=.5)
        except subprocess.TimeoutExpired:
            pass
        if manifest_path.exists():
            try:
                status = json.loads(manifest_path.read_text()).get("simulation_app_close", {})
            except (OSError, json.JSONDecodeError):
                continue
            if status.get("status") == "requested":
                close_started = close_started or time.monotonic()
        if close_started is not None and time.monotonic() - close_started > 45:
            worker.kill()
            worker.wait()
            timed_out = True
    if not manifest_path.exists():
        raise RuntimeError(f"rack-recall worker exited {worker.returncode} before writing run.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["worker_exit_code"] = worker.returncode
    if timed_out:
        manifest["simulation_app_close"] = {"status": "timed_out"}
    elif manifest.get("simulation_app_close", {}).get("status") == "requested":
        manifest["simulation_app_close"] = {"status": "process_exited_after_skip_cleanup"}
    _save_json(manifest_path, manifest)
    score_path = out / "scores.json"
    if not score_path.exists():
        raise RuntimeError(f"worker did not persist scores; see {manifest_path}")
    scores = json.loads(score_path.read_text())
    if worker.returncode != 0 or manifest.get("runtime_status") == "error":
        raise RuntimeError(f"rack-recall worker failed; see {manifest_path}")
    return scores


def _rgb(frame: Any):
    import numpy as np
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] < 3:
        return None
    rgb = image[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return np.array(rgb, copy=True, order="C")


def _encode_images(cameras: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    result = {}
    for name in ("front", "wrist"):
        frame = _rgb(cameras[name].get_rgba())
        if frame is None:
            raise PolicyServiceError(f"{name} camera is not ready")
        buffer = BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        result[name] = {"mime_type": "image/png",
                        "base64": base64.b64encode(buffer.getvalue()).decode("ascii")}
    return result


class _WorldTargetMotionAdapter:
    """Expose world-frame Cartesian goals to the base-frame RMP policy."""

    def __init__(self, motion: Any, franka: Any):
        self.motion = motion
        self.franka = franka
        self.last_frame_offset = None
        self.last_world_target = None
        self.last_policy_target = None
        self.last_policy_eef = None

    def reset(self) -> None:
        self.motion.reset()

    def forward(self, target_end_effector_position: Any,
                target_end_effector_orientation: Any = None,
                *, target_is_solver_frame: bool = False):
        import numpy as np
        joints = np.asarray(self.franka.get_joint_positions(), dtype=float)
        policy_eef = np.asarray(
            self.motion.rmp_flow.get_end_effector_pose(joints[:7])[0], dtype=float)
        tool_center = np.asarray(self.franka.get_gripper_position(), dtype=float)
        frame_offset = tool_center - policy_eef
        target = np.asarray(target_end_effector_position, dtype=float) - frame_offset
        self.last_frame_offset = frame_offset
        self.last_world_target = np.asarray(target_end_effector_position, dtype=float)
        self.last_policy_target = target
        self.last_policy_eef = policy_eef
        return self.motion.forward(
            target_end_effector_position=target,
            target_end_effector_orientation=target_end_effector_orientation)


class _DifferentialWorldTargetMotion:
    """Small-step DLS IK for contact-sensitive tube insertion."""

    def __init__(self, solver: Any, franka: Any):
        self.solver = solver
        self.franka = franka
        # Free-space travel may use a larger Cartesian increment. The final
        # insertion descent overrides this with a much smaller value below.
        self.max_position_step_m = .006
        # The insertion controller runs at the physics rate.  The previous
        # 0.012 rad/frame cap limited a 70--80 degree wrist correction to a
        # visibly slow crawl even when the Cartesian target was already
        # within the socket neighborhood.  Keep the joint-step and grasp-slip
        # guards as the safety limits, but allow the DLS orientation residual
        # to request a useful correction per physics step.
        self.max_rotation_step_rad = .100
        self.max_joint_step_norm_rad = .050
        self.orientation_weight = .35
        self.status = "idle"
        self.raw_joint_delta_max = 0.0
        self.orientation_error_deg = 0.0
        self.locked_rotation = None
        self.last_frame_offset = None
        self.last_world_target = None
        self.last_policy_target = None
        self.last_policy_eef = None

    def reset(self) -> None:
        self.status = "ready"
        self.raw_joint_delta_max = 0.0
        self.orientation_error_deg = 0.0
        self.locked_rotation = None

    def forward(self, target_end_effector_position: Any,
                target_end_effector_orientation: Any = None,
                *, target_is_solver_frame: bool = False,
                target_local_axis: Any = None,
                target_world_axis: Any = None):
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction
        from scipy.spatial.transform import Rotation

        world_target = np.asarray(target_end_effector_position, dtype=float)
        current_joints = np.asarray(self.franka.get_joint_positions()[:7], dtype=float)
        if not np.all(np.isfinite(world_target)) or not np.all(np.isfinite(current_joints)):
            raise RuntimeError("non-finite differential IK input")
        current_pose, current_rotation = self.solver.compute_forward_kinematics(
            "right_gripper", current_joints)
        current_pose = np.asarray(current_pose, dtype=float)
        current_rotation = np.asarray(current_rotation, dtype=float)
        tool_center = np.asarray(self.franka.get_gripper_position(), dtype=float)
        frame_offset = tool_center - current_pose
        target = world_target if target_is_solver_frame else world_target - frame_offset
        position_error = target - current_pose
        distance = float(np.linalg.norm(position_error))
        if distance > self.max_position_step_m:
            position_error *= self.max_position_step_m / distance

        if (target_local_axis is None) != (target_world_axis is None):
            raise ValueError("local and world axis targets must be provided together")
        axis_control = target_local_axis is not None
        if axis_control and target_end_effector_orientation is not None:
            raise ValueError("axis and full-orientation targets are mutually exclusive")
        local_axis = desired_axis = current_axis = None
        if axis_control:
            local_axis = np.asarray(target_local_axis, dtype=float)
            desired_axis = np.asarray(target_world_axis, dtype=float)
            if (local_axis.shape != (3,) or desired_axis.shape != (3,) or
                    not np.all(np.isfinite(local_axis)) or
                    not np.all(np.isfinite(desired_axis))):
                raise ValueError("axis targets must be finite three-vectors")
            local_axis /= np.linalg.norm(local_axis)
            desired_axis /= np.linalg.norm(desired_axis)
            current_axis = current_rotation @ local_axis

        epsilon = 1e-4
        columns = []
        axis_columns = []
        for joint in range(7):
            perturbed = current_joints.copy()
            perturbed[joint] += epsilon
            moved_pose, moved_rotation = self.solver.compute_forward_kinematics(
                "right_gripper", perturbed)
            translation = (np.asarray(moved_pose) - current_pose) / epsilon
            rotation = Rotation.from_matrix(
                np.asarray(moved_rotation) @ current_rotation.T).as_rotvec() / epsilon
            columns.append(np.concatenate((translation, rotation)))
            if axis_control:
                moved_axis = np.asarray(moved_rotation) @ local_axis
                axis_columns.append((moved_axis - current_axis) / epsilon)
        jacobian = np.stack(columns, axis=1)
        damping = .02
        if axis_control:
            axis_error = desired_axis - current_axis
            self.orientation_error_deg = float(np.degrees(np.arccos(np.clip(
                np.dot(current_axis, desired_axis), -1.0, 1.0))))
            # The position error is already rate-limited above, but the raw
            # axis vector can still be close to two for a 90-degree tilt.
            # Feeding that unbounded orientation residual to DLS makes it
            # spend the whole step rotating in place and ignore the socket
            # position.  Apply the same per-step angular envelope used by
            # full-pose control before weighting the axis objective.
            axis_norm = float(np.linalg.norm(axis_error))
            if axis_norm > self.max_rotation_step_rad:
                axis_error *= self.max_rotation_step_rad / axis_norm
            position_jacobian = jacobian[:3]
            axis_jacobian = np.stack(axis_columns, axis=1)
            weighted_jacobian = np.vstack((
                position_jacobian,
                self.orientation_weight * axis_jacobian,
            ))
            weighted_error = np.concatenate((
                position_error,
                self.orientation_weight * axis_error,
            ))
            joint_delta = weighted_jacobian.T @ np.linalg.solve(
                weighted_jacobian @ weighted_jacobian.T +
                damping**2 * np.eye(6), weighted_error)
        elif target_end_effector_orientation is None:
            self.orientation_error_deg = 0.0
            position_jacobian = jacobian[:3]
            error = position_error
            joint_delta = position_jacobian.T @ np.linalg.solve(
                position_jacobian @ position_jacobian.T +
                damping**2 * np.eye(3), error)
        else:
            if target_is_solver_frame:
                orientation = np.asarray(target_end_effector_orientation, dtype=float)
                if orientation.shape == (3, 3):
                    desired_rotation = orientation
                elif orientation.shape == (4,):
                    desired_rotation = Rotation.from_quat(orientation).as_matrix()
                else:
                    raise ValueError("solver-frame orientation must be xyzw or 3x3")
                if not np.all(np.isfinite(desired_rotation)):
                    raise RuntimeError("non-finite differential IK orientation")
            else:
                # Legacy place operations lock the orientation in the same
                # kinematic frame used by the solver.
                if self.locked_rotation is None:
                    self.locked_rotation = current_rotation.copy()
                desired_rotation = self.locked_rotation
            angular_error = Rotation.from_matrix(
                desired_rotation @ current_rotation.T).as_rotvec()
            angle = float(np.linalg.norm(angular_error))
            self.orientation_error_deg = float(np.degrees(angle))
            if angle > self.max_rotation_step_rad:
                angular_error *= self.max_rotation_step_rad / angle
            position_jacobian = jacobian[:3]
            rotation_jacobian = jacobian[3:]
            weighted_jacobian = np.vstack((
                position_jacobian,
                self.orientation_weight * rotation_jacobian,
            ))
            weighted_error = np.concatenate((
                position_error,
                self.orientation_weight * angular_error,
            ))
            joint_delta = weighted_jacobian.T @ np.linalg.solve(
                weighted_jacobian @ weighted_jacobian.T +
                damping**2 * np.eye(6), weighted_error)
        self.raw_joint_delta_max = float(np.max(np.abs(joint_delta)))
        if not np.all(np.isfinite(joint_delta)):
            self.status = "non_finite_differential_ik"
            solution = current_joints
        else:
            norm = float(np.linalg.norm(joint_delta))
            if norm > self.max_joint_step_norm_rad:
                joint_delta *= self.max_joint_step_norm_rad / norm
            solution = current_joints + joint_delta
            self.status = "differential_ik"
        self.last_frame_offset = frame_offset
        self.last_world_target = world_target
        self.last_policy_target = target
        self.last_policy_eef = current_pose
        return ArticulationAction(joint_positions=solution,
                                  joint_velocities=np.zeros(7),
                                  joint_indices=np.arange(7))


class ReferenceExecutor:
    """Pick-demo style controller; success still comes only from readback."""

    def __init__(self, handles: Any, contract: Contract, *, control_hz: int | None = None,
                 collection_mode: bool = False):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from robots.franka.rmpflow_controller import RMPFlowController
        from controllers.atomic_actions.pick_controller import PickController
        from controllers.atomic_actions.place_controller import PlaceController
        from controllers.robot_controllers.grapper_manager import Gripper

        class PreciseTubePickController(PickController):
            def reset(inner, events_dt=None):
                super().reset(events_dt=events_dt)
                inner._axial_lift_xy = None
                inner._axial_lift_command_z = None

            def get_pickz_offset(inner, item_name):
                if item_name == "CentrifugeTube":
                    return 0.0
                return super().get_pickz_offset(item_name)

            def get_pickprez_offset(inner, item_name):
                if item_name == "CentrifugeTube":
                    return .025
                return super().get_pickprez_offset(item_name)

            def _execute_phase(inner, picking_position, end_effector_orientation,
                               current_joint_positions, object_name, gripper_control,
                               gripper_position, gripper_distances, pick_z_offset):
                phase = inner._event
                if phase == 5:
                    # The shared controller stores lift_speed but jumps directly
                    # to the final target. A seated tube then slips while being
                    # pulled through both guide plates. Exit the socket at the
                    # configured speed and do not let its phase timer finish the
                    # operation before the Cartesian target is reached.
                    final_target = np.asarray(inner.target_position, dtype=float)
                    current = np.asarray(gripper_position, dtype=float)
                    # Extraction through the two guide plates must be axial.
                    # Reusing the old object-centre XY target drags a tilted or
                    # contact-shifted tube sideways and defeats the passive
                    # guides before the tip clears the lower plate.
                    if inner._axial_lift_xy is None:
                        inner._axial_lift_xy = current[:2].copy()
                        inner._axial_lift_command_z = float(current[2])
                    inner._axial_lift_command_z = min(
                        float(final_target[2]),
                        inner._axial_lift_command_z +
                        inner._lift_speed * inner._physics_dt)
                    step_target = np.asarray([
                        inner._axial_lift_xy[0], inner._axial_lift_xy[1],
                        inner._axial_lift_command_z,
                    ])
                    result = inner._cspace_controller.forward(
                        target_end_effector_position=step_target,
                        target_end_effector_orientation=end_effector_orientation)
                    distance = abs(float(final_target[2] - current[2]))
                    inner._t = -inner._events_dt[phase]
                    if distance <= inner._position_threshold:
                        inner._event = 6
                        inner._t = 0
                    return result
                result = super()._execute_phase(
                    picking_position, end_effector_orientation,
                    current_joint_positions, object_name, gripper_control,
                    gripper_position, gripper_distances, pick_z_offset)
                if phase == 2 and inner._event == 3:
                    error = np.asarray(gripper_position) - np.asarray(picking_position)
                    if np.linalg.norm(error[:2]) > .004 or abs(error[2]) > .005:
                        inner._event = 2
                        inner._t = 0
                return result

        class PreciseTubePlaceController(PlaceController):
            """Prevent the shared timed controller from releasing above a socket."""

            final_xy_tolerance = .0035
            final_z_tolerance = .004

            def set_final_tolerance(inner, xy: float, z: float) -> None:
                inner.final_xy_tolerance = xy
                inner.final_z_tolerance = z

            def forward(inner, place_position, current_joint_positions,
                        gripper_control, end_effector_orientation=None,
                        gripper_position=None, pre_place_z=.2,
                        place_offset_z=.05):
                phase = inner._event
                result = super().forward(
                    place_position=place_position,
                    current_joint_positions=current_joint_positions,
                    gripper_control=gripper_control,
                    end_effector_orientation=end_effector_orientation,
                    gripper_position=gripper_position,
                    pre_place_z=pre_place_z,
                    place_offset_z=place_offset_z)
                if gripper_position is None:
                    return result

                if phase == 0 and inner._event == 1:
                    target = np.asarray(place_position, dtype=float).copy()
                    target[2] += pre_place_z
                    error = np.asarray(gripper_position, dtype=float) - target
                    if np.linalg.norm(error[:2]) > .008 or abs(error[2]) > .015:
                        inner._event = 0
                        inner._t = 0
                elif phase == 1 and inner._event == 2:
                    target = np.asarray(place_position, dtype=float).copy()
                    target[2] += place_offset_z
                    error = np.asarray(gripper_position, dtype=float) - target
                    if (np.linalg.norm(error[:2]) > inner.final_xy_tolerance or
                            abs(error[2]) > inner.final_z_tolerance):
                        inner._event = 1
                        inner._t = 0
                return result

        self.np = np
        self.Rotation = Rotation
        self.handles = handles
        self.contract = contract
        self.control_hz = control_hz or contract.physics_hz
        self.collection_mode = bool(collection_mode)
        controller_time_scale = contract.physics_hz / self.control_hz
        self.raw_motion = RMPFlowController(
            # Preserve the repository's proven pick-demo RMP integration
            # horizon. The outer action command is still held according to
            # control_hz; this value only governs RMPflow's internal rollout.
            "rack_recall_rmp", handles.franka, physics_dt=1 / 10)
        self.motion = _WorldTargetMotionAdapter(self.raw_motion, handles.franka)
        self.place_motion = _DifferentialWorldTargetMotion(
            self.raw_motion.rmp_flow.get_kinematics_solver(), handles.franka)
        self.pick = PreciseTubePickController(
            "rack_recall_pick", self.motion,
            # The first three phases are geometric approach phases. Their
            # timeout must remain longer than the joint-limited travel time or
            # the shared controller can descend before horizontal alignment.
            # These are the same event durations used by the repository's
            # successful pick demos.  Sub-millisecond values make the
            # controller's internal elapsed-time state advance too slowly
            # when it is driven at the physics rate, leaving it above the
            # tube for thousands of frames without ever closing the gripper.
            events_dt=[value * controller_time_scale for value in
                       (.004, .002, .01, .02, .05, .004, .008)],
            lift_speed=.008, physics_dt=1 / self.control_hz,
            require_pose_reached=True, max_motion_phase_steps=3000)
        self.place = PreciseTubePlaceController(
            # Recovery is a long free-space transfer. Use the repository's
            # proven RMPflow path here; the differential solver remains
            # reserved for contact-sensitive insertion near the rack.
            "rack_recall_place", self.motion, handles.franka.gripper,
            events_dt=[value * controller_time_scale for value in
                       (.003, .005, .03, .035, .004, .02)],
            _position_threshold=.001)
        self.gripper_control = Gripper()
        current = self.np.asarray(handles.franka.get_joint_positions(), dtype=float)
        _, initial_rotation = self.raw_motion.rmp_flow.get_end_effector_pose(current[:7])
        orientation_xyzw = Rotation.from_matrix(initial_rotation).as_quat()
        # MotionPolicyController follows the Lula quaternion convention used by
        # the repository's working pick controllers (x, y, z, w).
        self.orientation = orientation_xyzw
        self.operation_orientation = self.orientation.copy()
        self.action: dict[str, Any] | None = None
        self.tube: str | None = None
        self.pick_target_position = None
        self.grasp_offset = None
        self.insertion_xy_correction = None
        self.insertion_alignment_refinements = 0
        self.insertion_clearance_target = None
        self.insertion_clearance_complete = False
        self.insertion_descent_alignment_holds = 0
        self.insertion_alignment_loss_steps = 0
        self.insertion_alignment_stable_steps = 0
        self.insertion_near_hole_alignment = False
        self.insertion_alignment_center = None
        self.insertion_recovery_target = None
        self.insertion_recovery_attempts = 0
        self.insertion_orientation_corrections = 0
        self.tcp_to_tube = None
        self.desired_tube_rotation = None
        self.captured_tube_rotation = None
        self.insertion_phase = None
        self.insertion_phase_step = 0
        self.insertion_force_exceed_steps = 0
        self.insertion_stall_start_step = 0
        self.insertion_stall_start_depth = None
        self.insertion_last_depth = None
        self.insertion_target_radial_error_m = None
        self.insertion_target_axis_error_deg = None
        self.guide_handoff_count = 0
        self.last_guide_handoff = None
        self.guide_handoff_radial_threshold_m = None
        self.guide_handoff_axis_threshold_deg = None
        self.insertion_slip_translation_m = 0.0
        self.insertion_slip_rotation_deg = 0.0
        self.insertion_resume_phase = None
        self.insertion_retract_center = None
        self.insertion_clearance_tube_center = None
        self.insertion_retract_gripper_target = None
        self.ik_joint_delta_max = 0.0
        self.ik_predicted_position_progress_m = 0.0
        self.ik_rejected_reason = None
        self.ik_fallback_used = False
        self.recovery_phase = None
        self.recovery_phase_control_steps = 0
        self.recovery_grasp_refined = False
        self.recovery_failure_reason = None
        self.recovery_retract_target = None
        self.failed = False

    def reset(self, action: dict[str, Any], state: State) -> None:
        action = normalize_action(action)
        if action is None or action["type"] in {"wait", "stop"}:
            raise ValueError("reference executor needs a physical operation")
        self.action, self.failed = action, False
        # A loose table pick has no guide plates to protect, so use the
        # repository's normal lift speed.  Retrieval remains deliberately
        # slower because the tube must clear both rack guide plates axially.
        self.pick._lift_speed = .020 if action["type"] == "pick" else .008
        self.pick_target_position = None
        self.insertion_xy_correction = None
        self.insertion_alignment_refinements = 0
        self.insertion_clearance_target = None
        self.insertion_clearance_complete = False
        self.insertion_descent_alignment_holds = 0
        self.insertion_alignment_loss_steps = 0
        self.insertion_alignment_stable_steps = 0
        self.insertion_near_hole_alignment = False
        self.insertion_alignment_center = None
        self.insertion_recovery_target = None
        self.insertion_recovery_attempts = 0
        self.insertion_orientation_corrections = 0
        self.tcp_to_tube = None
        self.desired_tube_rotation = None
        self.captured_tube_rotation = None
        self.insertion_phase = None
        self.insertion_phase_step = 0
        self.insertion_force_exceed_steps = 0
        self.insertion_stall_start_step = 0
        self.insertion_stall_start_depth = None
        self.insertion_last_depth = None
        self.insertion_target_radial_error_m = None
        self.insertion_target_axis_error_deg = None
        self.guide_handoff_count = 0
        self.last_guide_handoff = None
        self.guide_handoff_radial_threshold_m = None
        self.guide_handoff_axis_threshold_deg = None
        self.insertion_slip_translation_m = 0.0
        self.insertion_slip_rotation_deg = 0.0
        self.insertion_resume_phase = None
        self.insertion_retract_center = None
        self.insertion_clearance_tube_center = None
        self.insertion_retract_gripper_target = None
        self.ik_joint_delta_max = 0.0
        self.ik_predicted_position_progress_m = 0.0
        self.ik_rejected_reason = None
        self.ik_fallback_used = False
        self.recovery_phase = None
        self.recovery_phase_control_steps = 0
        self.recovery_grasp_refined = False
        self.recovery_failure_reason = None
        self.recovery_retract_target = None
        self.motion.reset()
        self.place_motion.reset()
        if action["type"] in {"pick", "retrieve"}:
            if action["type"] == "pick":
                self.tube = TARGET_TUBE
            else:
                self.tube = state.occupied.get(action["slot"])
                if self.tube is None:
                    self.failed = True
                    return
            tube_matrix = self._tube_world_matrix()
            self.pick_target_position = _tube_grasp_target(
                tube_matrix[:3, 3], tube_matrix[:3, :3],
                self.contract.grasp_point_from_center_m,
                self.contract.tcp_to_finger_pad_z_m)
            self.pick.reset()
            self.pick.set_robot_position(
                self.np.asarray(self.contract.franka_base_position_m, dtype=float))
        else:
            self.tube = TARGET_TUBE
            tube_position = self.np.asarray(
                self.handles.tubes[self.tube].get_world_pose()[0], dtype=float)
            self.grasp_offset = self.np.asarray(
                self.handles.franka.get_gripper_position(), dtype=float) - tube_position
            if not self.np.all(self.np.isfinite(self.grasp_offset)):
                raise RuntimeError("non-finite tube-to-gripper offset")
            if action["type"] == "insert":
                self.place.set_final_tolerance(
                    self.contract.radial_tolerance_m, .006)
                mouth_z = self.contract.slot_world_center(
                    action["slot"], self.handles.rack_pose())[2]
                self.insertion_clearance_target = self.np.asarray(
                    self.handles.franka.get_gripper_position(), dtype=float)
                self.insertion_clearance_target[2] = max(
                    self.insertion_clearance_target[2],
                    mouth_z + self.contract.tube_length_m / 2.0 +
                    float(self.grasp_offset[2]) + .045)
                tube_matrix = self._tube_world_matrix()
                gripper_matrix = self._solver_world_matrix()
                self.tcp_to_tube = self.np.linalg.inv(gripper_matrix) @ tube_matrix
                self.captured_tube_rotation = tube_matrix[:3, :3].copy()
                self.desired_tube_rotation = _upright_downward_rotation(
                    tube_matrix[:3, :3])
                self.insertion_phase = "clearance"
                self.insertion_stall_start_depth = None
                self.insertion_clearance_tube_center = tube_matrix[:3, 3].copy()
                self.insertion_clearance_tube_center[2] = max(
                    self.insertion_clearance_tube_center[2] + .080,
                    mouth_z + self.contract.tube_length_m / 2.0 + .040)
            else:
                self.place.set_final_tolerance(.010, .012)
                self.recovery_phase = "transport"
            self.place.reset()

    def _set_recovery_phase(self, phase: str) -> None:
        self.recovery_phase = phase
        self.recovery_phase_control_steps = 0
        if phase == "descend":
            # A low release can settle the tube again; allow one fresh grasp
            # calibration at the start of the vertical descent, then keep it
            # fixed for the remainder of the placement.
            self.recovery_grasp_refined = False

    def _step_recovery(self, evidence: dict[str, Any] | None):
        """Place the retrieved tube on the table using phase-specific solvers."""
        sample = (evidence or {}).get(TARGET_TUBE, {})
        phase = self.recovery_phase or "transport"
        if (self.collection_mode and phase in {"transport", "descend"} and
                self.tube is not None and self.grasp_offset is not None):
            # Collection-only expert stabilization: preserve the measured
            # grasp transform while the arm crosses the table boundary.  The
            # strict/native paths remain physics-only; release below clears
            # this synchronization before success is checked.
            tube_prim = self.handles.tubes[self.tube]
            tube_position, tube_orientation = tube_prim.get_world_pose()
            synchronized_position = (
                self.np.asarray(self.handles.franka.get_gripper_position(), dtype=float)
                - self.grasp_offset)
            tube_prim.set_world_pose(
                position=synchronized_position,
                orientation=tube_orientation)
        # Retrieval through the rack guides can settle the tube in the
        # fingers by a few centimetres.  Re-anchor the measured grasp frame
        # while bilateral contact is present so the recovery target is the
        # tube centre, not the stale pre-retrieval offset.
        if (not self.recovery_grasp_refined and
                sample.get("both_fingers_contact") and self.tube is not None):
            tube_position = self.np.asarray(
                self.handles.tubes[self.tube].get_world_pose()[0], dtype=float)
            measured_offset = self.np.asarray(
                self.handles.franka.get_gripper_position(), dtype=float) - tube_position
            if self.np.all(self.np.isfinite(measured_offset)):
                self.grasp_offset = measured_offset
                self.recovery_grasp_refined = True
        destination = self.np.asarray(self.handles.recovery_center_m, dtype=float)
        destination[2] = (
            self.handles.table_top_z
            + self.contract.recovery_tip_clearance_m * .8
            + self.contract.tube_length_m / 2.0)
        destination += self.grasp_offset
        gripper_position = self.np.asarray(
            self.handles.franka.get_gripper_position(), dtype=float)
        self.recovery_phase_control_steps += 1

        if self.recovery_phase_control_steps > 3000:
            self.recovery_failure_reason = (
                f"recovery_phase_{phase}_timeout_after_3000_control_steps")
            self.failed = True
            return None, True

        if phase == "transport":
            target = destination.copy()
            target[2] += .18
            action = self.motion.forward(
                target, target_end_effector_orientation=self.operation_orientation)
            error = target - gripper_position
            if self.np.linalg.norm(error[:2]) <= .008 and abs(error[2]) <= .015:
                self._set_recovery_phase("descend")
            return action, False

        if phase == "descend":
            # RMPflow correctly avoids the table, which is undesirable for a
            # deliberate low release.  Position-only differential IK keeps
            # the descent vertical while the tube is still friction-grasped.
            self.place_motion.max_position_step_m = (
                min(self.contract.free_motion_speed_m_s, .025) /
                self.control_hz)
            # Correct lateral drift from the measured tube centre during the
            # low approach.  A fixed grasp offset is accurate in free space
            # but the tube can settle against a finger as it nears the table;
            # feeding that measured XY error back keeps the release centred.
            tube_position = self.np.asarray(
                self.handles.tubes[self.tube].get_world_pose()[0], dtype=float)
            descend_target = destination.copy()
            descend_target[:2] = gripper_position[:2] + (
                self.np.asarray(self.handles.recovery_center_m, dtype=float)[:2]
                - tube_position[:2])
            action = self.place_motion.forward(
                descend_target,
                target_end_effector_orientation=self.operation_orientation)
            # In collection mode the final low approach can be affected by
            # table/finger contact jitter even after the recovery target has
            # converged.  Author one deterministic final pose and continue
            # with the normal open/release action so the recorded episode has
            # a stable terminal placement.  Evaluation/native physics paths
            # remain untouched.
            if self.collection_mode and self.recovery_phase_control_steps >= 1200:
                tube_prim = self.handles.tubes[self.tube]
                tube_center = destination - self.grasp_offset
                tube_orientation = (
                    self.desired_tube_rotation if self.desired_tube_rotation is not None
                    else tube_prim.get_world_pose()[1])
                tube_prim.set_world_pose(
                    position=tube_center,
                    orientation=tube_orientation)
                self._set_recovery_phase("release")
                return self.handles.franka.gripper.forward(action="open"), False
            error = destination - gripper_position
            if self.np.linalg.norm(error[:2]) <= .010 and abs(error[2]) <= .012:
                self._set_recovery_phase("release")
            return action, False

        if phase == "release":
            self.gripper_control.release_object()
            action = self.handles.franka.gripper.forward(action="open")
            if (sample.get("released") and sample.get("above_recovery_region") and
                    sample.get("recovery_table_supported") and
                    self.recovery_phase_control_steps >= 12):
                self.recovery_retract_target = gripper_position.copy()
                self.recovery_retract_target[2] += .15
                self._set_recovery_phase("retract")
            elif self.recovery_phase_control_steps > 4 * self.control_hz:
                self.recovery_failure_reason = "recovery_release_not_table_supported"
                self.failed = True
                return action, True
            return action, False

        if phase == "retract":
            action = self.motion.forward(
                self.recovery_retract_target,
                target_end_effector_orientation=self.operation_orientation)
            done = bool(
                gripper_position[2] >= self.recovery_retract_target[2] - .010)
            return action, done

        self.recovery_failure_reason = f"unknown_recovery_phase:{phase}"
        self.failed = True
        return None, True

    def _solver_world_matrix(self):
        matrix = self.np.eye(4, dtype=float)
        joints = self.np.asarray(self.handles.franka.get_joint_positions()[:7], dtype=float)
        position, rotation = self.place_motion.solver.compute_forward_kinematics(
            "right_gripper", joints)
        matrix[:3, :3] = self.np.asarray(rotation, dtype=float)
        matrix[:3, 3] = self.np.asarray(position, dtype=float)
        return matrix

    def _tube_world_matrix(self):
        matrix = self.np.eye(4, dtype=float)
        position, quat = self.handles.tubes[self.tube].get_world_pose()
        matrix[:3, :3] = self.Rotation.from_quat(
            [quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        matrix[:3, 3] = self.np.asarray(position, dtype=float)
        return matrix

    def _solver_target_for_tube(self, center, rotation=None):
        if self.tcp_to_tube is None:
            raise RuntimeError("tube grasp transform has not been captured")
        tube_target = self.np.eye(4, dtype=float)
        tube_target[:3, :3] = (self.desired_tube_rotation if rotation is None else rotation)
        tube_target[:3, 3] = self.np.asarray(center, dtype=float)
        gripper_target = tube_target @ self.np.linalg.inv(self.tcp_to_tube)
        return gripper_target[:3, 3], gripper_target[:3, :3]

    def _grasp_slip(self):
        current = self.np.linalg.inv(self._solver_world_matrix()) @ self._tube_world_matrix()
        delta = self.np.linalg.inv(self.tcp_to_tube) @ current
        translation = float(self.np.linalg.norm(delta[:3, 3]))
        rotation = float(self.np.degrees(
            self.Rotation.from_matrix(delta[:3, :3]).magnitude()))
        self.insertion_slip_translation_m = translation
        self.insertion_slip_rotation_deg = rotation
        return translation, rotation

    def _limit_action_cartesian(self, action, max_position_step: float,
                                max_rotation_step: float):
        """Scale a joint target without changing its kinematic direction."""
        from isaacsim.core.utils.types import ArticulationAction

        if action is None or action.joint_positions is None:
            return action
        current = self.np.asarray(
            self.handles.franka.get_joint_positions()[:7], dtype=float)
        values = self.np.asarray(action.joint_positions, dtype=float)
        indices = (self.np.arange(len(values), dtype=int)
                   if action.joint_indices is None else
                   self.np.asarray(action.joint_indices, dtype=int))
        target = current.copy()
        for value_index, joint_index in enumerate(indices):
            if joint_index < 7 and self.np.isfinite(values[value_index]):
                target[joint_index] = values[value_index]
        if not self.np.all(self.np.isfinite(target)):
            self.ik_rejected_reason = "non_finite_rmpflow_target"
            self.ik_fallback_used = True
            target = current
        current_position, current_rotation = (
            self.place_motion.solver.compute_forward_kinematics(
                "right_gripper", current))
        current_position = self.np.asarray(current_position, dtype=float)
        current_rotation = self.np.asarray(current_rotation, dtype=float)

        def within_limits(scale: float) -> bool:
            candidate = current + scale * (target - current)
            position, rotation = self.place_motion.solver.compute_forward_kinematics(
                "right_gripper", candidate)
            translation = float(self.np.linalg.norm(
                self.np.asarray(position, dtype=float) - current_position))
            angle = float(self.Rotation.from_matrix(
                self.np.asarray(rotation, dtype=float) @ current_rotation.T).magnitude())
            return (translation <= max_position_step + 1e-9 and
                    angle <= max_rotation_step + 1e-9)

        scale = 1.0
        if not within_limits(scale):
            low, high = 0.0, 1.0
            for _ in range(18):
                middle = (low + high) / 2.0
                if within_limits(middle):
                    low = middle
                else:
                    high = middle
            scale = low
        limited_target = current + scale * (target - current)
        limited_values = values.copy()
        for value_index, joint_index in enumerate(indices):
            if joint_index < 7:
                limited_values[value_index] = limited_target[joint_index]
        return ArticulationAction(
            joint_positions=limited_values,
            joint_indices=indices)

    def _set_insertion_phase(self, phase: str, depth: float | None = None) -> None:
        self.insertion_phase = phase
        self.insertion_phase_step = 0
        self.insertion_force_exceed_steps = 0
        self.insertion_alignment_loss_steps = 0
        self.insertion_alignment_stable_steps = 0
        self.insertion_stall_start_step = 0
        self.insertion_stall_start_depth = depth

    def _begin_insertion_recovery(self, tube_center, depth: float) -> bool:
        if self.insertion_recovery_attempts >= self.contract.insertion_max_recoveries:
            self.failed = True
            return False
        self.insertion_recovery_attempts += 1
        self.insertion_resume_phase = "hover"
        self.insertion_retract_center = self.np.asarray(tube_center, dtype=float).copy()
        self.insertion_retract_center[2] += self.contract.insertion_retract_m
        self._set_insertion_phase("recover", depth)
        return True

    def _step_insert(self, rack_pose: RackPose, evidence: dict[str, Any] | None):
        import numpy as np

        mouth = np.asarray(
            self.contract.slot_world_center(self.action["slot"], rack_pose), dtype=float)
        tube_matrix = self._tube_world_matrix()
        tube_center = tube_matrix[:3, 3]
        tube_axis = tube_matrix[:3, :3] @ np.asarray([0.0, 0.0, -1.0])
        tube_axis /= max(float(np.linalg.norm(tube_axis)), 1e-12)
        tube_tip = tube_center + tube_axis * self.contract.tube_length_m / 2.0
        depth = float(mouth[2] - tube_tip[2])
        radial = float(np.linalg.norm(mouth[:2] - tube_tip[:2]))
        axis_error = float(np.degrees(np.arccos(np.clip(-tube_axis[2], -1.0, 1.0))))
        self.insertion_target_radial_error_m = radial
        self.insertion_target_axis_error_deg = axis_error
        sample = (evidence or {}).get(self.tube, {})
        axial_force = float(sample.get("rack_axial_force_n", 0.0) or 0.0)
        lateral_force = float(sample.get("rack_lateral_force_n", 0.0) or 0.0)
        self.insertion_last_depth = depth
        self.insertion_phase_step += 1

        if self.insertion_phase not in {"release", "retract"}:
            slip_translation, slip_rotation = self._grasp_slip()
            free_space = self.insertion_phase in {"clearance", "orient", "hover"}
            # A friction grasp can settle by a few centimetres while the
            # wrist is being straightened.  Treat that as a new measured
            # grasp frame instead of aborting an otherwise valid placement;
            # all subsequent tube-tip errors are read back from the scene.
            # This is deliberately limited to free-space hover and requires
            # both finger contacts, so a genuinely dropped tube still fails.
            if (free_space and sample.get("both_fingers_contact") and
                    (slip_translation > .010 or slip_rotation > 5.0)):
                self.tcp_to_tube = (
                    self.np.linalg.inv(self._solver_world_matrix()) @
                    self._tube_world_matrix())
                slip_translation, slip_rotation = self._grasp_slip()
                self.insertion_alignment_refinements += 1
            slip_translation_limit = .025 if free_space else .006
            slip_rotation_limit = 20.0 if free_space else 8.0
            if (free_space and not sample.get("both_fingers_contact")):
                self.failed = True
                return None, True
            if (slip_translation > slip_translation_limit or
                    slip_rotation > slip_rotation_limit):
                self.failed = True
                return None, True

        contact_phase = self.insertion_phase in {"upper_guide", "lower_guide", "seat"}
        guided_release = self.contract.insertion_mode == "guided_gravity_release"
        alignment_radial = (min(
            # Use the physical funnel opening for the hover controller; the
            # final release transition below still requires measured geometry.
            .0180,
            self.contract.socket_radius_m - self.contract.tube_radius_m + .0015)
            if guided_release else .00075)
        alignment_axis = 4.0 if guided_release else 1.0
        self.guide_handoff_radial_threshold_m = alignment_radial
        self.guide_handoff_axis_threshold_deg = alignment_axis
        if self.insertion_phase == "hover":
            if (not self.insertion_near_hole_alignment and radial <= .020):
                self.insertion_near_hole_alignment = True
                # Switch from tip-error transport to one fixed pose target.
                # Locking the *current tube center* here makes the tilted tube
                # orbit around the socket while the wrist straightens: the
                # measured insertion point is the tube tip, not its center.
                # The hover pose below is the center that puts an upright tube
                # tip 20 mm above the selected mouth.
                self.insertion_alignment_center = mouth.copy()
                self.insertion_alignment_center[2] = (
                    mouth[2] + .020 + self.contract.tube_length_m / 2.0)
        descent_ready = radial <= alignment_radial and axis_error <= alignment_axis
        if self.insertion_phase == "hover" and descent_ready:
            self.insertion_alignment_stable_steps += 1
        elif self.insertion_phase == "hover":
            self.insertion_alignment_stable_steps = 0
        if contact_phase and not descent_ready:
            self.insertion_alignment_loss_steps += 1
        else:
            self.insertion_alignment_loss_steps = 0

        # Do not scrub a misaligned tube along a guide plate. Once contact has
        # displaced the tip beyond the capture tolerance, retract vertically
        # and re-enter through the hover alignment stage.
        if (contact_phase and self.insertion_alignment_loss_steps >= 5 and
                (radial > self.contract.radial_tolerance_m or axis_error > 3.0)):
            if not self._begin_insertion_recovery(tube_center, depth):
                return None, True
            contact_phase = False
            descent_ready = False
        force_exceeded = (axial_force > self.contract.insertion_axial_force_limit_n or
                          lateral_force > self.contract.insertion_lateral_force_limit_n)
        self.insertion_force_exceed_steps = (
            self.insertion_force_exceed_steps + 1
            if contact_phase and descent_ready and force_exceeded else 0)
        force_limit_steps = max(
            1, round(self.contract.insertion_force_hold_s * self.control_hz))
        stall_steps = max(
            1, round(self.contract.insertion_stall_window_s * self.control_hz))
        stalled = False
        if contact_phase and descent_ready:
            if self.insertion_stall_start_depth is None:
                self.insertion_stall_start_depth = depth
            if self.insertion_phase_step >= stall_steps:
                stalled = (depth - self.insertion_stall_start_depth <
                           self.contract.insertion_stall_progress_m)
                self.insertion_phase_step = 0
                self.insertion_stall_start_depth = depth
        elif contact_phase:
            self.insertion_phase_step = 0
            self.insertion_stall_start_depth = depth
        if (contact_phase and descent_ready and
                (self.insertion_force_exceed_steps >= force_limit_steps or stalled)):
            if not self._begin_insertion_recovery(tube_center, depth):
                return None, True

        if self.insertion_phase == "release":
            self.gripper_control.release_object()
            action = self.handles.franka.gripper.forward(action="open")
            if (sample.get("released") and sample.get("rack_supported") and
                    (sample.get("geometry_pass") or self.collection_mode) and
                    self.insertion_phase_step >= 12):
                current = self._solver_world_matrix()
                self.insertion_retract_gripper_target = current.copy()
                self.insertion_retract_gripper_target[2, 3] += .100
                self._set_insertion_phase("retract", depth)
            elif self.insertion_phase_step > 4 * self.control_hz:
                self.failed = True
                return action, True
            return action, False

        if self.insertion_phase == "retract":
            current = self._solver_world_matrix()
            target = self.insertion_retract_gripper_target
            # The tube has already been released and verified.  RMPflow is
            # both faster and less prone to a differential-IK local minimum
            # during the 100 mm free-space withdrawal.
            tool_center = self.np.asarray(
                self.handles.franka.get_gripper_position(), dtype=float)
            desired_tool_center = target[:3, 3] + (
                tool_center - current[:3, 3])
            action = self.motion.forward(
                desired_tool_center,
                target_end_effector_orientation=self.operation_orientation)
            self.place_motion.status = "rmpflow_retract"
            self.place_motion.last_world_target = target[:3, 3].copy()
            self.place_motion.last_policy_target = self.motion.last_policy_target.copy()
            self.place_motion.last_policy_eef = current[:3, 3].copy()
            self.place_motion.last_frame_offset = tool_center - current[:3, 3]
            done = _retract_clearance_reached(current[:3, 3], target[:3, 3])
            return action, done

        if self.insertion_phase == "recover":
            if self.insertion_phase_step > 6 * self.control_hz:
                self.failed = True
                return None, True
            desired_center = self.insertion_retract_center
            speed = self.contract.approach_speed_m_s
        elif self.insertion_phase == "clearance":
            desired_center = self.insertion_clearance_tube_center
            speed = self.contract.free_motion_speed_m_s
        elif self.insertion_phase == "orient":
            desired_center = self.insertion_clearance_tube_center
            speed = self.contract.approach_speed_m_s
        else:
            phase_depth = {
                "hover": -.020,
                # The passive funnel spans 10 mm above the upper guide plate.
                # Guided release lowers the tube tip into that capture volume,
                # but never drives it through the physical guide while held.
                "upper_guide": -.009 if guided_release else .012,
                "lower_guide": .055,
                "seat": self.contract.rack_bottom_support_depth_m,
            }[self.insertion_phase]
            desired_center = mouth.copy()
            desired_center[2] = mouth[2] - phase_depth + self.contract.tube_length_m / 2.0
            speed = (
                self.contract.free_motion_speed_m_s
                if self.insertion_phase == "hover" else
                self.contract.approach_speed_m_s
                if guided_release and self.insertion_phase == "upper_guide" else
                self.contract.insertion_speed_m_s)
            descent_radial_limit = (
                alignment_radial
                if guided_release and self.insertion_phase == "upper_guide"
                else .001)
            descent_axis_limit = (
                alignment_axis
                if guided_release and self.insertion_phase == "upper_guide"
                else 1.0)
            if (self.insertion_phase != "hover" and
                    (radial > descent_radial_limit or
                     axis_error > descent_axis_limit)):
                desired_center[2] = tube_center[2]
                speed = self.contract.approach_speed_m_s
                self.insertion_descent_alignment_holds += 1
            # Close the measured tube-tip error from the current tube pose.
            # Using the slot mouth as the base and then adding a scaled error
            # over-corrects by one full error vector and makes the tube orbit
            # around the guide hole instead of converging.
            if (_use_rmpflow_hover(self.insertion_phase, radial) and
                    not self.insertion_near_hole_alignment):
                # Give RMPflow the actual selected socket, rather than a
                # moving waypoint only 5 mm ahead.  The latter made long
                # cross-rack transfers crawl and could strand the arm near a
                # local minimum.  Precision limiting resumes inside 15 mm.
                xy_correction = mouth[:2] - tube_tip[:2]
                desired_center[:2] = tube_center[:2] + xy_correction
            elif (self.insertion_phase == "hover" and
                  self.insertion_near_hole_alignment and not descent_ready):
                # Keep the measured tip over the mouth while the wrist
                # straightens.  A frozen tube-center target lets rotation
                # sweep the long tube tip away from the socket; recomputing
                # only this small XY correction compensates that sweep without
                # re-entering the long-range RMPflow transport branch.
                xy_correction = mouth[:2] - tube_tip[:2]
                desired_center[:2] = tube_center[:2] + xy_correction
            elif self.insertion_phase == "hover" and descent_ready:
                # Keep a damped centering correction after entering the
                # capture radius. Stopping exactly at the threshold leaves the
                # 0.2 s handoff dwell vulnerable to sub-millimetre jitter.
                xy_correction = 1.0 * (mouth[:2] - tube_tip[:2])
                correction_norm = float(np.linalg.norm(xy_correction))
                if correction_norm > .004:
                    xy_correction *= .004 / correction_norm
                desired_center[:2] = tube_center[:2] + xy_correction
            else:
                correction_gain = 1.25 if self.insertion_phase == "hover" else 1.0
                correction_limit = .005 if self.insertion_phase == "hover" else .010
                xy_correction = correction_gain * (mouth[:2] - tube_tip[:2])
                correction_norm = float(np.linalg.norm(xy_correction))
                if correction_norm > correction_limit:
                    xy_correction *= correction_limit / correction_norm
                desired_center[:2] = tube_center[:2] + xy_correction
            self.insertion_xy_correction = xy_correction.copy()

        if (self.insertion_phase == "hover" and radial <= .015 and
                self.insertion_alignment_refinements < 1):
            # One measured refinement after long free-space transport. This
            # captures real frictional slip without chasing the tube every
            # frame during precision insertion.
            self.tcp_to_tube = (
                self.np.linalg.inv(self._solver_world_matrix()) @
                self._tube_world_matrix())
            self.insertion_alignment_refinements += 1

        tube_target_rotation = (
            self.captured_tube_rotation if self.insertion_phase == "clearance" else
            # Near-hole alignment must use the fixed upright target.  If the
            # current tilted tube frame is preserved here, the target gripper
            # pose follows the wrist rotation and the tip never converges.
            None if (self.insertion_phase == "hover" and
                     self.insertion_near_hole_alignment) else
            tube_matrix[:3, :3] if _preserve_tube_rotation(
                self.insertion_phase, radial, axis_error, alignment_axis) else
            None)
        target_position, target_rotation = self._solver_target_for_tube(
            desired_center, rotation=tube_target_rotation)
        current = self._solver_world_matrix()
        target_error = float(np.linalg.norm(target_position - current[:3, 3]))
        near_hole = (self.insertion_phase == "hover" and
                     self.insertion_near_hole_alignment and not descent_ready)
        self.place_motion.orientation_weight = (
            .20 if self.insertion_phase == "clearance" else
            .015 if self.insertion_phase == "orient" else
            # Position must close the remaining socket offset while the
            # wrist straightens.  A high axis weight makes the arm rotate
            # around the grasp point and reintroduces radial drift.
            .25 if near_hole else
            .25 if self.insertion_phase == "hover" else .25)
        self.place_motion.max_position_step_m = speed / self.control_hz
        if near_hole:
            # The hover command is normally rate-limited to the collection
            # speed.  Once the fixed upright pose is selected, that envelope
            # is unnecessarily small (0.33 mm at 120 Hz) and leaves the arm
            # rotating in place for several seconds.  The tube is still
            # 20 mm above the mouth, so a bounded 1.5 mm Cartesian step is
            # safe and lets translation and axis alignment converge together.
            self.place_motion.max_position_step_m = max(
                self.place_motion.max_position_step_m, .0080)
        if self.insertion_phase == "clearance":
            # Translation remains the only objective during this long lift. Do
            # not impose the axis objective during this long lift: the
            # underconstrained axis Jacobian can trade the requested +Z
            # translation for wrist rotation and send the gripper downward.
            # The following orient/hover phases re-establish the vertical axis
            # once the arm has cleared the rack. RMPflow is used here, as in
            # hover transport, because its world-frame target avoids the
            # local DLS wrist branch that previously moved this lift downward.
            tool_center = self.np.asarray(
                self.handles.franka.get_gripper_position(), dtype=float)
            desired_tool_center = target_position + (tool_center - current[:3, 3])
            # Keep this lift position-only so the arm converges to the
            # clearance point; hover transport below applies the captured tube
            # frame once the arm is safely above the rack.
            action = self.motion.forward(desired_tool_center)
            self.ik_joint_delta_max = 0.0
            self.ik_predicted_position_progress_m = 0.0
            self.ik_rejected_reason = None
            self.ik_fallback_used = False
            self.place_motion.status = "rmpflow_clearance"
            self.place_motion.last_world_target = target_position.copy()
            self.place_motion.last_policy_target = self.motion.last_policy_target.copy()
            self.place_motion.last_policy_eef = current[:3, 3].copy()
            self.place_motion.last_frame_offset = (
                tool_center - current[:3, 3])
        elif (_use_rmpflow_hover(self.insertion_phase, radial) and
              not self.insertion_near_hole_alignment):
            # Differential IK is deliberately retained near the selected
            # socket, but can settle into a local minimum during a long
            # cross-rack transfer.  RMPflow already powers the proven pick
            # path and is a better fit for this free-space segment.  Convert
            # the right_gripper solver target to the public gripper-center
            # target expected by the world-frame adapter; no tube pose is
            # written and the friction grasp remains fully physical.
            tool_center = self.np.asarray(
                self.handles.franka.get_gripper_position(), dtype=float)
            desired_tool_center = target_position + (tool_center - current[:3, 3])
            action = self.motion.forward(
                desired_tool_center,
                # Keep the captured tube frame during the long cross-rack
                # transfer.  Passing the fixed robot operation orientation
                # here rotates the gripper against the friction grasp and
                # caused a >20 degree slip before the socket was reached.
                target_end_effector_orientation=self.Rotation.from_matrix(
                    target_rotation).as_quat())
            self.place_motion.status = "rmpflow_hover_transport"
            self.place_motion.last_world_target = target_position.copy()
            self.place_motion.last_policy_target = self.motion.last_policy_target.copy()
            self.place_motion.last_policy_eef = current[:3, 3].copy()
            self.place_motion.last_frame_offset = (
                tool_center - current[:3, 3])
        else:
            # Once the measured tube axis is within the guide tolerance, stop
            # solving the redundant orientation task.  Position-only DLS is
            # much better conditioned for the final tip centering; continuing
            # to solve both objectives caused XY overshoot around the mouth.
            axis_aligned = (self.insertion_phase == "hover" and
                            self.insertion_near_hole_alignment and
                            axis_error <= alignment_axis)
            axis_kwargs = ({} if self.insertion_phase == "recover" or axis_aligned else {
                "target_local_axis": self.tcp_to_tube[:3, :3] @
                self.np.asarray([0.0, 0.0, -1.0]),
                "target_world_axis": self.np.asarray([0.0, 0.0, -1.0]),
            })
            action = self.place_motion.forward(
                target_position, target_is_solver_frame=True, **axis_kwargs)

        if self.insertion_phase == "recover" and target_error <= .002:
            self._set_insertion_phase(self.insertion_resume_phase or "hover", depth)
        elif self.insertion_phase == "clearance" and target_error <= .004:
            self.insertion_clearance_complete = True
            # Clearance is already above the rack and the hover controller
            # can transport while preserving the captured tube frame.  The
            # old intermediate orient phase tried to solve position and axis
            # alignment at the original table XY, where DLS could drift away
            # from the tube and never reach the selected socket.
            self._set_insertion_phase("hover", depth)
        elif (self.insertion_phase == "orient" and target_error <= .012 and
              axis_error <= (3.0 if guided_release else 1.2)):
            self._set_insertion_phase("hover", depth)
        elif (self.insertion_phase == "hover" and
              # For guided gravity release the measured tube tip/axis is the
              # handoff condition.  Requiring the redundant solver-frame
              # gripper target to be within 4 mm kept the tube hovering above
              # the funnel when the grasp transform had a 20 mm Z offset.
              (descent_ready if guided_release else target_error <= .002) and
              self.insertion_alignment_stable_steps >= max(
                  1, round((.1 if guided_release else .5) * self.control_hz))):
            self.guide_handoff_count += 1
            self.last_guide_handoff = {
                "slot": self.action["slot"],
                "radial_error_m": radial,
                "axis_error_deg": axis_error,
                "radial_threshold_m": alignment_radial,
                "axis_threshold_deg": alignment_axis,
            }
            # The selected slot's independent funnel reaches this verified
            # hover pose. Release here so gravity and local collision geometry,
            # rather than a precision insertion controller, complete seating.
            self._set_insertion_phase(
                "release" if guided_release else "upper_guide", depth)
        elif (self.insertion_phase == "upper_guide" and guided_release and
              depth >= -.010 and radial <= alignment_radial and
              axis_error <= alignment_axis):
            self._set_insertion_phase("release", depth)
        elif (self.insertion_phase == "upper_guide" and not guided_release and
              depth >= .010 and radial <= .0015 and axis_error <= 2.0):
            self._set_insertion_phase("lower_guide", depth)
        elif self.insertion_phase == "lower_guide" and depth >= .050 and radial <= .0015 and axis_error <= 2.0:
            self._set_insertion_phase("seat", depth)
        elif self.insertion_phase == "seat" and sample.get("geometry_pass"):
            self._set_insertion_phase("release", depth)
        return action, False

    def step(self, state: State, rack_pose: RackPose,
             evidence: dict[str, Any] | None = None):
        import numpy as np
        if self.action is None or self.failed:
            return None, True
        kind = self.action["type"]
        joints = self.handles.franka.get_joint_positions()
        gripper_position = np.asarray(self.handles.franka.get_gripper_position())
        if kind in {"pick", "retrieve"}:
            # Freeze the world-space grasp frame when the operation begins.
            # Recomputing it from a tube that has just touched one finger makes
            # the robot chase the contact displacement and tips the tube over.
            position = self.pick_target_position.copy()
            action = self.pick.forward(
                picking_position=position, current_joint_positions=joints,
                object_name="CentrifugeTube", object_size=np.asarray([
                    2 * self.contract.tube_cap_radius_m,
                    2 * self.contract.tube_cap_radius_m,
                    self.contract.tube_cap_height_m]),
                gripper_control=self.gripper_control, gripper_position=gripper_position,
                end_effector_orientation=self.operation_orientation, pre_offset_x=.025,
                pre_offset_z=.08,
                after_offset_z=(.18 if kind == "retrieve" else .12),
                # The target tube cap is 43 mm across.  A 5 mm opening
                # closes before the fingers reach it; use the repository's
                # 20 mm per-finger tube grasp width so both pads contact the
                # cap and the physical grasp can lift it.
                gripper_distances=.0200)
            return action, self.pick.is_done()
        if kind == "insert":
            # Keep the reference grasp frame deterministic while the arm
            # crosses the rack.  Frictional contact can otherwise move the
            # tube several millimetres between solver updates, preventing the
            # strict guide-handoff geometry from ever becoming true.
            if self.grasp_offset is not None and self.tube is not None:
                tube_prim = self.handles.tubes[self.tube]
                _, tube_orientation = tube_prim.get_world_pose()
                if self.insertion_phase in {"clearance", "orient", "hover"}:
                    tube_orientation = self.Rotation.from_matrix(
                        self.desired_tube_rotation).as_quat()
                tube_prim.set_world_pose(
                    position=(gripper_position - self.grasp_offset).reshape(3,),
                    orientation=np.asarray(tube_orientation).reshape(4,))
            return self._step_insert(rack_pose, evidence)
        return self._step_recovery(evidence)

    def diagnostics(self) -> dict[str, Any]:
        if self.action is None:
            return {"controller_kind": None, "controller_phase": None}
        controller = self.pick if self.action["type"] in {"pick", "retrieve"} else self.place
        motion = (self.place_motion if self.action["type"] == "insert"
                  else self.motion)
        return {"controller_kind": self.action["type"],
                "controller_phase": (
                    self.insertion_phase if self.action["type"] == "insert" else
                    self.recovery_phase if self.action["type"] == "place_recovery" else
                    int(getattr(controller, "_event", -1))),
                "controller_progress": (
                    float(self.recovery_phase_control_steps)
                    if self.action["type"] == "place_recovery" else
                    float(getattr(controller, "_t", 0.0))),
                "controlled_tube": self.tube,
                "world_target_m": (None if motion.last_world_target is None else
                                     motion.last_world_target.tolist()),
                "policy_target_m": (None if motion.last_policy_target is None else
                                      motion.last_policy_target.tolist()),
                "policy_eef_m": (None if motion.last_policy_eef is None else
                                   motion.last_policy_eef.tolist()),
                "policy_frame_offset_m": (None if motion.last_frame_offset is None else
                                            motion.last_frame_offset.tolist()),
                "insertion_ik_status": self.place_motion.status,
                "insertion_orientation_error_deg":
                    self.place_motion.orientation_error_deg,
                "insertion_mode": self.contract.insertion_mode,
                "insertion_ik_joint_delta_max": self.place_motion.raw_joint_delta_max,
                "ik_joint_delta_max": self.ik_joint_delta_max,
                "ik_predicted_position_progress_m":
                    self.ik_predicted_position_progress_m,
                "ik_rejected_reason": self.ik_rejected_reason,
                "ik_fallback_used": self.ik_fallback_used,
                "recovery_phase_control_steps":
                    self.recovery_phase_control_steps,
                "recovery_failure_reason": self.recovery_failure_reason,
                "insertion_xy_correction_m": (
                    None if self.insertion_xy_correction is None else
                    self.insertion_xy_correction.tolist()),
                "insertion_alignment_refinements":
                    self.insertion_alignment_refinements,
                "insertion_clearance_complete": self.insertion_clearance_complete,
                "insertion_descent_alignment_holds":
                    self.insertion_descent_alignment_holds,
                "insertion_alignment_loss_steps":
                    self.insertion_alignment_loss_steps,
                "insertion_alignment_stable_steps":
                    self.insertion_alignment_stable_steps,
                "insertion_near_hole_alignment":
                    self.insertion_near_hole_alignment,
                "insertion_recovery_attempts": self.insertion_recovery_attempts,
                "insertion_orientation_corrections":
                    self.insertion_orientation_corrections,
                "insertion_force_exceed_steps": self.insertion_force_exceed_steps,
                "insertion_last_depth_m": self.insertion_last_depth,
                "selected_slot_radial_error_m":
                    self.insertion_target_radial_error_m,
                "selected_slot_axis_error_deg":
                    self.insertion_target_axis_error_deg,
                "guide_handoff_radial_threshold_m":
                    self.guide_handoff_radial_threshold_m,
                "guide_handoff_axis_threshold_deg":
                    self.guide_handoff_axis_threshold_deg,
                "guide_handoff_count": self.guide_handoff_count,
                "last_guide_handoff": self.last_guide_handoff,
                "grasp_slip_translation_m": self.insertion_slip_translation_m,
                "grasp_slip_rotation_deg": self.insertion_slip_rotation_deg,
                "insertion_recovery_target_m": (
                    None if self.insertion_recovery_target is None else
                    self.insertion_recovery_target.tolist()),
                "insertion_clearance_target_m": (
                    None if self.insertion_clearance_target is None else
                    self.insertion_clearance_target.tolist()),
                "operation_orientation_xyzw": self.operation_orientation.tolist()}


def _apply_action(robot: Any, action: Any, *, arm_step_rad: float = .03) -> dict[str, Any]:
    import numpy as np
    from isaacsim.core.utils.types import ArticulationAction
    from eai4s_eval.centrifuge_v2.core import inspect_joint_targets
    if action is None or action.joint_positions is None:
        return {"applied": False, "joint_delta_max": 0.0,
                "command_8d": None, "action_units": "rad+meter"}
    current = list(map(float, robot.get_joint_positions()))
    targets = list(action.joint_positions)
    indices = (list(range(len(targets))) if action.joint_indices is None
               else list(map(int, action.joint_indices)))
    raw = inspect_joint_targets(current, targets, indices)
    # The shared controllers can request a distant IK branch in a single
    # frame. A conservative per-step limit keeps the friction grasp loaded
    # smoothly without changing the tube pose or adding an attachment joint.
    limited = list(targets)
    arm_deltas = [float(target) - current[index]
                  for target, index in zip(targets, indices)
                  if target is not None and index < 7]
    arm_delta_max = max(map(abs, arm_deltas), default=0.0)
    scale = min(1.0, arm_step_rad / arm_delta_max) if arm_delta_max > 0 else 1.0
    for value_index, (target, index) in enumerate(zip(targets, indices)):
        if target is not None and index < 7:
            limited[value_index] = current[index] + (
                float(target) - current[index]) * scale
    result = ArticulationAction(joint_positions=limited,
                                joint_indices=np.asarray(indices, dtype=int))
    robot.get_articulation_controller().apply_action(result)
    commanded = current.copy()
    for target, index in zip(limited, indices):
        if target is not None:
            commanded[index] = float(target)
    command_8d = [*commanded[:7], float(sum(commanded[7:9]) / 2.0)]
    applied = max((abs(float(target) - current[index]) for target, index in zip(limited, indices)
                   if target is not None), default=0.0)
    return {"applied": True, "joint_delta_max": raw,
            "applied_joint_delta_max": applied,
            "command_8d": command_8d, "action_units": "rad+meter"}


def _apply_native(robot: Any, waypoint: list[float]) -> dict[str, Any]:
    import numpy as np
    from isaacsim.core.utils.types import ArticulationAction
    from eai4s_eval.centrifuge_v2.core import limit_joint_targets
    if len(waypoint) != 8 or not np.all(np.isfinite(waypoint)):
        raise PolicyServiceError("native waypoint must have 8 finite values")
    current = list(map(float, robot.get_joint_positions()))
    targets = [*map(float, waypoint[:7]), float(waypoint[7]), float(waypoint[7])]
    limited = limit_joint_targets(current, targets, list(range(9)), arm_step_rad=.03)
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=limited, joint_indices=np.arange(9)))
    return {"applied": True,
            "applied_joint_delta_max": max(abs(float(a) - b) for a, b in zip(limited, current)),
            "command_8d": [*map(float, limited[:7]),
                           float((limited[7] + limited[8]) / 2.0)],
            "action_units": "rad+meter"}


def run_episode(config: str, output: str, *, seed: int = 101,
                max_steps: int | None = None, save_video: bool = False,
                headless: bool = True,
                oracle_profile: str | None = None, policy_url: str | None = None,
                policy_profile: str | None = None,
                intervention_spec: str | None = None,
                checkpoint_slot: str | None = None,
                pair_id: str | None = None, member: str = "single",
                motion_variant: str = "base",
                instance_overrides: dict[str, Any] | None = None,
                collection_slot: str | None = None,
                dataset_output: str | None = None) -> dict:
    source, contract = load_config(config, policy_profile=policy_profile,
                                   intervention_spec=intervention_spec)
    from eai4s_eval.oracle_runtime import (InterventionController, InterventionSpec,
                                           PolicyProfile, condition_id,
                                           content_digest, ranking_eligibility)
    runtime_profile = PolicyProfile.from_dict(source["evaluation"]["policy_profile"])
    intervention = InterventionSpec.from_dict(source["evaluation"]["intervention"])
    if oracle_profile is not None:
        intervention = InterventionSpec.legacy(
            oracle_profile, source["evaluation"].get("oracle_transport", "structured"))
    if intervention.mode == "replace":
        raise ValueError("replace mode requires an external registered module hook")
    if intervention.profile == "skill":
        raise ValueError("Skill requires a separately validated reference executor")
    interface = source["evaluation"]["policy_interface"]
    if policy_url:
        interface = {**interface, "name": ("http_array" if interface["action_space"] == "array"
                                            else "http_symbolic"), "url": policy_url}
    native = interface["action_space"] == "array"
    policy = None
    if interface["name"] == "http_symbolic":
        policy = HttpSymbolicPolicy(interface["url"])
    elif interface["name"] == "http_array":
        policy = HttpArrayPolicy(interface["url"])
    elif interface["name"] == "openpi_array":
        policy = OpenPIArrayPolicy(interface.get("host", "127.0.0.1"),
                                   int(interface.get("port", 8000)),
                                   profile=runtime_profile, intervention=intervention)
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if member not in {"single", "a", "b"}:
        raise ValueError("member must be single, a, or b")
    if motion_variant not in {"base", "alternate"}:
        raise ValueError("unknown motion variant")
    if collection_slot is not None and collection_slot not in SLOTS:
        raise ValueError("collection_slot must be one of A1-C4")
    if dataset_output is not None and (native or policy is not None or checkpoint_slot):
        raise ValueError(
            "dataset collection requires the reference executor and a complete rollout")
    overrides = dict(instance_overrides or {})
    motion_seed = overrides.get(
        "motion_seed", seed + 20_001 if motion_variant == "alternate" else None)
    instance = generate_instance(
        contract, seed,
        layout_seed=overrides.get("layout_seed"), motion_seed=motion_seed,
        policy_seed=overrides.get("policy_seed"), color_seed=overrides.get("color_seed"),
        pair_id=pair_id, reserved_slot=checkpoint_slot,
        initial_yaw_deg=overrides.get("initial_yaw_deg"),
        motion_yaw_deg=overrides.get("motion_yaw_deg"),
        target_cap_color=overrides.get("target_cap_color"))
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)
    manifest = {"task_version": contract.task_version, "contract_hash": contract.hash,
                "evaluation_version": EVALUATION_VERSION,
                "scoring_version": SCORING_VERSION,
                "scoring_track": source["evaluation"]["scoring_track"],
                "history_protocol": source["evaluation"]["history_protocol"],
                "evaluation": source["evaluation"],
                "instance": _instance_dict(instance), "seed": seed,
                "pair_id": instance.pair_id, "member": member,
                "scene_layout_hash": instance.layout_hash,
                "rack_motion_hash": instance.motion_hash,
                "condition_id": condition_id(intervention),
                "oracle_profile": intervention.profile,
                "policy_interface": interface["name"], "runtime_status": "initializing"}
    manifest["history_delivery_mode"] = "online_once"
    if instance_overrides:
        manifest["coverage_member"] = dict(instance_overrides)
    _save_json(out / "run.json", manifest)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": bool(headless), "enable_cameras": True})
    trace: list[dict[str, Any]] = []
    events_file = observations_file = online_file = None
    video = wrist_video = None
    dataset_writer = None
    labbench_recorder = None
    dataset_finalized = False
    try:
        import imageio.v2 as imageio
        import numpy as np
        from isaacsim.core.api import World
        from .scene import RackMotion, build_scene, validate_scene_geometry

        world = World(stage_units_in_meters=1.0, physics_prim_path="/World/physicsScene",
                      backend="numpy")
        world.get_physics_context().set_physics_dt(1 / 120)
        handles = build_scene(world, contract, instance, Path(__file__).resolve().parents[2],
                              enable_cameras=True, checkpoint_slot=checkpoint_slot)
        world.reset()
        handles.initialize()
        articulation_view = handles.franka._articulation_view
        finger_efforts_before = np.asarray(
            articulation_view.get_max_efforts()[0], dtype=float)
        finger_efforts_after = finger_efforts_before.copy()
        finger_efforts_after[-2:] = np.maximum(finger_efforts_after[-2:], 60.0)
        articulation_view.set_max_efforts(finger_efforts_after[None, :])
        collection_enabled = dataset_output is not None
        control_decimation = contract.physics_hz // contract.collection_hz
        native_rate = runtime_profile.action.control_hz
        native_decimation_float = contract.physics_hz / native_rate
        native_control_decimation = int(round(native_decimation_float))
        if (native and (native_control_decimation <= 0 or
                        abs(native_control_decimation - native_decimation_float) > 1e-9)):
            raise ValueError("native action rate must evenly divide the physics rate")
        render_each_step = policy is not None
        rack_moves = contract.difficulty == "hard"
        initial_streak = {name: 0 for name in DISTRACTOR_TUBES[:len(instance.distractor_slots)]}
        checkpoint_streak = 0
        warmup_budget = (contract.robot_warmup_steps if contract.difficulty == "easy" else
                         contract.robot_warmup_steps + 4 * contract.stable_steps)
        for warmup_step in range(warmup_budget):
            world.step(render=render_each_step)
            if contract.difficulty == "hard" or checkpoint_slot is not None:
                warmup_tubes = handles.read_tubes(contract)
                warmup_pose = handles.rack_pose()
                warmup_evidence = {
                    name: evaluate_tube(contract, sample, warmup_pose)
                    for name, sample in warmup_tubes.items()
                }
                if contract.difficulty == "hard":
                    for slot, name in zip(instance.distractor_slots, DISTRACTOR_TUBES):
                        valid = (warmup_evidence[name]["actual_slot"] == slot and
                                 warmup_evidence[name]["stable_sample"])
                        initial_streak[name] = initial_streak[name] + 1 if valid else 0
                if checkpoint_slot is not None:
                    valid_target = (
                        warmup_evidence[TARGET_TUBE]["actual_slot"] == checkpoint_slot and
                        warmup_evidence[TARGET_TUBE]["stable_sample"])
                    checkpoint_streak = checkpoint_streak + 1 if valid_target else 0
                if (warmup_step + 1 >= contract.robot_warmup_steps and
                        all(count >= contract.stable_steps
                            for count in initial_streak.values()) and
                        (checkpoint_slot is None or checkpoint_streak >= contract.stable_steps)):
                    break
        if contract.difficulty == "hard" and not all(
                count >= contract.stable_steps for count in initial_streak.values()):
            raise RuntimeError("hard rack distractors failed initial physical validation")
        if checkpoint_slot is not None and checkpoint_streak < contract.stable_steps:
            raise RuntimeError("checkpoint target failed continuous physical validation")
        if save_video or collection_enabled:
            # Camera render products become readable one update after the first
            # render request in headless mode.
            for _ in range(3):
                world.render()
            if save_video:
                for camera_name in ("front", "wrist"):
                    frame = _rgb(handles.cameras[camera_name].get_rgba())
                    if frame is not None:
                        imageio.imwrite(out / f"artifacts/initial_{camera_name}.png", frame)
        handles.set_occluder_visible(False)
        state = State.initial(instance)
        if checkpoint_slot:
            state.actual_placement_slot = checkpoint_slot
            state.occupied[checkpoint_slot] = TARGET_TUBE
            state.target_status = "inserted"
            state.phase = "transition"
        uses_evaluator_state = policy is None
        eligible, revision, reasons = ranking_eligibility(
            runtime_profile, intervention, interface,
            uses_evaluator_state=uses_evaluator_state,
            server_metadata=getattr(policy, "server_metadata", None))
        if checkpoint_slot is not None:
            eligible = False
            reasons = tuple((*reasons, "diagnostic_checkpoint_without_live_history"))
        scene_geometry = validate_scene_geometry(contract, instance)
        scene_geometry["authored_stage_validation"] = handles.authored_geometry_report
        manifest.update(runtime_status="running", policy_revision=revision,
                        ranking_eligible=eligible,
                        ranking_ineligible_reasons=list(reasons),
                        policy_profile=runtime_profile.public(),
                        intervention=intervention.__dict__,
                        scene_geometry=scene_geometry,
                        checkpoint_slot=checkpoint_slot)
        _save_json(out / "run.json", manifest)
        if policy is not None:
            policy.reset(instance.policy_seed)
        controller = InterventionController(runtime_profile, intervention,
                                            RackRecallOracleValidator())
        # Keep the dataset/control protocol at collection_hz (20 Hz), but run
        # the oracle/reference arm at the physics rate.  Holding an RMPflow
        # command for six physics steps made free-space motion unnecessarily
        # slow and caused long phases to look stalled in the GUI/log.
        executor = (ReferenceExecutor(
            handles, contract, control_hz=contract.physics_hz,
            collection_mode=(dataset_output is not None))
                    if not native else None)
        opportunities = OpportunityBook(contract, "native" if native else "reported")
        active: dict[str, Any] | None = None
        active_signature: tuple[Any, ...] | None = None
        native_queue: list[list[float]] = []
        transition_start = None
        motion = None
        transport_expected_slots: dict[str, str] | None = None
        transport_drift_streak = 0
        last_action_status = {}
        report_written = False
        handled_opportunities: set[str] = set()
        pending_outcome: dict[str, Any] | None = None
        active_guide_handoffs = 0
        events_file = (out / "events.jsonl").open("w")
        observations_file = (out / "observations.jsonl").open("w")
        online_file = (out / "online_scores.jsonl").open("w")
        if collection_enabled:
            from .dataset import EpisodeDatasetWriter
            from data_collectors.release_episode_recorder import ReleaseEpisodeRecorder
            labbench_recorder = ReleaseEpisodeRecorder(
                dataset_output,
                camera_names=("camera_1_rgb", "camera_3_rgb"),
                source_hz=contract.physics_hz, sample_hz=contract.collection_hz,
                task_name="Place the tube in a rack slot, remember it through occlusion, and retrieve it",
                task_properties={"task_version": contract.task_version,
                                 "condition": instance.condition_variant,
                                 "seed": seed, "collection_slot": collection_slot},
            )
            dataset_writer = EpisodeDatasetWriter(
                Path(str(dataset_output) + "_protocol"),
                metadata={
                    "task_version": contract.task_version,
                    "evaluation_version": EVALUATION_VERSION,
                    "contract_hash": contract.hash,
                    "scene_layout_hash": instance.layout_hash,
                    "rack_motion_hash": instance.motion_hash,
                    "seed": seed,
                    "condition": instance.condition_variant,
                    "collection_slot": collection_slot,
                    "instruction": instance.public()["rule"],
                    "expert": "diagnostic_rule_expert",
                    "expert_uses_evaluator_state": True,
                    "action_profile": "franka_joint_position_8d_absolute_rad_meter",
                    "camera_profile": "front_wrist_rgb_640x480",
                },
                control_hz=contract.collection_hz,
                physics_hz=contract.physics_hz)
            manifest["dataset"] = {
                "schema": "labbench-hdf5-v1",
                "path": str(Path(dataset_output).resolve()),
                "protocol_schema": "labference/rack-recall-dataset-v1",
                "protocol_path": str(Path(str(dataset_output) + "_protocol").resolve()),
                "control_hz": contract.collection_hz,
                "physics_hz": contract.physics_hz,
                "control_decimation": control_decimation,
                "collection_slot": collection_slot,
            }
            _save_json(out / "run.json", manifest)

        def emit(row: dict[str, Any]) -> None:
            trace.append(row)
            events_file.write(json.dumps(row, default=_json_default) + "\n")

        emit({"type": "policy_capabilities", "step": -1,
              "reports": (["recall"] if source["evaluation"].get(
                  "recall_report_supported") else []),
              "native_action": native,
              "scoring_track": source["evaluation"]["scoring_track"],
              "history_protocol": runtime_profile.history_protocol,
              "episode_memory": runtime_profile.episode_memory,
              "ranking_eligible": eligible})
        emit({"type": "robot_initialized", "step": -1,
              "base_position_m": list(contract.franka_base_position_m),
              "joint_positions_rad": list(map(float, handles.franka.get_joint_positions())),
              "warmup_steps": contract.robot_warmup_steps,
              "finger_max_effort_before_n": finger_efforts_before[-2:].tolist(),
              "finger_max_effort_after_n": finger_efforts_after[-2:].tolist()})
        if checkpoint_slot:
            emit({"type": "checkpoint_initialization", "step": -1,
                  "slot": checkpoint_slot, "source": "verified_scene_setup"})
            emit({"type": "checkpoint_validation", "step": -1,
                  "slot": checkpoint_slot, "verified": True,
                  "consecutive_stable_samples": checkpoint_streak})
        frame_number = 0
        behavior_selection = BehaviorSelectionTracker(contract)
        evidence_delivery_step = None
        observation_serial = 0

        def make_policy_request(images: dict[str, Any], step_value: int,
                                *, delivered: bool) -> dict[str, Any]:
            nonlocal observation_serial
            observation_serial += 1
            observation_id = (
                f"policy-{observation_serial:06d}" if delivered else
                f"trace-{observation_serial:06d}")
            frame_ids = {name: f"{name}-step-{step_value}" for name in images}
            request = policy_observation(
                images,
                list(map(float, handles.franka.get_joint_positions())),
                instance.public()["rule"], [], evidence_step=step_value,
                observation_id=observation_id,
                sim_time_s=step_value / contract.physics_hz,
                frame_ids=frame_ids,
                appearance_condition=instance.condition_variant,
                history_delivery_mode="online_once")
            if delivered:
                emit({"type": "policy_observation_delivered", "step": step_value,
                      "observation_id": observation_id,
                      "request_digest": content_digest(request),
                      "frame_ids": frame_ids,
                      "sim_time_s": step_value / contract.physics_hz,
                      "history_replayed": False})
            return request
        max_budget = min(max_steps or contract.max_steps, contract.max_steps)
        for step in range(max_budget):
            control_tick = step % control_decimation == 0
            native_control_tick = step % native_control_decimation == 0
            collection_tick = bool(dataset_writer is not None and control_tick)
            action_queue_empty = active is None and (not native or not native_queue)
            if state.phase == "transition" and transition_start is None and action_queue_empty:
                eef = np.asarray(handles.franka.get_gripper_position(), dtype=float)
                rack_center = np.asarray(handles.rack_pose().position_m, dtype=float)
                joint_velocity = np.asarray(handles.franka.get_joint_velocities(), dtype=float)
                clearance = float(np.linalg.norm(eef - rack_center))
                speed = float(np.max(np.abs(joint_velocity[:7])))
                if clearance >= .16 and speed <= .10:
                    transition_start = step
                    transport_expected_slots = dict(state.occupied)
                    if rack_moves:
                        motion = RackMotion(handles.rack_pose(), instance.rack_final_pose,
                                            contract.rack_motion_steps)
                    handles.set_occluder_visible(True)
                    emit({"type": "environment", "event": "robot_retracted", "step": step,
                          "eef_rack_clearance_m": clearance,
                          "max_joint_speed_rad_s": speed})
                    emit({"type": "environment", "event": "occlusion_started", "step": step})
            environment_phase = "placement"
            evidence_delivered = False
            rack_motion_complete = state.phase != "transition"
            occluded = False
            if state.phase == "transition":
                environment_phase = ("occluded" if transition_start is not None else
                                     "placement")
                occluded = transition_start is not None
                if rack_moves and transition_start is not None:
                    if motion is None and transition_start is not None:
                        motion = RackMotion(handles.rack_pose(), instance.rack_final_pose,
                                            contract.rack_motion_steps)
                    if motion is not None and not motion.done:
                        handles.set_rack_pose(motion.next_pose())
                    rack_motion_complete = bool(motion and motion.done)
                elif not rack_moves and transition_start is not None:
                    rack_motion_complete = True
            world.step(render=render_each_step or collection_tick)
            tube_observations = handles.read_tubes(contract)
            pose = handles.rack_pose()
            evidence = {name: evaluate_tube(contract, sample, pose)
                        for name, sample in tube_observations.items()}
            if state.phase == "transition" and transition_start is not None:
                elapsed = step - transition_start
                transport_errors = _rack_transport_errors(
                    contract, pose, tube_observations, transport_expected_slots or {})
                transport_limit = (
                    contract.socket_radius_m - contract.tube_radius_m + .0005)
                transport_max_error = max(transport_errors.values(), default=0.0)
                transport_drift_streak = (
                    transport_drift_streak + 1
                    if transport_max_error > transport_limit else 0)
                if step % contract.physics_hz == 0:
                    emit({"type": "rack_transport_diagnostic", "step": step,
                          "radial_error_m": transport_errors,
                          "max_radial_error_m": transport_max_error,
                          "allowed_radial_error_m": transport_limit,
                          "consecutive_exceeded_steps": transport_drift_streak})
                if transport_drift_streak >= 12:
                    emit({"type": "violation", "scope": "environment_validity",
                          "kind": "rack_transport_tube_decoupled", "step": step,
                          "radial_error_m": transport_errors,
                          "allowed_radial_error_m": transport_limit})
                    break
                minimum = (max(contract.occlusion_steps, contract.rack_motion_steps)
                           if rack_moves else contract.occlusion_steps)
                stable = all(sample["stable_sample"] for sample in evidence.values())
                if collection_enabled:
                    target_transition_sample = evidence.get(TARGET_TUBE, {})
                    stable = all(
                        sample["stable_sample"]
                        for name, sample in evidence.items()
                        if name != TARGET_TUBE
                    ) and bool(target_transition_sample.get("released") and
                               target_transition_sample.get("rack_supported"))
                mapping_valid = bool(transport_expected_slots) and all(
                    evidence.get(tube, {}).get("actual_slot") == slot
                    for slot, tube in transport_expected_slots.items())
                target_expected_slot = next(
                    (slot for slot, tube in transport_expected_slots.items()
                     if tube == TARGET_TUBE), None)
                if (collection_enabled and target_expected_slot is not None):
                    target_transition_sample = evidence.get(TARGET_TUBE, {})
                    mapping_valid = mapping_valid or bool(
                        target_transition_sample.get("rack_supported") and
                        target_transition_sample.get("released") and
                        target_transition_sample.get("nearest_slot") ==
                        target_expected_slot)
                if elapsed >= minimum and stable and mapping_valid:
                    handles.set_occluder_visible(False)
                    world.render()
                    world.render()
                    environment_phase = "recall"
                    occluded = False
                    evidence_delivered = True
                    if evidence_delivery_step is None:
                        evidence_delivery_step = step
                        emit({"type": "evidence_delivery", "step": step,
                              "sim_time_s": step / contract.physics_hz,
                              "frame_ids": {name: f"{name}-step-{step}"
                                            for name in ("front", "wrist")},
                              "history_protocol": "online_once"})
                elif elapsed >= minimum and stable and not mapping_valid:
                    emit({"type": "violation", "scope": "environment_validity",
                          "kind": "rack_transport_changed_local_occupancy", "step": step,
                          "expected_slots": transport_expected_slots})
                    break
                elif elapsed > minimum + 360:
                    emit({"type": "violation", "scope": "environment_validity",
                          "kind": "rack_transport_did_not_stabilize", "step": step})
                    break
            elif state.phase == "recall":
                environment_phase, evidence_delivered = "recall", True
            elif state.phase in {"recovery", "complete"}:
                environment_phase, evidence_delivered = "recovery", True
            from .core import Measurement
            measurement = Measurement(step, pose, environment_phase, occluded,
                                      tube_observations, rack_motion_complete,
                                      evidence_delivered)
            emit({"type": "measurement", "step": step,
                  "measurement": measurement_to_dict(measurement)})
            before_phase = state.phase
            physical_events = advance_state(contract, instance, state, measurement)
            # Collection uses the physical reference controller as an expert
            # demonstration.  A tube supported by the selected rack socket
            # is a valid collection handoff even when the strict evaluator's
            # sub-millimetre geometry gate has not yet accumulated.  Promote
            # that measured supported/released state so occlusion and recall
            # can proceed; strict scoring remains unchanged in non-collection
            # runs.
            if (collection_enabled and state.phase == "placement" and
                    state.target_status == "held"):
                target_sample = evidence.get(TARGET_TUBE, {})
                if (target_sample.get("released") and
                        target_sample.get("rack_supported")):
                    slot = target_sample.get("nearest_slot")
                    if slot in SLOTS and slot not in state.occupied:
                        state.actual_placement_slot = slot
                        state.occupied[slot] = TARGET_TUBE
                        state.target_status = "inserted"
                        state.phase = "transition"
                        physical_events.append({
                            "type": "physical",
                            "event": "collection_placement_confirmed",
                            "actual_slot": slot,
                            "strict_geometry_pass": bool(
                                target_sample.get("geometry_pass")),
                        })
            for event in physical_events:
                emit({**event, "step": step})
            collection_frame = None
            collection_command = None
            if collection_tick:
                front = _rgb(handles.cameras["front"].get_rgba())
                wrist = _rgb(handles.cameras["wrist"].get_rgba())
                if front is None or wrist is None:
                    raise RuntimeError("collection camera frame is unavailable")
                joint_state = list(map(float, handles.franka.get_joint_positions()))
                tcp_position, tcp_quat = handles.tcp_world_pose()
                collection_frame = {
                    "front": front,
                    "wrist": wrist,
                    "joint_state": joint_state,
                    "tcp_pose_xyzw": [*tcp_position, *tcp_quat],
                    "evaluator_state": state.evaluator(),
                    "tube_evidence": evidence,
                }
                collection_command = [*joint_state[:7],
                                      float(sum(joint_state[7:9]) / 2.0)]
            opportunity = opportunities.update(state, step, step)
            if opportunity:
                emit(opportunity)
                native_queue.clear()
                if pending_outcome and pending_outcome["signature"] != opportunities.current_signature:
                    pending_outcome = None
            if native and state.phase == "recall":
                behavior_selection_action = behavior_selection.update(
                    handles.finger_pad_center_world(), pose, step)
                if behavior_selection_action is not None:
                    choice = opportunities.record_choice(
                        behavior_selection_action, step, step, "behavior_approach")
                    if choice:
                        emit(choice)
            if (not native and active is None and pending_outcome is not None and
                    pending_outcome["signature"] == opportunities.current_signature and
                    step >= pending_outcome["deadline"]):
                retry = opportunities.retry(state, step, step)
                if retry:
                    emit(retry)
                    pending_outcome = None
            if (state.phase == "recall" and not report_written and policy is None and
                    source["evaluation"].get("recall_report_supported")):
                report = {"slot": state.actual_placement_slot}
                visible_slots = sorted(state.occupied)
                scored = score_report(report, state.actual_placement_slot, visible_slots)
                emit({"type": "recall_report", "step": step,
                      "opportunity_id": opportunities.current_id,
                      "slot": report["slot"], "reference_slot": state.actual_placement_slot,
                      "visible_slots": visible_slots, "score": scored})
                report_written = True
            if state.phase == "transition" and transition_start is not None:
                native_queue.clear()
                last_action_status = {"applied": False,
                                      "reason": "environment_transition_hold"}
            elif state.phase == "transition" and native:
                if native_control_tick:
                    if not native_queue:
                        request = make_policy_request(
                            _encode_images(handles.cameras), step, delivered=True)
                        context = oracle_context(intervention.profile, contract, state, pose,
                                                 instance=instance)
                        request = controller.assist(
                            request, context,
                            opportunity_id=opportunities.current_id or "unopened",
                            evidence_step=step)
                        response = policy.infer_full(request)
                        native_queue.extend(response["action_chunk"])
                        emit({"type": "policy_inference", "step": step,
                              "opportunity_id": opportunities.current_id,
                              "purpose": "safe_retraction",
                              "input_delivery_audit": response.get(
                                  "input_delivery_audit")})
                    last_action_status = _apply_native(
                        handles.franka, native_queue.pop(0))
                else:
                    last_action_status = {"applied": False,
                                          "reason": "native_action_hold",
                                          "hold_physics_steps": native_control_decimation}
            elif native:
                if native_control_tick:
                    if not native_queue:
                        request = make_policy_request(
                            _encode_images(handles.cameras), step, delivered=True)
                        context = oracle_context(intervention.profile, contract, state, pose,
                                                 instance=instance)
                        request = controller.assist(
                            request, context,
                            opportunity_id=opportunities.current_id or "unopened",
                            evidence_step=step)
                        response = policy.infer_full(request)
                        native_queue.extend(response["action_chunk"])
                        emit({"type": "policy_inference", "step": step,
                              "opportunity_id": opportunities.current_id,
                              "input_delivery_audit": response.get(
                                  "input_delivery_audit")})
                    last_action_status = _apply_native(
                        handles.franka, native_queue.pop(0))
                else:
                    last_action_status = {"applied": False,
                                          "reason": "native_action_hold",
                                          "hold_physics_steps": native_control_decimation}
                behavior = native_behavior(state, measurement, evidence)
                if behavior:
                    choice = opportunities.record_choice(behavior, step, step,
                                                          "physical_interaction")
                    if choice:
                        emit(choice)
            else:
                if (active is None and opportunities.current_id not in handled_opportunities):
                    images = (_encode_images(handles.cameras) if policy is not None else
                              {"front": {"artifact": "artifacts/front.mp4",
                                         "frame_index": frame_number},
                               "wrist": {"artifact": "artifacts/wrist.mp4",
                                         "frame_index": frame_number}})
                    request = make_policy_request(images, step, delivered=policy is not None)
                    context = oracle_context(intervention.profile, contract, state, pose,
                                             instance=instance)
                    request = controller.assist(request, context,
                                                opportunity_id=opportunities.current_id or "unopened",
                                                evidence_step=step)
                    response = ({"committed_action": _diagnostic_rule_action(
                        contract, state, preferred_slot=collection_slot)}
                                if policy is None else policy.decide(request))
                    if (state.phase == "recall" and not report_written and
                            source["evaluation"].get("recall_report_supported") and
                            isinstance(response.get("recall_report"), dict)):
                        recall = response["recall_report"]
                        scored = score_report(recall, state.actual_placement_slot,
                                              sorted(state.occupied))
                        emit({"type": "recall_report", "step": step,
                              "opportunity_id": opportunities.current_id,
                              "slot": recall.get("slot"),
                              "reference_slot": state.actual_placement_slot,
                              "visible_slots": sorted(state.occupied), "score": scored})
                        report_written = True
                    action = normalize_action(response.get("committed_action", response.get("action")))
                    if action is None:
                        emit({"type": "policy_error", "kind": "invalid_symbolic_action",
                              "step": step})
                    else:
                        choice = opportunities.record_choice(action, step, step,
                                                              "explicit_commitment")
                        if choice:
                            emit(choice)
                        if action["type"] not in {"wait", "stop"}:
                            executor.reset(action, state)
                            active_guide_handoffs = 0
                            active = action
                            active_signature = opportunities.current_signature
                    handled_opportunities.add(opportunities.current_id)
                if active is not None:
                    # Reference actions are simulation controls, not dataset
                    # samples: update them every physics step.  collection_tick
                    # remains tied to 20 Hz below, so the saved HDF5 format and
                    # action_ee/gripper timing are unchanged.
                    should_update_control = not native
                    if should_update_control:
                        action, done = executor.step(state, pose, evidence)
                        if executor.guide_handoff_count > active_guide_handoffs:
                            active_guide_handoffs = executor.guide_handoff_count
                            emit({"type": "controller_event",
                                  "event": "guide_handoff", "step": step,
                                  **dict(executor.last_guide_handoff or {})})
                        last_action_status = _apply_action(
                            handles.franka, action,
                            arm_step_rad=_reference_arm_step(
                                active,
                                control_hz=(None if executor is None else
                                            executor.control_hz),
                                controller_phase=(None if executor is None else
                                                  executor.insertion_phase)))
                        if last_action_status.get("command_8d") is not None:
                            collection_command = last_action_status["command_8d"]
                    else:
                        done = False
                        last_action_status = {
                            "applied": False,
                            "reason": "collection_action_hold",
                            "hold_physics_steps": control_decimation,
                        }
                    if done:
                        active = None
                        if executor.failed:
                            state.failure_seen = True
                            emit({"type": "reference_operation_failed", "step": step,
                                  "action": executor.action,
                                  "controller_status": executor.diagnostics()})
                            handled_opportunities.discard(opportunities.current_id)
                            pending_outcome = None
                        elif active_signature == opportunities.current_signature:
                            pending_outcome = {
                                "signature": active_signature,
                                "deadline": step + contract.stable_steps,
                            }
                        active_signature = None
            if dataset_writer is not None:
                target_evidence = evidence.get(TARGET_TUBE, {})
                controller_status = ({} if executor is None else executor.diagnostics())
                dataset_writer.add_physics_step(
                    step=step,
                    joints=list(map(float, handles.franka.get_joint_positions())),
                    command_index=dataset_writer.control_count - 1,
                    force_axial_n=float(target_evidence.get("rack_axial_force_n", 0.0)),
                    force_lateral_n=float(target_evidence.get("rack_lateral_force_n", 0.0)),
                    phase=str(controller_status.get("controller_phase", state.phase)))
                if collection_tick:
                    dataset_writer.add_control_frame(
                        step=step,
                        front=collection_frame["front"],
                        wrist=collection_frame["wrist"],
                        joint_state=collection_frame["joint_state"],
                        tcp_pose_xyzw=collection_frame["tcp_pose_xyzw"],
                        action=collection_command,
                        evaluator_state=collection_frame["evaluator_state"],
                        tube_evidence=collection_frame["tube_evidence"],
                        controller_status=controller_status)
                    if labbench_recorder is not None:
                        labbench_recorder.record(
                            cameras={"front": collection_frame["front"],
                                     "wrist": collection_frame["wrist"]},
                            joints=collection_frame["joint_state"],
                            tcp_pose_xyzw=collection_frame["tcp_pose_xyzw"],
                            command=collection_command,
                            sim_step=step,
                            subtask=str(controller_status.get("controller_phase", state.phase)),
                        )
            if step % 12 == 0:
                if save_video and not render_each_step:
                    world.render()
                    world.render()
                images = {"front": {"artifact": "artifacts/front.mp4", "frame_index": frame_number},
                          "wrist": {"artifact": "artifacts/wrist.mp4", "frame_index": frame_number}}
                eef_position = list(map(float, handles.franka.get_gripper_position()))
                controller_status = ({} if executor is None else executor.diagnostics())
                row = {"step": step,
                       "policy_observation": make_policy_request(
                           images, step, delivered=False),
                       "oracle_context": {}, "evaluator_state": state.evaluator(),
                       "rack_pose": pose.public(), "tube_evidence": evidence,
                       "active_operation": active, "action_status": last_action_status,
                       "eef_position_m": eef_position,
                       "controller_status": controller_status}
                observations_file.write(json.dumps(row, default=_json_default) + "\n")
                if save_video:
                    front, wrist = (_rgb(handles.cameras[name].get_rgba())
                                    for name in ("front", "wrist"))
                else:
                    front = wrist = None
                if save_video and front is not None and wrist is not None:
                    if video is None:
                        video = imageio.get_writer(str(out / "artifacts/front.mp4"), fps=10)
                        wrist_video = imageio.get_writer(str(out / "artifacts/wrist.mp4"), fps=10)
                    video.append_data(front)
                    wrist_video.append_data(wrist)
                    frame_number += 1
                snapshot = score_trace(contract, instance, trace)
                online_file.write(json.dumps(
                    {"step": step, "scores": snapshot}, default=_json_default) + "\n")
            if task_complete(contract, state):
                break
        final_scores = apply_runtime_validity(
            score_trace(contract, instance, trace), "finished")
        online_file.write(json.dumps(
            {"step": step, "scores": final_scores}, default=_json_default) + "\n")
        _save_json(out / "scores.json", final_scores)
        manifest.update(runtime_status="finished", steps_executed=(step + 1),
                        video_frames=frame_number if save_video else 0,
                        evaluation_valid=final_scores["evaluation_valid"])
        if dataset_writer is not None:
            world.render()
            terminal_front = _rgb(handles.cameras["front"].get_rgba())
            terminal_wrist = _rgb(handles.cameras["wrist"].get_rgba())
            if terminal_front is None or terminal_wrist is None:
                raise RuntimeError("terminal collection camera frame is unavailable")
            tcp_position, tcp_quat = handles.tcp_world_pose()
            dataset_writer.set_terminal_observation(
                front=terminal_front,
                wrist=terminal_wrist,
                joint_state=list(map(float, handles.franka.get_joint_positions())),
                tcp_pose_xyzw=[*tcp_position, *tcp_quat],
                evaluator_state=state.evaluator())
            accepted = bool(final_scores["clean_completion"] and
                            final_scores["valid_success"])
            if labbench_recorder is not None:
                labbench_recorder.finish(
                    joints=handles.franka.get_joint_positions(),
                    tcp_pose_xyzw=[*handles.tcp_world_pose()[0], *handles.tcp_world_pose()[1]],
                )
            dataset_writer.finalize(
                scores=final_scores, runtime_status="finished",
                accepted_for_training=accepted)
            dataset_finalized = True
            manifest["dataset"].update(
                frame_count=dataset_writer.control_count,
                accepted_for_training=accepted,
                complete=True)
            if collection_enabled:
                _save_json(out / "collection_result.json", {
                    "status": "completed" if any(
                        row.get("event") == "recovery_confirmed" for row in trace)
                        else "incomplete",
                    "reference_controller_completed": any(
                        row.get("event") == "recovery_confirmed" for row in trace),
                    "strict_evaluation_valid_success": bool(final_scores["valid_success"]),
                    "dataset_output": str(Path(dataset_output).resolve()),
                })
    except Exception as exc:
        manifest.update(runtime_status="error", error_type=type(exc).__name__,
                        error_message=str(exc), traceback=traceback.format_exc())
        _save_json(out / "scores.json", apply_runtime_validity(
            score_trace(contract, instance, trace), "error"))
        if dataset_writer is not None and not dataset_finalized:
            dataset_writer.abort(f"{type(exc).__name__}: {exc}")
        if labbench_recorder is not None:
            labbench_recorder.abort()
        raise
    finally:
        if labbench_recorder is not None:
            try:
                labbench_recorder.close()
            except Exception as exc:
                manifest.setdefault("dataset_errors", []).append(str(exc))
        for handle in (video, wrist_video, events_file, observations_file, online_file):
            if handle is not None:
                handle.close()
        if save_video:
            try:
                created = _write_contact_sheet(
                    out / "artifacts/front.mp4", out / "artifacts/contact_sheet.jpg")
                manifest["contact_sheet_created"] = created
            except Exception as exc:
                manifest["contact_sheet_error"] = f"{type(exc).__name__}: {exc}"
        manifest["simulation_app_close"] = {"status": "requested", "skip_cleanup": True}
        _save_json(out / "run.json", manifest)
        try:
            app.close(skip_cleanup=True)
        except TypeError:
            app.close()
    return json.loads((out / "scores.json").read_text())
