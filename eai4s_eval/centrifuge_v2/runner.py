"""Single-episode Isaac Sim smoke for the versioned centrifuge task."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import random
import time
import traceback

from .cli import load_evaluation_config
from .core import (SLOTS, State, TUBES, advance_state, cartesian_waypoint,
                   cartesian_waypoint_reached, choose_accessible_first_slot,
                   gripper_target_for_insertion,
                   insertion_matches_commit, inspect_joint_targets, legal_actions,
                   insertion_clearance_target_z, joint_limit_avoidance_delta,
                   limit_joint_velocities,
                   joint_waypoint_is_local, limit_joint_targets,
                   wrap_joint_targets_near_current,
                   operation_step_limit, slew_lift_height)
from .diagnostics import (OnlineEvaluationMonitor, PolicyCapabilities, check_policy_request, deliver_input, normalize_action,
                          oracle_context, policy_observation, score_checkpoint, score_trace,
                          validated_geometry_report)
from .policy import HttpSymbolicPolicy, PolicyServiceError
from .policy import HttpArrayPolicy, OpenPIArrayPolicy
from .online_v2 import EVALUATION_VERSION, OpportunityBook, native_behavior


def camera_rgb(frame):
    """Return a writable RGB frame when an Isaac camera has rendered one."""
    import numpy as np

    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] < 3:
        return None
    rgb = image[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return rgb


def scipy_xyzw_to_lula_wxyz(quaternion):
    """Convert a SciPy xyzw quaternion to the scalar-first order used by Lula."""
    import numpy as np

    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("orientation quaternion must be a finite xyzw vector")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("orientation quaternion must be non-zero")
    value = value / norm
    return value[[3, 0, 1, 2]]


def run(config_path: str, output_dir: str, seed: int = 101, max_steps: int | None = None,
        save_video: bool = False, oracle_profile: str | None = None,
        policy_url: str | None = None, track: str | None = None,
        checkpoint_slot: str | None = None, checkpoint_kind: str = "behavior",
        policy_profile: str | None = None, intervention_spec: str | None = None,
        dataset_output: str | None = None, headless: bool = True) -> dict:
    evaluation_config, contract = load_evaluation_config(
        config_path, policy_profile=policy_profile, intervention_spec=intervention_spec)
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    from eai4s_eval.oracle_runtime import (
        InterventionSpec, PolicyProfile, condition_id, content_digest,
        ranking_eligibility, validate_connection_binding,
    )
    runtime_profile = PolicyProfile.from_dict(evaluation_config["policy_profile"])
    intervention = InterventionSpec.from_dict(evaluation_config["intervention"])
    if oracle_profile is not None:
        if intervention_spec is not None and oracle_profile != intervention.profile:
            raise ValueError("--oracle-profile conflicts with --intervention-spec")
        evaluation_config["oracle_profile"] = oracle_profile
        if intervention_spec is None:
            intervention = InterventionSpec.legacy(
                oracle_profile, evaluation_config.get("oracle_transport", "structured"))
            evaluation_config["intervention"] = intervention.__dict__
    if policy_url is not None:
        configured_interface = evaluation_config.get("policy_interface", {})
        requested_native = (track == "native" or
                            (track is None and configured_interface.get("action_space") == "array"))
        evaluation_config["policy_interface"] = {
            "name": "http_array" if requested_native else "http_symbolic",
            "action_space": "array" if requested_native else "symbolic",
            "supported_reports": configured_interface.get("supported_reports", []),
            "oracle_transports": configured_interface.get("oracle_transports", []),
            "resettable": True, "url": policy_url,
            "coordinate_input": configured_interface.get("coordinate_input", False),
            "fixed_subgoal_input": configured_interface.get("fixed_subgoal_input", False),
            "policy_revision": configured_interface.get("policy_revision")}
        if policy_profile is None:
            runtime_profile = PolicyProfile.legacy(
                evaluation_config["policy_interface"], ("front", "wrist"))
            evaluation_config["policy_profile"] = runtime_profile.public()
    interface = evaluation_config.get("policy_interface", {})
    validate_connection_binding(runtime_profile, interface)
    native = interface.get("action_space") == "array"
    selected_track = "native" if native else "reported"
    if track is not None and track != selected_track:
        raise ValueError(f"configured interface is {selected_track}, not {track}")
    evaluation_version = evaluation_config.get("evaluation_version")
    if checkpoint_slot is not None and (checkpoint_slot not in SLOTS or evaluation_version != EVALUATION_VERSION):
        raise ValueError("checkpoint_slot requires a registered slot and online-v2 evaluation")
    if checkpoint_kind not in {"behavior", "execution"}:
        raise ValueError("unsupported online checkpoint kind")
    if checkpoint_kind == "execution" and (checkpoint_slot is None or not native or
            not interface.get("coordinate_input") or not interface.get("fixed_subgoal_input")):
        raise ValueError("execution checkpoint requires a native policy with declared fixed-subgoal and coordinate inputs")
    if checkpoint_kind == "execution" and evaluation_config.get("oracle_profile", "none") != "none":
        raise ValueError("fixed-subgoal execution checkpoints must not mix Oracle conditions")
    profile = evaluation_config.get("oracle_profile", "none")
    transport = intervention.transport if intervention.mode == "assist" else "none"
    if profile not in {"none", "semantic", "decision", "geometry", "skill"}:
        raise ValueError(f"unknown oracle profile: {profile}")
    if interface.get("name") not in {"diagnostic_rule_expert", "http_symbolic", "http_array", "openpi_array"}:
        raise ValueError("unsupported policy interface")
    capabilities = PolicyCapabilities(runtime_profile.reports,
                                      runtime_profile.oracle_transports,
                                      interface.get("action_space", "symbolic"),
                                      True)
    if profile != "none" and intervention.mode == "assist" and intervention.transport not in runtime_profile.oracle_transports:
        raise ValueError("selected policy does not support Oracle transport")
    if runtime_profile.adapter in {"rpent", "vla_evaluation_harness"}:
        raise ValueError("external agent/harness profiles must run through eai4s_eval.integrations")
    if intervention.mode == "replace":
        raise ValueError("interface replacement requires an external agent or declared module-hook adapter")
    if profile == "skill":
        raise ValueError("Skill Oracle rollout requires a validated reference executor")
    if profile == "semantic" and not evaluation_config.get("semantic_visible_slots"):
        raise ValueError("Semantic Oracle requires registered visible-slot evidence")
    if interface["name"] in {"http_symbolic", "http_array"}:
        policy_class = HttpArrayPolicy if native else HttpSymbolicPolicy
        policy_service = policy_class(interface["url"], float(interface.get("timeout_s", 30)))
    elif interface["name"] == "openpi_array":
        from eai4s_eval.special_policy import openpi_policy
        policy_service = openpi_policy(OpenPIArrayPolicy, interface.get("host", "127.0.0.1"),
                                           int(interface.get("port", 8000)), interface.get("api_key"),
                                           profile=runtime_profile, intervention=intervention,
                                           options=json.loads(Path(config_path).read_text()).get("labbench_runtime"))
    else:
        policy_service = None
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)
    (out / "run.json").write_text(json.dumps({
        "task_version": contract.task_version, "contract_hash": contract.hash,
        "evaluation": evaluation_config, "seed": seed, "runtime_status": "initializing"}, indent=2))
    from isaacsim import SimulationApp

    try:
        app = SimulationApp({"headless": bool(headless), "enable_cameras": True})
    except Exception as exc:
        manifest = json.loads((out / "run.json").read_text())
        manifest.update(runtime_status="error", error_type=type(exc).__name__, error_message=str(exc))
        (out / "run.json").write_text(json.dumps(manifest, indent=2))
        (out / "scores.json").write_text(json.dumps(
            score_trace(contract, [], set(capabilities.reports)), indent=2))
        raise
    try:
        import numpy as np
        import imageio.v2 as imageio
        from PIL import Image
        from omegaconf import OmegaConf
        import omni.usd
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.utils.types import ArticulationAction
        from scipy.spatial.transform import Rotation
        from factories.robot_factory import create_robot
        from tasks.base_task import BaseTask
        from utils.object_utils import ObjectUtils
        from robots.franka.rmpflow_controller import RMPFlowController
        from controllers.atomic_actions.pick_controller import PickController
        from controllers.robot_controllers.grapper_manager import Gripper
        from .sim_bridge import PhysxReadback, author_slot_labels, author_tube_identity_marks, slot_frames

        class SceneTask(BaseTask):
            def setup_scene(self):
                super().setup_scene()
                rotor = self.stage.GetPrimAtPath("/World/Centrifugal")
                fixed = 0
                collision_meshes = 0
                for prim in Usd.PrimRange(rotor):
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                        fixed += 1
                    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                        UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr("none")
                        collision_meshes += 1
                if fixed == 0:
                    raise RuntimeError("centrifuge USD has no rigid body to immobilize")
                if collision_meshes == 0:
                    raise RuntimeError("centrifuge USD has no collision mesh")
                author_slot_labels(self.stage, contract)
                author_tube_identity_marks(self.stage)
                for prim in Usd.PrimRange(self.stage.GetPrimAtPath("/World/Franka")):
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
                for tube in TUBES:
                    for prim in Usd.PrimRange(self.stage.GetPrimAtPath(f"/World/{tube}")):
                        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
                if initial_slot is not None:
                    mouths, axis = slot_frames(self.stage, contract)
                    position = np.asarray(mouths[initial_slot]) + .70 * contract.socket_depth_m * np.asarray(axis)
                    self.object_utils.set_object_position("/World/NMR_01", position)
                for root_path in ("/World/NMR_01", "/World/NMR_02",
                                  "/World/Centrifugal", "/World/Franka/panda_leftfinger",
                                  "/World/Franka/panda_rightfinger"):
                    for prim in Usd.PrimRange(self.stage.GetPrimAtPath(root_path)):
                        if prim.HasAPI(UsdPhysics.CollisionAPI):
                            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                            collision.CreateContactOffsetAttr(.0005)
                            collision.CreateRestOffsetAttr(0.)
                            self.add_friction_to_prim(str(prim.GetPath()), 2.0, 1.6, 0.0)

            def step(self):
                return {}

        class SlowPickController(PickController):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._slow_z = None

            def reset(self, *args, **kwargs):
                super().reset(*args, **kwargs)
                self._slow_z = None

            def _execute_phase(self, picking_position, end_effector_orientation,
                               current_joint_positions, object_name, gripper_control,
                               gripper_position, gripper_distances, pick_z_offset):
                if self._event != 5:
                    return super()._execute_phase(
                        picking_position, end_effector_orientation, current_joint_positions,
                        object_name, gripper_control, gripper_position, gripper_distances,
                        pick_z_offset)
                if self._slow_z is None:
                    self._slow_z = float(gripper_position[2])
                self._slow_z = slew_lift_height(self._slow_z, float(self.target_position[2]))
                target = self.target_position.copy()
                target[2] = self._slow_z
                action = self._cspace_controller.forward(
                    target_end_effector_position=target,
                    target_end_effector_orientation=end_effector_orientation)
                if self._slow_z >= self.target_position[2] and np.linalg.norm(
                        gripper_position - self.target_position) < self._position_threshold:
                    self._event += 1
                    self._t = 0
                return action

        class FeedbackIKMotion:
            def __init__(self, solver, position_provider, joint_provider, max_step_m=0.0025,
                         rmp_controller=None, gripper_position_provider=None,
                         home_joint_provider=None):
                self.solver = solver
                self.position_provider = position_provider
                self.joint_provider = joint_provider
                self.rmp_controller = rmp_controller
                self.gripper_position_provider = gripper_position_provider
                self.home_joint_provider = home_joint_provider
                self.max_step_m = max_step_m
                self.axis_step_limits = None
                self.orientation_weight = .2
                self.status = "idle"
                self.raw_joint_delta_max = 0.0
                self.last_target_position = None
                self.last_current_position = None
                self._free_space_phase = None
                self._free_space_target = None
                self._free_space_waypoint = None
                self._free_space_solution = None
                self._free_space_yaw_deg = None
                self._free_space_transport_orientation = None
                self._position_ik_target = None
                self._position_ik_solution = None
                self._position_ik_orientation = None
                self._preferred_ik_seed = None
                self.last_target_position = None
                self.last_current_position = None

            def forward_free_space(self, target_end_effector_position,
                                   target_end_effector_orientation, phase,
                                   position_only_fallback=False):
                from isaacsim.core.utils.types import ArticulationAction
                solver_position = np.asarray(self.position_provider(), dtype=float)
                gripper_center = np.asarray(self.gripper_position_provider(), dtype=float)
                frame_offset = gripper_center - solver_position
                target = np.asarray(target_end_effector_position, dtype=float)
                desired_position = target + frame_offset
                if phase == "align":
                    self._free_space_waypoint = desired_position.copy()
                    return self.forward(
                        target_end_effector_position=desired_position,
                        target_end_effector_orientation=target_end_effector_orientation)
                desired_orientation = Rotation.from_quat(
                    np.asarray(target_end_effector_orientation, dtype=float))
                # A direct joint-space interpolation to a distant reachable
                # endpoint can pass through the rotor/table and physically
                # stall. Walk a short Cartesian waypoint path instead; solve
                # each segment from the measured arm state so the next IK
                # target follows the actually executed path.
                phase_changed = self._free_space_phase != phase
                replan = (self._free_space_target is None or
                          phase_changed or
                          np.linalg.norm(desired_position - self._free_space_target) > .015)
                if replan:
                    self._free_space_target = desired_position.copy()
                    self._free_space_phase = phase
                    self._free_space_waypoint = None
                    self._free_space_solution = None
                    if phase_changed:
                        self._free_space_yaw_deg = None
                        self._free_space_transport_orientation = None
                        if phase in {"clearance_lift", "transit", "align"}:
                            current = np.asarray(self.joint_provider()[:7], dtype=float)
                            _, current_rotation = self.solver.compute_forward_kinematics(
                                "right_gripper", current)
                            self._free_space_transport_orientation = (
                                Rotation.from_matrix(current_rotation).as_quat())
                waypoint_reached = (
                    self._free_space_waypoint is not None and
                    cartesian_waypoint_reached(
                        solver_position.tolist(), self._free_space_waypoint.tolist(), .003))
                if self._free_space_waypoint is None or waypoint_reached:
                    current = np.asarray(self.joint_provider()[:7], dtype=float)
                    waypoint = np.asarray(cartesian_waypoint(
                        solver_position, desired_position, .03), dtype=float)
                    self._free_space_waypoint = waypoint.copy()
                    self._free_space_solution = None
                if phase in {"clearance_lift", "transit"} and self.rmp_controller is not None:
                    # Use the same obstacle-aware motion policy as the
                    # proven collection controllers for lift, transit and
                    # above-socket alignment. The phase planner already breaks
                    # motion into vertical/outside-housing legs, so give RMPflow
                    # the full phase destination instead of a moving local carrot.
                    self.status = "rmpflow_free_space_waypoint"
                    self._free_space_waypoint = desired_position.copy()
                    rmp_orientation = None
                    return self.rmp_controller.forward(
                        target_end_effector_position=desired_position,
                        # The kinematic grasp fixes the tube's world attitude,
                        # so constraining wrist orientation is unnecessary and
                        # can pin the seventh joint at its limit in this narrow
                        # workspace. Let RMPflow prioritize the waypoint.
                        target_end_effector_orientation=rmp_orientation)
                if phase == "align":
                    # RMPflow can become trapped against the rotor housing
                    # after a failed insertion, leaving the tube far below
                    # the clearance pose. Use position-only differential IK
                    # here so the redundant joints can recenter while lifting
                    # the rigidly held tube back above the socket.
                    self.status = "differential_ik_align"
                    self._free_space_waypoint = desired_position.copy()
                    return self.forward(
                        target_end_effector_position=desired_position,
                        target_end_effector_orientation=target_end_effector_orientation)
                if self._free_space_solution is None:
                    current = np.asarray(self.joint_provider()[:7], dtype=float)
                    properties = robot.dof_properties
                    limits = np.column_stack((properties["lower"], properties["upper"]))[:7]
                    waypoint = self._free_space_waypoint
                    home = (np.asarray(self.home_joint_provider()[:7], dtype=float)
                            if self.home_joint_provider is not None else current.copy())
                    seeds = [current.copy(), home.copy()]
                    for base in (current, home):
                        for joint, offset in ((3, -.45), (3, .45), (5, -.7), (5, .7)):
                            seed = base.copy()
                            seed[joint] = np.clip(seed[joint] + offset,
                                                  limits[joint, 0] + .02,
                                                  limits[joint, 1] - .02)
                            seeds.append(seed)
                    candidates = []
                    candidate_base_orientation = desired_orientation
                    if phase in {"clearance_lift", "transit"}:
                        # Do not ask free-space IK to simultaneously translate
                        # and snap to the nominal wrist pose. That combination
                        # has no nearby solution in part of the centrifuge
                        # workspace and unnecessarily sends control to DLS.
                        # Hold the hand orientation computed from the rigid
                        # grasp transform and the vertical tube-axis target.
                        candidate_base_orientation = Rotation.from_quat(
                            self._free_space_transport_orientation)
                    yaw_candidates = ((self._free_space_yaw_deg,)
                                      if self._free_space_yaw_deg is not None else
                                      (0.0, 90.0, -90.0, 180.0))
                    for yaw in yaw_candidates:
                        world_yaw = Rotation.from_euler("z", yaw, degrees=True)
                        orientation = (world_yaw * candidate_base_orientation).as_quat()
                        quaternion_wxyz = orientation[[3, 0, 1, 2]]
                        target_rotation = Rotation.from_quat(orientation).as_matrix()
                        for seed in seeds:
                            solution, solved = self.solver.compute_inverse_kinematics(
                                "right_gripper", waypoint, quaternion_wxyz, seed)
                            if (not solved or not np.all(np.isfinite(solution)) or
                                    not joint_waypoint_is_local(current, solution, .10) or
                                    np.any(solution < limits[:, 0] + .01) or
                                    np.any(solution > limits[:, 1] - .01)):
                                continue
                            fk_position, fk_rotation = self.solver.compute_forward_kinematics(
                                "right_gripper", solution)
                            position_error = float(np.linalg.norm(fk_position - waypoint))
                            rotation_error = float(Rotation.from_matrix(
                                fk_rotation @ target_rotation.T).magnitude())
                            if position_error > .008 or rotation_error > np.deg2rad(5.0):
                                continue
                            range_width = np.maximum(limits[:, 1] - limits[:, 0], .1)
                            normalized_step = (solution - current) / range_width
                            limit_margin = np.minimum(
                                solution - limits[:, 0], limits[:, 1] - solution) / range_width
                            cost = float(np.linalg.norm(normalized_step) +
                                         .15 * np.sum(1.0 / np.maximum(limit_margin, .02)))
                            candidates.append((cost, yaw, solution.copy()))
                    if candidates:
                        _, self._free_space_yaw_deg, self._free_space_solution = min(
                            candidates, key=lambda item: item[0])
                    else:
                        self._free_space_solution = None
                if self._free_space_solution is not None:
                    self.status = "verified_free_space_ik_waypoint"
                    return ArticulationAction(
                        joint_positions=self._free_space_solution,
                        joint_indices=np.arange(7))
                # Preserve the grasp orientation during free-space Cartesian
                # fallback. Position-only DLS leaves four redundant joints
                # unconstrained, and the wrist can accumulate whole revolutions
                # while the TCP appears to move toward the waypoint. The
                # measured transport orientation is captured at phase entry;
                # the low orientation weight still lets translation dominate.
                self.status = "differential_ik_free_space_waypoint"
                if phase in {"clearance_lift", "transit"}:
                    fallback_orientation = self._free_space_transport_orientation
                elif position_only_fallback:
                    fallback_orientation = None
                else:
                    fallback_orientation = target_end_effector_orientation
                return self.forward(
                    target_end_effector_position=self._free_space_waypoint,
                    target_end_effector_orientation=fallback_orientation)

            def reset(self):
                if self.rmp_controller is not None:
                    self.rmp_controller.reset()
                self.status = "ready"
                self.raw_joint_delta_max = 0.0
                self._free_space_phase = None
                self._free_space_target = None
                self._free_space_waypoint = None
                self._free_space_solution = None
                self._free_space_yaw_deg = None
                self._free_space_transport_orientation = None
                self._position_ik_target = None
                self._position_ik_solution = None
                self._position_ik_orientation = None

            def forward(self, target_end_effector_position, target_end_effector_orientation):
                target = np.asarray(target_end_effector_position, dtype=float)
                self.last_target_position = target.copy()
                if not np.all(np.isfinite(target)):
                    raise ValueError("non-finite Franka end-effector target")
                current_joints = np.asarray(self.joint_provider()[:7], dtype=float)
                current_pose, current_rotation = self.solver.compute_forward_kinematics(
                    "right_gripper", current_joints)
                self.last_current_position = np.asarray(current_pose, dtype=float).copy()
                if target_end_effector_orientation is not None or self.axis_step_limits is not None:
                    # Hold a solved joint target until the physical arm reaches
                    # it. Repeated current+tiny-delta targets leave a steady
                    # tracking error under load and never center the tube.
                    desired_quat = (None if target_end_effector_orientation is None else
                                    np.asarray(target_end_effector_orientation, dtype=float))
                    orientation_changed = (
                        desired_quat is not None and
                        (self._position_ik_orientation is None or
                         (Rotation.from_quat(desired_quat) *
                          Rotation.from_quat(self._position_ik_orientation).inv()).magnitude() > .01))
                    if (self._position_ik_target is None or orientation_changed or
                            np.linalg.norm(target - self._position_ik_target) > .001):
                        self._position_ik_target = target.copy()
                        self._position_ik_solution = None
                        self._position_ik_orientation = (None if desired_quat is None else desired_quat.copy())
                    if self._position_ik_solution is None:
                        waypoint = np.asarray(cartesian_waypoint(
                            current_pose, target, .012), dtype=float)
                        solution, solved = self.solver.compute_inverse_kinematics(
                            "right_gripper", waypoint,
                            (scipy_xyzw_to_lula_wxyz(target_end_effector_orientation)
                             if target_end_effector_orientation is not None else None),
                            (self._preferred_ik_seed if self._preferred_ik_seed is not None else current_joints),
                            position_tolerance=.0005)
                        if solved and np.all(np.isfinite(solution)):
                            predicted, _ = self.solver.compute_forward_kinematics(
                                "right_gripper", solution)
                            if (np.linalg.norm(predicted - waypoint) <= .001 and
                                    np.max(np.abs(solution - current_joints)) <= 2.5):
                                self._position_ik_solution = np.asarray(solution).copy()
                    if self._position_ik_solution is not None:
                        predicted, _ = self.solver.compute_forward_kinematics(
                            "right_gripper", self._position_ik_solution)
                        solution = self._position_ik_solution.copy()
                        if (np.linalg.norm(current_pose - predicted) <= .004 and
                                np.linalg.norm(current_pose - target) > .004):
                            self._position_ik_solution = None
                        self.status = "held_position_ik"
                        self.raw_joint_delta_max = float(np.max(np.abs(solution - current_joints)))
                        return ArticulationAction(joint_positions=solution,
                            joint_velocities=np.zeros(7), joint_indices=np.arange(7))
                delta = target - current_pose
                distance = float(np.linalg.norm(delta))
                if self.axis_step_limits is None:
                    position_error = delta * min(1.0, self.max_step_m / distance) if distance else delta
                else:
                    xy_step, z_step = self.axis_step_limits
                    position_error = np.clip(delta, [-xy_step, -xy_step, -z_step],
                                             [xy_step, xy_step, z_step])
                epsilon = 1e-4
                columns = []
                for joint in range(7):
                    perturbed = current_joints.copy()
                    perturbed[joint] += epsilon
                    moved_pose, moved_rotation = self.solver.compute_forward_kinematics(
                        "right_gripper", perturbed)
                    translation = (moved_pose - current_pose) / epsilon
                    rotation = Rotation.from_matrix(moved_rotation @ current_rotation.T).as_rotvec() / epsilon
                    columns.append(np.concatenate((translation, self.orientation_weight * rotation)))
                jacobian = np.stack(columns, axis=1)
                if target_end_effector_orientation is None:
                    jacobian = jacobian[:3]
                    error = position_error
                else:
                    desired_rotation = Rotation.from_quat(
                        np.asarray(target_end_effector_orientation, dtype=float)).as_matrix()
                    angular_error = Rotation.from_matrix(desired_rotation @ current_rotation.T).as_rotvec()
                    angle = float(np.linalg.norm(angular_error))
                    if angle > .025:
                        angular_error *= .025 / angle
                    error = np.concatenate((position_error, self.orientation_weight * angular_error))
                damping = .015
                task_inverse = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping**2 * np.eye(jacobian.shape[0]),
                    np.eye(jacobian.shape[0]))
                joint_delta = task_inverse @ error
                properties = robot.dof_properties
                limits = np.column_stack((properties["lower"], properties["upper"]))[:7]
                if target_end_effector_orientation is None:
                    joint_delta += np.asarray(joint_limit_avoidance_delta(
                        current_joints, jacobian, limits, gain=.04), dtype=float)
                self.raw_joint_delta_max = float(np.max(np.abs(joint_delta)))
                if not np.all(np.isfinite(joint_delta)):
                    self.status = "non_finite_differential_ik"
                    solution = current_joints
                else:
                    norm = float(np.linalg.norm(joint_delta))
                    solution = current_joints + joint_delta * min(1.0, .06 / norm) if norm else current_joints
                    solution = np.clip(solution, limits[:, 0] + .005, limits[:, 1] - .005)
                    self.status = "differential_ik"
                return ArticulationAction(joint_positions=solution,
                                          joint_velocities=np.zeros(7), joint_indices=np.arange(7))

        class FeedbackInsertController:
            def __init__(self, motion, gripper, orientation, collection_mode=False):
                self.motion = motion
                self.gripper = gripper
                self.orientation = orientation
                self.phase = "idle"
                self.phase_steps = 0
                self.goal = None
                self.safe_z = None
                self.clearance_xy = None
                self.retreat_xy = None
                self.mouth = None
                self.axis = None
                self.grasp_offset = np.zeros(3, dtype=float)
                self._descent_target_z = None
                self.transport_orientation = np.asarray(orientation, dtype=float)
                self.insert_orientation = None
                self.approach_orientation = None
                self.collection_mode = bool(collection_mode)

            def reset(self, goal, current_gripper, clearance_z, mouth, axis,
                      grasp_offset=None, transit_clearance_x=None):
                self.motion.reset()
                self.goal = np.asarray(goal, dtype=float)
                self.mouth = np.asarray(mouth, dtype=float)
                self.axis = np.asarray(axis, dtype=float)
                self.grasp_offset = (np.asarray(grasp_offset, dtype=float)
                                     if grasp_offset is not None else np.zeros(3, dtype=float))
                # Raise the tube above the rotor's accessible rim before
                # crossing its footprint. The asset bounds include the open
                # lid and are not a reachable clearance target in this scene;
                # cap the transfer height at 240 mm above the socket mouth.
                # In the hard scene, the previous 270 mm target was outside
                # the reachable wrist workspace: the arm plateaued ~28 mm
                # below it, preventing the planner from entering transit.
                # The measured TCP-to-tube offset is included so the tube,
                # not merely the gripper, clears the housing.
                self.safe_z = insertion_clearance_target_z(
                    float(self.mouth[2]), float(clearance_z),
                    float(self.grasp_offset[2]), .24)
                self.clearance_xy = np.asarray(current_gripper[:2], dtype=float).copy()
                self.transit_clearance_x = (None if transit_clearance_x is None else
                                            float(transit_clearance_x))
                self.transit_route_stage = 0
                self.last_phase_goal = None
                self.retreat_xy = self.clearance_xy + np.array([-0.15, -0.15], dtype=float)
                self.phase = "clearance_lift"
                self.phase_steps = 0
                self.capture_stable_steps = 0
                self._descent_target_z = None
                self._centering_integral_xy = np.zeros(2)
                _, grasp_rotation = self.motion.solver.compute_forward_kinematics(
                    "right_gripper", np.asarray(self.motion.joint_provider()[:7], dtype=float))
                self.transport_orientation = Rotation.from_matrix(grasp_rotation).as_quat()
                self.insert_orientation = None
                self.approach_orientation = None

            def forward(self, gripper_position, evidence, actual_slot, state):
                self.phase_steps += 1
                current = np.asarray(gripper_position, dtype=float)
                # The phase planner reasons about the physical gripper centre,
                # while the kinematics solver targets ``right_gripper``.  The
                # Franka model has a small but important solver/tool-frame
                # offset.  Passing the gripper-centre goal directly to the
                # solver accumulated that offset on every feedback step and
                # left the tube several centimetres away from the socket.
                solver_position = np.asarray(self.motion.position_provider(), dtype=float)
                solver_frame_offset = current - solver_position

                def solver_goal(gripper_goal):
                    return np.asarray(gripper_goal, dtype=float) - solver_frame_offset

                tube_position = np.asarray(evidence["position_world_m"], dtype=float)
                command_orientation = (self.insert_orientation if self.insert_orientation is not None
                                       else self.transport_orientation)
                if self.approach_orientation is not None:
                    if self.phase in {"transit", "align"}:
                        command_orientation = self.approach_orientation.copy()
                    elif self.phase == "descend":
                        approach_fraction = np.clip(-evidence["insertion_depth_m"] / .025, 0., 1.)
                        tilt = Rotation.from_rotvec(self.approach_tilt_rotvec * approach_fraction)
                        command_orientation = (tilt * Rotation.from_quat(self.insert_orientation)).as_quat()
                precision_approach = (self.phase in {"align", "descend"} or
                                      (self.phase == "transit" and self.transit_route_stage >= 2))
                if precision_approach and self.insert_orientation is None:
                    # Correct tube-axis error relative to the *current* TCP
                    # orientation.  Composing the correction with a nominal
                    # wrist pose is wrong because the tube-to-gripper grasp
                    # transform is not identity and can amplify tilt.
                    measured_axis = np.asarray(evidence["axis_world"], dtype=float)
                    if np.linalg.norm(measured_axis) > 1e-8 and np.linalg.norm(self.axis) > 1e-8:
                        target_tube_axis = self.axis.copy()
                        if float(np.dot(target_tube_axis, measured_axis)) < 0.0:
                            target_tube_axis *= -1.0
                        joints_now = np.asarray(self.motion.joint_provider()[:7], dtype=float)
                        _, tcp_rotation = self.motion.solver.compute_forward_kinematics(
                            "right_gripper", joints_now)
                        correction, _ = Rotation.align_vectors(
                            target_tube_axis.reshape(1, 3), measured_axis.reshape(1, 3))
                        upright_rotation = correction * Rotation.from_matrix(tcp_rotation)
                        # Use the physical fingers, not a presumed TCP axis:
                        # right_gripper and panda_hand differ by a fixed yaw.
                        finger_origins = [np.asarray(UsdGeom.Xformable(
                            stage.GetPrimAtPath(f"/World/Franka/panda_{side}finger")
                        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())
                                          for side in ("left", "right")]
                        finger_axis = correction.apply(finger_origins[0] - finger_origins[1])
                        machine = stage.GetPrimAtPath("/World/Centrifugal")
                        machine_transform = UsdGeom.Xformable(machine).ComputeLocalToWorldTransform(
                            Usd.TimeCode.Default())
                        hinge_axis = np.asarray(machine_transform.TransformDir(Gf.Vec3d(1, 0, 0)))
                        hinge_axis /= np.linalg.norm(hinge_axis)
                        # Lean the carried tube toward the front while still
                        # above the socket. Straighten it before crossing the
                        # mouth plane, keeping the larger wrist below the lid.
                        approach_tilt = Rotation.from_rotvec(hinge_axis * np.deg2rad(20.))
                        yaw = np.arctan2(hinge_axis[1], hinge_axis[0]) - np.arctan2(
                            finger_axis[1], finger_axis[0])
                        wrist_position = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(
                            "/World/Franka/panda_link7")).ComputeLocalToWorldTransform(
                                Usd.TimeCode.Default()).ExtractTranslation())
                        wrist_offset = correction.apply(wrist_position - current)
                        front_axis = np.asarray(machine_transform.TransformDir(Gf.Vec3d(0, 1, 0)))
                        # The two parallel finger orientations differ in which
                        # side the bulky wrist occupies. Keep it away from the
                        # rear lid, not merely the shortest axial rotation.
                        candidates = []
                        candidate_seeds = {}
                        from omni.physx import get_physx_scene_query_interface
                        collision_query = get_physx_scene_query_interface()
                        link_frames = self.motion.solver.get_all_frame_names()

                        def candidate_obstructions(q):
                            import carb
                            count = 0
                            for frame, local_center, half_size, correction_r, correction_t in robot_collision_boxes:
                                origin, rotation = self.motion.solver.compute_forward_kinematics(frame, q)
                                box_rotation = rotation @ correction_r
                                center = origin + rotation @ (correction_t + correction_r @ local_center)
                                overlaps = []

                                def report_overlap(hit):
                                    if str(hit.rigid_body).startswith("/World/Centrifugal") or str(
                                            hit.collision).startswith("/World/Centrifugal"):
                                        overlaps.append(True)
                                    return True

                                quat = Rotation.from_matrix(box_rotation).as_quat()
                                collision_query.overlap_box(
                                    Gf.Vec3f(*map(float, half_size)), Gf.Vec3f(*map(float, center)),
                                    carb.Float4(*map(float, quat)), report_overlap)
                                count += bool(overlaps)
                            return count

                        dof_limits = np.column_stack((robot.dof_properties["lower"],
                                                      robot.dof_properties["upper"]))[:7]
                        posture_seeds = [joints_now, np.asarray(ready_arm_joints)]
                        for shoulder in (-.8, .8):
                            for elbow in (-1.2, 0., 1.2):
                                for wrist in (-1.8, 1.8):
                                    posture_seed = np.asarray(ready_arm_joints).copy()
                                    posture_seed[[0, 2, 6]] = [shoulder, elbow, wrist]
                                    posture_seeds.append(posture_seed)
                        for angle in (yaw + np.arange(24) * (np.pi / 12)):
                            axial_rotation = Rotation.from_rotvec([0., 0., angle])
                            candidate_rotation = axial_rotation * upright_rotation
                            candidate_quat = candidate_rotation.as_quat()
                            rotated_offset = axial_rotation.apply(correction.apply(current - tube_position))
                            entry_rotation = approach_tilt * candidate_rotation
                            entry_quat = entry_rotation.as_quat()
                            entry_offset = approach_tilt.apply(rotated_offset)
                            entry = current.copy()
                            entry[2] = self.mouth[2] + .025 + entry_offset[2]
                            above_mouth = self.mouth + np.array([0., 0., .025]) + entry_offset
                            seated = self.mouth + .75 * contract.socket_depth_m * self.axis + rotated_offset
                            solutions = []
                            for seed_joints in posture_seeds:
                                q = seed_joints.copy()
                                valid = True
                                blocked = 0
                                entry_solution = None
                                for point, orientation in ((entry, entry_quat), (above_mouth, entry_quat),
                                                           (seated, candidate_quat)):
                                    q, solved = self.motion.solver.compute_inverse_kinematics(
                                        "right_gripper", solver_goal(point),
                                        scipy_xyzw_to_lula_wxyz(orientation), q,
                                        position_tolerance=.001)
                                    if not solved or not np.all(np.isfinite(q)):
                                        valid = False
                                        break
                                    margin = np.minimum(q - dof_limits[:, 0], dof_limits[:, 1] - q)
                                    if np.min(margin) < .08:
                                        valid = False
                                        break
                                    blocked += candidate_obstructions(q)
                                    if entry_solution is None:
                                        entry_solution = np.asarray(q).copy()
                                if valid:
                                    solutions.append((np.asarray(q), blocked, entry_solution))
                            if solutions:
                                wrist_frontage = float(np.dot(axial_rotation.apply(wrist_offset), front_axis))
                                best_solution = min(solutions, key=lambda solution:
                                    float(np.linalg.norm(solution[0] - joints_now)) + 100 * solution[1])
                                cost = float(np.linalg.norm(best_solution[0] - joints_now)) + 100 * best_solution[1]
                                candidates.append((cost - 10 * wrist_frontage, float(angle)))
                                candidate_seeds[float(angle)] = best_solution[2]
                        if candidates:
                            yaw = min(candidates)[1]
                            self.motion._preferred_ik_seed = candidate_seeds[yaw].copy()
                            self.entry_joint_target = candidate_seeds[yaw].copy()
                        else:
                            yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
                            self.entry_joint_target = None
                        emit({"type": "insertion_orientation_candidates",
                              "candidates": candidates, "selected_yaw_rad": float(yaw)})
                        command_orientation = (Rotation.from_rotvec([0., 0., yaw]) *
                                               upright_rotation).as_quat()
                        # Keep the chosen axial yaw fixed. Rebuilding this
                        # target from the moving wrist on every frame let
                        # twist accumulate even after the tube was vertical.
                        self.insert_orientation = command_orientation.copy()
                        self.approach_orientation = (approach_tilt * Rotation.from_quat(
                            self.insert_orientation)).as_quat()
                        self.approach_tilt_rotvec = approach_tilt.as_rotvec()
                        self.upright_tube_axis = target_tube_axis.copy()
                        command_orientation = self.approach_orientation.copy()
                motion_axis_error = float(evidence["axis_error_deg"])
                if self.approach_orientation is not None:
                    expected_tilt = (Rotation.from_rotvec(self.approach_tilt_rotvec)
                                     if self.phase in {"transit", "align"} else
                                     Rotation.from_rotvec(self.approach_tilt_rotvec * np.clip(
                                         -evidence["insertion_depth_m"] / .025, 0., 1.)))
                    desired_axis = expected_tilt.apply(self.upright_tube_axis)
                    actual_axis = np.asarray(evidence["axis_world"])
                    motion_axis_error = float(np.degrees(np.arccos(np.clip(
                        np.dot(desired_axis, actual_axis), -1., 1.))))
                goal = self.goal.copy()
                if self.phase == "clearance_lift":
                    goal[:2] = self.clearance_xy
                    goal[2] = self.safe_z
                    if abs(current[2] - self.safe_z) <= .015:
                        self.phase, self.phase_steps = "transit", 0
                elif self.phase == "transit":
                    goal[2] = self.safe_z
                    grasp_xy = current[:2] - tube_position[:2]
                    if self.transit_clearance_x is None:
                        goal[:2] = self.mouth[:2] + grasp_xy
                    elif self.transit_route_stage == 0:
                        # First clear the centrifuge footprint while remaining
                        # on the source side of its housing.
                        goal[:2] = [self.transit_clearance_x + grasp_xy[0],
                                    self.clearance_xy[1]]
                        if tube_position[0] <= self.transit_clearance_x + .01:
                            self.transit_route_stage = 1
                    elif self.transit_route_stage == 1:
                        # Travel along the robot-facing outside edge, then
                        # approach the selected socket from that side.
                        goal[:2] = [self.transit_clearance_x + grasp_xy[0],
                                    self.mouth[1] + grasp_xy[1]]
                        if abs(float(tube_position[1] - self.mouth[1])) <= .012:
                            self.transit_route_stage = 2
                    elif self.transit_route_stage == 2:
                        # The open lid leans over the rim at its top edge.
                        # Descend outside the housing, then enter sideways
                        # below that overhang instead of crossing the lid.
                        goal[:2] = [self.transit_clearance_x + grasp_xy[0],
                                    self.mouth[1] + grasp_xy[1]]
                        goal[2] = float(self.mouth[2] + .025 + self.grasp_offset[2])
                        _, measured_rotation = self.motion.solver.compute_forward_kinematics(
                            "right_gripper", np.asarray(self.motion.joint_provider()[:7], dtype=float))
                        orientation_error = (Rotation.from_quat(command_orientation) *
                                             Rotation.from_matrix(measured_rotation).inv()).magnitude()
                        if self.entry_joint_target is not None:
                            measured_joints = np.asarray(self.motion.joint_provider()[:7], dtype=float)
                            if np.max(np.abs(measured_joints - self.entry_joint_target)) > .025:
                                self.last_phase_goal = goal.copy()
                                return ArticulationAction(
                                    joint_positions=self.entry_joint_target.copy(),
                                    joint_velocities=np.zeros(7), joint_indices=np.arange(7)), False
                        if (abs(current[2] - goal[2]) <= .008 and
                                orientation_error <= np.deg2rad(3.0)):
                            self.safe_z = float(goal[2])
                            self.transit_route_stage = 3
                            self.motion._preferred_ik_seed = None
                            self.motion._position_ik_solution = None
                    else:
                        goal[:2] = self.mouth[:2] + grasp_xy
                    if (self.transit_route_stage == 3 and
                            np.linalg.norm(tube_position[:2] - self.mouth[:2]) <= .012 and
                            abs(current[2] - self.safe_z) <= .012):
                        self.phase, self.phase_steps = "align", 0
                elif self.phase == "align":
                    goal[:2] = self.mouth[:2] + current[:2] - tube_position[:2]
                    goal[2] = self.safe_z
                    tube_xy_error = np.linalg.norm(tube_position[:2] - self.mouth[:2])
                    _, actual_rotation = self.motion.solver.compute_forward_kinematics(
                        "right_gripper", np.asarray(self.motion.joint_provider()[:7], dtype=float))
                    wrist_error = (Rotation.from_quat(command_orientation) *
                                   Rotation.from_matrix(actual_rotation).inv()).magnitude()
                    if (tube_xy_error <= .015 and
                            motion_axis_error <= 5.0 and
                            wrist_error <= np.deg2rad(5.0) and
                            abs(current[2] - self.safe_z) <= .008):
                        self.phase, self.phase_steps = "descend", 0
                        self._descent_target_z = float(current[2])
                elif self.phase == "descend":
                    goal = np.asarray(gripper_target_for_insertion(
                        # The tube-tip readback includes the gripper/tool
                        # offset; using 70% left the measured tip only a few
                        # millimetres inside the socket.  Compensate while
                        # retaining the strict evaluator depth bounds.
                        self.mouth, self.axis, .20 * contract.socket_depth_m,
                        self.grasp_offset, 0.0))
                    # The public right-gripper frame is ~20 mm below the
                    # tube centre in this asset; compensate the tool-frame
                    # offset while keeping insertion geometry non-negative.
                    # The wrist/tool centre sits above the rigid tube centre
                    # by roughly 80 mm in this asset.  A 20 mm correction
                    # leaves the tube at the rotor lip; command the remaining
                    # tool-frame offset so the physical tube can reach the
                    # measured 26.6--38 mm insertion band.
                    # Close the measured tube-tip depth error. A fixed 80 mm
                    # tool offset kept driving after contact and pushed a
                    # seated tube beyond the contract's depth band.
                    desired_tip = self.mouth + .75 * contract.socket_depth_m * self.axis
                    goal = current + desired_tip - tube_position
                    tube_xy_error = np.linalg.norm(tube_position[:2] - self.mouth[:2])
                    # Once vertical, keep descending through small contact
                    # jitter instead of bouncing back to the high clearance
                    # pose whenever the tube drifts a few millimetres.
                    if tube_xy_error > .03 or motion_axis_error > 15.0:
                        self.phase, self.phase_steps = "align", 0
                        self._descent_target_z = None
                        goal[2] = self.safe_z
                    elif motion_axis_error > 5.0:
                        # Re-orient at the current height before descending;
                        # pushing a tilted tube against the rotor lip was the
                        # failure mode seen in the prior hard-task run.
                        goal[2] = current[2]
                    # Continue the insertion descent while the tube is within
                    # the socket's capture funnel.  Freezing Z at 3 mm was
                    # too conservative: contact jitter kept the measured
                    # radial error oscillating between 3–8 mm, so the tube
                    # never reached the required depth.  Final success still
                    # requires the strict evaluator radial/depth/support and
                    # release checks below.
                    elif tube_xy_error > .015:
                        goal[2] = current[2]
                    elif (evidence["geometry_pass"] and evidence["supported"] and
                          actual_slot == evidence["actual_slot"]):
                        self.phase, self.phase_steps = "release", 0
                    else:
                        # Fingers can meet the rotor lip before the tube reaches
                        # the final support depth. Release only inside the
                        # measured capture opening and let gravity seat it.
                        # Success still requires the unchanged strict 60%-depth,
                        # support, released and stationary readback predicates.
                        captured = (tube_xy_error <= .0005 and
                                    evidence["insertion_depth_m"] >= .0228 and
                                    evidence["axis_error_deg"] <= 1.0 and
                                    evidence["linear_speed_m_s"] <= .01)
                        self.capture_stable_steps = self.capture_stable_steps + 1 if captured else 0
                        if self.capture_stable_steps >= 6:
                            self.phase, self.phase_steps = "release", 0
                if self.phase == "descend":
                    tube_xy_error = float(np.linalg.norm(
                        tube_position[:2] - self.mouth[:2]))
                    # Decouple horizontal centering from vertical insertion.
                    # Keep centering at the actual socket clearance before
                    # descending. Small XY increments plus a locked height
                    # prevent the arm from oscillating vertically while it
                    # converges to the strict physical radial tolerance.
                    goal = current.copy()
                    self._centering_integral_xy = np.clip(
                        self._centering_integral_xy + .02 * (self.mouth[:2] - tube_position[:2]),
                        -.012, .012)
                    if motion_axis_error > 5.0:
                        self._descent_target_z = float(current[2])
                    elif (tube_xy_error > .005 and
                          (tube_xy_error > .01 or evidence["insertion_depth_m"] >= -.005)):
                        goal[:2] = self.mouth[:2] + self.grasp_offset[:2]
                        if self._descent_target_z is None:
                            self._descent_target_z = float(current[2])
                        goal[2] = self._descent_target_z
                    else:
                        goal[:2] = self.mouth[:2] + self.grasp_offset[:2]
                        if self._descent_target_z is None:
                            self._descent_target_z = float(current[2])
                        elif current[2] <= self._descent_target_z + .008:
                            self._descent_target_z -= .005
                        goal[2] = self._descent_target_z
                    goal[:2] += self._centering_integral_xy
                    if evidence["insertion_depth_m"] >= .027:
                        self._descent_target_z = float(current[2])
                        goal[2] = current[2]
                elif self.phase == "release":
                    if state.tube_status[evidence["tube"]] == "inserted":
                        self.phase, self.phase_steps = "retreat", 0
                    elif self.phase_steps <= 120:
                        # Clear the previous arm target while opening; do not
                        # continue the final downward insertion command.
                        positions = np.asarray(robot.get_joint_positions(), dtype=float).copy()
                        positions[7:9] = .04
                        return ArticulationAction(joint_positions=positions,
                                                  joint_velocities=np.zeros_like(positions)), False
                    else:
                        self.phase = "failed"
                        return None, True
                elif self.phase == "retreat":
                    if self.retreat_xy is not None:
                        goal[:2] = self.retreat_xy
                    goal[2] = self.safe_z
                    # The high clearance waypoint is outside the reachable
                    # wrist workspace in the hard scene.  Once the released
                    # tube has been physically contact-free for a short
                    # dwell, the retreat is complete; waiting for the
                    # unreachable absolute Z target would block the next
                    # legal operation indefinitely.
                    tube_xy_distance = float(np.linalg.norm(
                        np.asarray(current[:2]) - np.asarray(tube_position[:2])))
                    if (self.phase_steps >= 120 and tube_xy_distance >= .04 and
                            not evidence.get("gripper_contact", False)):
                        self.phase = "done"
                        return None, True
                if self.phase in {"done", "failed"}:
                    return None, True
                self.last_phase_goal = goal.copy()
                self.motion.max_step_m = {"clearance_lift": .006, "transit": .025, "align": .02,
                                          "descend": .003, "release": .003,
                                          "retreat": .003}[self.phase]
                self.motion.axis_step_limits = (.01, .008) if self.phase == "descend" else None
                # Precision insertion must preserve the grasped tube attitude;
                # the former .08 weight let the XY correction rotate the wrist
                # and left the tube tilted before descent.
                self.motion.orientation_weight = .9 if self.phase in {"align", "descend"} else .25
                solver_target = solver_goal(goal)
                motion_orientation = command_orientation
                if (self.phase in {"clearance_lift", "transit", "align"} and
                        not (self.phase == "transit" and self.transit_route_stage >= 2)):
                    action = self.motion.forward_free_space(
                        target_end_effector_position=solver_target,
                        target_end_effector_orientation=command_orientation,
                        phase=self.phase,
                        position_only_fallback=motion_orientation is None)
                else:
                    action = self.motion.forward(
                        target_end_effector_position=solver_target,
                        target_end_effector_orientation=motion_orientation)
                return action, False

        random.seed(seed)
        np.random.seed(seed)
        repo = Path(__file__).resolve().parents[2]
        source_config = json.loads(Path(config_path).read_text())["legacy_scene"]
        cfg = OmegaConf.load(repo / source_config)
        cfg.multi_run.run_dir = str(out)
        cfg.max_episodes = 1
        cfg.robot.position = list(contract.franka_base_position_m)
        cfg.cameras[0].translation = list(contract.front_camera_position_m)
        cfg.cameras[0].orientation = list(contract.front_camera_orientation_wxyz)
        cfg.cameras[0].focal_length = contract.front_camera_focal_length_mm
        cfg.scene = [obj for obj in cfg.scene if obj.prim_path in {
            "/World/table", "/World/NMR_01", "/World/NMR_02", "/World/Centrifugal"}]
        for obj in cfg.scene:
            if obj.prim_path == "/World/NMR_01" and contract.difficulty == "hard":
                obj.position = [0.20, -0.25, 0.76]
                obj.orientation = [0.0, 1.0, 0.0, 0.0]
            elif obj.prim_path == "/World/NMR_02":
                # Keep the second pickup inside the reachable, collision-free
                # part of the Franka workspace in the hard scene.  The
                # previous y=-0.10 pose drove the wrist to its joint limit
                # after the first tube was seated, without bilateral contact.
                obj.position = [0.15, -0.05, 0.76]
                obj.orientation = [0.0, 1.0, 0.0, 0.0]
        rotor_cfg = next(obj for obj in cfg.scene if obj.prim_path == "/World/Centrifugal")
        rotor_cfg.position = list(contract.rotor_spawn_position_m)
        if evaluation_version == EVALUATION_VERSION:
            scene_random = random.Random(f"scene:{seed}")
            decision_random = random.Random(f"decision:{seed}")
            yaw = scene_random.randrange(contract.slot_count) * (360.0 / contract.slot_count)
            initial_slot = SLOTS[decision_random.randrange(len(SLOTS))] if contract.difficulty == "easy" else None
        else:
            yaw = (seed % contract.slot_count) * (360.0 / contract.slot_count)
            initial_slot = SLOTS[seed % len(SLOTS)] if contract.difficulty == "easy" else None
        if checkpoint_slot is not None:
            initial_slot = checkpoint_slot
        rotor_xyzw = (Rotation.from_euler("z", yaw, degrees=True) *
                      Rotation.from_euler("x", 180, degrees=True)).as_quat()
        rotor_cfg.orientation = [float(rotor_xyzw[3]), *map(float, rotor_xyzw[:3])]
        cfg.task.obj_paths = []
        physics_substeps = 4
        world = World(stage_units_in_meters=1.0, physics_prim_path="/World/physicsScene",
                      backend="numpy", physics_dt=1.0 / (60 * physics_substeps),
                      rendering_dt=1.0 / 60)
        stage = omni.usd.get_context().get_stage()
        robot = create_robot("franka", position=np.array(cfg.robot.position))
        ObjectUtils.get_instance(stage)
        scene = SceneTask(cfg, world, stage, robot)
        rotor_prim = stage.GetPrimAtPath("/World/Centrifugal")
        rotor_bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]) \
            .ComputeWorldBound(rotor_prim).ComputeAlignedRange()
        rotor_top_z = float(rotor_bounds.GetMax()[2])
        if not np.isfinite(rotor_top_z):
            raise RuntimeError("invalid centrifuge world-space bounds")
        slot_marker_count = sum(stage.GetPrimAtPath(
            f"/World/Centrifugal/DiagnosticLabels/{slot}").IsValid() for slot in SLOTS)
        if slot_marker_count != len(SLOTS):
            raise RuntimeError(f"only {slot_marker_count}/{len(SLOTS)} visible slot markers were authored")
        tube_marker_count = sum(stage.GetPrimAtPath(f"/World/{tube}/IdentityBand").IsValid()
                                for tube in TUBES)
        if tube_marker_count != len(TUBES):
            raise RuntimeError("tube identity markers were not authored")
        reader = PhysxReadback(contract, stage)
        world.reset()
        robot.initialize()
        ready_joint_positions = np.asarray(cfg.get("collection_startup", {}).get(
            "ready_joint_positions", [0., -.7854, 0., -2.3562, 0., 1.5708, .7854, .04, .04]),
            dtype=float)
        robot.set_joints_default_state(positions=ready_joint_positions)
        robot.set_joint_positions(ready_joint_positions)
        robot.set_joint_velocities(np.zeros_like(ready_joint_positions))
        reader.initialize()
        state = State(contract.difficulty)
        uses_evaluator_state = policy_service is None
        eligible, policy_revision, ineligible_reasons = ranking_eligibility(
            runtime_profile, intervention, interface, uses_evaluator_state=uses_evaluator_state,
            server_metadata=getattr(policy_service, "server_metadata", None))
        run_manifest = {"task_version": contract.task_version, "contract_hash": contract.hash,
                        "evaluation": evaluation_config,
                        "evaluation_version": evaluation_version, "policy_track": selected_track,
                        "difficulty": contract.difficulty, "seed": seed, "policy": interface["name"],
                        "policy_id": interface.get("policy_id", interface.get("url", interface["name"])),
                        "policy_revision": policy_revision,
                        "policy_revision_declared": policy_revision != "unversioned",
                        "policy_profile_id": runtime_profile.name,
                        "policy_profile_hash": runtime_profile.hash,
                        "policy_system_type": runtime_profile.system_type,
                        "policy_profile": runtime_profile.public(),
                        "intervention": intervention.__dict__,
                        "condition_id": condition_id(intervention),
                        "condition_supported": True,
                        "action_spec_hash": runtime_profile.action.hash,
                        "toolset_hash": content_digest(list(runtime_profile.agent_tools)),
                        "reset_protocol_hash": content_digest(runtime_profile.reset.__dict__),
                        "memory_revision": interface.get("memory_revision", "episode-isolated-v1"),
                        "policy_reset_scope": interface.get("reset_scope", "session"),
                        "pair_id": f"{contract.difficulty}-online-{seed:04d}",
                        "instance_id": f"{contract.difficulty}-online-{seed:04d}:single",
                        "scene_seed": seed, "mass_seed": None, "decision_seed": seed,
                        "pair_member": "single",
                        "scene_layout_hash": content_digest({
                            "initial_slot": initial_slot, "rotor_yaw_deg": yaw,
                            "franka_base_position_m": contract.franka_base_position_m}),
                        "initial_state_hash": content_digest({
                            "task_version": contract.task_version, "seed": seed,
                            "initial_slot": initial_slot, "rotor_yaw_deg": yaw}),
                        "max_steps_budget": min(max_steps if max_steps is not None else contract.max_steps,
                                                contract.max_steps),
                        "budget_id": (
                            f"steps={min(max_steps if max_steps is not None else contract.max_steps, contract.max_steps)};"
                            f"hz={runtime_profile.action.control_hz:g};chunk={runtime_profile.action.max_chunk}"),
                        "sensor_condition": "front+wrist_rgb",
                        "oracle_profile": profile, "oracle_transport": transport,
                        "policy_uses_evaluator_state": uses_evaluator_state,
                        "oracle_delivered_to_policy": policy_service is not None and profile != "none",
                        "oracle_consumed_by_policy": None,
                        "evaluation_valid": False,
                        "execution_geometry_source": ("native_joint_actions" if native else
                                                      "policy_geometry_report" if policy_service is not None and
                                                      "geometry" in capabilities.reports else "evaluator_reference"),
                        "initial_slot": initial_slot, "rotor_yaw_deg": yaw,
                        "initial_condition_authoring": "pre_reset_usd",
                        "run_scope": "verified_preload_checkpoint" if checkpoint_slot else "full_rollout",
                        "checkpoint_requested_slot": checkpoint_slot,
                        "checkpoint_kind": checkpoint_kind if checkpoint_slot else None,
                        "checkpoint_verified": False if checkpoint_slot else None,
                        "checkpoint_setup_actor": "reference_pick_controller" if checkpoint_slot else None,
                        "checkpoint_decision_window_steps": 180 if checkpoint_slot else None,
                        "franka_base_position_m": list(contract.franka_base_position_m),
                        "front_camera": {"translation_m": list(contract.front_camera_position_m),
                                         "orientation_wxyz": list(contract.front_camera_orientation_wxyz),
                                         "focal_length_mm": contract.front_camera_focal_length_mm},
                        "rotor_world_bounds_m": {"min": list(map(float, rotor_bounds.GetMin())),
                                                 "max": list(map(float, rotor_bounds.GetMax()))},
                        "slot_marker_count": slot_marker_count,
                        "tube_marker_count": tube_marker_count,
                        "uses_evaluator_state": uses_evaluator_state,
                        "ranking_eligible": eligible,
                        "ranking_ineligible_reasons": list(ineligible_reasons),
                        "runtime_status": "running"}
        (out / "run.json").write_text(json.dumps(run_manifest, indent=2))
        motion = RMPFlowController(name="diagnostic_rmp", robot_articulation=robot)
        from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats

        def motion_ee_pose():
            joints_now = robot.get_joint_positions()
            return motion.rmp_flow.get_end_effector_pose(np.asarray(joints_now[:7], dtype=float))

        pick = SlowPickController(name="diagnostic_pick", cspace_controller=motion,
                                  events_dt=[.004, .002, .005, .02, .05, .002, .006],
                                  require_pose_reached=True)
        place_motion = FeedbackIKMotion(
            motion.rmp_flow.get_kinematics_solver(), lambda: motion_ee_pose()[0],
            robot.get_joint_positions, rmp_controller=motion,
            gripper_position_provider=robot.get_gripper_position,
            home_joint_provider=lambda: ready_arm_joints)
        run_manifest["robot_dof_names"] = list(robot.dof_names)
        run_manifest["ik_joint_names"] = list(place_motion.solver.get_joint_names())
        run_manifest["arm_joint_limits_rad"] = np.column_stack((
            robot.dof_properties["lower"], robot.dof_properties["upper"]))[:7].tolist()
        gains = robot.get_articulation_controller().get_gains()
        run_manifest["joint_drive_gains"] = [np.asarray(gain).tolist() for gain in gains]
        # The asset's very stiff wrist drives oscillate at the 60 Hz physics
        # rate when IK sends zero-velocity position commands. Use a damped
        # position servo with arm gravity compensation; object dynamics and
        # physical insertion/contact scoring are unaffected.
        servo_kp, servo_kd = [np.asarray(gain, dtype=float).copy() for gain in gains]
        arm_inertia = np.diag(np.asarray(
            robot._articulation_view.get_mass_matrices()[0], dtype=float))[:7]
        # The imported Panda's seventh-axis inertia is almost zero. Model the
        # motor rotor inertia explicitly so PD targets cannot excite a
        # massless wrist at the physics timestep.
        armatures = np.array([[.05, .05, .05, .05, .02, .02, .01, 0., 0.]], dtype=np.float32)
        robot._articulation_view.set_armatures(armatures)
        servo_kp[:7] = [2000., 2000., 2000., 2000., 400., 200., 80.]
        servo_kd[:7] = [20., 20., 20., 20., 4., 4., 2.]
        robot.get_articulation_controller().set_gains(kps=servo_kp, kds=servo_kd)
        robot.disable_gravity()
        run_manifest["body_disable_gravity"] = np.asarray(
            robot._articulation_view.get_body_disable_gravity()).tolist()
        robot.set_solver_position_iteration_count(32)
        robot.set_solver_velocity_iteration_count(8)
        robot._articulation_view.set_max_joint_velocities(
            np.array([[1.0] * 7 + [0.2, 0.2]], dtype=np.float32))
        run_manifest["effective_joint_drive_gains"] = [servo_kp.tolist(), servo_kd.tolist()]
        run_manifest["arm_gravity_compensation"] = True
        run_manifest["arm_initial_joint_inertia"] = arm_inertia.tolist()
        run_manifest["joint_motor_armature_kg_m2"] = armatures[0].tolist()
        run_manifest["physics_substeps_per_control_step"] = physics_substeps
        run_manifest["ready_joint_positions"] = ready_joint_positions.tolist()
        robot.get_articulation_controller().apply_action(ArticulationAction(
            joint_positions=ready_joint_positions, joint_velocities=np.zeros_like(ready_joint_positions)))
        for warmup_step in range(120 * physics_substeps):
            world.step(render=(save_video and warmup_step % physics_substeps == physics_substeps - 1))
        robot_collision_boxes = []
        box_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        calibration_joints = np.asarray(robot.get_joint_positions()[:7], dtype=float)
        solver_frames = place_motion.solver.get_all_frame_names()
        for frame in ("panda_link3", "panda_link4", "panda_link5", "panda_link6",
                      "panda_link7", "panda_hand"):
            if frame not in solver_frames:
                continue
            body = stage.GetPrimAtPath(f"/World/Franka/{frame}")
            body_matrix = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            body_rotation = np.asarray(body_matrix)[:3, :3].T
            body_origin = np.asarray(body_matrix.ExtractTranslation())
            fk_origin, fk_rotation = place_motion.solver.compute_forward_kinematics(frame, calibration_joints)
            correction_r = fk_rotation.T @ body_rotation
            correction_t = fk_rotation.T @ (body_origin - fk_origin)
            for collider in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if not collider.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                bounds = box_cache.ComputeRelativeBound(collider, body).ComputeAlignedBox()
                low, high = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
                if not np.all(np.isfinite(low)) or np.any(high <= low):
                    continue
                robot_collision_boxes.append((frame, (low + high) / 2, (high - low) / 2 + .001,
                                              correction_r, correction_t))
        run_manifest["planning_collision_box_count"] = len(robot_collision_boxes)
        run_manifest["planning_kinematics_frames"] = list(solver_frames)
        if not robot_collision_boxes:
            raise RuntimeError("robot collision geometry missing from insertion planner")
        (out / "run.json").write_text(json.dumps(run_manifest, indent=2))
        # Keep the tube's long axis vertical while grasping. The former
        # [0, 90, 30] wrist pose introduced a large tube tilt after pickup.
        orientation_pick = Rotation.from_euler("xyz", np.radians([0, 180, 0])).as_quat()
        orientation_pick_wxyz = scipy_xyzw_to_lula_wxyz(orientation_pick)
        # Keep the proven pickup wrist pose for transport and insertion; the
        # tube is held upright by the grasp constraint and aligned by its axis.
        orientation_insert = orientation_pick.copy()
        collection_mode = dataset_output is not None and policy_service is None
        place = FeedbackInsertController(place_motion, robot.gripper, orientation_pick,
                                         collection_mode=collection_mode)
        gripper = Gripper()
        recorder = None
        if dataset_output is not None and policy_service is None:
            from data_collectors.release_episode_recorder import ReleaseEpisodeRecorder
            recorder = ReleaseEpisodeRecorder(
                dataset_output, camera_names=("camera_1_rgb", "camera_3_rgb"),
                source_hz=60, sample_hz=10,
                task_name="Place two equal-mass tubes in opposite centrifuge sockets",
                task_properties={"task_version": contract.task_version,
                                 "difficulty": contract.difficulty, "seed": seed},
            )
        if policy_service is not None:
            policy_service.reset(seed)
            if hasattr(policy_service, "session"):
                run_manifest["policy_capability_check"] = {
                    "compatible": policy_service.session.capability_check.compatible,
                    "errors": list(policy_service.session.capability_check.errors),
                    "warnings": list(policy_service.session.capability_check.warnings),
                    "verified_from_server": policy_service.session.capability_check.verified_from_server,
                }
                (out / "run.json").write_text(json.dumps(run_manifest, indent=2))
        active = None
        active_steps = 0
        collection_done = False
        observations = (out / "observations.jsonl").open("w")
        events = (out / "events.jsonl").open("w")
        online_scores = (out / "online_scores.jsonl").open("w")
        video = None
        wrist_video = None
        video_frames = 0
        video_status = "not_requested" if not save_video else "waiting_for_cameras"
        trace = []
        def emit(row):
            trace.append(row)
            events.write(json.dumps(row) + "\n")

        def emit_intervention(step: int, context: dict, opportunity_id: str | None) -> None:
            if intervention.mode != "assist":
                return
            emit({"type": "oracle_intervention", "step": step,
                  "evidence_step": step, "opportunity_id": opportunity_id,
                  "profile": intervention.profile, "mode": intervention.mode,
                  "site": intervention.site, "transport": intervention.transport,
                  "status": "delivered_to_adapter",
                  "oracle_digest": content_digest(context)})

        online_monitor = OnlineEvaluationMonitor(contract, trace, online_scores,
                                                 set(capabilities.reports), events,
                                                 evaluation_version=evaluation_version)
        opportunities = OpportunityBook(contract, selected_track)
        native_action_queue: list[list[float]] = []
        reported_opportunities: set[str] = set()
        checkpoint_setup_steps = 0
        fixed_subgoal_event = None
        fixed_subgoal_delivery_recorded = False
        # Collection is an expert-demonstration run, not a fixed-budget
        # leaderboard rollout.  Permit the caller's larger budget so both
        # long-range tube placements can finish in the hard scene.
        steps = (max_steps if collection_mode and max_steps is not None else
                 min(max_steps if max_steps is not None else contract.max_steps,
                     contract.max_steps))
        initial_heights = {}
        ready_arm_joints = np.asarray(robot.get_joint_positions()[:7], dtype=float).copy()
        branch_options = json.loads(Path(config_path).read_text()).get("labbench_runtime", {})
        branch_budget = branch_options.get("max_action_steps")
        native_action_count = 0
        native_decimation = max(1, round(60 / runtime_profile.action.control_hz))
        # Once a real bilateral contact/lift is observed, keep the tube as a
        # kinematic rigid body coupled to the hand during the long transit.
        # This models a stable grasp constraint, not success promotion:
        # insertion still requires measured socket geometry, support, release
        # and stable dwell. Keep the tube vertical while following the gripper
        # translation; the scene's randomized contact can otherwise tip it
        # before the arm has a chance to align over the rotor.
        kinematic_grasps: dict[str, np.ndarray] = {}
        kinematic_grasp_orientations: dict[str, np.ndarray] = {}
        kinematic_grasp_hand_rotations: dict[str, np.ndarray] = {}

        def set_tube_kinematic(tube: str, enabled: bool) -> None:
            root = stage.GetPrimAtPath(f"/World/{tube}")
            if not root.IsValid():
                return
            # An attached kinematic body must not push back against its own
            # robot. Keep rotor/environment collisions active, and restore
            # robot contact when handing the tube back to rigid-body physics.
            robot_bodies = [
                prim.GetPath() for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Franka"))
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ]
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    pairs = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
                    pairs.SetTargets(robot_bodies if enabled else [])
                    body = UsdPhysics.RigidBodyAPI(prim)
                    body.CreateKinematicEnabledAttr().Set(bool(enabled))
                    if enabled:
                        body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        last_joint_target_delta_max = 0.0
        last_joint_applied_delta_max = 0.0
        # Read-only diagnostics: identify the actual obstructing collider instead
        # of compensating for a collision with increasingly stiff joint drives.
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        contact_pairs = {}

        def record_robot_contacts(headers, data):
            for header in headers:
                pair = tuple(str(PhysicsSchemaTools.intToSdfPath(path))
                             for path in (header.collider0, header.collider1))
                if not any(path.startswith("/World/Franka/") for path in pair):
                    continue
                impulse = sum(float(np.linalg.norm(data[index].impulse)) for index in range(
                    header.contact_data_offset,
                    header.contact_data_offset + header.num_contact_data))
                if impulse > 1e-5:
                    contact_pairs[pair] = {
                        "impulse": impulse,
                        "positions": [list(data[index].position) for index in range(
                            header.contact_data_offset,
                            header.contact_data_offset + min(header.num_contact_data, 3))]}

        contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            record_robot_contacts)
        try:
            frame_event = {"type": "geometry_frame", "step": -1,
                           "slot_mouths_world_m": reader.slots_world,
                           "insertion_axis_world": reader.insertion_axis_world}
            trace.append(frame_event)
            events.write(json.dumps(frame_event) + "\n")
            for step in range(steps):
                for substep in range(physics_substeps):
                    world.step(render=(substep == physics_substeps - 1 and
                                       (save_video or policy_service is not None)))
                joints = robot.get_joint_positions()
                if step % 120 == 0:
                    emit({"type": "robot_contact_diagnostic", "step": step,
                          "contacts": [{"colliders": list(pair), **contact}
                                       for pair, contact in contact_pairs.items()]})
                    contact_pairs.clear()
                if 2000 <= step < 2020:
                    emit({"type": "servo_diagnostic", "step": step,
                          "q": np.asarray(joints[:7]).tolist(),
                          "dq": np.asarray(robot.get_joint_velocities()[:7]).tolist()})
                if joints is None or not np.all(np.isfinite(joints)):
                    row = {"type": "violation", "kind": "runtime_invalid_joint_state", "step": step}
                    events.write(json.dumps(row) + "\n")
                    trace.append(row)
                    online_monitor.snapshot(step, "runtime_invalid_state")
                    break
                # Apply the measured grasp constraint after PhysX advances
                # the frame and before readback.  Doing this after readback
                # let rotor collision resolution push the kinematic tube back
                # out of the socket on the next frame, so the evaluator never
                # observed the commanded insertion depth.
                for held_tube, grasp_offset in list(kinematic_grasps.items()):
                    if state.tube_status[held_tube] != "inserted":
                        hand = np.asarray(robot.get_gripper_position(), dtype=float)
                        _, hand_rotation = motion_ee_pose()
                        rotation_delta = (hand_rotation @
                            kinematic_grasp_hand_rotations[held_tube].T)
                        scene.object_utils.set_object_position(
                            f"/World/{held_tube}", hand - rotation_delta @ grasp_offset)
                        tube_rotation = Rotation.from_quat(
                            kinematic_grasp_orientations[held_tube][[1, 2, 3, 0]]).as_matrix()
                        tube_orientation = Rotation.from_matrix(
                            rotation_delta @ tube_rotation).as_quat()[[3, 0, 1, 2]]
                        scene.object_utils.set_object_orientation(
                            f"/World/{held_tube}", tube_orientation)
                    else:
                        set_tube_kinematic(held_tube, False)
                        kinematic_grasps.pop(held_tube, None)
                        kinematic_grasp_orientations.pop(held_tube, None)
                evidence = {tube: reader.measure(tube) for tube in TUBES}
                if active is not None and active.get("type") == "insert":
                    held_tube = active.get("tube")
                    if held_tube in kinematic_grasps:
                        current_offset = (np.asarray(robot.get_gripper_position(), dtype=float) -
                                          np.asarray(evidence[held_tube]["position_world_m"],
                                                     dtype=float))
                        active["gripper_minus_tube_world_m"] = list(map(float, current_offset))
                        place.grasp_offset = current_offset.copy()
                measurement = {"type": "measurement", "step": step,
                               "tubes": {tube: {"actual_slot": evidence[tube]["actual_slot"],
                                                "stable_sample": evidence[tube]["stable_sample"],
                                                "geometry_pass": evidence[tube]["geometry_pass"],
                                                "released": evidence[tube]["released"],
                                                "supported": evidence[tube]["supported"],
                                                "gripper_contact": evidence[tube]["gripper_contact"],
                                                "left_finger_contact": evidence[tube]["left_finger_contact"],
                                                "right_finger_contact": evidence[tube]["right_finger_contact"],
                                                "both_fingers_contact": evidence[tube]["both_fingers_contact"],
                                                "nearest_slot": evidence[tube]["nearest_slot"],
                                                "radial_error_m": evidence[tube]["radial_error_m"],
                                                "insertion_depth_m": evidence[tube]["insertion_depth_m"],
                                                "axis_error_deg": evidence[tube]["axis_error_deg"],
                                                "position_world_m": evidence[tube]["position_world_m"],
                                                "axis_world": evidence[tube]["axis_world"],
                                                "linear_speed_m_s": evidence[tube]["linear_speed_m_s"],
                                                "position_delta_m": evidence[tube]["position_delta_m"]}
                                         for tube in TUBES},
                               "gripper_width_m": float(sum(joints[-2:])),
                               "finger_joints_m": list(map(float, joints[-2:]))}
                trace.append(measurement)
                events.write(json.dumps(measurement) + "\n")
                if (step > 30 and native and evaluation_version == EVALUATION_VERSION and
                        opportunities.current_id is not None and
                        (checkpoint_slot is None or run_manifest["checkpoint_verified"])):
                    behavior = native_behavior(state, evidence, contract, float(sum(joints[-2:])))
                    if behavior is not None:
                        choice = opportunities.record_choice(behavior, step, "physical_interaction", step)
                        if choice is not None:
                            choice["reference_step"] = step - 1
                            emit(choice)
                            online_monitor.snapshot(step, "native_choice")
                for tube in TUBES:
                    z = evidence[tube]["position_world_m"][2]
                    if step == 30:
                        initial_heights[tube] = z
                    if step > 30 and state.tube_status[tube] == "table" and z - initial_heights[tube] >= .05 and evidence[tube]["gripper_contact"]:
                        state.tube_status[tube] = "held"
                        hand = np.asarray(robot.get_gripper_position(), dtype=float)
                        _, hand_rotation_matrix = place_motion.solver.compute_forward_kinematics(
                            "right_gripper", np.asarray(joints[:7], dtype=float))
                        kinematic_grasps[tube] = (
                            hand - np.asarray(evidence[tube]["position_world_m"], dtype=float))
                        kinematic_grasp_hand_rotations[tube] = hand_rotation_matrix.copy()
                        # Restore the intended upright initial tube attitude
                        # after fingertip collision has completed the grasp.
                        kinematic_grasp_orientations[tube] = np.asarray(
                            [0.0, 1.0, 0.0, 0.0], dtype=float)
                        scene.object_utils.set_object_orientation(
                            f"/World/{tube}", kinematic_grasp_orientations[tube])
                        set_tube_kinematic(tube, True)
                        row = {"type": "physical", "event": "lifted", "tube": tube, "step": step,
                               "time_s": step / 60.0, "contact_pairs": evidence[tube]["contact_pair_count"]}
                        trace.append(row)
                        events.write(json.dumps(row) + "\n")
                        online_monitor.snapshot(step, "physical:lifted")
                    if (not collection_mode and step > 30 and
                            state.tube_status[tube] == "held" and
                            z <= initial_heights[tube] + .025 and not evidence[tube]["gripper_contact"]):
                        state.tube_status[tube] = "table"
                        row = {"type": "physical", "event": "grasp_lost", "tube": tube,
                               "step": step, "time_s": step / 60.0,
                               "position_world_m": evidence[tube]["position_world_m"]}
                        trace.append(row)
                        events.write(json.dumps(row) + "\n")
                        online_monitor.snapshot(step, "physical:grasp_lost")
                        if active is not None and active.get("tube") == tube:
                            active = None
                # Collection and evaluation use the same measured physical
                # state machine.  A controller handoff alone is not an
                # insertion; the tube must satisfy socket geometry, release,
                # support and the required stable dwell.
                new_events = advance_state(contract, state, evidence)
                for new in new_events:
                    selected = state.selected_slot.get(new["tube"])
                    actual = new["slot"]
                    precision_pass = (new["type"] == "inserted" and
                                      insertion_matches_commit(contract, new["tube"], selected, actual) and
                                      (new["tube"] == "NMR_01" or
                                       (state.first_actual_slot in SLOTS and
                                        actual == contract.opposite(state.first_actual_slot))))
                    row = {"type": "physical", "event": new["type"], "tube": new["tube"],
                           "slot": actual, "committed_slot": selected,
                           "execution_deviation": selected is not None and selected != actual,
                           "precision_pass": precision_pass, "step": step, "time_s": step / 60.0,
                           "evidence": evidence[new["tube"]]}
                    trace.append(row)
                    events.write(json.dumps(row) + "\n")
                    online_monitor.snapshot(step, f"physical:{new['type']}")
                    # A measured insertion supersedes any stale retry of the
                    # same operation.  Without clearing it here, a previous
                    # timeout could keep driving NMR_01 after it had already
                    # become physically inserted, preventing the legal policy
                    # sequence from advancing to NMR_02.
                    if (new["type"] == "inserted" and active is not None and
                            active.get("tube") == new["tube"]):
                        # Keep the operation alive only for the physical
                        # retreat.  Clearing it in-place leaves the gripper
                        # over the released tube and can cause a measured
                        # insertion_lost on the next frame.
                        place.phase = "retreat"
                        place.phase_steps = 0
                if step % 180 == 0:
                    online_monitor.snapshot(step, "periodic")
                if step % 6 == 0:
                    front_frame = camera_rgb(scene.cameras[0].get_rgba()) if save_video else None
                    wrist_frame = camera_rgb(scene.cameras[-1].get_rgba()) if save_video else None
                    frames_ready = front_frame is not None and wrist_frame is not None
                    if collection_mode and frames_ready and step % 600 == 0:
                        Image.fromarray(front_frame).save(out / "artifacts/diagnostic_latest_front.png")
                        Image.fromarray(wrist_frame).save(out / "artifacts/diagnostic_latest_wrist.png")
                    images = ({"front": {"artifact": "artifacts/front.mp4", "frame_index": video_frames},
                               "wrist": {"artifact": "artifacts/wrist.mp4", "frame_index": video_frames}}
                              if frames_ready else {})
                    policy_view = policy_observation(
                        images, list(map(float, joints)),
                        "Place two equal-mass tubes in opposite centrifuge sockets. NMR_01 has a teal band; NMR_02 has an amber band. Stop only after both are stable.",
                        list(SLOTS))
                    check_policy_request(policy_view)
                    row = {"step": step, "policy_observation": policy_view,
                           "oracle_context": {},
                           "evaluator_state": {"physics": evidence, "task_state": state.as_dict()},
                           "active_operation": active,
                           "joint_target_delta_max_rad": last_joint_target_delta_max,
                           "joint_applied_delta_max_rad": last_joint_applied_delta_max,
                           "arm_joint_velocities_rad_s": np.asarray(robot.get_joint_velocities()[:7]).tolist(),
                           "arm_measured_efforts_nm": np.asarray(robot.get_measured_joint_efforts()[:7]).tolist(),
                           "finger_contacts": {tube: {"left": evidence[tube]["left_finger_contact"],
                                                      "right": evidence[tube]["right_finger_contact"]}
                                               for tube in TUBES},
                           "controller_phase": pick._event if active and active["type"] == "pick" else
                                               (place.phase if active and active["type"] == "insert" else None),
                           "insertion_ee_goal_world_m": place.goal.tolist() if active and active["type"] == "insert" else None,
                           "insertion_phase_goal_world_m": (place.last_phase_goal.tolist()
                               if active and active["type"] == "insert" and
                               place.last_phase_goal is not None else None),
                           "insertion_transit_route_stage": (place.transit_route_stage
                               if active and active["type"] == "insert" else None),
                           "insertion_transit_clearance_x_m": (place.transit_clearance_x
                               if active and active["type"] == "insert" else None),
                           "insertion_ee_waypoint_world_m": (place_motion._free_space_waypoint.tolist()
                               if active and active["type"] == "insert" and
                               place_motion._free_space_waypoint is not None else None),
                           "insertion_safe_z_m": place.safe_z if active and active["type"] == "insert" else None,
                           "insertion_ik_status": place_motion.status if active and active["type"] == "insert" else None,
                           "insertion_held_ik_joint_target": (
                               place_motion._position_ik_solution.tolist()
                               if place_motion._position_ik_solution is not None else None),
                           "insertion_ik_raw_joint_delta_max_rad": place_motion.raw_joint_delta_max
                           if active and active["type"] == "insert" else None,
                           "insertion_motion_target_world_m": (
                               place_motion.last_target_position.tolist()
                               if active and active["type"] == "insert" and
                               place_motion.last_target_position is not None else None),
                           "insertion_motion_fk_world_m": (
                               place_motion.last_current_position.tolist()
                               if active and active["type"] == "insert" and
                               place_motion.last_current_position is not None else None),
                           "gripper_position_world_m": list(map(float, robot.get_gripper_position()))}
                    row["motion_ee_position_world_m"] = list(map(float, motion_ee_pose()[0]))
                    row["ik_fk_position_world_m"] = list(map(float,
                        place_motion.solver.compute_forward_kinematics(
                            "right_gripper", np.asarray(joints[:7], dtype=float))[0]))
                    observations.write(json.dumps(row) + "\n")
                    if frames_ready:
                        if video is None:
                            video = imageio.get_writer(str(out / "artifacts/front.mp4"), fps=10)
                            wrist_video = imageio.get_writer(str(out / "artifacts/wrist.mp4"), fps=10)
                        video.append_data(front_frame)
                        wrist_video.append_data(wrist_frame)
                        video_frames += 1
                        video_status = "recording"
                if step < 30:
                    continue
                if checkpoint_slot is not None and not run_manifest["checkpoint_verified"]:
                    if state.first_actual_slot != checkpoint_slot and step >= 120:
                        raise RuntimeError("preloaded checkpoint did not reach measured stable insertion")
                    if state.first_actual_slot != checkpoint_slot:
                        continue
                    if state.tube_status["NMR_02"] != "held":
                        checkpoint_setup_steps += 1
                        if checkpoint_setup_steps > 1300:
                            raise RuntimeError("checkpoint setup did not physically lift NMR_02")
                        setup_action = pick.forward(
                            picking_position=np.array(evidence["NMR_02"]["position_world_m"]),
                            current_joint_positions=joints, object_name="NMR_02",
                            object_size=np.array([.015, .015, .075]),
                            gripper_control=gripper,
                            gripper_position=np.asarray(robot.get_gripper_position()),
                            end_effector_orientation=orientation_pick_wxyz,
                            pre_offset_x=.03, pre_offset_z=.1, after_offset_z=.15,
                            gripper_distances=.001)
                        if setup_action is not None:
                            if setup_action.joint_positions is not None:
                                setup_action.joint_positions = limit_joint_targets(
                                    list(map(float, joints)), list(setup_action.joint_positions),
                                    None if setup_action.joint_indices is None else list(setup_action.joint_indices))
                            robot.get_articulation_controller().apply_action(setup_action)
                        continue
                    run_manifest["checkpoint_verified"] = True
                    run_manifest["checkpoint_evidence_step"] = step
                    (out / "run.json").write_text(json.dumps(run_manifest, indent=2))
                    if checkpoint_kind == "execution":
                        target = contract.opposite(state.first_actual_slot)
                        fixed_subgoal_event = {
                            "type": "fixed_subgoal", "step": step, "evidence_step": step,
                            "action": {"type": "insert", "tube": "NMR_02", "slot": target},
                            "interaction_pose": {"slot": target, "frame": "world", "units": "m",
                                                 "mouth_world_m": list(reader.slots_world[target]),
                                                 "axis_world": list(reader.insertion_axis_world)}}
                        emit(fixed_subgoal_event)
                        online_monitor.snapshot(step, "fixed_subgoal_selected")
                if evaluation_version == EVALUATION_VERSION:
                    if native and state.terminal_steps >= contract.final_stable_steps:
                        break
                    opportunity = (opportunities.update(state, step, evidence_step=step)
                                   if native or active is None else None)
                    if opportunity is not None:
                        emit(opportunity)
                        if native_action_queue:
                            emit({"type": "queued_actions_invalidated", "step": step,
                                  "opportunity_id": opportunities.current_id,
                                  "discarded_action_count": len(native_action_queue),
                                  "reason": "new_evidence"})
                        native_action_queue.clear()
                        online_monitor.snapshot(step, "opportunity_open")
                if native:
                    if step % native_decimation:
                        continue
                    if branch_budget is not None and native_action_count >= branch_budget:
                        emit({"type": "action_budget_exhausted", "step": step, "action_count": native_action_count})
                        break
                    if not native_action_queue:
                        encoded_images = {}
                        for name, camera in (("front", scene.cameras[0]), ("wrist", scene.cameras[-1])):
                            rgb = camera_rgb(camera.get_rgba())
                            if rgb is None:
                                raise PolicyServiceError(f"{name} camera is not ready at inference step {step}")
                            buffer = BytesIO()
                            Image.fromarray(rgb).save(buffer, format="PNG")
                            encoded_images[name] = {"mime_type": "image/png",
                                                    "base64": base64.b64encode(buffer.getvalue()).decode("ascii")}
                        native_view = policy_observation(
                            encoded_images, list(map(float, joints)), policy_view["task_rule"], list(SLOTS))
                        visible_occupancy = {slot: tube for slot, tube in state.occupied.items()
                                             if slot in evaluation_config.get("semantic_visible_slots", [])}
                        candidate_frames = {slot: {"mouth_world_m": list(reader.slots_world[slot]),
                                                   "axis_world": list(reader.insertion_axis_world)} for slot in SLOTS}
                        context = oracle_context(profile, contract, state, visible_occupancy, candidate_frames)
                        request = native_view if profile == "none" else deliver_input(
                            native_view, context, transport, set(capabilities.oracle_transports))
                        emit_intervention(step, context, opportunities.current_id)
                        if checkpoint_kind == "execution":
                            request["selected_operation"] = fixed_subgoal_event["action"]
                            request["interaction_pose"] = fixed_subgoal_event["interaction_pose"]
                        check_policy_request(request, allow_fixed_subgoal=checkpoint_kind == "execution")
                        inference = policy_service.infer_full(request)
                        native_action_queue = inference["action_chunk"]
                        if inference.get("input_delivery_audit"):
                            emit({"type": "input_delivery_audit", "step": step,
                                  "opportunity_id": opportunities.current_id,
                                  **inference["input_delivery_audit"]})
                        if capabilities.reports and opportunities.current_id not in reported_opportunities:
                            report = {key: inference[key] for key in
                                      ("semantic_report", "committed_action", "geometry_report")
                                      if key in inference}
                            try:
                                json.dumps(report, allow_nan=False)
                            except (TypeError, ValueError) as exc:
                                raise PolicyServiceError("native policy report must be finite JSON") from exc
                            emit({"type": "policy_report", "step": step,
                                  "evidence_step": step,
                                  "opportunity_id": opportunities.current_id,
                                  "visible_slots": list(evaluation_config.get("semantic_visible_slots", [])),
                                  "report": report})
                            reported_opportunities.add(opportunities.current_id)
                            online_monitor.snapshot(step, "policy_report")
                        emit({"type": "policy_inference", "step": step,
                              "opportunity_id": opportunities.current_id,
                              "oracle_profile": profile, "action_count": len(native_action_queue),
                              "server_timing": inference.get("server_timing"),
                              "policy_timing": inference.get("policy_timing"),
                              "policy_request": request})
                        if checkpoint_kind == "execution" and not fixed_subgoal_delivery_recorded:
                            online_monitor.snapshot(step, "fixed_subgoal_delivered")
                            fixed_subgoal_delivery_recorded = True
                    waypoint = native_action_queue.pop(0)
                    native_action_count += 1
                    from eai4s_eval.special_policy import joint_waypoint
                    waypoint = joint_waypoint(robot, waypoint, branch_options)
                    try:
                        dof_properties = robot.dof_properties
                        waypoint[:7] = wrap_joint_targets_near_current(
                            list(map(float, joints)), waypoint[:7],
                            list(map(float, dof_properties["lower"])),
                            list(map(float, dof_properties["upper"])), list(range(7)))
                        last_joint_target_delta_max = inspect_joint_targets(
                            list(map(float, joints)), waypoint[:7], list(range(7)))
                        limited = limit_joint_targets(list(map(float, joints)), waypoint[:7],
                                                      list(range(7)))
                        last_joint_applied_delta_max = inspect_joint_targets(
                            list(map(float, joints)), limited, list(range(7)))
                    except ValueError as exc:
                        emit({"type": "violation", "kind": "runtime_invalid_joint_target",
                              "step": step, "reason": str(exc)})
                        online_monitor.snapshot(step, "runtime_invalid_joint_target")
                        break
                    robot.get_articulation_controller().apply_action(
                        ArticulationAction(joint_positions=np.asarray(limited + [waypoint[7]] * 2)))
                    continue
                if (evaluation_version == EVALUATION_VERSION and active is None and
                        opportunities.first_choice_recorded):
                    continue
                if active is None:
                    choices = legal_actions(contract, state)
                    if (collection_mode and state.first_actual_slot is None and
                            any(item.get("type") == "insert" and
                                item.get("tube") == "NMR_01" for item in choices)):
                        # Prefer an approach-facing rotor socket over simply
                        # choosing the nearest Euclidean pair. The latter can
                        # send the held tube straight into the centrifuge's
                        # front housing before it reaches the opening.
                        base_xy = np.asarray(contract.franka_base_position_m[:2], dtype=float)
                        source_xy = evidence["NMR_01"]["position_world_m"][:2]
                        preferred_slot = choose_accessible_first_slot(
                            reader.slots_world, base_xy.tolist(), source_xy)
                        choices.sort(key=lambda item: (
                            item.get("slot") != preferred_slot, item.get("slot", "")))
                    if policy_service is None:
                        chosen = choices[0] if choices else {"type": "wait"}
                        response = {"committed_action": chosen}
                        decision_view = policy_view
                    else:
                        encoded_images = {}
                        for name, camera in (("front", scene.cameras[0]), ("wrist", scene.cameras[-1])):
                            rgb = camera_rgb(camera.get_rgba())
                            if rgb is None:
                                raise PolicyServiceError(f"{name} camera is not ready at decision step {step}")
                            buffer = BytesIO()
                            Image.fromarray(rgb).save(buffer, format="PNG")
                            encoded_images[name] = {"mime_type": "image/png",
                                                    "base64": base64.b64encode(buffer.getvalue()).decode("ascii")}
                        decision_view = policy_observation(
                            encoded_images, list(map(float, joints)), policy_view["task_rule"], list(SLOTS))
                    visible_occupancy = {slot: tube for slot, tube in state.occupied.items()
                                         if slot in evaluation_config.get("semantic_visible_slots", [])}
                    candidate_frames = {slot: {"mouth_world_m": list(reader.slots_world[slot]),
                                               "axis_world": list(reader.insertion_axis_world)} for slot in SLOTS}
                    context = oracle_context(profile, contract, state, visible_occupancy, candidate_frames)
                    request = (decision_view if profile == "none" else deliver_input(
                        decision_view, context, transport, set(capabilities.oracle_transports)))
                    emit_intervention(step, context, opportunities.current_id)
                    check_policy_request(request)
                    if policy_service is not None:
                        response = policy_service.decide(request)
                    chosen = normalize_action(response.get("committed_action"))
                    row = {"type": "decision", "action": chosen, "legal_actions": choices, "step": step,
                           "supported_reports": sorted(capabilities.reports),
                           "oracle_profile": profile, "policy_request": request,
                           "policy_response": response}
                    trace.append(row)
                    events.write(json.dumps(row) + "\n")
                    if evaluation_version == EVALUATION_VERSION and chosen is not None:
                        choice = opportunities.record_choice(chosen, step, "explicit_commitment", step)
                        if choice is not None:
                            emit(choice)
                    reference_geometry = (dict(candidate_frames[chosen["slot"]], slot=chosen["slot"])
                                          if chosen is not None and chosen.get("type") == "insert" and
                                          chosen.get("slot") in candidate_frames else None)
                    checkpoint_record = {
                        "instance_id": f"{contract.difficulty}-{seed}-{step}",
                        "supported_reports": sorted(capabilities.reports - {"execution"}),
                        "expected_semantic": visible_occupancy,
                        "semantic_report": response.get("semantic_report"),
                        "legal_actions": choices,
                        "committed_action": chosen,
                        "geometry_applicable": chosen is not None and chosen.get("type") == "insert",
                        "reference_geometry": reference_geometry,
                        "geometry_report": response.get("geometry_report"),
                    }
                    checkpoint_event = {"type": "checkpoint", "step": step,
                                        "scores": score_checkpoint(checkpoint_record)}
                    trace.append(checkpoint_event)
                    events.write(json.dumps(checkpoint_event) + "\n")
                    online_monitor.snapshot(step, "decision")
                    if chosen is None or (chosen.get("tube") not in TUBES and chosen["type"] in {"pick", "insert"}) or (
                            chosen["type"] == "insert" and chosen.get("slot") not in SLOTS):
                        row = {"type": "violation", "kind": "invalid_policy_action", "step": step}
                        trace.append(row)
                        events.write(json.dumps(row) + "\n")
                        online_monitor.snapshot(step, "invalid_policy_action")
                        break
                    active = dict(chosen)
                    active_steps = 0
                    if chosen["type"] == "pick":
                        pick.reset()
                        if chosen.get("tube") == "NMR_02":
                            active["recover_arm_posture"] = True
                            # Route the second grasp through a clear waypoint
                            # instead of sweeping the seated NMR_01.  Give the
                            # approach phases enough physical time to reach
                            # the waypoint and the table tube.
                            pick.reset_phase_state(
                                events_dt=[.0013, .0013, .005, .02, .05, .002, .006],
                                initial_pregrasp_target=np.array([0.15, -0.05, 0.95], dtype=float))
                    if chosen["type"] == "insert":
                        state.selected_slot[chosen["tube"]] = chosen["slot"]
                        plan_mouth = reader.slots_world[chosen["slot"]]
                        plan_axis = reader.insertion_axis_world
                        if policy_service is not None and "geometry" in capabilities.reports:
                            try:
                                plan_mouth, plan_axis = validated_geometry_report(
                                    response.get("geometry_report"), chosen["slot"])
                            except ValueError as exc:
                                row = {"type": "violation", "kind": "invalid_geometry_report",
                                       "step": step, "reason": str(exc)}
                                trace.append(row)
                                events.write(json.dumps(row) + "\n")
                                online_monitor.snapshot(step, "invalid_geometry_report")
                                break
                        motion_position, motion_rotation = motion_ee_pose()
                        # The kinematic grasp follows robot.get_gripper_position()
                        # below; use the same frame for the committed offset so
                        # insertion XY does not inherit the ~3 mm motion-pose
                        # versus physical-gripper discrepancy.
                        grasp_position = np.asarray(robot.get_gripper_position(), dtype=float)
                        active["gripper_minus_tube_world_m"] = list(map(float,
                            grasp_position -
                            np.asarray(evidence[chosen["tube"]]["position_world_m"])))
                        active["both_fingers_contact_at_commit"] = evidence[chosen["tube"]]["both_fingers_contact"]
                        place.orientation = orientation_insert
                        place_goal = np.asarray(gripper_target_for_insertion(
                            plan_mouth, plan_axis,
                            .20 * contract.socket_depth_m,
                            active["gripper_minus_tube_world_m"], 0.0), dtype=float)
                        place_goal[2] -= .080
                        place.reset(place_goal, grasp_position, rotor_top_z + .05,
                                    plan_mouth, plan_axis,
                                    active["gripper_minus_tube_world_m"],
                                    # Clear the whole hand/wrist envelope, not
                                    # only the held tube's small radius.
                                    transit_clearance_x=min(float(rotor_bounds.GetMin()[0]) - .15,
                                                            float(contract.franka_base_position_m[0]) - .35))
                active_steps += 1
                kind = active["type"]
                if kind == "stop":
                    break
                if kind == "wait":
                    if active_steps >= 30:
                        active = None
                    continue
                tube = active["tube"]
                gripper_position = np.asarray(robot.get_gripper_position())
                if kind == "pick":
                    pick_position = np.array(evidence[tube]["position_world_m"], dtype=float)
                    recovering = bool(active.get("recover_arm_posture"))
                    if recovering and np.max(np.abs(np.asarray(joints[:7]) - ready_arm_joints)) > .03:
                        # First insertion can end on a wrist/elbow joint limit.
                        # Return the released arm to its initial waiting posture
                        # through bounded motor commands before the second grasp.
                        target = np.asarray(joints[:7]) + np.clip(
                            ready_arm_joints - np.asarray(joints[:7]), -.025, .025)
                        action = ArticulationAction(joint_positions=np.r_[target, .04, .04])
                        done = False
                    else:
                        if recovering:
                            active["recover_arm_posture"] = False
                            motion.reset()
                        action = pick.forward(
                            picking_position=pick_position,
                            current_joint_positions=joints, object_name=tube,
                            object_size=np.array([.015, .015, .075]),
                            gripper_control=gripper, gripper_position=gripper_position,
                            end_effector_orientation=orientation_pick,
                            pre_offset_x=.03, pre_offset_z=.1, after_offset_z=.15,
                            pick_z_offset=.060,
                            gripper_distances=.001,
                        )
                        done = pick.is_done()
                else:
                    action, done = place.forward(
                        robot.get_gripper_position(), evidence[tube], active["slot"], state)
                    # The kinematic grasp is only a transport constraint. As
                    # soon as the inserter reaches its measured descent
                    # phase, return the tube to ordinary rigid-body physics
                    # so socket support/contact remains genuine.
                    handoff_ready = (
                        place.phase == "descend"
                        and evidence[tube]["radial_error_m"] <= .0005
                        and evidence[tube]["axis_error_deg"] <= 1.0
                        and contract.socket_depth_m * contract.insertion_depth_fraction
                        <= evidence[tube]["insertion_depth_m"] <= contract.socket_depth_m)
                    if handoff_ready and tube in kinematic_grasps:
                        # Opening must precede restoring robot-object contact.
                        # The transport constraint allows closed fingers to
                        # overlap the tube; re-enabling contact first ejects it
                        # from an otherwise correctly reached socket.
                        place.phase, place.phase_steps = "release", 0
                    release_width = float(np.sum(np.asarray(robot.get_joint_positions())[7:9]))
                    if (tube in kinematic_grasps and
                            place.phase in {"release", "retreat"} and release_width >= .03):
                        set_tube_kinematic(tube, False)
                        kinematic_grasps.pop(tube, None)
                        emit({"type": "physical_grasp_handoff", "step": step, "tube": tube,
                              "gripper_width_m": release_width,
                              "insertion_depth_m": evidence[tube]["insertion_depth_m"]})
                if action is not None:
                    if action.joint_positions is not None:
                        try:
                            dof_properties = robot.dof_properties
                            action.joint_positions = wrap_joint_targets_near_current(
                                list(map(float, joints)), list(action.joint_positions),
                                list(map(float, dof_properties["lower"])),
                                list(map(float, dof_properties["upper"])),
                                None if action.joint_indices is None else
                                list(map(int, action.joint_indices)))
                            last_joint_target_delta_max = inspect_joint_targets(
                                list(map(float, joints)), list(action.joint_positions),
                                None if action.joint_indices is None else list(action.joint_indices))
                            action.joint_positions = limit_joint_targets(
                                list(map(float, joints)), list(action.joint_positions),
                                None if action.joint_indices is None else list(action.joint_indices))
                            action_indices = (list(range(len(action.joint_positions)))
                                              if action.joint_indices is None else
                                              list(map(int, action.joint_indices)))
                            arm_limits = np.column_stack((dof_properties["lower"],
                                                          dof_properties["upper"]))[:7]
                            # Clamp bounded revolute joints only. Continuous
                            # joints were already wrapped to the nearest
                            # equivalent angle; clamping them to an arbitrary
                            # +/-2pi interval would reintroduce a huge jump.
                            bounded_positions = list(action.joint_positions)
                            for offset, joint_index in enumerate(action_indices):
                                if joint_index < 7 and bounded_positions[offset] is not None:
                                    lower, upper = arm_limits[joint_index]
                                    if upper - lower < 2.0 * np.pi - 1e-3:
                                        bounded_positions[offset] = float(np.clip(
                                            bounded_positions[offset], lower + .005,
                                            upper - .005))
                            # Absolute limit clipping above can move a target
                            # far from an already out-of-range continuous-joint
                            # state. Reapply the slew limit last so the actual
                            # command sent to PhysX is always bounded.
                            bounded_positions = limit_joint_targets(
                                list(map(float, joints)), bounded_positions,
                                action_indices)
                            action.joint_positions = bounded_positions
                            last_joint_applied_delta_max = inspect_joint_targets(
                                list(map(float, joints)), list(action.joint_positions),
                                None if action.joint_indices is None else list(action.joint_indices))
                            if action.joint_velocities is not None:
                                velocity_indices = (list(range(len(action.joint_velocities)))
                                                    if action.joint_indices is None else
                                                    list(map(int, action.joint_indices)))
                                action.joint_velocities = np.asarray(limit_joint_velocities(
                                    list(map(float, joints)),
                                    list(map(float, action.joint_velocities)),
                                    velocity_indices), dtype=float)
                        except ValueError as exc:
                            row = {"type": "violation", "kind": "runtime_invalid_joint_target",
                                   "step": step, "reason": str(exc)}
                            trace.append(row)
                            events.write(json.dumps(row) + "\n")
                            online_monitor.snapshot(step, "runtime_invalid_joint_target")
                            break
                    robot.get_articulation_controller().apply_action(action)
                    if recorder is not None and policy_service is None:
                        if not save_video:
                            world.render()
                        tcp_position, tcp_matrix = motion_ee_pose()
                        recorder.record(
                            cameras={"front": camera_rgb(scene.cameras[0].get_rgba()),
                                     "wrist": camera_rgb(scene.cameras[-1].get_rgba())},
                            joints=joints,
                            tcp_pose_xyzw=[*tcp_position, *Rotation.from_matrix(tcp_matrix).as_quat()],
                            command=(list(action.joint_positions)
                                     if action.joint_positions is not None else joints),
                            sim_step=step,
                            subtask=f"{kind}_{place.phase if kind == 'insert' else pick._event}",
                        )
                # Long cross-rack transit can exceed the evaluation action
                # budget during a recorded demonstration. Keep benchmark
                # rollouts bounded; collection gets a larger insert budget.
                operation_limit = operation_step_limit(
                    kind, tube, collection_mode=collection_mode)
                if done or active_steps >= operation_limit:
                    operation_failed = ((kind == "pick" and state.tube_status[tube] != "held") or
                                        (kind == "insert" and state.tube_status[tube] != "inserted"))
                    collection_pick_promoted = False
                    if kind == "pick" and state.tube_status[tube] != "held":
                        row = {"type": "physical", "event": "grasp_attempt_failed", "tube": tube,
                               "step": step, "time_s": step / 60.0,
                               "reason": "controller_done_without_lift" if done else "operation_timeout"}
                        trace.append(row)
                        events.write(json.dumps(row) + "\n")
                        online_monitor.snapshot(step, "physical:grasp_attempt_failed")
                    if kind == "insert" and place.phase == "failed":
                        row = {"type": "violation", "kind": "insertion_not_stable", "operation": active,
                               "step": step}
                        trace.append(row)
                        events.write(json.dumps(row) + "\n")
                        online_monitor.snapshot(step, "insertion_not_stable")
                    if active_steps >= operation_limit:
                        row = {"type": "violation", "kind": "operation_timeout", "operation": active, "step": step}
                        trace.append(row)
                        events.write(json.dumps(row) + "\n")
                        online_monitor.snapshot(step, "operation_timeout")
                    active = None
                    if evaluation_version == EVALUATION_VERSION and operation_failed:
                        emit(opportunities.retry_after_failure(state, step, evidence_step=step))
                        online_monitor.snapshot(step, "observable_operation_failure")
                if collection_done:
                    state.terminal_steps = contract.final_stable_steps
                    stop_row = {"type": "decision", "action": {"type": "stop"},
                                "legal_actions": [{"type": "stop"}],
                                "step": step + 1, "supported_reports": sorted(capabilities.reports),
                                "policy_response": {"committed_action": {"type": "stop"}}}
                    trace.append(stop_row)
                    events.write(json.dumps(stop_row) + "\n")
                    break
            snapshot = {"type": "snapshot", "terminal_steps": state.terminal_steps,
                        "task_state": state.as_dict(), "step": step}
            trace.append(snapshot)
            events.write(json.dumps(snapshot) + "\n")
            scores = online_monitor.snapshot(step, "terminal")
            run_manifest.update(
                evaluation_valid=bool(scores.get("evaluation_valid")),
                online_legal_next_action_rate=scores.get("online_legal_next_action_rate"),
                physical_completion=float(bool(scores.get("physical_completion"))),
            )
            (out / "scores.json").write_text(json.dumps(scores, indent=2))
            if collection_mode:
                (out / "collection_result.json").write_text(json.dumps({
                    "status": "completed" if collection_done else "incomplete",
                    "reference_controller_completed": bool(collection_done),
                    "strict_evaluation_valid_success": bool(scores.get("valid_success")),
                    "dataset_output": str(Path(dataset_output).resolve()) if dataset_output else None,
                }, indent=2))
            completed = scores.get("task_completion", scores.get("valid_success", False))
            run_manifest["runtime_status"] = "finished" if completed else (
                "time_limit" if step + 1 == steps else "stopped_after_violation")
            run_manifest["valid_success"] = completed
            run_manifest["steps_executed"] = step + 1
            run_manifest["online_evaluation_samples"] = online_monitor.sample_count
            run_manifest["video_frames"] = video_frames
            run_manifest["video_status"] = "complete" if video_frames else video_status
            if recorder is not None and policy_service is None:
                tcp_position, tcp_matrix = motion_ee_pose()
                if scores.get("valid_success") and collection_done:
                    recorder.collector.set_task_properties({
                        "task_version": contract.task_version, "seed": seed,
                        "source_commit": "3d828af", "physical_success": True,
                        "valid_success": True})
                    recorder.finish(
                        joints=robot.get_joint_positions(),
                        tcp_pose_xyzw=[*tcp_position, *Rotation.from_matrix(tcp_matrix).as_quat()],
                    )
                else:
                    recorder.abort()
            (out / "run.json").write_text(json.dumps(run_manifest, indent=2))
            return scores
        except Exception as exc:
            traceback.print_exc()
            if recorder is not None:
                try:
                    recorder.abort()
                except Exception:
                    pass
            run_manifest["runtime_status"] = "service_error" if isinstance(exc, PolicyServiceError) else "error"
            run_manifest["error_type"] = type(exc).__name__
            run_manifest["error_message"] = str(exc)
            snapshot = {"type": "snapshot", "terminal_steps": state.terminal_steps,
                        "task_state": state.as_dict(), "step": locals().get("step", -1)}
            trace.append(snapshot)
            events.write(json.dumps(snapshot) + "\n")
            (out / "scores.json").write_text(json.dumps(
                online_monitor.snapshot(snapshot["step"], "error"), indent=2))
            run_manifest["online_evaluation_samples"] = online_monitor.sample_count
            run_manifest["video_frames"] = video_frames
            run_manifest["video_status"] = "partial" if video_frames else video_status
            (out / "run.json").write_text(json.dumps(run_manifest, indent=2))
            raise
        finally:
            observations.close()
            events.close()
            online_scores.close()
            close_errors = []
            for name, writer in (("front", video), ("wrist", wrist_video)):
                if writer is not None:
                    try:
                        writer.close()
                    except Exception as exc:
                        close_errors.append({"stream": name, "error": str(exc)})
            if recorder is not None:
                try:
                    recorder.close()
                except Exception as exc:
                    close_errors.append({"stream": "dataset", "error": str(exc)})
            if close_errors:
                run_manifest["video_close_errors"] = close_errors
            run_manifest["video_close"] = {"status": "error" if close_errors else "closed",
                                           "streams": [name for name, writer in (("front", video), ("wrist", wrist_video))
                                                       if writer is not None]}
            (out / "run.json").write_text(json.dumps(run_manifest, indent=2))
    except Exception as exc:
        manifest = json.loads((out / "run.json").read_text())
        if manifest.get("runtime_status") not in {"error", "service_error", "time_limit", "finished"}:
            manifest.update(runtime_status="error", error_type=type(exc).__name__, error_message=str(exc))
            (out / "run.json").write_text(json.dumps(manifest, indent=2))
            (out / "scores.json").write_text(json.dumps(
                score_trace(contract, locals().get("trace", []), set(capabilities.reports)), indent=2))
        raise
    finally:
        close_started = time.monotonic()
        if "out" in locals() and (out / "run.json").exists():
            manifest = json.loads((out / "run.json").read_text())
            manifest["simulation_app_close"] = {"status": "requested"}
            (out / "run.json").write_text(json.dumps(manifest, indent=2))
        try:
            app.close(skip_cleanup=True)
        except Exception as exc:
            close_result = {"status": "error", "error": str(exc)}
        else:
            close_result = {"status": "closed"}
        close_result["duration_s"] = round(time.monotonic() - close_started, 3)
        if "out" in locals() and (out / "run.json").exists():
            manifest = json.loads((out / "run.json").read_text())
            manifest["simulation_app_close"] = close_result
            (out / "run.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--save-video", action="store_true")
    args = p.parse_args()
    print(json.dumps(run(args.config, args.output, args.seed, args.max_steps, args.save_video), indent=2))


if __name__ == "__main__":
    main()
