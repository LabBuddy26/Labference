"""Adapt the current branch's model observation and EE-action contract to release scenes."""
from __future__ import annotations

import base64
from dataclasses import replace
from io import BytesIO

import numpy as np
from PIL import Image


class BranchArrayPolicy:
    def __init__(self, host, port, api_key=None, *, profile, intervention, options):
        from eai4s_eval.policy import JointPolicyAdapter, load_openpi_profile
        from eai4s_eval.oracle_runtime.session import PolicySession

        branch_profile = load_openpi_profile(options.get(
            "policy_profile", "eai4s_eval/configs/policies/openpi_franka_ee_pos_v1.yaml"))
        mapping = {source: destination for source, destination in branch_profile.observation_map.items()}
        if "display" in profile.sensors.rgb_channels:
            mapping.update(display_rgb="observation/display_image", weights_rgb="observation/weights_image")
        branch_profile = replace(branch_profile, observation_map=mapping,
                                 control_hz=profile.action.control_hz,
                                 action_space=options.get("action_space", "ee_pos"))
        self.adapter = JointPolicyAdapter(host=host, port=port, api_key=api_key, profile=branch_profile,
                                         backend=options.get("backend", "websocket"))
        self.profile = profile
        self.server_metadata = self.adapter.server_metadata
        self.session = PolicySession(profile, intervention, server_metadata=self.server_metadata)
        self.horizon = options.get("execution_horizon")

    def reset(self, seed):
        self.adapter.reset()
        self.session.reset(seed)

    def infer_full(self, request):
        mapping = {"front": "camera_1_rgb", "wrist": "camera_3_rgb",
                   "display": "display_rgb", "weights": "weights_rgb"}
        frames = {}
        for key, image in request["images"].items():
            if key in mapping:
                frames[mapping[key]] = np.asarray(Image.open(BytesIO(base64.b64decode(
                    image if isinstance(image, str) else image["base64"])))).copy()
        chunk = self.adapter.predict({"camera_data": frames,
                                      "joint_positions": request["joint_positions"],
                                      "language_instruction": request["task_rule"]})
        actions = chunk.joint_positions[:self.horizon] if self.horizon else chunk.joint_positions
        return {"action_chunk": actions.tolist(), "policy_timing": {"latency_ms": chunk.latency_ms}}


def openpi_policy(base_class, host, port, api_key=None, *, profile, intervention, options=None):
    if options:
        return BranchArrayPolicy(host, port, api_key, profile=profile, intervention=intervention, options=options)
    return base_class(host, port, api_key, profile=profile, intervention=intervention)


def joint_waypoint(robot, waypoint, options):
    """Convert xyz+wxyz+finger-position to joints only when explicitly declared."""
    if options.get("action_space") != "ee_pos":
        return waypoint
    values = np.asarray(waypoint, dtype=float)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("EE action must contain 8 finite values")
    norm = np.linalg.norm(values[3:7])
    if norm < 1e-6:
        raise ValueError("EE quaternion cannot be zero")
    from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
    from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_lula_kinematics_solver_config

    if not hasattr(robot, "_release_ee_solver"):
        solver = LulaKinematicsSolver(**load_supported_lula_kinematics_solver_config("Franka"))
        robot._release_ee_solver = solver, ArticulationKinematicsSolver(robot, solver, "right_gripper")
    solver, articulation = robot._release_ee_solver
    solver.set_robot_base_pose(*robot.get_world_pose())
    action, valid = articulation.compute_inverse_kinematics(
        target_position=values[:3], target_orientation=values[3:7] / norm)
    if not valid or not np.isfinite(action.joint_positions).all():
        raise ValueError("EE inverse kinematics did not converge")
    return [*map(float, action.joint_positions[:7]), float(np.clip(values[7], 0, .04))]
