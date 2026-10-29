"""Core World rollout with live load-cell readback and online event scoring."""

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
from .core import (Contract, Instance, Measurement, State, WEIGHTS, advance,
                   generate_instance, legal_actions, normalize_action,
                   paired_instances, task_complete)
from .online import (EVALUATION_VERSION, OpportunityBook, execution_subgoal_confirmed,
                     physical_choice, score_trace)


def _save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def run_supervised(config: str, output: str, *, seed: int = 101,
                   mass_seed: int | None = None, scene_seed: int | None = None,
                   pair_seed: int | None = None, pair_member: str = "a",
                   max_steps: int | None = None, save_video: bool = False,
                   headless: bool = True,
                   oracle_profile: str | None = None, policy_url: str | None = None,
                   checkpoint_kind: str = "behavior", policy_profile: str | None = None,
                   intervention_spec: str | None = None,
                   dataset_output: str | None = None) -> dict:
    """A worker process isolates Kit shutdown from persisted rollout results."""
    if pair_seed is not None and (mass_seed is not None or scene_seed is not None):
        raise ValueError("planned pair seeds cannot be overridden")
    out = Path(output).resolve()
    cmd = [sys.executable, "-m", "eai4s_eval.weighing_v1.cli", "run", "--worker",
           "--config", str(Path(config).resolve()), "--output", str(out), "--seed", str(seed)]
    if headless:
        cmd.append("--headless")
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    if mass_seed is not None:
        cmd += ["--mass-seed", str(mass_seed)]
    if scene_seed is not None:
        cmd += ["--scene-seed", str(scene_seed)]
    if pair_seed is not None:
        cmd += ["--pair-seed", str(pair_seed), "--pair-member", pair_member]
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
    if dataset_output:
        cmd += ["--dataset-output", str(Path(dataset_output).resolve())]
    if checkpoint_kind != "behavior":
        cmd += ["--checkpoint-kind", checkpoint_kind]
    worker = subprocess.Popen(cmd)
    close_started = None
    timed_out = False
    manifest_path = out / "run.json"
    while worker.poll() is None:
        try:
            worker.wait(timeout=0.5)
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
        raise RuntimeError(f"weighing worker exited {worker.returncode} before writing manifest")
    manifest = json.loads(manifest_path.read_text())
    if timed_out:
        manifest["simulation_app_close"] = {"status": "timed_out", "worker_exit_code": worker.returncode}
    elif manifest.get("simulation_app_close", {}).get("status") == "requested":
        manifest["simulation_app_close"] = {
            "status": "process_exited_after_skip_cleanup", "worker_exit_code": worker.returncode}
    manifest["worker_exit_code"] = worker.returncode
    _save_json(manifest_path, manifest)
    scores_path = out / "scores.json"
    if not scores_path.exists():
        raise RuntimeError(f"worker did not persist scores; see {manifest_path}")
    scores = json.loads(scores_path.read_text())
    if worker.returncode != 0:
        scores["worker_exit_code"] = worker.returncode
    if manifest.get("runtime_status") == "error" or worker.returncode != 0:
        raise RuntimeError(f"weighing worker failed; see {manifest_path} and {scores_path}")
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


def _encoded_images(cameras: dict[str, Any], *, frames: dict[str, Any] | None = None) -> dict[str, Any]:
    from PIL import Image

    result = {}
    for name, camera in cameras.items():
        frame = _rgb(frames[name] if frames is not None else camera.get_rgba())
        if frame is None:
            raise RuntimeError(f"{name} camera has no RGB frame")
        buffer = BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        result[name] = {"mime_type": "image/png", "base64": base64.b64encode(buffer.getvalue()).decode("ascii")}
    return result


def _contact_names(frame: dict) -> list[str]:
    return [str(contact.get(key, "")) for contact in frame.get("contacts", [])
            for key in ("body0", "body1")]


def _contact_report(stage: Any, handles: Any) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Attach sensors before reset, then initialize them after World.reset()."""
    import numpy as np
    from isaacsim.sensors.physics import ContactSensor
    from pxr import PhysxSchema

    result = {}
    # ContactSensor expects its parent to be the rigid-body prim. Attaching
    # it below the visual/collision child (``.../Body``) makes Kit add the
    # contact-report API while the simulation is already live, which produces
    # the "stop and play" error and leaves the sensor uninitialized.
    paths = {"sample": "/World/Sample", "tray": "/World/Scale/Tray"}
    paths.update({f"{weight}g": f"/World/weight_{weight}g" for weight in WEIGHTS})
    for name, path in paths.items():
        if not stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"missing contact collider: {path}")
        prim = stage.GetPrimAtPath(path)
        if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0)
        sensor = ContactSensor(prim_path=f"{path}/WeighingContact", name=f"weighing_contact_{name}",
                               frequency=handles.world.get_physics_dt() ** -1,
                               translation=np.zeros(3), min_threshold=0, max_threshold=1e7, radius=-1)
        sensor.add_raw_contact_data_to_frame()
        result[name] = sensor
    return result, {weight: {} for weight in WEIGHTS}


def _read_contacts(sensors: dict[str, Any], supported_guess: tuple[int, ...],
                   previous_width: float, current_width: float) -> tuple[dict, dict[int, dict]]:
    details = {}
    fingers = {}
    for name, sensor in sensors.items():
        frame = sensor.get_current_frame() or {}
        names = _contact_names(frame)
        details[name] = {
            "tray": any("/World/Scale/Tray" in path for path in names),
            "table": any("/World/table" in path for path in names),
            "left": any("panda_leftfinger" in path for path in names),
            "right": any("panda_rightfinger" in path for path in names),
            "franka_contact": any("/World/Franka/" in path for path in names),
            "force_n": float(frame.get("force", 0.0) or 0.0),
            "contact_bodies": names,
        }
        if name.endswith("g") and name[:-1].isdigit():
            fingers[int(name[:-1])] = {
                "left_finger_contact": details[name]["left"],
                "right_finger_contact": details[name]["right"],
                "gripper_closing": current_width + 0.00002 < previous_width,
            }
    return details, fingers


def _read_measurement(step: int, handles: Any, reading: Any,
                      contacts: dict, fingers: dict) -> tuple[Measurement, dict]:
    import numpy as np

    positions = {"sample": handles.sample.get_world_pose()[0]}
    positions.update({f"{weight}g": body.get_world_pose()[0] for weight, body in handles.weights.items()})
    supported = tuple(weight for weight in WEIGHTS if contacts[f"{weight}g"]["tray"] and
                      not contacts[f"{weight}g"]["left"] and not contacts[f"{weight}g"]["right"] and
                      not contacts[f"{weight}g"]["table"])
    sample_contact = contacts["sample"]
    sample_untouched = not sample_contact["left"] and not sample_contact["right"]
    released = not any(contacts[name]["left"] or contacts[name]["right"]
                       for name in ("sample", *(f"{weight}g" for weight in WEIGHTS)))
    no_external_support = not any(contacts[name]["table"] and contacts[name]["tray"]
                                  for name in ("sample", *(f"{weight}g" for weight in WEIGHTS)))
    speeds = []
    angular = []
    for body in (handles.sample, *handles.weights.values()):
        speeds.append(float(np.linalg.norm(body.get_linear_velocity())))
        angular.append(float(np.linalg.norm(body.get_angular_velocity())))
    # A moving object off the scale must not prevent a genuinely stable pan reading.
    on_pan = (handles.sample, *(handles.weights[weight] for weight in supported))
    pan_linear = max(float(np.linalg.norm(body.get_linear_velocity())) for body in on_pan)
    pan_angular = max(float(np.linalg.norm(body.get_angular_velocity())) for body in on_pan)
    valid = bool(reading.valid)
    measurement = Measurement(step, valid,
                              float(reading.raw_grams) if valid else 0.0,
                              float(reading.displayed_grams) if valid else 0.0,
                              supported, sample_contact["tray"], sample_untouched,
                              released, no_external_support,
                              not contacts["tray"]["franka_contact"],
                              pan_linear, pan_angular, bool(reading.overloaded))
    detail = asdict(measurement)
    detail.update({"sensor_source": reading.sensor_source,
                   "raw_force_n": reading.raw_force_n,
                   "zero_force_n": reading.zero_force_n,
                   "display_text": reading.display_text,
                   "sensor_error": reading.error,
                   "finger_contacts": {str(weight): data for weight, data in fingers.items()},
                   "object_positions_world_m": {name: list(map(float, pose)) for name, pose in positions.items()},
                   "contacts": contacts})
    return measurement, detail


def _candidate_geometry(handles: Any) -> dict[str, Any]:
    from .scene import initial_load_positions

    pan_positions = initial_load_positions(handles.tray_center, handles.tray_top_z, WEIGHTS)
    def frame(position):
        return {"position_world_m": list(map(float, position)),
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
    return {"frame": "world", "units": "m",
            "grasp_frames": {str(weight): frame(handles.weights[weight].get_world_pose()[0])
                             for weight in WEIGHTS},
            "pan_frames": {str(weight): frame(pan_positions[f"{weight}g"]) for weight in WEIGHTS},
            "return_frames": {str(weight): frame(handles.object_start_poses[f"{weight}g"])
                              for weight in WEIGHTS}}


class PhysicalTransfer:
    """Reference add/remove attempt; every target is sent through the Franka controller."""

    PHASES = (("above_source", 240), ("descend", 240), ("close", 100),
              ("lift", 220), ("above_destination", 280), ("lower", 240),
              ("open", 110), ("retreat", 160))

    def __init__(self, action: dict, handles: Any, motion: Any):
        import numpy as np
        from scipy.spatial.transform import Rotation

        self.action = action
        self.handles = handles
        self.motion = motion
        self.weight = handles.weights[action["weight_g"]]
        self.origin = np.asarray(self.weight.get_world_pose()[0], dtype=float)
        if action["type"] == "add":
            from .scene import initial_load_positions
            destination = initial_load_positions(handles.tray_center, handles.tray_top_z,
                                                 (action["weight_g"],))[f"{action['weight_g']}g"]
        else:
            destination = handles.object_start_poses[f"{action['weight_g']}g"]
        self.destination = np.asarray(destination, dtype=float)
        self.orientation = None
        self.phase_index = 0
        self.phase_step = 0
        self.initial_supported = action["weight_g"] in handles.instance.initial_weights_g
        self.failure_reason: str | None = None
        self.last_command = None

    @property
    def phase(self) -> str:
        return self.PHASES[min(self.phase_index, len(self.PHASES) - 1)][0]

    def step(self) -> bool:
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction
        from eai4s_eval.centrifuge_v2.core import limit_joint_targets

        name, duration = self.PHASES[self.phase_index]
        current = np.asarray(self.handles.franka.get_joint_positions(), dtype=float)
        if not np.isfinite(current).all():
            raise RuntimeError("non-finite Franka joint state")
        source = np.asarray(self.weight.get_world_pose()[0], dtype=float)
        clearance = max(self.origin[2], self.destination[2]) + .16
        anchor = source if name in {"above_source", "descend", "close", "lift"} else self.destination
        target = np.asarray((anchor[0], anchor[1], clearance), dtype=float)
        if name in {"descend", "close"}:
            target[2] = source[2] + .027
        elif name == "lower" or name == "open":
            target[2] = self.destination[2] + .027
        elif name == "retreat":
            target[2] = clearance
        policy_eef = np.asarray(self.motion.rmp_flow.get_end_effector_pose(current[:7])[0], dtype=float)
        tool_center = np.asarray(self.handles.franka.get_gripper_position(), dtype=float)
        offset = tool_center - policy_eef
        if not np.isfinite(offset).all():
            raise RuntimeError("non-finite motion-policy to gripper frame offset")
        policy_target = target - offset
        joint_action = self.motion.forward(target_end_effector_position=policy_target,
                                           target_end_effector_orientation=self.orientation)
        joint_targets = list(joint_action.joint_positions) if joint_action.joint_positions is not None else []
        if joint_targets:
            joint_targets = limit_joint_targets(list(map(float, current)), joint_targets,
                                                 None if joint_action.joint_indices is None else
                                                 list(joint_action.joint_indices), arm_step_rad=.04)
            joint_action.joint_positions = joint_targets
            self.handles.franka.get_articulation_controller().apply_action(joint_action)
            self.last_command = list(map(float, joint_targets)) + [
                .004 if name in {"close", "lift", "above_destination", "lower"} else .04]
        if name in {"close", "lift", "above_destination", "lower"}:
            self.handles.franka.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=np.array([.004, .004]),
                                   joint_indices=np.array([7, 8])))
        else:
            self.handles.franka.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=np.array([.04, .04]),
                                   joint_indices=np.array([7, 8])))
        self.phase_step += 1
        if self.phase_step >= duration:
            self.phase_step = 0
            self.phase_index += 1
            if name == "lift" and self.phase_index < len(self.PHASES):
                lifted = float(self.weight.get_world_pose()[0][2] - self.origin[2])
                if lifted < .05:
                    self.failure_reason = f"grasp_not_lifted:{lifted:.4f}m"
                    self.phase_index = len(self.PHASES)
        return self.phase_index >= len(self.PHASES)


def run_episode(config: str, output: str, *, seed: int = 101,
                mass_seed: int | None = None, scene_seed: int | None = None,
                pair_seed: int | None = None, pair_member: str = "a",
                max_steps: int | None = None, save_video: bool = False,
                headless: bool = True,
                oracle_profile: str | None = None, policy_url: str | None = None,
                checkpoint_kind: str = "behavior", policy_profile: str | None = None,
                intervention_spec: str | None = None,
                dataset_output: str | None = None) -> dict:
    source, contract = load_config(config, policy_profile=policy_profile,
                                   intervention_spec=intervention_spec)
    ev = dict(source["evaluation"])
    interface = dict(ev["policy_interface"])
    if policy_url:
        native = interface["action_space"] == "array"
        interface.update(name="http_array" if native else "http_symbolic", url=policy_url)
    from eai4s_eval.oracle_runtime import (
        InterventionSpec, PolicyProfile, condition_id, content_digest,
        ranking_eligibility, validate_connection_binding,
    )
    runtime_profile = (PolicyProfile.legacy(interface, ("front", "wrist", "display", "weights"))
                       if policy_url is not None and policy_profile is None else
                       PolicyProfile.from_dict(ev["policy_profile"]))
    validate_connection_binding(runtime_profile, interface)
    intervention = InterventionSpec.from_dict(ev["intervention"])
    profile = oracle_profile or intervention.profile
    if oracle_profile is not None and intervention_spec is not None and oracle_profile != intervention.profile:
        raise ValueError("--oracle-profile conflicts with --intervention-spec")
    if oracle_profile is not None and intervention_spec is None:
        intervention = InterventionSpec.legacy(oracle_profile, ev.get("oracle_transport", "structured"))
    if profile not in {"none", "semantic", "decision", "geometry", "skill"}:
        raise ValueError("unknown Oracle profile")
    if profile != "none" and interface["name"] == "diagnostic_rule_expert":
        raise ValueError("GT diagnostic expert cannot be used for an Oracle comparison")
    if profile != "none" and intervention.mode == "assist" and intervention.transport not in runtime_profile.oracle_transports:
        raise ValueError("selected adapter does not support Oracle transport")
    if runtime_profile.adapter in {"rpent", "vla_evaluation_harness"}:
        raise ValueError("external agent/harness profiles must run through eai4s_eval.integrations")
    if intervention.mode == "replace":
        raise ValueError("interface replacement requires an external agent or declared module-hook adapter")
    if profile == "skill":
        raise ValueError("Skill Oracle requires a separately validated reference executor")
    if checkpoint_kind not in {"behavior", "semantic", "decision", "geometry", "execution"}:
        raise ValueError("unknown checkpoint kind")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if pair_member not in {"a", "b"} or pair_seed is None and pair_member != "a":
        raise ValueError("pair member requires a pair seed and must be a or b")
    if pair_seed is not None and (mass_seed is not None or scene_seed is not None):
        raise ValueError("planned pair seeds cannot be overridden")
    instance = (paired_instances(contract, pair_seed)[0 if pair_member == "a" else 1]
                if pair_seed is not None else generate_instance(
                    contract, seed, mass_seed=mass_seed, scene_seed=scene_seed))
    from .scene import generate_scene_layout
    scene_layout = generate_scene_layout(instance.scene_seed)
    episode_instance_id = (
        f"{instance.pair_id}:{pair_member if pair_seed is not None else 'single'}:"
        f"mass-{instance.mass_seed}:scene-{instance.scene_seed}")
    initial_state_hash = content_digest({"instance": asdict(instance),
                                         "scene_layout": scene_layout.public()})
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)
    limit = min(max_steps or contract.max_steps, contract.max_steps)
    ev["oracle_profile"] = profile
    ev["intervention"] = intervention.__dict__
    ev["policy_profile"] = runtime_profile.public()
    ev["policy_interface"] = interface
    uses_evaluator_state = interface["name"] == "diagnostic_rule_expert"
    eligible, policy_revision, ineligible_reasons = ranking_eligibility(
        runtime_profile, intervention, interface, uses_evaluator_state=uses_evaluator_state)
    manifest = {"task_version": contract.task_version, "evaluation_version": EVALUATION_VERSION,
                "contract_hash": contract.hash, "evaluation": ev, "instance": asdict(instance),
                "seed": instance.seed, "scene_seed": instance.scene_seed, "mass_seed": instance.mass_seed,
                "pair_id": instance.pair_id, "instance_id": episode_instance_id,
                "pair_seed": pair_seed,
                "pair_member": pair_member if pair_seed is not None else None,
                "difficulty": contract.difficulty,
                "scene_revision": source["scene"]["scene_revision"],
                "scene_layout": scene_layout.public(),
                "scene_layout_hash": scene_layout.hash,
                "initial_state_hash": initial_state_hash,
                "policy_id": interface.get("policy_id", interface.get("name")),
                "policy_revision": policy_revision,
                "policy_revision_declared": policy_revision != "unversioned",
                "policy_profile_id": runtime_profile.name,
                "policy_profile_hash": runtime_profile.hash,
                "policy_system_type": runtime_profile.system_type,
                "intervention": intervention.__dict__,
                "condition_id": condition_id(intervention),
                "condition_supported": True,
                "action_spec_hash": runtime_profile.action.hash,
                "toolset_hash": content_digest(list(runtime_profile.agent_tools)),
                "reset_protocol_hash": content_digest(asdict(runtime_profile.reset)),
                "memory_revision": interface.get("memory_revision", "episode-isolated-v1"),
                "action_space": interface["action_space"], "oracle_profile": profile,
                "sensor_condition": "front+wrist+display+weights_rgb",
                "checkpoint_kind": checkpoint_kind, "max_steps_budget": limit,
                "budget_id": f"steps={limit};hz={runtime_profile.action.control_hz:g};chunk={runtime_profile.action.max_chunk}",
                "uses_evaluator_state": uses_evaluator_state,
                "ranking_eligible": eligible,
                "ranking_ineligible_reasons": list(ineligible_reasons),
                "evaluation_valid": False,
                "runtime_status": "initializing"}
    _save_json(out / "run.json", manifest)
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": bool(headless), "enable_cameras": True,
                             "anti_aliasing": 2})
    except Exception as exc:
        manifest.update(runtime_status="error", error_type=type(exc).__name__, error_message=str(exc))
        _save_json(out / "run.json", manifest)
        _save_json(out / "scores.json", score_trace(contract, instance, []))
        raise
    trace: list[dict] = []
    writers: dict[str, Any] = {}
    events_file = None
    observations_file = None
    online_file = None
    # Scene construction can fail before the dataset recorder is created.
    recorder = None
    error = None
    try:
        import numpy as np
        import imageio.v2 as imageio
        from PIL import Image
        from scipy.spatial.transform import Rotation
        from isaacsim.core.api import World
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.physics import ContactSensor
        from robots.franka.rmpflow_controller import RMPFlowController
        from eai4s_eval.centrifuge_v2.core import limit_joint_targets
        from .scene import build_scene, update_display
        from .loadcell import PhysxLoadCell
        from utils.object_utils import ObjectUtils
        from .policy import HttpArrayPolicy, HttpSymbolicPolicy, OpenPIArrayPolicy, PolicyServiceError
        from .diagnostics import (PolicyCapabilities, policy_observation, check_policy_request,
                                  oracle_context, deliver_input, evaluate_policy_checkpoint,
                                  score_checkpoint)

        world = World(stage_units_in_meters=1.0, physics_prim_path="/World/physicsScene", backend="numpy")
        repo = Path(__file__).resolve().parents[2]
        handles = build_scene(world, contract, instance, repo)
        if handles.layout.hash != scene_layout.hash:
            raise RuntimeError("authored scene layout differs from run manifest")
        ObjectUtils.get_instance(handles.stage)
        sensors, _ = _contact_report(handles.stage, handles)
        world.reset()
        handles.initialize()
        manifest["camera_poses_usd"] = {
            name: {"position_m": list(map(float, camera.get_world_pose(camera_axes="usd")[0])),
                   "quaternion_wxyz": list(map(float, camera.get_world_pose(camera_axes="usd")[1]))}
            for name, camera in handles.cameras.items()}
        manifest["camera_intrinsics_m"] = {
            name: {"focal_length": float(camera.get_focal_length()),
                   "horizontal_aperture": float(camera.get_horizontal_aperture())}
            for name, camera in handles.cameras.items()}
        manifest["tray_center_world_m"] = list(handles.tray_center)
        manifest["object_start_poses_world_m"] = {
            name: list(position) for name, position in handles.object_start_poses.items()}
        for sensor in sensors.values():
            sensor.initialize()
        cell = PhysxLoadCell(handles.stage, handles.loadcell_joint_path,
                             tray_path="/World/Scale/Tray", filter_tau_s=contract.readout_tau_s,
                             display_quantum_g=contract.display_quantum_g,
                             stable_window_s=contract.stable_window_s,
                             stable_range_g=contract.stable_range_g)
        cell.initialize()
        zero = cell.calibrate(world)
        setup_positions = handles.place_initial_loads()
        initial_settle_steps = math.ceil(1.5 * contract.physics_hz)
        for settle_step in range(initial_settle_steps):
            world.step(render=False)
            initial_reading = cell.update(settle_step, 1.0 / contract.physics_hz)
            if not initial_reading.valid or initial_reading.overloaded:
                raise RuntimeError("initial scale load did not produce a valid reading")
            update_display(handles, initial_reading)
        if save_video or interface["name"] != "diagnostic_rule_expert" or checkpoint_kind != "behavior":
            for _ in range(3):
                world.render()
        manifest.update(runtime_status="running", empty_pan_zero_force_n=zero,
                        checkpoint_setup_positions_world_m=setup_positions,
                        initial_condition_authoring="pre_scored_once",
                        initial_load_settle_steps=initial_settle_steps,
                        sensor_source="physx_loadcell_joint")
        _save_json(out / "run.json", manifest)
        if interface["name"] == "diagnostic_rule_expert":
            policy = None
        elif interface["name"] == "http_symbolic":
            policy = HttpSymbolicPolicy(interface["url"], interface.get("timeout_s", 30))
        elif interface["name"] == "http_array":
            policy = HttpArrayPolicy(interface["url"], interface.get("timeout_s", 30))
        else:
            from eai4s_eval.special_policy import openpi_policy
            policy = openpi_policy(OpenPIArrayPolicy, interface.get("host", "127.0.0.1"),
                                       int(interface.get("port", 8000)), interface.get("api_key"),
                                       profile=runtime_profile, intervention=intervention,
                                       options=source.get("labbench_runtime"))
        if policy is not None:
            eligible, policy_revision, ineligible_reasons = ranking_eligibility(
                runtime_profile, intervention, interface, uses_evaluator_state=False,
                server_metadata=getattr(policy, "server_metadata", None))
            manifest.update(
                policy_revision=policy_revision,
                policy_revision_declared=policy_revision != "unversioned",
                ranking_eligible=eligible,
                ranking_ineligible_reasons=list(ineligible_reasons),
            )
            policy.reset(instance.seed)
            if hasattr(policy, "session"):
                manifest["policy_capability_check"] = asdict(policy.session.capability_check)
                _save_json(out / "run.json", manifest)
        motion = RMPFlowController(name="weighing_reference_rmp", robot_articulation=handles.franka,
                                   physics_dt=1.0 / contract.physics_hz)
        if dataset_output is not None and policy is None:
            from data_collectors.release_episode_recorder import ReleaseEpisodeRecorder
            recorder = ReleaseEpisodeRecorder(
                dataset_output,
                camera_names=("camera_1_rgb", "camera_3_rgb", "display_rgb", "weights_rgb"),
                source_hz=contract.physics_hz, sample_hz=10,
                task_name="Adjust the scale to the instructed target weight",
                task_properties={"task_version": contract.task_version, "seed": instance.seed},
            )
        events_file = (out / "events.jsonl").open("w")
        observations_file = (out / "observations.jsonl").open("w")
        online_file = (out / "online_scores.jsonl").open("w")

        def emit(row: dict) -> None:
            trace.append(row)
            events_file.write(json.dumps(row, default=str) + "\n")

        def emit_intervention(step: int, context: dict, opportunity_id: str | None) -> None:
            if intervention.mode != "assist":
                return
            emit({"type": "oracle_intervention", "step": step,
                  "evidence_step": step, "opportunity_id": opportunity_id,
                  "profile": intervention.profile, "mode": intervention.mode,
                  "site": intervention.site, "transport": intervention.transport,
                  "status": "delivered_to_adapter",
                  "oracle_digest": content_digest(context)})

        def snapshot(step: int, reason: str) -> None:
            result = score_trace(contract, instance, trace)
            online_file.write(json.dumps({"step": step, "reason": reason, "scores": result}) + "\n")

        state = State()
        book = OpportunityBook(contract, instance, interface["action_space"] == "array" and "native" or "reported")
        active: PhysicalTransfer | None = None
        active_start = None
        native_pending: dict[str, Any] | None = None
        native_pending_start: int | None = None
        native_queue: list[list[float]] = []
        native_action_count = 0
        branch_budget = source.get("labbench_runtime", {}).get("max_action_steps")
        native_decimation = max(1, round(contract.physics_hz / runtime_profile.action.control_hz))
        checkpoint_record: dict[str, Any] | None = None
        checkpoint_response: dict[str, Any] | None = None
        execution_checkpoint_action: dict[str, Any] | None = None
        execution_checkpoint_pending: dict[str, Any] | None = None
        previous_width = .08
        frames_seen = 0
        thumbnail_frames = []
        started = time.monotonic()
        video_stride = max(1, contract.physics_hz // 10)
        for step in range(limit):
            world.step(render=False)
            reading = cell.update(step, 1.0 / contract.physics_hz)
            update_display(handles, reading)
            capture_phase = 1 if save_video and video_stride > 1 else 0
            video_prepare = save_video and step % video_stride == 0
            video_sample = save_video and step % video_stride == capture_phase
            if (policy is not None or
                    (checkpoint_kind != "behavior" and checkpoint_record is None and
                     step % video_stride == 0) or video_sample):
                world.render()
            elif video_prepare:
                world.render()
            joints = np.asarray(handles.franka.get_joint_positions(), dtype=float)
            if not np.isfinite(joints).all():
                raise RuntimeError("non-finite Franka joint state")
            width = float(joints[7] + joints[8])
            contacts, fingers = _read_contacts(sensors, state.supported_weights_g, previous_width, width)
            previous_width = width
            measured, detail = _read_measurement(step, handles, reading, contacts, fingers)
            advance(contract, instance, state, measured)
            emit({"type": "measurement", "step": step, "measurement": detail})
            if execution_checkpoint_pending is not None and checkpoint_record is not None and state.reading_stable:
                action = execution_checkpoint_pending["action"]
                confirmed = execution_subgoal_confirmed(
                    action, state, observed_change=execution_checkpoint_pending["changed"])
                checkpoint_record["scores"] = score_checkpoint({
                    "instance_id": instance.pair_id, "layer": "execution", "supported": True,
                    "physical_subgoal_confirmed": confirmed})
                checkpoint_record.update(status="scored", physical_action=action,
                                         physical_subgoal_confirmed=confirmed, settled_step=step)
                execution_checkpoint_pending = None
                _save_json(out / "checkpoints.json", checkpoint_record)
            if not measured.sensor_valid:
                emit({"type": "sensor_invalid", "step": step, "error": reading.error})
                raise RuntimeError(f"physical scale readback failed: {reading.error}")
            if measured.overloaded:
                emit({"type": "scale_overload", "step": step,
                      "raw_force_n": reading.raw_force_n,
                      "raw_grams": reading.raw_grams})
            if not measured.sample_untouched or not measured.no_hand_press:
                emit({"type": "safety_violation", "step": step, "kind": "sample_or_pan_touched"})
            if step % video_stride == capture_phase:
                observation = {"step": step, "policy_observation": {
                    "task_rule": instance.public()["rule"], "target_g": instance.target_g,
                    "display_reading_g": None, "stable_indicator": None,
                    "visible_weight_labels_g": list(WEIGHTS)},
                    "evaluator_state": {"supported_weights_g": list(state.supported_weights_g),
                                        "desired_weights_g": list(instance.desired_weights_g),
                                        "sample_mass_g": instance.sample_mass_g,
                                        "measured_display_g": measured.displayed_g,
                                        "measured_stable_indicator": reading.stable,
                                        "state": state.public()},
                    "load_cell": reading.as_dict(), "controller_phase": active.phase if active else None,
                    "gripper_position_world_m": list(map(float, handles.franka.get_gripper_position())),
                    "motion_eef_position_world_m": list(map(float,
                        motion.rmp_flow.get_end_effector_pose(joints[:7])[0])),
                    "target_weight_position_world_m": (list(map(float, active.weight.get_world_pose()[0]))
                                                       if active else None)}
                observations_file.write(json.dumps(observation, default=str) + "\n")
                if save_video:
                    for name in ("front", "wrist", "display", "weights"):
                        frame = _rgb(handles.cameras[name].get_rgba())
                        if frame is None:
                            continue
                        if name not in writers:
                            writers[name] = imageio.get_writer(str(out / f"artifacts/{name}.mp4"), fps=10)
                        writers[name].append_data(frame)
                        if name == "front":
                            frames_seen += 1
                            if frames_seen % 10 == 0:
                                thumbnail_frames.append(Image.fromarray(frame).resize((320, 240)))
            if step % contract.physics_hz == 0:
                snapshot(step, "periodic")
            if recorder is not None and policy is None and step % video_stride == 0:
                world.render()
                joints_now = np.asarray(handles.franka.get_joint_positions(), dtype=float)
                tcp_position, tcp_matrix = motion.rmp_flow.get_end_effector_pose(joints_now[:7])
                recorder.record(
                    cameras={name: _rgb(handles.cameras[name].get_rgba())
                             for name in ("front", "wrist", "display", "weights")},
                    joints=joints_now,
                    tcp_pose_xyzw=[*tcp_position, *Rotation.from_matrix(tcp_matrix).as_quat()],
                    command=active.last_command if active is not None else None,
                    sim_step=step,
                    subtask=f"{active.action['type']}_{active.phase}" if active else "stable_readout",
                )
            if active is not None:
                finished = active.step()
                if finished:
                    chosen = active.action
                    weight = chosen["weight_g"]
                    changed = active.failure_reason is None and (
                        (weight in state.supported_weights_g) == (chosen["type"] == "add"))
                    emit({"type": "operation_result", "step": step, "action": chosen,
                          "started_step": active_start,
                          "failure_reason": active.failure_reason,
                          "status": "confirmed" if changed else "failed",
                          "physical_supported_weights_g": list(state.supported_weights_g)})
                    if execution_checkpoint_action is not None and checkpoint_record is not None:
                        execution_checkpoint_pending = {
                            "action": chosen, "changed": changed and chosen == execution_checkpoint_action}
                        checkpoint_record["status"] = "awaiting_stable_load" if changed else "scored"
                        if not changed:
                            checkpoint_record["scores"] = score_checkpoint({
                                "instance_id": instance.pair_id, "layer": "execution", "supported": True,
                                "physical_subgoal_confirmed": False})
                            checkpoint_record["physical_subgoal_confirmed"] = False
                            execution_checkpoint_pending = None
                        execution_checkpoint_action = None
                        _save_json(out / "checkpoints.json", checkpoint_record)
                    active = None
                    active_start = None
                    if not changed:
                        opened = book.retry_after_failure(state, step, evidence_step=step)
                        emit(opened)
                        snapshot(step, "operation_failed")
                continue
            if task_complete(contract, instance, state) and interface["action_space"] == "array":
                break
            native_candidate = (physical_choice(fingers, book.current_signature[0])
                                if interface["action_space"] == "array" and
                                book.current_signature is not None else None)
            if interface["action_space"] == "array" and native_pending is not None and (
                    state.reading_stable and measured.released):
                weight = native_pending["weight_g"]
                changed = (weight in state.supported_weights_g) == (native_pending["type"] == "add")
                emit({"type": "operation_result", "step": step, "action": native_pending,
                      "started_step": native_pending_start,
                      "status": "confirmed" if changed else "failed",
                      "physical_supported_weights_g": list(state.supported_weights_g)})
                if execution_checkpoint_action is not None and checkpoint_record is not None:
                    checkpoint_record["scores"] = score_checkpoint({
                        "instance_id": instance.pair_id, "layer": "execution", "supported": True,
                        "physical_subgoal_confirmed": execution_subgoal_confirmed(
                            native_pending, state,
                            observed_change=changed and native_pending == execution_checkpoint_action)})
                    checkpoint_record["status"] = "scored"
                    checkpoint_record["physical_action"] = native_pending
                    checkpoint_record["physical_subgoal_confirmed"] = (
                        checkpoint_record["scores"]["execution"] == "pass")
                    execution_checkpoint_action = None
                    _save_json(out / "checkpoints.json", checkpoint_record)
                native_pending = None
                native_pending_start = None
                opened = (book.update(state, step, evidence_step=step) if changed else
                          book.retry_after_failure(state, step, evidence_step=step))
            else:
                opened = book.update(state, step, evidence_step=step,
                                     operation_in_flight=native_pending is not None or
                                     (native_candidate is not None and book.first_choice is None))
            if opened:
                emit(opened)
                snapshot(step, "opportunity_open")
                if interface["action_space"] == "array":
                    native_queue.clear()
            if (checkpoint_kind != "behavior" and checkpoint_record is None and
                    state.reading_stable and measured.released):
                eligible_actions = [action for action in legal_actions(contract, instance, state)
                                    if action["type"] in {"add", "remove"}]
                selected = eligible_actions[0] if eligible_actions else None
                capabilities = PolicyCapabilities(
                    runtime_profile.reports,
                    runtime_profile.oracle_transports,
                    interface["action_space"], True)
                if checkpoint_kind == "execution" and not (
                        interface["name"] == "diagnostic_rule_expert" or
                        interface.get("selected_operation_input", False)):
                    checkpoint_record = {"instance_id": instance.pair_id, "layer": "execution",
                                         "status": "unsupported", "reason": "selected_operation_input"}
                elif selected is None and checkpoint_kind in {"geometry", "execution"}:
                    checkpoint_record = {"instance_id": instance.pair_id, "layer": checkpoint_kind,
                                         "status": "unavailable", "reason": "no_add_remove_subgoal"}
                else:
                    images = _encoded_images(handles.cameras)
                    view = policy_observation(
                        images, list(map(float, joints)), instance.public()["rule"], list(WEIGHTS),
                        None, None, history=[])
                    callback = ((lambda request: {"committed_action": legal_actions(contract, instance, state)[0]})
                                if policy is None else
                                (policy.infer_full if interface["action_space"] == "array" else policy.decide))
                    reset = ((lambda: None) if policy is None or runtime_profile.reset.mode == "stateless"
                             else (lambda: policy.reset(instance.seed)))
                    checkpoint_record = evaluate_policy_checkpoint(
                        contract, instance, state, view,
                        {"visible_weight_locations_g": {},
                         "display_reading_g": None if reading.overloaded else reading.displayed_grams,
                         "stable_indicator": bool(reading.stable)}, _candidate_geometry(handles),
                        callback, capabilities, instance_id=instance.pair_id,
                        layer=checkpoint_kind, profile=profile,
                        transport=intervention.transport if intervention.mode == "assist" else "structured",
                        selected_action=selected, reset_policy=reset)
                    if checkpoint_record.get("status") == "service_error":
                        raise PolicyServiceError(checkpoint_record.get("message", "checkpoint policy service failed"))
                    if checkpoint_record.get("status") == "scored":
                        checkpoint_response = checkpoint_record.get("response")
                        checkpoint_record = {key: value for key, value in checkpoint_record.items()
                                             if key not in {"request", "response"}}
                        checkpoint_record["request_audited"] = True
                        if checkpoint_kind == "execution":
                            execution_checkpoint_action = selected
                            checkpoint_record["status"] = "awaiting_physical_result"
                checkpoint_record["step"] = step
                _save_json(out / "checkpoints.json", checkpoint_record)
                emit({"type": "checkpoint", "step": step, "layer": checkpoint_kind,
                      "status": checkpoint_record["status"]})
            if interface["action_space"] == "array":
                if step % native_decimation:
                    continue
                if branch_budget is not None and native_action_count >= branch_budget:
                    emit({"type": "action_budget_exhausted", "step": step, "action_count": native_action_count})
                    break
                candidate = native_candidate
                if candidate and book.first_choice is None and native_pending is None:
                    chosen = book.record_choice(candidate, step,
                                                evidence_step=(book.current_evidence_step
                                                               if book.current_evidence_step is not None else step),
                                                source="physical_interaction")
                    if chosen:
                        emit(chosen)
                        snapshot(step, "physical_choice")
                        native_pending = candidate
                        native_pending_start = step
                        emit({"type": "operation_start", "step": step, "action": candidate,
                              "source": "physical_interaction"})
                if not native_queue:
                    if checkpoint_response is not None and "action_chunk" in checkpoint_response:
                        inference = checkpoint_response
                        checkpoint_response = None
                    else:
                        images = _encoded_images(handles.cameras)
                        view = policy_observation(images, list(map(float, joints)), instance.public()["rule"],
                                                  list(WEIGHTS), None, None, history=[])
                        context = oracle_context(profile, contract, instance, state, view,
                                                 visible_facts={"visible_weight_locations_g": {},
                                                                "display_reading_g": None if reading.overloaded else reading.displayed_grams,
                                                                "stable_indicator": bool(reading.stable)},
                                                 candidate_geometry=_candidate_geometry(handles))
                        request = view if profile == "none" else deliver_input(
                            view, context, intervention.transport,
                            set(runtime_profile.oracle_transports))
                        emit_intervention(step, context, book.current_id)
                        check_policy_request(request)
                        inference = policy.infer_full(request)
                    native_queue = inference["action_chunk"]
                    if inference.get("input_delivery_audit"):
                        emit({"type": "input_delivery_audit", "step": step,
                              "opportunity_id": book.current_id,
                              **inference["input_delivery_audit"]})
                    emit({"type": "policy_inference", "step": step,
                          "opportunity_id": book.current_id,
                          "oracle_profile": profile, "action_count": len(native_queue),
                          "server_timing": inference.get("server_timing"),
                          "policy_timing": inference.get("policy_timing")})
                waypoint = native_queue.pop(0)
                native_action_count += 1
                from eai4s_eval.special_policy import joint_waypoint
                waypoint = joint_waypoint(handles.franka, waypoint, source.get("labbench_runtime", {}))
                limited = limit_joint_targets(list(map(float, joints)), waypoint[:7],
                                              list(range(7)), arm_step_rad=.08)
                handles.franka.get_articulation_controller().apply_action(
                    ArticulationAction(joint_positions=np.asarray(limited + [waypoint[7], waypoint[7]])))
                continue
            if book.first_choice is not None:
                continue
            if policy is None:
                action = legal_actions(contract, instance, state)[0]
                response = {"committed_action": action}
                request = None
            elif checkpoint_response is not None and "committed_action" in checkpoint_response:
                response = checkpoint_response
                checkpoint_response = None
                action = normalize_action(response.get("committed_action"))
                request = None
            else:
                images = _encoded_images(handles.cameras)
                view = policy_observation(images, list(map(float, joints)), instance.public()["rule"],
                                          list(WEIGHTS), None, None, history=[])
                context = oracle_context(profile, contract, instance, state, view,
                                         visible_facts={"visible_weight_locations_g": {},
                                                        "display_reading_g": None if reading.overloaded else reading.displayed_grams,
                                                        "stable_indicator": bool(reading.stable)},
                                         candidate_geometry=_candidate_geometry(handles))
                request = view if profile == "none" else deliver_input(
                    view, context, intervention.transport,
                    set(runtime_profile.oracle_transports))
                emit_intervention(step, context, book.current_id)
                check_policy_request(request)
                response = policy.decide(request)
                action = normalize_action(response.get("committed_action"))
            if action is None:
                emit({"type": "invalid_policy_response", "step": step})
                break
            emit({"type": "decision", "step": step, "action": action,
                  "opportunity_id": book.current_id,
                  "policy_response": response,
                  "policy_request_audited": request is not None})
            choice = book.record_choice(action, step, evidence_step=step, source="explicit_commitment")
            if choice:
                emit(choice)
                snapshot(step, "explicit_choice")
            if action["type"] == "stop":
                break
            if action["type"] == "wait":
                continue
            active = PhysicalTransfer(action, handles, motion)
            active_start = step
            emit({"type": "operation_start", "step": step, "action": action})
        manifest.update(runtime_status="finished" if task_complete(contract, instance, state) else "time_limit",
                        steps=step + 1, duration_s=round(time.monotonic() - started, 3),
                        video_frames=frames_seen)
        emit({"type": "terminal_measurement", "step": state.last_step,
              "physical_state": state.public()})
        if checkpoint_kind != "behavior" and checkpoint_record is None:
            checkpoint_record = {"instance_id": instance.pair_id, "layer": checkpoint_kind,
                                 "status": "unavailable", "reason": "no_stable_checkpoint"}
            _save_json(out / "checkpoints.json", checkpoint_record)
        elif (execution_checkpoint_action is not None or execution_checkpoint_pending is not None) and checkpoint_record is not None:
            checkpoint_record["status"] = "unavailable"
            checkpoint_record["reason"] = "physical_subgoal_not_observed"
            _save_json(out / "checkpoints.json", checkpoint_record)
        snapshot(state.last_step, "terminal")
        scores = score_trace(contract, instance, trace)
        manifest.update(
            evaluation_valid=bool(scores.get("evaluation_valid")),
            online_legal_next_action_rate=scores.get("online_legal_next_action_rate"),
            physical_completion=float(bool(scores.get("task_completion"))),
        )
        _save_json(out / "scores.json", scores)
        _save_json(out / "run.json", manifest)
        if recorder is not None and policy is None:
            joints_now = np.asarray(handles.franka.get_joint_positions(), dtype=float)
            tcp_position, tcp_matrix = motion.rmp_flow.get_end_effector_pose(joints_now[:7])
            if scores.get("valid_success"):
                recorder.collector.set_task_properties({
                    "task_version": contract.task_version, "seed": instance.seed,
                    "source_commit": "3d828af", "physical_success": True,
                    "sensor_source": "physical_load_cell", "valid_success": True})
                recorder.finish(
                    joints=joints_now,
                    tcp_pose_xyzw=[*tcp_position, *Rotation.from_matrix(tcp_matrix).as_quat()],
                )
            else:
                recorder.abort()
        if thumbnail_frames:
            selected = [thumbnail_frames[round(index * (len(thumbnail_frames) - 1) / 4)] for index in range(5)]
            grid = Image.new("RGB", (320 * len(selected), 240), "white")
            for index, frame in enumerate(selected):
                grid.paste(frame, (index * 320, 0))
            grid.save(out / "artifacts/contact_sheet.jpg", quality=90)
        return scores
    except Exception as exc:
        error = exc
        if recorder is not None:
            try:
                recorder.abort()
            except Exception:
                pass
        manifest.update(runtime_status="error", error_type=type(exc).__name__,
                        error_message=str(exc), traceback=traceback.format_exc())
        _save_json(out / "run.json", manifest)
        if events_file is not None:
            trace.append({"type": "runtime_error", "message": str(exc)})
            events_file.write(json.dumps(trace[-1]) + "\n")
        _save_json(out / "scores.json", score_trace(contract, instance, trace))
        raise
    finally:
        writer_errors = {}
        for writer in writers.values():
            try:
                writer.close()
            except Exception as exc:
                writer_errors[str(len(writer_errors))] = str(exc)
        for stream in (events_file, observations_file, online_file):
            if stream is not None:
                stream.close()
        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:
                manifest.setdefault("dataset_errors", []).append(str(exc))
        manifest["video_encoder_close"] = {"status": "failed" if writer_errors else "completed",
                                           "errors": writer_errors}
        manifest["simulation_app_close"] = {"status": "requested"}
        _save_json(out / "run.json", manifest)
        close = {"status": "completed"}
        try:
            app.close(skip_cleanup=True)
        except TypeError:
            try:
                app.close()
            except Exception as exc:
                close = {"status": "failed", "error": str(exc)}
        except Exception as exc:
            close = {"status": "failed", "error": str(exc)}
        manifest["simulation_app_close"] = close
        _save_json(out / "run.json", manifest)
