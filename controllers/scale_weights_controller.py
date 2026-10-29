from __future__ import annotations

import numpy as np

from .multi_object_task_controller import MultiObjectTaskController, PhaseSpec


class ScaleWeightsTaskController(MultiObjectTaskController):
    """Move three weights onto the three scale pads.

    Uses the corrected ``weight_01/02_*`` state keys instead of the vial keys
    that LabBench copied from the cohort task.
    """

    # Place offsets follow the LabBench originals: the heavier weights sit
    # higher on their pads, so each place target is raised less; the third
    # weight is lifted lower because it is already near the scale.
    PHASE_SPECS = (
        PhaseSpec(
            "pick_weight_0", "pick",
            object_key="object_position", size_key="object_size",
            gripper_distance=0.002,
        ),
        PhaseSpec(
            "place_weight_0", "place",
            object_key="object_position", target_key="target_position",
            place_offset_z=0.01,
        ),
        PhaseSpec(
            "pick_weight_1", "pick",
            object_key="weight_01_position", size_key="weight_01_size",
            name_key="weight_01_name",
            # The scaled 20 g weight is about 28.6 mm wide. A 20 mm
            # per-finger target leaves the jaws open too far to establish
            # contact, so command 10 mm per finger instead.
            gripper_distance=0.010,
        ),
        PhaseSpec(
            "place_weight_1", "place",
            object_key="weight_01_position", target_key="target_01_position",
            place_offset_z=0.02,
        ),
        PhaseSpec(
            "pick_weight_2", "pick",
            object_key="weight_02_position", size_key="weight_02_size",
            name_key="weight_02_name",
            # The default offset is 2/5 of the 75 mm object height (30 mm),
            # which pinches the 50 g weight too close to its top. Grasp 20 mm
            # lower, at 10 mm above the geometry center.
            pick_z_offset=0.010,
            # Lift beyond the configured 0.12 m pick-success threshold.
            after_offset_z=0.15,
            # The scaled 50 g weight is about 45 mm wide.
            gripper_distance=0.010,
        ),
        PhaseSpec("place_weight_2", "place", object_key="weight_02_position", target_key="target_02_position"),
    )
    SUCCESS_CFG_SECTION: str = "scale_weights_success"
    LANGUAGE_INSTRUCTION: str = "Move the three weights onto the scale"

    @staticmethod
    def _contact_force(sensor):
        """Return the latest scalar contact force, or ``None`` if unavailable."""
        try:
            frame = sensor.get_current_frame() or {}
            force = frame.get("force")
            return None if force is None else float(force)
        except Exception as exc:
            print(
                "[ScaleWeightsTaskController] failed to read contact sensor: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return None

    def _log_third_pick_diagnostics(self, state, spec: PhaseSpec) -> None:
        initial = self._initial_positions.get(spec.name)
        current = np.asarray(state[spec.object_key], dtype=np.float64)
        initial_z = None if initial is None else float(initial[2])
        current_z = float(current[2])
        lift = None if initial_z is None else current_z - initial_z

        left_sensor, right_sensor = self.robot.get_contact_sensor()
        left_force = self._contact_force(left_sensor)
        right_force = self._contact_force(right_sensor)

        joints = np.asarray(state["joint_positions"], dtype=np.float64)
        finger_indices = list(self.robot.gripper.joint_dof_indicies)
        finger_positions = [
            float(joints[index]) if 0 <= index < joints.size else None
            for index in finger_indices
        ]
        required_lift = float(self._success_cfg().pick_lift_height_m)
        print(
            "[ScaleWeightsTaskController] pick_weight_2 completion diagnostics: "
            f"initial_z={initial_z}, current_z={current_z}, "
            f"lift_height={lift}, required_lift={required_lift}, "
            f"left_contact_force_n={left_force}, "
            f"right_contact_force_n={right_force}, "
            f"finger_joint_indices={finger_indices}, "
            f"finger_joint_positions={finger_positions}",
            flush=True,
        )

    def _on_phase_completion(self, state, spec: PhaseSpec) -> None:
        if spec.name == "pick_weight_2":
            self._log_third_pick_diagnostics(state, spec)
