from __future__ import annotations

from .multi_object_task_controller import MultiObjectTaskController, PhaseSpec


class CentrifugeLoadHardTaskController(MultiObjectTaskController):
    """Load the two loose NMR tubes into the occupied centrifuge rotor."""

    SUCCESS_CFG_SECTION = "centrifuge_success"
    LANGUAGE_INSTRUCTION = (
        "Load both NMR tubes from the table into the two empty centrifuge slots"
    )
    PHASE_SPECS = (
        PhaseSpec(
            "pick_nmr_01", "pick", "object_position",
            gripper_distance=0.001, orientation_deg=(0.0, 90.0, 30.0),
            pre_offset_x=0.03, pre_offset_z=0.10, after_offset_z=0.20,
        ),
        PhaseSpec(
            "place_nmr_01", "place", "object_position",
            target_key="target1_position", orientation_deg=(0.0, 90.0, 90.0),
            # Keep the hard two-tube protocol on the proven L07 insertion
            # trajectory: the hand stays 70 mm above the tube centre at
            # release so the tube, rather than the fingers, enters the hole.
            pre_place_z=0.20, place_offset_z=0.07,
        ),
        PhaseSpec(
            "pick_nmr_02", "pick", "object2_position",
            size_key="object2_size", name_key="object2_name",
            gripper_distance=0.001, orientation_deg=(0.0, 90.0, 30.0),
            pre_offset_x=0.03, pre_offset_z=0.10, after_offset_z=0.20,
        ),
        PhaseSpec(
            "place_nmr_02", "place", "object2_position",
            target_key="target2_position", orientation_deg=(0.0, 90.0, 90.0),
            pre_place_z=0.20, place_offset_z=0.07,
        ),
    )

    def _check_phase_success(self, state, spec):
        if spec.kind == "place":
            import numpy as np
            obj = np.asarray(state.get(spec.object_key), dtype=float)
            target = np.asarray(state.get(spec.target_key), dtype=float)
            xy_error = float(np.linalg.norm(obj[:2] - target[:2]))
            z_error = float(abs(obj[2] - target[2]))
            return bool(xy_error <= 0.04 and z_error <= 0.12)
        return super()._check_phase_success(state, spec)
