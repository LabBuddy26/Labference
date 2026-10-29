from __future__ import annotations

import numpy as np

from .multi_object_task_controller import MultiObjectTaskController, PhaseSpec


class ClearResetTaskController(MultiObjectTaskController):
    """Clear the bench: put the vial into the sample box and the tube onto the
    tube rack.
    """

    PHASE_SPECS = (
        PhaseSpec("pick_vial", "pick", object_key="object_position", size_key="object_size"),
        PhaseSpec(
            "place_vial", "insert",
            object_key="object_position", target_key="target_position",
            socket_key="vial_socket_position",
            # Release above the settled centre and let gravity complete the
            # last part instead of holding the vial against the aperture.
            pre_release_offset_z=0.025,
        ),
        PhaseSpec("pick_tube", "pick", object_key="object1_position", size_key="object1_size", name_key="object1_name"),
        # The tube must sink deeper into the rack slot than the vial does
        # into the sample box (LabBench used 0.06 here).
        PhaseSpec(
            "place_tube", "insert",
            object_key="object1_position", size_key="object1_size",
            target_key="target1_position",
            inserted_center_key="target1_inserted_center",
            socket_key="tube_socket_position",
            # At this centre the tube bottom is about 20 mm through the upper
            # guide plate. Gravity can then seat it on the lower support.
            pre_release_offset_z=0.115,
            # The target rack is near Franka's forward workspace boundary.
            # A 0.30 m approach raises the Cartesian target beyond reliable
            # reach; 0.18 m still clears the rack while remaining reachable.
            pre_place_z=0.18,
            place_offset_z=0.06,
        ),
    )
    SUCCESS_CFG_SECTION: str = "clear_reset_success"
    LANGUAGE_INSTRUCTION: str = "Clear the bench: put the vial into the sample box and the tube onto the rack"

    def _check_phase_success(self, state, spec: PhaseSpec) -> bool:
        if spec.name != "place_tube":
            return super()._check_phase_success(state, spec)

        tube_center = np.asarray(state[spec.object_key], dtype=np.float64)
        tube_size = np.asarray(state[spec.size_key], dtype=np.float64)
        socket = np.asarray(state[spec.socket_key], dtype=np.float64)
        section = self._success_cfg()
        xy_error = float(np.linalg.norm(tube_center[:2] - socket[:2]))
        tube_bottom_z = float(tube_center[2] - 0.5 * tube_size[2])
        insertion_depth = float(socket[2] - tube_bottom_z)
        return bool(
            xy_error <= float(section.place_xy_tolerance_m)
            and insertion_depth
            >= float(section.tube_minimum_insertion_depth_m)
        )

    def _on_phase_completion(self, state, spec: PhaseSpec) -> None:
        if spec.name != "place_tube":
            return
        tube_center = np.asarray(state[spec.object_key], dtype=np.float64)
        tube_size = np.asarray(state[spec.size_key], dtype=np.float64)
        socket = np.asarray(state[spec.socket_key], dtype=np.float64)
        xy_error = float(np.linalg.norm(tube_center[:2] - socket[:2]))
        tube_bottom_z = float(tube_center[2] - 0.5 * tube_size[2])
        insertion_depth = float(socket[2] - tube_bottom_z)
        print(
            "[ClearResetTaskController] tube placement diagnostics: "
            f"socket={np.array2string(socket, precision=6)}, "
            f"tube_center={np.array2string(tube_center, precision=6)}, "
            f"tube_bottom_z={tube_bottom_z:.6f}, "
            f"insertion_depth={insertion_depth:.6f} m/"
            f"{float(self._success_cfg().tube_minimum_insertion_depth_m):.6f} m, "
            f"xy_error={xy_error:.6f} m/"
            f"{float(self._success_cfg().place_xy_tolerance_m):.6f} m",
            flush=True,
        )
