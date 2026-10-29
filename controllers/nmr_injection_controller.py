from __future__ import annotations

from typing import Optional

from .single_object_place_controller import SingleObjectPlaceTaskController


class NMRInjectionTaskController(SingleObjectPlaceTaskController):
    """Pick the NMR machine from the table and insert it into the sample box.

    The place target is the box opening (``target_position`` carries the box
    offset from the task); the deeper ``place_offset_z`` lets the machine sink
    into the box before the gripper opens.
    """

    SUCCESS_CFG_SECTION: str = "nmr_injection_success"
    LANGUAGE_INSTRUCTION: str = (
        "Pick up the NMR machine from the table and insert it into the sample box"
    )
    PLACE_OFFSET_Z: float = 0.07

    def get_language_instruction(self) -> Optional[str]:
        if self.state and self.state.get("task_instruction"):
            return str(self.state["task_instruction"])
        return self.LANGUAGE_INSTRUCTION
