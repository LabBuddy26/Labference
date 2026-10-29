from __future__ import annotations

from typing import Optional

from .single_object_place_controller import SingleObjectPlaceTaskController


class SampleOrganitionTaskController(SingleObjectPlaceTaskController):
    """Pick the vial and place it into the target sample box.

    The language instruction is dynamic: it names the box chosen by the
    episode seed (A or B).
    """

    SUCCESS_CFG_SECTION: str = "sample_organization_success"
    LANGUAGE_INSTRUCTION: str = "Pick up the vial and place it into the target sample box"
    # Each Franka finger closes by an additional 1 mm, reducing the total
    # gripper opening by 2 mm compared with the default vial setting.
    GRIPPER_DISTANCE: float = 0.002

    def get_language_instruction(self) -> Optional[str]:
        if self.state is not None and self.state.get("target_box_name"):
            return (
                f"Pick up the vial and place it into the "
                f"{self.state['target_box_name']} sample box"
            )
        return self.LANGUAGE_INSTRUCTION or None
