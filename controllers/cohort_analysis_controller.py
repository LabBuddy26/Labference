from __future__ import annotations

from .multi_object_task_controller import MultiObjectTaskController, PhaseSpec


class CohortAnalysisTaskController(MultiObjectTaskController):
    """Move three vials into the three slots of the sample box.

    Each vial uses its own position/size state key (fixing the LabBench bug
    where every phase read the shared ``object_position`` for a different
    object).
    """

    PHASE_SPECS = (
        PhaseSpec("pick_vial_00", "pick", object_key="object_position", size_key="object_size"),
        PhaseSpec("place_vial_00", "place", object_key="object_position", target_key="target_position"),
        PhaseSpec(
            "pick_vial_01", "pick",
            object_key="vial_01_position", size_key="vial_01_size",
            name_key="vial_01_name",
        ),
        PhaseSpec("place_vial_01", "place", object_key="vial_01_position", target_key="target_01_position"),
        PhaseSpec(
            "pick_vial_02", "pick",
            object_key="vial_02_position", size_key="vial_02_size",
            name_key="vial_02_name",
        ),
        PhaseSpec("place_vial_02", "place", object_key="vial_02_position", target_key="target_02_position"),
    )
    SUCCESS_CFG_SECTION: str = "cohort_success"
    LANGUAGE_INSTRUCTION: str = "Move the three vials into the sample box slots in order"
