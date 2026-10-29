from __future__ import annotations

from .nmr_reasoning_task import NMRReasoningTask


class NMRRackTask(NMRReasoningTask):
    """R09: map a grid label to the corresponding physical rack socket."""

    TARGET_LABELS = (
        "column 5 row 3",
        "column 4 row 3",
        "column 3 row 3",
        "column 5 row 2",
        "column 5 row 4",
    )
    TASK_TEMPLATE = "Insert the NMR tube into rack {target_label}"
