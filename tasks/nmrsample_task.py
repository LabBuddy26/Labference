from __future__ import annotations

from .nmr_reasoning_task import NMRReasoningTask


class NMRSampleTask(NMRReasoningTask):
    """R08: map a sample-slot instruction to a physical socket."""

    TARGET_LABELS = ("slot 1", "slot 2", "slot 3", "slot 4", "slot 5")
    TASK_TEMPLATE = "Read the sample instruction and insert the NMR tube into {target_label}"
