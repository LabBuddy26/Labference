from __future__ import annotations

import random

from .display_button_task import DisplayButtonTask


class DeviceStatusTask(DisplayButtonTask):
    """R05: choose Wait/Open from a randomized device status."""

    def _select_episode(self) -> tuple[str, str, str]:
        status = random.choice(("Running", "Done"))
        button = "Open" if status == "Done" else "Wait"
        return status, button, f"The device status is {status}; press the {button} button"
