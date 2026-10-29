from __future__ import annotations

import random

from .display_button_task import DisplayButtonTask


class CentrifugeControlTask(DisplayButtonTask):
    """R04: choose increase, decrease, or confirm from the displayed RPM."""

    INSTRUMENT_NAME = "RpmInstrument"
    BUTTON_TARGETS = {
        "Increase": "TargetPoint",
        "Decrease": "TargetPoint_01",
        "Confirm": "TargetPoint_02",
    }
    FONT_SIZE = 100
    RPM_TO_BUTTON = {2000: "Increase", 3000: "Confirm", 4000: "Decrease"}

    def _select_episode(self) -> tuple[str, str, str]:
        rpm = random.choice(tuple(self.RPM_TO_BUTTON))
        button = self.RPM_TO_BUTTON[rpm]
        value = f"{rpm} RPM"
        return value, button, f"The centrifuge displays {value}; press {button}"
