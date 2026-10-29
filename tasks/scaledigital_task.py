from __future__ import annotations

import random

from .display_button_task import DisplayButtonTask


class ScaleDigitalTask(DisplayButtonTask):
    """R01: read a displayed weight and press its matching button."""

    INSTRUMENT_NAME = "ScaleInstrument"
    BUTTON_TARGETS = {
        "100 g": "TargetPoint",
        "200 g": "TargetPoint_01",
        "300 g": "TargetPoint_02",
        "400 g": "TargetPoint_03",
    }
    FONT_SIZE = 100

    def _select_episode(self) -> tuple[str, str, str]:
        button = random.choice(tuple(self.BUTTON_TARGETS))
        return button, button, f"The scale displays {button}; press the {button} button"
