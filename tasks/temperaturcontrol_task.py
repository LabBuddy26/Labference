from __future__ import annotations

import random

from .display_button_task import DisplayButtonTask


class TemperaturControlTask(DisplayButtonTask):
    """R03: choose Wait/Open from a randomized temperature threshold."""

    def _select_episode(self) -> tuple[str, str, str]:
        temperature = random.uniform(0.0, 50.0)
        button = "Open" if temperature > 30.0 else "Wait"
        value = f"{temperature:.1f} C"
        instruction = (
            f"The display reads {temperature:.1f} degrees Celsius; "
            f"press the {button} button"
        )
        return value, button, instruction
