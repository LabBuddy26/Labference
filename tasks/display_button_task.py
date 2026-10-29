from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from PIL import Image, ImageDraw, ImageFont
from pxr import Sdf, UsdShade

from .base_task import BaseTask


class DisplayButtonTask(BaseTask):
    """Shared safe display rendering and semantic button selection."""

    INSTRUMENT_NAME = "TempInstrument"
    BUTTON_TARGETS = {
        "Wait": "WaitButton/TargetPoint",
        "Open": "OpenButton/TargetPoint",
    }
    FONT_SIZE = 112

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.root_sub_obj_path = str(cfg.sub_obj_path)
        prefix = f"{self.root_sub_obj_path}/{self.INSTRUMENT_NAME}/FrontPanel"
        self.button_paths = {
            label: f"{prefix}/{relative_path}"
            for label, relative_path in self.BUTTON_TARGETS.items()
        }
        self.selected_button = next(iter(self.button_paths))
        self.display_value = ""
        self.task_instruction = ""
        self.sub_obj_path = self.button_paths[self.selected_button]
        self._texture_generation = 0

    def _select_episode(self) -> tuple[str, str, str]:
        raise NotImplementedError

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        value, button, instruction = self._select_episode()
        self.display_value = value
        self.selected_button = button
        self.task_instruction = instruction
        self.sub_obj_path = self.button_paths[button]
        self._render_display(value)

    @staticmethod
    def _font(size: int):
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ):
            if Path(path).is_file():
                return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    def _render_display(self, text: str) -> None:
        screen_path = (
            f"{self.root_sub_obj_path}/{self.INSTRUMENT_NAME}/FrontPanel/Screen"
        )
        if not self.stage.GetPrimAtPath(screen_path).IsValid():
            raise RuntimeError(f"Display prim not found: {screen_path}")

        image = Image.new("RGB", (640, 320), color=(10, 25, 40))
        draw = ImageDraw.Draw(image)
        font = self._font(self.FONT_SIZE)
        bounds = draw.textbbox((0, 0), text, font=font)
        x = (640 - (bounds[2] - bounds[0])) // 2
        y = (320 - (bounds[3] - bounds[1])) // 2 - 10
        draw.text((x, y), text, fill=(0, 255, 190), font=font)

        texture_dir = Path(str(self.cfg.multi_run.run_dir)) / "generated_textures"
        texture_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in text)
        # Never overwrite a texture that RTX may still be reading.  R05 has
        # only two display values (Done/Running), so reusing a value-based
        # filename during reset can race the renderer's resource thread and
        # trigger a native carb mutex assertion.  Publish every texture under
        # a new path, and make the file visible only after the PNG is complete.
        self._texture_generation += 1
        texture_path = texture_dir / (
            f"display_{safe_name}_{self._texture_generation:06d}.png"
        )
        temporary_path = texture_path.with_suffix(".png.tmp")
        image.save(temporary_path, format="PNG")
        os.replace(temporary_path, texture_path)

        shader_path = f"{screen_path}/Looks/Material/DiffuseTexture"
        shader = UsdShade.Shader(self.stage.GetPrimAtPath(shader_path))
        file_input = shader.GetInput("file") if shader else None
        if not file_input:
            raise RuntimeError(f"Display texture shader input not found: {shader_path}")
        file_input.Set(Sdf.AssetPath(str(texture_path.resolve())))

    def step(self) -> Dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None
        target = self.object_utils.get_object_xform_position(self.sub_obj_path)
        if target is None:
            raise RuntimeError(f"Button target prim has no transform: {self.sub_obj_path}")
        return self.get_basic_state_info(
            additional_info={
                "object_position": target,
                "object_path": self.sub_obj_path,
                "object_name": self.selected_button,
                "selected_button": self.selected_button,
                "display_value": self.display_value,
                "task_instruction": self.task_instruction,
            }
        )
