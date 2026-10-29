import os
from .base_task import BaseTask
from typing import Dict, Any
from pxr import UsdShade, Sdf, Gf
from PIL import Image, ImageDraw, ImageFont
from isaacsim.core.utils.prims import set_prim_visibility
import numpy as np
import random


status_list = ["Open", "Close"]

class DeviceCoverStatusTask(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        """Initialize the Pick and Pour task.

        Args:
            cfg: Configuration object for the task.
            world: The simulation world instance.
            stage: The USD stage for the simulation.
            robot: The robot instance used in the task.
        """
        super().__init__(cfg, world, stage, robot)
        self.root_sub_obj_path = cfg.sub_obj_path
        self.sub_obj_wait_path = cfg.sub_obj_path + "/TempInstrument/FrontPanel/WaitButton/TargetPoint"
        self.sub_obj_open_path = cfg.sub_obj_path + "/TempInstrument/FrontPanel/OpenButton/TargetPoint"
        self._status_texture_paths = {
            status: self._ensure_status_texture(status) for status in status_list
        }
        self._applied_texture_path = None

    def reset(self):
        """Reset the task state."""
        super().reset()
        self.robot.initialize()
        status_index = random.randint(0, 1)
        status = status_list[status_index]
        self.cover_status = status
        self.update_temperature(status)
        self.sub_obj_path = self.sub_obj_open_path if  status=="Open"  else self.sub_obj_wait_path
    
    @staticmethod
    def _ensure_status_texture(status: str) -> str:
        """Create immutable status textures once, never during rendering."""
        save_dir = os.path.join(os.path.expanduser("~"), "isaac_temp_textures")
        os.makedirs(save_dir, exist_ok=True)
        tex_path = os.path.join(save_dir, f"labbench_r12_{status}_v1.png")
        if os.path.exists(tex_path):
            return tex_path

        text = str(status)
        img = Image.new("RGB", (640, 320), color=(10, 25, 40))
        draw = ImageDraw.Draw(img)
        font = None
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        for fp in candidates:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, 120)
                    break
                except OSError:
                    pass
        if font is None:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (640 - tw) // 2
        y = (320 - th) // 2 - 10
        draw.text((x, y), text, fill=(0, 255, 190), font=font)
        img.save(tex_path)
        return tex_path

    def update_temperature(self, new_status: str):
        screen_path = f"{self.root_sub_obj_path}/TempInstrument/FrontPanel/Screen"
        if not self.stage.GetPrimAtPath(screen_path).IsValid():
            return

        tex_path = self._status_texture_paths[str(new_status)]
        if tex_path == self._applied_texture_path:
            return
        texture_shader_path = f"{screen_path}/Looks/Material/DiffuseTexture"
        tex_shader = UsdShade.Shader(self.stage.GetPrimAtPath(texture_shader_path))
        if tex_shader:
            tex_shader.GetInput("file").Set(Sdf.AssetPath(tex_path))
            self._applied_texture_path = tex_path
        else:
            return
        
        # screen_prim = self.stage.GetPrimAtPath(screen_path)
        # if screen_prim.HasAttribute("temperature"):
        #     screen_prim.GetAttribute("temperature").Set(float(new_status))
        # else:
        #     screen_prim.CreateAttribute("temperature", Sdf.ValueTypeNames.Float).Set(float(new_status))


    def get_basic_state_info(self, joint_positions: np.ndarray = None, object_path: str = None, 
                           target_path: str = None, additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get basic state information containing common state data for all tasks.
        
        Args:
            joint_positions: Robot joint positions
            object_path: Object path
            target_path: Target object path
            additional_info: Additional state information
            
        Returns:
            Dict[str, Any]: Basic state information dictionary
        """
        
        if joint_positions is None:
            joint_positions = self.robot.get_joint_positions()
            if joint_positions is None:
                return None

        
        camera_data, display_data = self.get_camera_data()
        state = {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': display_data,
            'done': self.reset_needed,
            'gripper_position': self.robot.get_gripper_position(),
        }
        
        if object_path:
            state.update({
                'object_position': self.object_utils.get_object_xform_position(object_path),
                'object_size': self.object_utils.get_object_size(object_path=object_path),
                'object_path': object_path,
                'object_name': object_path.split("/")[-1]
            })
        
        
        if target_path:
            state.update({
                'target_position': self.object_utils.get_geometry_center(object_path=target_path),
                'target_size': self.object_utils.get_object_size(object_path=target_path),
                'target_path': target_path,
                'target_name': target_path.split("/")[-1]
            })
        
        if additional_info:
            state.update(additional_info)
            
        return state
    
    def step(self):
        """Execute one simulation step.

        Returns:
            dict: A dictionary containing simulation state data, or None if not ready.
        """
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        # source_quaternion = self.object_utils.get_transform_quat(object_path=self.sub_obj_path)
        
        return self.get_basic_state_info(
            object_path=self.sub_obj_path,
            additional_info={
                "cover_status": self.cover_status,
                "expected_action": "close" if self.cover_status == "Open" else "start",
                "task_state": {
                    "cover_open": self.cover_status == "Open",
                    "expected_action": "close" if self.cover_status == "Open" else "start",
                },
            }
        )

        
