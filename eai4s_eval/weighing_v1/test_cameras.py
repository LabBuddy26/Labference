"""Camera recovery tests without Isaac Sim or GPU access."""

import base64
from io import BytesIO
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from .cameras import capture_frames, initialize_camera
from .runner import _encoded_images
from .scene import SceneHandles


class CameraTests(unittest.TestCase):
    def camera(self, frames):
        camera = Mock()
        camera.get_render_product_path.return_value = "/Render/Camera"
        camera.get_rgba.side_effect = frames
        return camera

    def test_reset_rebinds_existing_rgb_before_initialize(self):
        camera = Mock()
        camera.get_render_product_path.return_value = "/Render/Camera"
        initialize_camera(camera)
        calls = [call[0] for call in camera.mock_calls]
        self.assertLess(calls.index("remove_rgb_from_frame"), calls.index("initialize"))
        camera.set_frequency.assert_called_once_with(-1)
        camera.resume.assert_called_once()

    def test_first_initialization_does_not_detach_missing_render_product(self):
        camera = Mock()
        camera.get_render_product_path.return_value = None
        initialize_camera(camera)
        camera.remove_rgb_from_frame.assert_not_called()
        camera.initialize.assert_called_once()

    def test_scene_reset_rebinds_all_camera_outputs(self):
        handles = SceneHandles.__new__(SceneHandles)
        handles.franka = Mock()
        handles.sample = Mock()
        handles.weights = {}
        handles.tray = Mock()
        handles.cameras = {name: self.camera([]) for name in ("front", "wrist", "display")}
        handles.initialize()
        handles.initialize()
        for camera in handles.cameras.values():
            self.assertEqual(camera.remove_rgb_from_frame.call_count, 2)
            self.assertEqual(camera.initialize.call_count, 2)
            self.assertEqual(camera.resume.call_count, 2)

    def test_waits_until_every_camera_has_rgb_even_if_alpha_is_nonzero(self):
        black = np.zeros((2, 3, 4), dtype=np.uint8)
        black[:, :, 3] = 255
        ready = np.full((2, 3, 4), 120, dtype=np.uint8)
        cameras = {"front": self.camera([black, ready, ready]),
                   "display": self.camera([None, black, ready]),
                   "wrist": self.camera([ready, ready, ready])}
        render = Mock()
        frames = capture_frames(cameras, render, max_render_frames=5)
        self.assertEqual(render.call_count, 3)
        self.assertEqual(set(frames), set(cameras))
        for camera in cameras.values():
            camera.initialize.assert_not_called()
        ready[:] = 0
        self.assertTrue(np.all(frames["front"] == 120))

    def test_rebinds_only_black_cameras_and_recaptures_current_frames(self):
        black = np.zeros((2, 3, 4), dtype=np.uint8)
        ready = np.full((2, 3, 4), 120, dtype=np.uint8)
        front = self.camera([black, black, ready])
        wrist = self.camera([ready, ready, ready])
        with self.assertLogs(level="WARNING"):
            frames = capture_frames({"front": front, "wrist": wrist}, Mock(), max_render_frames=2)
        front.remove_rgb_from_frame.assert_called_once()
        front.initialize.assert_called_once()
        wrist.initialize.assert_not_called()
        self.assertTrue(np.any(frames["front"][:, :, :3]))

    def test_persistent_black_or_invalid_frames_fail_with_camera_name(self):
        for bad in (np.zeros((2, 3, 4), dtype=np.uint8),
                    np.full((2, 3, 4), np.nan), np.empty((0, 0, 4))):
            with self.subTest(shape=bad.shape):
                camera = self.camera([bad] * 4)
                render = Mock()
                with self.assertLogs(level="WARNING"), self.assertRaisesRegex(RuntimeError, "display"):
                    capture_frames({"display": camera}, render, max_render_frames=2)
                self.assertEqual(render.call_count, 4)
                camera.initialize.assert_called_once()

    def test_encoding_uses_validated_snapshot_not_new_sensor_read(self):
        cameras = {name: Mock() for name in ("front", "wrist", "display")}
        frames = {name: np.full((2, 3, 4), 120, dtype=np.uint8) for name in cameras}
        images = _encoded_images(cameras, frames=frames)
        for name, camera in cameras.items():
            camera.get_rgba.assert_not_called()
            decoded = np.asarray(Image.open(BytesIO(base64.b64decode(images[name]["base64"]))))
            self.assertEqual(decoded.shape, (2, 3, 3))
            self.assertTrue(np.all(decoded == 120))


if __name__ == "__main__":
    unittest.main()
