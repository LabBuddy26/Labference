"""Camera reset and bounded RGB readiness checks for the weighing scene."""

import logging

import numpy as np


def initialize_camera(camera):
    if camera.get_render_product_path():
        camera.remove_rgb_from_frame()
    camera.initialize()
    camera.set_frequency(-1)
    camera.resume()


def capture_frames(cameras, render, max_render_frames=60):
    if max_render_frames < 1:
        raise ValueError("max_render_frames must be positive")
    pending = list(cameras)
    for recovery in range(2):
        if recovery:
            logging.warning("Rebinding cameras with missing/black RGB frames: %s", ", ".join(pending))
            for name in pending:
                initialize_camera(cameras[name])
        for _ in range(max_render_frames):
            render()
            frames = {}
            pending = []
            for name, camera in cameras.items():
                frame = np.asarray(camera.get_rgba())
                valid_shape = (frame.ndim == 3 and frame.shape[0] > 0
                               and frame.shape[1] > 0 and frame.shape[2] >= 3)
                if (not valid_shape or not np.isfinite(frame[:, :, :3]).all()
                        or not np.any(frame[:, :, :3] > 0)):
                    pending.append(name)
                else:
                    frames[name] = np.array(frame, copy=True)
            if not pending:
                return frames
    raise RuntimeError(
        f"Camera RGB unavailable or entirely black after {2 * max_render_frames} renders: "
        f"{', '.join(pending)}. Check camera render products and RTX renderer; "
        "observation rejected before Codex inference."
    )
