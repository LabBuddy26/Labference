"""Stream one camera mosaic per episode, sampled in simulation time."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .cameras import capture_frames
from .runner import _rgb


class EpisodeVideo:
    def __init__(self, path, cameras, render, physics_hz, fps=10):
        if not 0 < fps <= physics_hz:
            raise ValueError("video fps must be positive and no higher than physics_hz")
        if not cameras:
            raise ValueError("episode video requires at least one camera")
        import imageio.v2 as imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cameras = cameras
        self.render = render
        self.physics_hz = physics_hz
        self.fps = fps
        self.last_step = None
        self.frame_count = 0
        self.writer = imageio.get_writer(
            str(self.path), format="FFMPEG", mode="I", fps=fps,
            codec="libx264", macro_block_size=2, ffmpeg_log_level="error",
            output_params=["-movflags", "+faststart"],
        )

    def capture(self, step, *, force=False):
        if self.writer is None or step == self.last_step:
            return
        # Integer time buckets also handle rates that do not divide physics_hz.
        # Repeated renders during policy/network waits never add video frames.
        if (not force and self.last_step is not None
                and step * self.fps // self.physics_hz
                <= self.last_step * self.fps // self.physics_hz):
            return
        frames = capture_frames(self.cameras, self.render)
        images = {name: _rgb(frame) for name, frame in frames.items()}
        width = max(frame.shape[1] for frame in images.values())
        height = max(frame.shape[0] for frame in images.values())
        # Even dimensions are required by the H.264 yuv420p encoder.
        width += width % 2
        height += height % 2
        columns = min(2, len(images))
        rows = (len(images) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * width, rows * height + 24))
        draw = ImageDraw.Draw(canvas)
        for index, (name, frame) in enumerate(images.items()):
            x, y = index % columns * width, index // columns * height
            canvas.paste(Image.fromarray(frame), (x, y))
            draw.rectangle((x, y, x + 100, y + 20), fill="black")
            draw.text((x + 4, y + 4), name, fill="white")
        draw.text((4, rows * height + 4),
                  f"step={step}  simulation time={step / self.physics_hz:.3f}s",
                  fill="white")
        self.writer.append_data(np.asarray(canvas))
        self.last_step = step
        self.frame_count += 1

    def close(self):
        writer, self.writer = self.writer, None
        if writer is not None:
            writer.close()
