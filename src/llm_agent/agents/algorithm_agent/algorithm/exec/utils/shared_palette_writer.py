"""A ``PillowWriter`` variant that renders GIFs with one shared palette across all frames."""

from matplotlib.animation import PillowWriter
from PIL import Image


def _stack_images(frames: list[Image.Image]) -> Image.Image:
    """Vertically stack frames so one palette can be sampled from all their colors."""
    if len(frames) == 1:
        return frames[0]
    width = max(frame.width for frame in frames)
    total_height = sum(frame.height for frame in frames)
    stacked = Image.new("RGB", (width, total_height))
    offset = 0
    for frame in frames:
        stacked.paste(frame, (0, offset))
        offset += frame.height
    return stacked


class SharedPalettePillowWriter(PillowWriter):
    """PillowWriter that renders the GIF with one shared palette across all frames.

    matplotlib's default PillowWriter quantizes each frame to its own adaptive 256-color palette,
    so a fixed colormap value (e.g. a point on the colorbar) maps to slightly different RGB colors
    from frame to frame, making the color scale appear to drift even when the norm is fixed.
    Quantizing every frame against a single palette built from the whole animation keeps colors
    stable. Dithering is disabled to avoid per-frame shimmer in smooth gradients.

    Like the base PillowWriter, all frames are held in memory (``self._frames``) until ``finish``;
    this writer does not change that, it only re-quantizes the already-collected frames.
    """

    def __init__(self, *, fps: int, duration_ms: int) -> None:
        """Construct with an explicit per-frame duration (ms), independent of the integer fps."""
        super().__init__(fps=fps)
        self._duration_ms = duration_ms

    def finish(self) -> None:
        """Quantize every frame against one shared palette and write the GIF."""
        frames: list[Image.Image] = [frame.convert("RGB") for frame in self._frames]  # type: ignore[reportUnknownMemberType]
        palette_img = _stack_images(frames).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        quantized = [
            frame.quantize(palette=palette_img, dither=Image.Dither.NONE) for frame in frames
        ]
        quantized[0].save(
            self.outfile,  # type: ignore[reportUnknownArgumentType]
            save_all=True,
            append_images=quantized[1:],
            duration=self._duration_ms,
            loop=0,
        )
