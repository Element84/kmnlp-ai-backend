# ruff: noqa: D102

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from functools import partial
from os.path import basename
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

import cartopy.crs  # pyright: ignore[reportMissingTypeStubs]
import matplotlib as mpl
import numpy as np
import seaborn as sns
from anyio import to_thread
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.axes import Axes
from pydantic import BaseModel

from llm_agent.agents.algorithm_agent.algorithm.exec.base import ExecutionContext, eval_expression
from llm_agent.agents.algorithm_agent.algorithm.exec.data_array import (
    DataArrayShapeError,
    materialize_data_array_var,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.plot import (
    annotate_plot,
    is_lat_lon_data_array,
    setup_geospatial_axes,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.utils.shared_palette_writer import (
    SharedPalettePillowWriter,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    AnimationSpec,
    BarPlotSpec,
    GeneralRasterPlotSpec,
    GeospatialRasterPlotSpec,
    LinePlotSpec,
    MakeAnimation,
    MultiLinePlotSpec,
    PlotSpec,
)
from llm_agent.tools.data_analysis_tool.types import AnimationRef

mpl.use("Agg")

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from matplotlib.collections import QuadMesh
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

ANIMATION_FORMAT: Literal["gif", "mp4"] = "mp4"


class AnimatorRegistry:
    """Registry for mapping AnimationSpec types to Animator classes."""

    def __init__(self) -> None:
        """Constructor."""
        self._registry: dict[str, type[Animator[Any]]] = {}

    def register[T: PlotSpec](self, spec_type: type[T], animator_cls: "type[Animator[T]]") -> None:
        """Register an Animator class for a given PlotSpec type."""
        self._registry[spec_type.__name__] = animator_cls

    def get_animator[T: PlotSpec](self, spec_type: type[T]) -> "type[Animator[T]]":
        """Get the Animator class for a given PlotSpec type."""
        if spec_type.__name__ not in self._registry:
            msg = (
                f"No animator registered for spec type: {spec_type.__name__}."
                f"Registered spec types: {list(self._registry.keys())}."
            )
            raise ValueError(msg)
        return self._registry[spec_type.__name__]


animator_registry = AnimatorRegistry()


@eval_expression.register(MakeAnimation)
async def _(expr: MakeAnimation[Any], context: ExecutionContext) -> AnimationRef:
    """Evaluate expression."""
    anim = await make_animation(expr.animation_spec, context)

    match ANIMATION_FORMAT:
        case "gif":
            writer = _make_gif_writer(expr.animation_spec.frame_interval_seconds)
            file_ext = ".gif"
        case "mp4":
            writer = _make_mp4_writer(expr.animation_spec.frame_interval_seconds)
            file_ext = ".mp4"

    with NamedTemporaryFile(prefix="animation_", suffix=file_ext) as tmpfile:
        log.info("Saving animation to temporary file: %s", tmpfile.name)

        tic = perf_counter()
        # Run in separate thread to keep the event loop free to process progress updates.
        await to_thread.run_sync(lambda: anim.save(tmpfile.name, writer=writer))
        toc = perf_counter()
        log.info("Saved animation in %.2f seconds.", toc - tic)

        ref_key = context.tool_call_context.chat_context.persist_data(
            data_or_path=tmpfile.name, data_type="animation"
        )
        return AnimationRef(
            key=ref_key,
            name=basename(tmpfile.name),
            format=ANIMATION_FORMAT,
        )


def _make_gif_writer(frame_interval_seconds: float) -> SharedPalettePillowWriter:
    interval_seconds = frame_interval_seconds
    duration_ms = int(interval_seconds * 1000) if interval_seconds > 0 else 200
    fps = max(1, round(1000 / duration_ms))
    # Use a shared-palette writer so the color scale stays constant across frames.
    writer = SharedPalettePillowWriter(fps=fps, duration_ms=duration_ms)
    return writer


def _make_mp4_writer(frame_interval_seconds: float) -> FFMpegWriter:
    fps = max(1, round(1 / frame_interval_seconds)) if frame_interval_seconds > 0 else 5
    # H.264 in an MP4 plays reliably in browsers; yuv420p and +faststart are needed for that.
    return FFMpegWriter(
        fps=fps,
        codec="h264",
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )


class AnimationProgressUpdate(BaseModel):
    """Progress update for animation creation."""

    frame_num: int
    total_frames: int


async def make_animation[T: PlotSpec](
    animation_spec: AnimationSpec[T], context: ExecutionContext
) -> FuncAnimation:
    """Make animation."""
    fig = plt.figure()  # type: ignore[reportUnknownMemberType]

    def create_ax_func(**kwargs: Any) -> Axes:  # noqa: ANN401
        return fig.add_subplot(1, 1, 1, **kwargs)  # type: ignore[reportUnknownMemberType]

    animator_cls = animator_registry.get_animator(type(animation_spec.plot_spec))
    animator = animator_cls(animation_spec, context, create_ax_func)
    await animator.init_data()

    # Call init() here instead of passing init_func to FuncAnimation seems to work better.
    # With init_func, the first frame doesn't seem to get cleared.
    animator.init()

    return FuncAnimation(
        fig,
        func=partial(animator.update_with_progress, update_every=5),
        frames=animator.nframes,  # type: ignore[reportUnknownMemberType]
        blit=True,
        interval=int(animation_spec.frame_interval_seconds * 1000),
    )


def _compute_trendline_y(
    y: "np.ndarray[Any, Any]",
) -> "tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]":
    """Compute trendline y-values for integer-indexed data."""
    x = np.arange(len(y))
    mask = np.isfinite(y)
    coeffs = np.polyfit(x[mask], y[mask], 1)
    return np.polyval(coeffs, x), coeffs  # type: ignore[reportReturnType]


class Animator[T: PlotSpec](ABC):
    """Animator base class."""

    spec: AnimationSpec[T]
    context: ExecutionContext
    create_ax_func: Callable[..., Axes]

    def __init__(  # noqa: D107
        self,
        spec: AnimationSpec[T],
        context: ExecutionContext,
        create_ax_func: Callable[..., Axes],
    ) -> None:
        self.spec = spec
        self.context = context
        self.create_ax_func = create_ax_func

    @property
    @abstractmethod
    def nframes(self) -> int:
        """Number of frames in the animation."""

    @abstractmethod
    def init(self) -> Iterable[Any]:
        """Create first frame."""

    @abstractmethod
    async def init_data(self) -> None:
        """Load and validate data arrays. Called by make_animation after construction."""

    @abstractmethod
    def update(self, frame: int) -> Iterable[Any]:
        """Create i-th frame."""

    def update_with_progress(self, frame: int, update_every: int) -> Iterable[Any]:
        """Update frame and emit progress update."""
        if frame % update_every == 0:
            update = AnimationProgressUpdate(frame_num=frame + 1, total_frames=self.nframes)
            log.info("Animation progress update: %s", update)
            self.context.tool_call_context.update_progress(update)
        return self.update(frame)


def _errorbar_artists(container: Any) -> list[Any]:  # noqa: ANN401
    """Extract non-None artists from an ErrorbarContainer.

    ErrorbarContainer iterates as (data_line, caplines, barlinecols).
    data_line is None when fmt="none", which would crash FuncAnimation's
    set_animated call.
    """
    data_line, caplines, barlinecols = container
    artists: list[Any] = []
    if data_line is not None:
        artists.append(data_line)
    artists.extend(caplines)
    artists.extend(barlinecols)
    return artists


class BarPlotAnimator(Animator[BarPlotSpec]):
    """Bar plot animator."""

    # TODO: resolve type
    height_arrays: Any

    def __init__(  # noqa: D107
        self,
        spec: AnimationSpec[BarPlotSpec],
        context: ExecutionContext,
        create_ax_func: Callable[..., Axes],
    ) -> None:
        super().__init__(spec, context, create_ax_func)

    async def init_data(self) -> None:
        self.height_arrays = [
            await materialize_data_array_var(h, self.context) for h in self.spec.plot_spec.heights
        ]
        if not all(h.ndim == 1 for h in self.height_arrays):
            msg = (
                "Expected all bar plot height sequences to be 1D. "
                f"Found shapes: {[h.shape for h in self.height_arrays]}."
            )
            raise ValueError(msg)

        lens = [len(h) for h in self.height_arrays]
        if min(lens) != max(lens):
            msg = f"Expected all bar plot height sequences to be of the same length. Found: {lens}."
            raise ValueError(msg)
        self.frame_dim = self.spec.frame_dim
        self.frame_vals = self.height_arrays[0][self.frame_dim].values  # type: ignore[reportUnknownMemberType]

    @property
    def nframes(self) -> int:
        """Number of frames in the animation."""
        return len(self.height_arrays[0])

    def init(self) -> "Iterable[Rectangle]":
        """Create first frame."""
        self.ax = self.create_ax_func()
        y_min = min(float(h.min()) for h in self.height_arrays)  # type: ignore[reportUnknownMemberType]
        y_max = max(float(h.max()) for h in self.height_arrays)  # type: ignore[reportUnknownMemberType]
        margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        self.ax.set_ylim(y_min - margin, y_max + margin)  # type: ignore[reportUnknownMemberType]
        self.bars = self.ax.bar(  # type: ignore[reportUnknownMemberType]
            x=self.spec.plot_spec.labels,
            height=[h[0].values.item() for h in self.height_arrays],
        )
        annotate_plot(
            self.ax,
            self.spec.plot_spec.annotation,
            title_postfix=f"\n{self.frame_dim}: {self.frame_vals[0]}",
        )
        sns.despine(ax=self.ax)
        return self.bars.patches

    def update(self, frame: int) -> "Iterable[Rectangle]":
        """Create i-th frame."""
        new_heights = [h[frame].values.item() for h in self.height_arrays]
        for bar, new_height in zip(self.bars.patches, new_heights, strict=True):
            bar.set_height(new_height)
        annotate_plot(
            self.ax,
            self.spec.plot_spec.annotation,
            title_postfix=f"\n{self.frame_dim}: {self.frame_vals[frame]}",
        )
        return self.bars.patches


class LinePlotAnimator(Animator[LinePlotSpec]):
    """Line plot animator for animating over a dimension of a 2D DataArray."""

    def __init__(  # noqa: D107
        self,
        spec: AnimationSpec[LinePlotSpec],
        context: ExecutionContext,
        create_ax_func: Callable[..., Axes],
    ) -> None:
        super().__init__(spec, context, create_ax_func)

    async def init_data(self) -> None:
        data_array = await materialize_data_array_var(self.spec.plot_spec.data_array, self.context)
        data_array = data_array.squeeze().load()  # type: ignore[reportUnknownMemberType]
        if data_array.ndim != 2:  # noqa: PLR2004
            raise DataArrayShapeError.with_shape(
                func_name="MakeAnimation",
                dims=data_array.dims,  # pyright: ignore[reportArgumentType]
                shape=data_array.shape,
                msg_prefix="Expected a 2D DataArray for line animation.",
            )
        self.data_array = data_array
        self.frame_dim = self.spec.frame_dim
        self.frame_vals = data_array[self.frame_dim].values  # type: ignore[reportUnknownMemberType]

        # Resolve optional error array
        self.error_array = None
        if self.spec.plot_spec.data_array_error is not None:
            err = await materialize_data_array_var(
                self.spec.plot_spec.data_array_error, self.context
            )
            err = err.squeeze().load()  # type: ignore[reportUnknownMemberType]
            if err.ndim != 2:  # noqa: PLR2004
                raise DataArrayShapeError.with_shape(
                    func_name="MakeAnimation",
                    dims=err.dims,  # pyright: ignore[reportArgumentType]
                    shape=err.shape,
                    msg_prefix="Expected a 2D error DataArray for line animation.",
                )
            self.error_array = err

        # Pre-compute global y-limits across all frames (including error)
        y_min = float(data_array.min())  # type: ignore[reportUnknownMemberType]
        y_max = float(data_array.max())  # type: ignore[reportUnknownMemberType]
        if self.error_array is not None:
            max_err = float(self.error_array.max())  # type: ignore[reportUnknownMemberType]
            y_min -= max_err
            y_max += max_err
        self._y_min = y_min
        self._y_max = y_max

    @property
    def nframes(self) -> int:
        """Number of frames in the animation."""
        return len(self.frame_vals)

    def init(self) -> Iterable[Any]:
        """Create first frame."""
        self.ax = self.create_ax_func()
        margin = (self._y_max - self._y_min) * 0.1 if self._y_max > self._y_min else 1.0
        self.ax.set_ylim(self._y_min - margin, self._y_max + margin)  # type: ignore[reportUnknownMemberType]
        frame_data = self.data_array.isel({self.frame_dim: 0})
        (self.line,) = self.ax.plot(  # type: ignore[reportUnknownMemberType]
            frame_data,
            color=self.spec.plot_spec.style.color,
            ls=self.spec.plot_spec.style.linestyle,
            lw=self.spec.plot_spec.style.linewidth,
        )

        artists: list[Any] = [self.line]
        self._error_artist = None
        if self.error_array is not None:
            err_frame = self.error_array.isel({self.frame_dim: 0}).values  # type: ignore[reportUnknownMemberType]
            y = frame_data.values  # type: ignore[reportUnknownMemberType]
            x = np.arange(len(y))
            match self.spec.plot_spec.error_display_type:
                case "shaded":
                    self._error_artist = self.ax.fill_between(  # type: ignore[reportUnknownMemberType]
                        x,
                        y - err_frame,
                        y + err_frame,
                        alpha=0.2,
                        color=self.spec.plot_spec.style.color,
                    )
                    artists.append(self._error_artist)
                case "errorbars":
                    self._error_artist = self.ax.errorbar(  # type: ignore[reportUnknownMemberType]
                        x,
                        y,
                        yerr=err_frame,
                        ecolor=self.spec.plot_spec.style.color,
                        fmt="none",
                        capsize=2,
                    )
                    artists.extend(_errorbar_artists(self._error_artist))

        self._trend_line: Line2D | None = None
        if self.spec.plot_spec.add_linear_trendline:
            y = frame_data.values  # type: ignore[reportUnknownMemberType]
            trend_y, coeffs = _compute_trendline_y(y)
            label = f"y={coeffs[0]:.2f}x+{coeffs[1]:.2f}"
            (self._trend_line,) = self.ax.plot(  # type: ignore[reportUnknownMemberType]
                np.arange(len(y)),
                trend_y,
                color=self.spec.plot_spec.style.color,
                ls=":",
                lw=1,
                alpha=0.7,
                label=label,
            )
            artists.append(self._trend_line)

        annotate_plot(
            self.ax,
            self.spec.plot_spec.annotation,
            title_postfix=f"\n{self.frame_dim}: {self.frame_vals[0]}",
        )
        sns.despine(ax=self.ax)
        return artists

    def update(self, frame: int) -> Iterable[Any]:
        """Create i-th frame."""
        frame_data = self.data_array.isel({self.frame_dim: frame})
        self.line.set_ydata(frame_data.values)  # type: ignore[reportUnknownMemberType]

        artists: list[Any] = [self.line]
        if self.error_array is not None and self._error_artist is not None:
            self._error_artist.remove()  # type: ignore[reportUnknownMemberType]
            err_frame = self.error_array.isel({self.frame_dim: frame}).values  # type: ignore[reportUnknownMemberType]
            y = frame_data.values  # type: ignore[reportUnknownMemberType]
            x = np.arange(len(y))
            match self.spec.plot_spec.error_display_type:
                case "shaded":
                    self._error_artist = self.ax.fill_between(  # type: ignore[reportUnknownMemberType]
                        x,
                        y - err_frame,
                        y + err_frame,
                        alpha=0.2,
                        color=self.spec.plot_spec.style.color,
                    )
                    artists.append(self._error_artist)
                case "errorbars":
                    self._error_artist = self.ax.errorbar(  # type: ignore[reportUnknownMemberType]
                        x,
                        y,
                        yerr=err_frame,
                        ecolor=self.spec.plot_spec.style.color,
                        fmt="none",
                        capsize=2,
                    )
                    artists.extend(_errorbar_artists(self._error_artist))

        if self._trend_line is not None:
            y = frame_data.values  # type: ignore[reportUnknownMemberType]
            trend_y, coeffs = _compute_trendline_y(y)
            self._trend_line.set_ydata(trend_y)  # type: ignore[reportUnknownMemberType]
            self._trend_line.set_label(f"y={coeffs[0]:.2f}x+{coeffs[1]:.2f}")
            artists.append(self._trend_line)

        annotate_plot(
            self.ax,
            self.spec.plot_spec.annotation,
            title_postfix=f"\n{self.frame_dim}: {self.frame_vals[frame]}",
        )
        return artists


class MultiLinePlotAnimator(Animator[MultiLinePlotSpec]):
    """Multi-line plot animator for animating multiple lines over a shared dimension."""

    def __init__(  # noqa: D107
        self,
        spec: AnimationSpec[MultiLinePlotSpec],
        context: ExecutionContext,
        create_ax_func: Callable[..., Axes],
    ) -> None:
        super().__init__(spec, context, create_ax_func)

    async def init_data(self) -> None:
        self.frame_dim = self.spec.frame_dim

        self.data_arrays: list[Any] = []
        for line_spec in self.spec.plot_spec.specs:
            data_array = await materialize_data_array_var(line_spec.data_array, self.context)
            data_array = data_array.squeeze().load()  # type: ignore[reportUnknownMemberType]
            if data_array.ndim != 2:  # noqa: PLR2004
                raise DataArrayShapeError.with_shape(
                    func_name="MakeAnimation",
                    dims=data_array.dims,  # pyright: ignore[reportArgumentType]
                    shape=data_array.shape,
                    msg_prefix="Expected a 2D DataArray for multi-line animation.",
                )
            self.data_arrays.append(data_array)

        self.frame_vals: Any = self.data_arrays[0][self.frame_dim].values  # type: ignore[reportUnknownMemberType]

        # Resolve optional error arrays for each line
        self.error_arrays: list[Any | None] = []
        for line_spec in self.spec.plot_spec.specs:
            if line_spec.data_array_error is not None:
                err = await materialize_data_array_var(line_spec.data_array_error, self.context)
                err = err.squeeze().load()  # type: ignore[reportUnknownMemberType]
                if err.ndim != 2:  # noqa: PLR2004
                    raise DataArrayShapeError.with_shape(
                        func_name="MakeAnimation",
                        dims=err.dims,  # pyright: ignore[reportArgumentType]
                        shape=err.shape,
                        msg_prefix="Expected a 2D error DataArray for multi-line animation.",
                    )
                self.error_arrays.append(err)
            else:
                self.error_arrays.append(None)

        # Pre-compute global y-limits across all lines and frames (including error)
        self._y_min = min(float(da.min()) for da in self.data_arrays)  # type: ignore[reportUnknownMemberType]
        self._y_max = max(float(da.max()) for da in self.data_arrays)  # type: ignore[reportUnknownMemberType]
        for err in self.error_arrays:
            if err is not None:
                max_err = float(err.max())  # type: ignore[reportUnknownMemberType]
                self._y_min -= max_err
                self._y_max += max_err

    @property
    def nframes(self) -> int:
        """Number of frames in the animation."""
        return len(self.frame_vals)

    def init(self) -> Iterable[Any]:
        """Create first frame."""
        self.ax = self.create_ax_func()
        margin = (self._y_max - self._y_min) * 0.1 if self._y_max > self._y_min else 1.0
        self.ax.set_ylim(self._y_min - margin, self._y_max + margin)  # type: ignore[reportUnknownMemberType]
        self.lines: list[Line2D] = []
        palette = sns.color_palette("deep", n_colors=len(self.spec.plot_spec.specs))  # type: ignore[reportUnknownMemberType]
        artists: list[Any] = []
        self._error_artists: list[Any | None] = []
        for i, (line_spec, data_array, error_array) in enumerate(
            zip(self.spec.plot_spec.specs, self.data_arrays, self.error_arrays, strict=True)
        ):
            frame_data = data_array.isel({self.frame_dim: 0})
            (line,) = self.ax.plot(  # type: ignore[reportUnknownMemberType]
                frame_data,
                color=palette[i],
                ls=line_spec.style.linestyle,
                lw=line_spec.style.linewidth,
            )
            self.lines.append(line)
            artists.append(line)

            if error_array is not None:
                err_frame = error_array.isel({self.frame_dim: 0}).values  # type: ignore[reportUnknownMemberType]
                y = frame_data.values  # type: ignore[reportUnknownMemberType]
                x = np.arange(len(y))
                match line_spec.error_display_type:
                    case "shaded":
                        error_artist = self.ax.fill_between(  # type: ignore[reportUnknownMemberType]
                            x,
                            y - err_frame,
                            y + err_frame,
                            alpha=0.2,
                            color=palette[i],
                        )
                        artists.append(error_artist)
                    case "errorbars":
                        error_artist = self.ax.errorbar(  # type: ignore[reportUnknownMemberType]
                            x,
                            y,
                            yerr=err_frame,
                            ecolor=palette[i],
                            fmt="none",
                            capsize=2,
                        )
                        artists.extend(_errorbar_artists(error_artist))
                self._error_artists.append(error_artist)
            else:
                self._error_artists.append(None)

        self._trend_lines: list[Line2D | None] = []
        for i, (line_spec, data_array) in enumerate(
            zip(self.spec.plot_spec.specs, self.data_arrays, strict=True)
        ):
            if line_spec.add_linear_trendline:
                y = data_array.isel({self.frame_dim: 0}).values  # type: ignore[reportUnknownMemberType]
                trend_y, coeffs = _compute_trendline_y(y)
                (trend_line,) = self.ax.plot(  # type: ignore[reportUnknownMemberType]
                    np.arange(len(y)),
                    trend_y,
                    color=palette[i],
                    ls=":",
                    lw=1,
                    alpha=0.7,
                    label=f"y={coeffs[0]:.2f}x+{coeffs[1]:.2f}",
                )
                self._trend_lines.append(trend_line)
                artists.append(trend_line)
            else:
                self._trend_lines.append(None)

        for line, line_spec in zip(self.lines, self.spec.plot_spec.specs, strict=True):
            if line_spec.annotation.title:
                line.set_label(line_spec.annotation.title)
        self.ax.legend()  # type: ignore[reportUnknownMemberType]

        annotate_plot(
            self.ax,
            self.spec.plot_spec.annotation,
            title_postfix=f"\n{self.frame_dim}: {self.frame_vals[0]}",
        )
        sns.despine(ax=self.ax)
        return artists

    def update(self, frame: int) -> Iterable[Any]:
        """Create i-th frame."""
        palette = sns.color_palette("deep", n_colors=len(self.spec.plot_spec.specs))  # type: ignore[reportUnknownMemberType]
        artists: list[Any] = []
        for i, (line, line_spec, data_array, error_array) in enumerate(
            zip(
                self.lines,
                self.spec.plot_spec.specs,
                self.data_arrays,
                self.error_arrays,
                strict=True,
            )
        ):
            frame_data = data_array.isel({self.frame_dim: frame})
            line.set_ydata(frame_data.values)  # type: ignore[reportUnknownMemberType]
            artists.append(line)

            if error_array is not None and self._error_artists[i] is not None:
                self._error_artists[i].remove()  # type: ignore[reportUnknownMemberType]
                err_frame = error_array.isel({self.frame_dim: frame}).values  # type: ignore[reportUnknownMemberType]
                y = frame_data.values  # type: ignore[reportUnknownMemberType]
                x = np.arange(len(y))
                match line_spec.error_display_type:
                    case "shaded":
                        error_artist = self.ax.fill_between(  # type: ignore[reportUnknownMemberType]
                            x,
                            y - err_frame,
                            y + err_frame,
                            alpha=0.2,
                            color=palette[i],
                        )
                        artists.append(error_artist)
                    case "errorbars":
                        error_artist = self.ax.errorbar(  # type: ignore[reportUnknownMemberType]
                            x,
                            y,
                            yerr=err_frame,
                            ecolor=palette[i],
                            fmt="none",
                            capsize=2,
                        )
                        artists.extend(_errorbar_artists(error_artist))
                self._error_artists[i] = error_artist

            if (trend_line := self._trend_lines[i]) is not None:
                y = frame_data.values  # type: ignore[reportUnknownMemberType]
                trend_y, coeffs = _compute_trendline_y(y)
                trend_line.set_ydata(trend_y)  # type: ignore[reportUnknownMemberType]
                trend_line.set_label(f"y={coeffs[0]:.2f}x+{coeffs[1]:.2f}")
                artists.append(trend_line)

        annotate_plot(
            self.ax,
            self.spec.plot_spec.annotation,
            title_postfix=f"\n{self.frame_dim}: {self.frame_vals[frame]}",
        )
        return artists


class GeneralRasterPlotAnimator(Animator[GeneralRasterPlotSpec]):
    """General raster plot animator for animating over a dimension of a 3D DataArray."""

    def __init__(  # noqa: D107
        self,
        spec: AnimationSpec[GeneralRasterPlotSpec],
        context: ExecutionContext,
        create_ax_func: Callable[..., Axes],
    ) -> None:
        super().__init__(spec, context, create_ax_func)

    async def init_data(self) -> None:
        data_array = await materialize_data_array_var(self.spec.plot_spec.data_array, self.context)
        data_array = data_array.squeeze().load()  # type: ignore[reportUnknownMemberType]
        if data_array.ndim != 3:  # noqa: PLR2004
            raise DataArrayShapeError.with_shape(
                func_name="MakeAnimation",
                dims=data_array.dims,  # pyright: ignore[reportArgumentType]
                shape=data_array.shape,
                msg_prefix="Expected a 3D DataArray for raster animation.",
            )
        self.data_array = data_array
        self.frame_dim = self.spec.frame_dim
        self.frame_vals = data_array[self.frame_dim].values  # type: ignore[reportUnknownMemberType]

        # Pre-compute global color limits when not explicitly set
        self._vmin = self.spec.plot_spec.style.cmap_vmin
        self._vmax = self.spec.plot_spec.style.cmap_vmax
        if self._vmin is None:
            self._vmin = float(data_array.min())  # type: ignore[reportUnknownMemberType]
        if self._vmax is None:
            self._vmax = float(data_array.max())  # type: ignore[reportUnknownMemberType]

    @property
    def nframes(self) -> int:
        """Number of frames in the animation."""
        return len(self.frame_vals)

    def init(self) -> "Iterable[AxesImage]":
        """Create first frame."""
        self.ax = self.create_ax_func()
        self.im = self.data_array.isel({self.frame_dim: 0}).plot.imshow(  # type: ignore[reportUnknownMemberType]
            ax=self.ax,
            add_colorbar=True,
            cmap=self.spec.plot_spec.style.cmap,
            vmin=self._vmin,
            vmax=self._vmax,
        )
        annotate_plot(
            self.ax,
            self.spec.plot_spec.annotation,
            title_postfix=f"\n{self.frame_dim}: {self.frame_vals[0]}",
        )
        return [self.im]

    def update(self, frame: int) -> "Iterable[AxesImage]":
        """Create i-th frame."""
        self.im.set_data(self.data_array.isel({self.frame_dim: frame}))  # type: ignore[reportUnknownMemberType]
        annotate_plot(
            self.ax,
            self.spec.plot_spec.annotation,
            title_postfix=f"\n{self.frame_dim}: {self.frame_vals[frame]}",
        )
        return [self.im]


class GeospatialRasterPlotAnimator(Animator[GeospatialRasterPlotSpec]):
    """Geospatial raster plot animator."""

    def __init__(  # noqa: D107
        self,
        spec: AnimationSpec[GeospatialRasterPlotSpec],
        context: ExecutionContext,
        create_ax_func: Callable[..., Axes],
    ) -> None:
        super().__init__(spec, context, create_ax_func)

    async def init_data(self) -> None:
        data_array = await materialize_data_array_var(self.spec.plot_spec.data_array, self.context)
        data_array = data_array.squeeze().load()  # type: ignore[reportUnknownMemberType]

        if data_array.ndim != 3:  # noqa: PLR2004
            raise DataArrayShapeError.with_shape(
                func_name="MakeAnimation",
                dims=data_array.dims,  # pyright: ignore[reportArgumentType]
                shape=data_array.shape,
                msg_prefix="Expected a 3D DataArray.",
            )

        if not is_lat_lon_data_array(data_array):
            raise DataArrayShapeError.with_shape(
                func_name="MakeAnimation",
                dims=data_array.dims,  # pyright: ignore[reportArgumentType]
                shape=data_array.shape,
                msg_prefix="Expected a DataArray with 'lat' and 'lon' dimensions.",
            )

        # Trim NaN values from the edges of the data array
        data_array = data_array.dropna(dim="lat", how="all").dropna(dim="lon", how="all")

        self.data_array = data_array

        self.frame_dim = self.spec.frame_dim
        frame_vals = data_array[self.frame_dim].values  # type: ignore[reportUnknownMemberType]
        if np.issubdtype(frame_vals.dtype, np.timedelta64):
            frame_vals = [f"{t:.2f} hours" for t in frame_vals / np.timedelta64(1, "h")]
        self.frame_vals = frame_vals

        # Pre-compute global color limits when not explicitly set
        self._vmin = self.spec.plot_spec.style.cmap_vmin
        self._vmax = self.spec.plot_spec.style.cmap_vmax
        if self._vmin is None:
            self._vmin = float(data_array.min())  # type: ignore[reportUnknownMemberType]
        if self._vmax is None:
            self._vmax = float(data_array.max())  # type: ignore[reportUnknownMemberType]

    @property
    def nframes(self) -> int:
        """Number of frames in the animation."""
        return len(self.frame_vals)

    def init(self) -> "Iterable[QuadMesh]":
        """Create first frame."""
        self.ax = setup_geospatial_axes(
            self.data_array, self.spec.plot_spec.geospatial, self.create_ax_func
        )  # type: ignore[reportUnknownMemberType]
        self.im = self.data_array.isel({self.frame_dim: 0}).plot.pcolormesh(  # type: ignore[reportUnknownMemberType]
            ax=self.ax,
            add_colorbar=True,
            cmap=self.spec.plot_spec.style.cmap,
            vmin=self._vmin,
            vmax=self._vmax,
            transform=cartopy.crs.PlateCarree(),  # type: ignore[reportUnknownMemberType]
        )
        annotate_plot(
            self.ax,
            self.spec.plot_spec.annotation,
            title_postfix=f"\n{self.frame_dim}: {self.frame_vals[0]}",
        )
        return [self.im]

    def update(self, frame: int) -> "Iterable[QuadMesh]":
        """Create i-th frame."""
        self.im.set_array(self.data_array.isel({self.frame_dim: frame}).values.ravel())  # type: ignore[reportUnknownMemberType]
        annotate_plot(
            self.ax,
            self.spec.plot_spec.annotation,
            title_postfix=f"\n{self.frame_dim}: {self.frame_vals[frame]}",
        )
        return [self.im]


animator_registry.register(BarPlotSpec, BarPlotAnimator)
animator_registry.register(LinePlotSpec, LinePlotAnimator)
animator_registry.register(MultiLinePlotSpec, MultiLinePlotAnimator)
animator_registry.register(GeneralRasterPlotSpec, GeneralRasterPlotAnimator)
animator_registry.register(GeospatialRasterPlotSpec, GeospatialRasterPlotAnimator)
