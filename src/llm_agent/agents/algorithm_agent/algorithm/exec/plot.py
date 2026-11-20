import logging
import textwrap
from collections.abc import Callable
from functools import singledispatch
from os.path import basename
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Literal, cast

import cartopy  # pyright: ignore[reportMissingTypeStubs]
import cartopy.crs  # pyright: ignore[reportMissingTypeStubs]
import cartopy.mpl.geoaxes  # pyright: ignore[reportMissingTypeStubs]
import dask.array as darr
import matplotlib as mpl
import numpy as np
import seaborn as sns
import xarray as xr
from matplotlib import path as mpath
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from llm_agent.agents.algorithm_agent.algorithm.exec.base import (
    AlgorithmExecutionError,
    ExecutionContext,
    eval_expression,
    summarize_value_for_update,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.data_array import (
    DataArrayShapeError,
    materialize_data_array,
    materialize_data_array_var,
    resolve_data_array_var,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.plot_utils import (
    deoverlap_x_tick_labels,
    fit_figure_height,
    geospatial_cbar_kwargs,
    sanitize_display_text,
    wrap_titles_to_axes_width,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    AnnotationSpec,
    BarPlotSpec,
    GeneralRasterPlotSpec,
    GeospatialRasterPlotSpec,
    GeospatialSpec,
    HistogramPlotSpec,
    LinePlotSpec,
    MakePlot,
    MultiLinePlotSpec,
    PlotGridSpec,
    PlotSpec,
    ScatterPlotSpec,
)
from llm_agent.tools.data_analysis_tool.types import ImageRef
from llm_agent.utils.dask import validate_data_array_size_within_limit

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection, QuadMesh
    from matplotlib.container import BarContainer, ErrorbarContainer
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D

mpl.use("Agg")
sns.set_theme(style="whitegrid", palette="deep")

log = logging.getLogger(__name__)


def _save_figure(fig: Figure, context: ExecutionContext) -> ImageRef:
    """Persist a Matplotlib figure and return an ImageRef."""
    wrap_titles_to_axes_width(fig)
    fit_figure_height(fig)
    deoverlap_x_tick_labels(fig)
    with NamedTemporaryFile(prefix="figure_", suffix=".png", delete=False) as tmp_file:
        fig.savefig(tmp_file.name, bbox_inches="tight", pad_inches=0.2)  # pyright: ignore[reportUnknownMemberType]
        ref_key = context.tool_call_context.chat_context.persist_data(
            tmp_file.name, data_type="image"
        )
        return ImageRef(key=ref_key, name=basename(tmp_file.name))


@eval_expression.register(MakePlot)
async def _[T: PlotSpec](expr: MakePlot[T], context: ExecutionContext) -> ImageRef:
    """Evaluate expression."""
    fig = plt.figure(layout="constrained")  # type: ignore[reportUnknownMemberType]

    match expr.plot_spec:
        case PlotGridSpec():
            await make_plot_grid(expr.plot_spec, context, fig=fig)
        case _:

            def create_ax_func(**kwargs: Any) -> "Axes":  # noqa: ANN401
                return fig.add_subplot(1, 1, 1, **kwargs)  # type: ignore[reportUnknownMemberType]

            await make_plot(expr.plot_spec, context, create_ax_func=create_ax_func)

    return _save_figure(fig, context)


@singledispatch
def make_plot(
    spec: Any,  # noqa: ANN401
    context: ExecutionContext,
    create_ax_func: Callable[..., "Axes"],
) -> Any:  # noqa: ANN401
    """Create a single plot based on the provided specification."""
    raise NotImplementedError(f"Unsupported plot spec type: {type(spec)}.")


@summarize_value_for_update.register(Figure)
def _(value: Figure) -> str | None:  # noqa: ARG001
    """Helper to create a summary for a Figure."""
    return None


def annotate_plot(ax: "Axes", annotation: AnnotationSpec, title_postfix: str = "") -> None:
    """Annotate the plot with title and axis labels."""
    if annotation.title:
        title_wrapped = "\n".join(textwrap.wrap(sanitize_display_text(annotation.title), width=40))
        ax.set_title(f"{title_wrapped}{title_postfix}", fontdict={"fontsize": 10})  # type: ignore[reportUnknownMemberType]
    if annotation.xlabel:
        ax.set_xlabel(sanitize_display_text(annotation.xlabel))  # type: ignore[reportUnknownMemberType]
    if annotation.ylabel:
        ax.set_ylabel(sanitize_display_text(annotation.ylabel))  # type: ignore[reportUnknownMemberType]


def is_lat_lon_data_array(data_array: xr.DataArray) -> bool:
    """Check if the DataArray has lat and lon dimensions."""
    return "lat" in data_array.dims and "lon" in data_array.dims


@make_plot.register(BarPlotSpec)
async def make_bar_plot(
    spec: BarPlotSpec,
    context: ExecutionContext,
    create_ax_func: Callable[..., "Axes"],
) -> "tuple[Axes, BarContainer]":
    """Evaluate expression."""
    heights = [
        (await materialize_data_array_var(h, context)).values.item()
        if isinstance(h, xr.DataArray)
        else await eval_expression(h, context)
        for h in spec.heights
    ]
    ax = create_ax_func()
    sns.barplot(x=spec.labels, y=heights, ax=ax)  # type: ignore[reportUnknownMemberType]
    bar = cast("BarContainer", ax.containers[-1])  # type: ignore[reportUnknownMemberType]
    annotate_plot(ax, spec.annotation)
    return ax, bar


async def _histogram_bin_range(
    data_array: xr.DataArray, spec: HistogramPlotSpec, context: ExecutionContext
) -> tuple[float, float]:
    """Find the binning range, gathering only the two bounds."""
    bounds = await materialize_data_array(
        xr.concat([data_array.min(), data_array.max()], dim="stat"),  # type: ignore[reportUnknownMemberType]
        context,
        progress_label=f"range({spec.data_array.name})",
    )
    low, high = (float(v) for v in np.asarray(bounds.values))  # type: ignore[reportUnknownArgumentType]
    if not (np.isfinite(low) and np.isfinite(high)):
        msg = "the DataArray has no finite values to bin."
        raise AlgorithmExecutionError(msg, func_name=spec.spec_name)
    if low == high:
        # np.histogram widens a degenerate range; dask leaves zero-width bins, so do it here.
        low, high = low - 0.5, high + 0.5
    return low, high


@make_plot.register(HistogramPlotSpec)
async def make_histogram_plot(
    spec: HistogramPlotSpec,
    context: ExecutionContext,
    create_ax_func: Callable[..., "Axes"],
) -> "Axes":
    """Plot a histogram of a DataArray's flattened, finite values."""
    data_array = await resolve_data_array_var(spec.data_array, context)
    # Binning streams over chunks, so allow the same larger limit as streamable reductions.
    validate_data_array_size_within_limit(
        data_array, size_limit=context.config.data_array_compute_size_limit * 5
    )
    low, high = await _histogram_bin_range(data_array, spec, context)

    # Values outside the range are dropped, which excludes NaNs.
    if data_array.chunks is None:
        counts, edges = np.histogram(
            np.asarray(data_array.values),  # type: ignore[reportUnknownArgumentType]
            bins=spec.bins,
            range=(low, high),
        )
    else:
        lazy_counts, edges = darr.histogram(  # type: ignore[reportUnknownMemberType]
            data_array.data, bins=spec.bins, range=(low, high)
        )
        materialized = await materialize_data_array(
            xr.DataArray(lazy_counts, dims=["bin"]),
            context,
            progress_label=f"histogram({spec.data_array.name})",
        )
        counts = np.asarray(materialized.values)  # type: ignore[reportUnknownArgumentType]

    ax = create_ax_func()
    # Replay the precomputed counts through seaborn so styling matches the other plots.
    edge_values: Any = np.asarray(edges)  # type: ignore[reportUnknownArgumentType]
    bin_centers = (edge_values[:-1] + edge_values[1:]) / 2
    sns.histplot(  # type: ignore[reportUnknownMemberType]
        x=bin_centers,
        weights=counts,
        bins=spec.bins,
        binrange=(low, high),
        stat="density" if spec.density else "count",
        ax=ax,
    )
    annotate_plot(ax, spec.annotation)
    return ax


@make_plot.register(ScatterPlotSpec)
async def make_scatter_plot(
    spec: ScatterPlotSpec,
    context: ExecutionContext,
    create_ax_func: Callable[..., "Axes"],
) -> "Axes":
    """Plot one 1D DataArray against another as a scatter plot."""
    x_da = (await materialize_data_array_var(spec.x_data_array, context)).squeeze()
    y_da = (await materialize_data_array_var(spec.y_data_array, context)).squeeze()
    for name, da in (("x_data_array", x_da), ("y_data_array", y_da)):
        if da.ndim != 1:
            raise DataArrayShapeError.with_shape(
                func_name=spec.spec_name,
                dims=da.dims,  # pyright: ignore[reportArgumentType]
                shape=da.shape,
                msg_prefix=f"Expected {name} to be squeezable to 1D.",
            )
    if x_da.size != y_da.size:
        msg = (
            f"{spec.spec_name}: x_data_array (size {x_da.size}) and y_data_array "
            f"(size {y_da.size}) must be the same length."
        )
        raise DataArrayShapeError(msg, func_name=spec.spec_name)

    # Flatten and filter out any NaN values, so the scatter plot only shows valid points.
    x = np.asarray(x_da.values).ravel()
    y = np.asarray(y_da.values).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    ax = create_ax_func()
    sns.scatterplot(x=x, y=y, ax=ax, s=15)  # type: ignore[reportUnknownMemberType]

    if spec.add_regression_line and len(x) >= 2:  # noqa: PLR2004
        coeffs: Any = np.polyfit(x, y, 1)  # type: ignore[reportUnknownMemberType]
        endpoints = np.array([x.min(), x.max()])
        label = f"y={coeffs[0]:.2f}x+{coeffs[1]:.2f}"
        ax.plot(endpoints, np.polyval(coeffs, endpoints), ls="--", color="0.3", label=label)  # type: ignore[reportUnknownMemberType]
        ax.legend()  # type: ignore[reportUnknownMemberType]

    annotate_plot(ax, spec.annotation)
    return ax


@make_plot.register(LinePlotSpec)
async def make_line_plot(
    spec: LinePlotSpec,
    context: ExecutionContext,
    ax: "Axes | None" = None,
    create_ax_func: Callable[..., "Axes"] | None = None,
    color: "str | tuple[float, ...] | None" = None,
) -> "tuple[Axes, Line2D]":
    """Evaluate expression."""
    data_array = await materialize_data_array_var(spec.data_array, context)
    data_array = data_array.squeeze()

    if data_array.ndim != 1:
        raise DataArrayShapeError.with_shape(
            func_name=spec.spec_name,
            dims=data_array.dims,  # pyright: ignore[reportArgumentType]
            shape=data_array.shape,
            msg_prefix="Expected a 1D DataArray.",
        )

    if ax is None:
        if create_ax_func is None:
            msg = "Either an Axes object or a create_ax_func must be provided."
            raise ValueError(msg)
        ax = create_ax_func()

    effective_color = color or spec.style.color or plt.get_cmap("tab10")(0)
    (line,) = data_array.plot.line(  # type: ignore[reportUnknownMemberType]
        ax=ax, color=effective_color, ls=spec.style.linestyle, lw=spec.style.linewidth
    )

    if spec.data_array_error is not None:
        error_array = await materialize_data_array_var(spec.data_array_error, context)
        error_array = error_array.squeeze()
        _plot_error(ax, data_array, error_array, spec.error_display_type, effective_color)

    if spec.add_linear_trendline:
        _plot_trendline(ax, data_array, effective_color, time_unit=spec.linear_trendline_time_unit)

    annotate_plot(ax, spec.annotation)
    return ax, line


def _plot_error(
    ax: "Axes",
    data_array: xr.DataArray,
    error_array: xr.DataArray,
    display_type: Literal["shaded", "errorbars"],
    color: "str | tuple[float, ...] | None",
) -> "PolyCollection | ErrorbarContainer":
    """Plot error as shaded region or error bars."""
    y = data_array.values  # type: ignore[reportUnknownMemberType]
    err = error_array.values  # type: ignore[reportUnknownMemberType]
    dim_name = data_array.dims[0]
    if dim_name in data_array.coords:  # pyright: ignore[reportUnknownMemberType]
        x = data_array.coords[dim_name].values  # type: ignore[reportUnknownMemberType]
    else:
        x = np.arange(len(data_array))

    match display_type:
        case "shaded":
            return ax.fill_between(x, y - err, y + err, alpha=0.2, color=color)  # type: ignore[reportUnknownMemberType]
        case "errorbars":
            return ax.errorbar(x, y, yerr=err, ecolor=color, fmt="none", capsize=2)  # type: ignore[reportUnknownMemberType]


def _plot_trendline(
    ax: "Axes",
    data_array: xr.DataArray,
    color: "str | tuple[float, ...] | None",
    time_unit: Literal["day", "year", "decade"] = "year",
) -> "Line2D":
    """Add a linear trendline to the plot."""
    _ns_per_unit_time: dict[str, float] = {
        "day": 24 * 3600 * 1e9,
        "year": 365.25 * 24 * 3600 * 1e9,
        "decade": 365.25 * 24 * 3600 * 1e9 * 10,
    }

    dim_name = data_array.dims[0]
    y: Any = data_array.values  # type: ignore[reportUnknownMemberType]
    if dim_name in data_array.coords:  # pyright: ignore[reportUnknownMemberType]
        x_coords: Any = data_array.coords[dim_name].values  # type: ignore[reportUnknownMemberType]
    else:
        x_coords = np.arange(len(data_array))

    # Convert to numeric for polyfit
    if np.issubdtype(x_coords.dtype, np.datetime64):  # type: ignore[reportUnknownMemberType]
        x_numeric: Any = (x_coords - x_coords[0]).astype("timedelta64[ns]").astype(float)  # type: ignore[reportUnknownMemberType]
    elif np.issubdtype(x_coords.dtype, np.timedelta64):  # type: ignore[reportUnknownMemberType]
        x_numeric = x_coords.astype("timedelta64[ns]").astype(float)  # type: ignore[reportUnknownMemberType]
    else:
        x_numeric = x_coords.astype(float)  # type: ignore[reportUnknownMemberType]

    mask = np.isfinite(x_numeric) & np.isfinite(y)  # type: ignore[reportUnknownMemberType]
    coeffs: Any = np.polyfit(x_numeric[mask], y[mask], 1)  # type: ignore[reportUnknownMemberType]

    slope, intercept = coeffs  # pyright: ignore[reportUnknownVariableType]
    if np.issubdtype(x_coords.dtype, np.datetime64) or np.issubdtype(  # type: ignore[reportUnknownMemberType]
        x_coords.dtype,  # type: ignore[reportUnknownMemberType]
        np.timedelta64,
    ):
        # Scale slope from per-nanosecond to (more useful) per-day/year/decade
        slope_scaled = slope * _ns_per_unit_time[time_unit]  # pyright: ignore[reportUnknownVariableType]
        label = f"y = {slope_scaled:.2f}x + {intercept:.2f}"
    else:
        label = f"y = {slope:.2f}x + {intercept:.2f}"

    trend_y: Any = np.polyval(coeffs, x_numeric)  # type: ignore[reportUnknownMemberType]
    (trend_line,) = ax.plot(x_coords, trend_y, color=color, ls=":", lw=1, alpha=0.7, label=label)  # type: ignore[reportUnknownMemberType]

    # Ensure the trendline is included in the legend, even if the original line has no label.
    if not ax.get_legend():  # type: ignore[reportUnknownMemberType]
        ax.legend()  # type: ignore[reportUnknownMemberType]

    return trend_line


@make_plot.register(MultiLinePlotSpec)
async def make_multi_line_plot(
    spec: MultiLinePlotSpec,
    context: ExecutionContext,
    create_ax_func: Callable[..., "Axes"],
) -> "tuple[Axes, list[Line2D]]":
    """Evaluate expression."""
    ax = create_ax_func()
    palette = sns.color_palette("deep", n_colors=len(spec.specs))  # type: ignore[reportUnknownMemberType]
    lines = [
        (await make_line_plot(line_spec, context, ax=ax, color=palette[i]))[1]
        for i, line_spec in enumerate(spec.specs)
    ]
    for line, line_spec in zip(lines, spec.specs, strict=True):
        if line_spec.annotation.title:
            line.set_label(line_spec.annotation.title)
    ax.legend()  # type: ignore[reportUnknownMemberType]
    annotate_plot(ax, spec.annotation)
    return ax, lines


@make_plot.register(GeneralRasterPlotSpec)
async def make_general_raster_plot(
    spec: GeneralRasterPlotSpec,
    context: ExecutionContext,
    create_ax_func: Callable[..., "Axes"],
) -> "tuple[Axes, AxesImage]":
    """Evaluate expression."""
    data_array = await materialize_data_array_var(spec.data_array, context)
    data_array = data_array.squeeze()

    if data_array.ndim != 2:  # noqa: PLR2004
        raise DataArrayShapeError.with_shape(
            func_name=spec.spec_name,
            dims=data_array.dims,  # pyright: ignore[reportArgumentType]
            shape=data_array.shape,
            msg_prefix="Expected a DataArray squeezable to 2D.",
        )

    ax = create_ax_func()

    im = data_array.plot.imshow(  # type: ignore[reportUnknownMemberType]
        ax=ax,
        add_colorbar=True,
        cmap=spec.style.cmap,
        vmin=spec.style.cmap_vmin,
        vmax=spec.style.cmap_vmax,
        cbar_kwargs={"label": sanitize_display_text(spec.style.colorbar_label)}
        if spec.style.colorbar_label
        else None,
    )

    annotate_plot(ax, spec.annotation)

    return ax, im


# Projections that recenter cleanly on a region via central_longitude, so an antimeridian-crossing
# footprint can be shown as one contiguous strip. Excludes the polar projections (which show the
# full disc regardless) and InterruptedGoodeHomolosine (interrupted by design).
_ANTIMERIDIAN_RECENTER_PROJECTIONS = frozenset(
    {"PlateCarree", "Robinson", "EqualEarth", "AlbersEqualArea"}
)


def _antimeridian_lon_extent(data_array: xr.DataArray) -> "tuple[float, float] | None":
    """Return a (west, east) longitude extent when the data wraps across the antimeridian.

    Looks at the longitudes that actually contain data and finds the largest circular gap between
    them. If that gap is an interior one (i.e. the data hugs both sides of +/-180 with an empty span
    across the rest of the globe), the region wraps the antimeridian and the covering arc is
    returned, with ``east`` possibly exceeding 180. Otherwise (the largest gap is the ordinary
    empty span, so the region does not wrap) returns None.
    """
    reduce_dims = [d for d in data_array.dims if d != "lon"]
    has_data = data_array.notnull().any(dim=reduce_dims) if reduce_dims else data_array.notnull()  # type: ignore[reportUnknownMemberType]
    lon_values = np.asarray(data_array["lon"].values)  # type: ignore[reportUnknownArgumentType]
    lons = np.sort(lon_values[np.asarray(has_data.values)])  # type: ignore[reportUnknownArgumentType]
    if lons.size < 2:  # noqa: PLR2004
        return None

    diffs = np.diff(lons)
    wrap_gap = float(lons[0] + 360.0 - lons[-1])
    max_gap_idx = int(np.argmax(diffs))
    if float(diffs[max_gap_idx]) <= wrap_gap:
        # The largest gap is the ordinary empty span; the region does not wrap the antimeridian.
        return None
    west = float(lons[max_gap_idx + 1])
    east = float(lons[max_gap_idx]) + 360.0
    return west, east


def setup_geospatial_axes(
    data_array: xr.DataArray, geospatial_spec: GeospatialSpec, create_ax_func: Callable[..., "Axes"]
) -> "cartopy.mpl.geoaxes.GeoAxes":
    """Set up a cartopy figure for the given DataArray."""
    projection_spec = geospatial_spec.map_projection

    lat_min, lat_max = (float(data_array["lat"].min()), float(data_array["lat"].max()))
    lon_min, lon_max = (float(data_array["lon"].min()), float(data_array["lon"].max()))
    central_longitude = projection_spec.central_longitude
    lon_extent = (max(lon_min, -180.0), min(lon_max, 180.0))

    # If the map's data wraps across the antimeridian (e.g. the Chukchi Sea), recenter the map on
    # the region so it renders as one contiguous strip instead of spanning the whole globe. Only
    # done when the caller left the projection at the default center (so an explicit center is
    # respected) and only for projections that recenter cleanly (see the constant above).
    if (
        projection_spec.projection in _ANTIMERIDIAN_RECENTER_PROJECTIONS
        and projection_spec.central_longitude == 0.0
    ):
        wrapped = _antimeridian_lon_extent(data_array)
        if wrapped is not None:
            west, east = wrapped
            center = (west + east) / 2.0
            # Normalize to [-180, 180); cartopy treats -180 and 180 identically.
            central_longitude = ((center + 180.0) % 360.0) - 180.0
            lon_extent = (west, east)

    projection = getattr(cartopy.crs, projection_spec.projection)(  # type: ignore[reportUnknownMemberType]
        central_longitude=central_longitude,
    )
    ax = create_ax_func(projection=projection)  # type: ignore[reportUnknownMemberType]
    ax = cast("cartopy.mpl.geoaxes.GeoAxes", ax)  # type: ignore[reportUnknownMemberType]

    ax.stock_img()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    if geospatial_spec.add_coastlines:
        ax.coastlines()  # type: ignore[reportUnknownMemberType]

    ax.add_feature(cartopy.feature.OCEAN, edgecolor="w", linewidth=0.01)  # type: ignore[reportUnknownMemberType]
    ax.add_feature(cartopy.feature.LAND, edgecolor="w", linewidth=0.01)  # type: ignore[reportUnknownMemberType]

    if geospatial_spec.add_borders:
        ax.add_feature(cartopy.feature.BORDERS, linestyle=":")  # type: ignore[reportUnknownMemberType]
        ax.add_feature(cartopy.feature.STATES, linestyle="--")  # type: ignore[reportUnknownMemberType]
    if geospatial_spec.add_rivers:
        ax.add_feature(cartopy.feature.RIVERS, linestyle="--")  # type: ignore[reportUnknownMemberType]
    if geospatial_spec.add_lakes:
        ax.add_feature(cartopy.feature.LAKES, linestyle="--")  # type: ignore[reportUnknownMemberType]

    if geospatial_spec.add_gridlines:
        # Cartopy maps have no tick labels by default; labelled gridlines act as the lat/lon ticks.
        try:
            gridlines = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)  # type: ignore[reportUnknownMemberType]
            gridlines.top_labels = False  # type: ignore[reportAttributeAccessIssue]
            gridlines.right_labels = False  # type: ignore[reportAttributeAccessIssue]
        except Exception:  # noqa: BLE001
            # Some projections don't support labelled gridlines; fall back to plain gridlines.
            ax.gridlines(linewidth=0.5, color="gray", alpha=0.5)  # type: ignore[reportUnknownMemberType]

    ax.set_extent(  # type: ignore[reportUnknownMemberType]
        [
            lon_extent[0],
            lon_extent[1],
            max(lat_min, -90.0),
            min(lat_max, 90.0),
        ],
        crs=cartopy.crs.PlateCarree(),  # type: ignore[reportUnknownMemberType]
    )

    if projection_spec.projection in ("NorthPolarStereo", "SouthPolarStereo"):
        # Apply circular boundary for polar stereographic projections
        # We define a circle in the axes coordinates
        theta = np.linspace(0, 2 * np.pi, 100)
        center = (0.5, 0.5)
        radius = 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)  # type: ignore[reportUnknownMemberType]

    return ax


@make_plot.register(GeospatialRasterPlotSpec)
async def make_geospatial_raster_plot(
    spec: GeospatialRasterPlotSpec,
    context: ExecutionContext,
    create_ax_func: Callable[..., "Axes"],
) -> "tuple[Axes, QuadMesh]":
    """Make a geospatial raster plot."""
    data_array = await materialize_data_array_var(spec.data_array, context)
    data_array = data_array.squeeze()

    if data_array.ndim != 2:  # noqa: PLR2004
        raise DataArrayShapeError.with_shape(
            func_name=spec.spec_name,
            dims=data_array.dims,  # pyright: ignore[reportArgumentType]
            shape=data_array.shape,
            msg_prefix="Expected a DataArray squeezable to 2D.",
        )

    if not is_lat_lon_data_array(data_array):
        raise DataArrayShapeError.with_shape(
            func_name=spec.spec_name,
            dims=data_array.dims,  # pyright: ignore[reportArgumentType]
            shape=data_array.shape,
            msg_prefix="Expected a DataArray with 'lat' and 'lon' dimensions for "
            "geospatial raster plot.",
        )

    # Trim NaN values from the edges of the data array
    data_array = data_array.dropna(dim="lat", how="all").dropna(dim="lon", how="all")

    ax = setup_geospatial_axes(data_array, spec.geospatial, create_ax_func)  # type: ignore[reportUnknownMemberType]
    im = data_array.plot.pcolormesh(  # type: ignore[reportUnknownMemberType]
        ax=ax,
        add_colorbar=True,
        cmap=spec.style.cmap,
        vmin=spec.style.cmap_vmin,
        vmax=spec.style.cmap_vmax,
        transform=cartopy.crs.PlateCarree(),  # type: ignore[reportUnknownMemberType]
        cbar_kwargs=geospatial_cbar_kwargs(data_array, label=spec.style.colorbar_label or None),
    )
    annotate_plot(ax, spec.annotation)

    return ax, im


@make_plot.register(PlotGridSpec)
async def make_plot_grid[T: PlotSpec](
    spec: PlotGridSpec[T], context: ExecutionContext, fig: "Figure"
) -> None:
    """Make a grid of subplots based on the provided specifications."""
    ncols = spec.ncols
    nrows = (len(spec.subplot_specs) + ncols - 1) // ncols
    for i, subplot_spec in enumerate(spec.subplot_specs, start=1):

        def create_ax_func(**kwargs: Any) -> "Axes":  # noqa: ANN401
            return fig.add_subplot(nrows, ncols, i, **kwargs)  # type: ignore[reportUnknownMemberType]  # noqa: B023

        await make_plot(subplot_spec, context, create_ax_func=create_ax_func)

    if spec.super_title:
        title_wrapped = "\n".join(textwrap.wrap(sanitize_display_text(spec.super_title), width=40))
        fig.suptitle(title_wrapped, fontsize=12)  # type: ignore[reportUnknownMemberType]
