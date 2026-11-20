# ruff: noqa: D102

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

import cartopy  # pyright: ignore[reportMissingTypeStubs]
import cartopy.crs  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import xarray as xr
from e84_geoai_common.tracing import timed_function
from matplotlib.axes import Axes

from llm_agent.agents.algorithm_agent.algorithm.exec.animation import (
    AnimationSpec,
    Animator,
    animator_registry,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.base import ExecutionContext
from llm_agent.agents.algorithm_agent.algorithm.exec.data_array import (
    DataArrayShapeError,
    materialize_data_array_var,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.plot import (
    annotate_plot,
    is_lat_lon_data_array,
    make_plot,
    setup_geospatial_axes,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.plot_utils import (
    format_coordinate_label,
    geospatial_cbar_kwargs,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    AnnotationSpec,
    GeospatialSpec,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.wind import WindVectorPlotSpec

if TYPE_CHECKING:
    from matplotlib.quiver import Barbs, Quiver


def _validate_wind_component(
    data_array: xr.DataArray, func_name: str, component: str, expected_ndim: int
) -> None:
    """Validate that a wind component has the expected dimensionality and lat/lon dims."""
    if data_array.ndim != expected_ndim:
        raise DataArrayShapeError.with_shape(
            func_name=func_name,
            dims=data_array.dims,  # pyright: ignore[reportArgumentType]
            shape=data_array.shape,
            msg_prefix=f"Expected a {expected_ndim}D {component} wind component DataArray.",
        )
    if not is_lat_lon_data_array(data_array):
        raise DataArrayShapeError.with_shape(
            func_name=func_name,
            dims=data_array.dims,  # pyright: ignore[reportArgumentType]
            shape=data_array.shape,
            msg_prefix=f"Expected {component} wind component with 'lat' and 'lon' dimensions.",
        )


def _subsample(data_array: xr.DataArray, subsample: int) -> xr.DataArray:
    """Keep every Nth grid point along lat and lon to reduce visual clutter."""
    if subsample <= 1:
        return data_array
    return data_array.isel(lat=slice(None, None, subsample), lon=slice(None, None, subsample))


def _draw_wind_vectors(
    ax: Axes,
    u: xr.DataArray,
    v: xr.DataArray,
    spec: WindVectorPlotSpec,
) -> "Quiver | Barbs":
    """Draw wind arrows or barbs on a geospatial axes and return the drawn artist."""
    lon = u["lon"].values
    lat = u["lat"].values
    uu = u.values
    vv = v.values
    transform = cartopy.crs.PlateCarree()
    if spec.barbs:
        return ax.barbs(lon, lat, uu, vv, transform=transform)  # type: ignore[reportUnknownMemberType]
    if spec.color_by_speed:
        speed = np.hypot(uu, vv)
        return ax.quiver(lon, lat, uu, vv, speed, transform=transform, cmap="plasma")  # type: ignore[reportUnknownMemberType]
    return ax.quiver(lon, lat, uu, vv, transform=transform)  # type: ignore[reportUnknownMemberType]


def _add_speed_colorbar(ax: Axes, artist: "Quiver", data_array: xr.DataArray) -> None:
    """Add a wind-speed colorbar for a color-mapped quiver artist."""
    fig = ax.get_figure()
    if fig is None:
        return
    fig.colorbar(artist, ax=ax, **geospatial_cbar_kwargs(data_array, label="Wind speed"))  # type: ignore[reportUnknownMemberType]


@make_plot.register(WindVectorPlotSpec)
async def make_wind_vector_plot(
    spec: WindVectorPlotSpec,
    context: ExecutionContext,
    create_ax_func: Callable[..., Axes],
) -> Axes:
    """Make a 2D wind vector plot (arrows or barbs) on a map."""
    u = (await materialize_data_array_var(spec.u_data_array, context)).squeeze()
    v = (await materialize_data_array_var(spec.v_data_array, context)).squeeze()

    _validate_wind_component(u, spec.spec_name, "eastward (u)", expected_ndim=2)
    _validate_wind_component(v, spec.spec_name, "northward (v)", expected_ndim=2)
    if u.shape != v.shape:
        msg = (
            f"{spec.spec_name}: expected u and v wind components to share the same grid, "
            f"but found shapes {u.shape} and {v.shape}."
        )
        raise DataArrayShapeError(msg, func_name=spec.spec_name)

    u = _subsample(u.transpose("lat", "lon"), spec.subsample)
    v = _subsample(v.transpose("lat", "lon"), spec.subsample)

    geospatial = GeospatialSpec(map_projection=spec.map_projection)
    ax = setup_geospatial_axes(u, geospatial, create_ax_func)  # type: ignore[reportUnknownMemberType]
    artist = _draw_wind_vectors(ax, u, v, spec)
    if spec.color_by_speed and not spec.barbs:
        _add_speed_colorbar(ax, cast("Quiver", artist), u)

    annotate_plot(ax, AnnotationSpec(title=spec.title or "Wind"))
    return ax


class WindVectorPlotAnimator(Animator[WindVectorPlotSpec]):
    """Animator for wind vector plots over a frame dimension (e.g. time or lead time)."""

    def __init__(  # noqa: D107
        self,
        spec: AnimationSpec[WindVectorPlotSpec],
        context: ExecutionContext,
        create_ax_func: Callable[..., Axes],
    ) -> None:
        super().__init__(spec, context, create_ax_func)

    async def init_data(self) -> None:
        plot_spec = self.spec.plot_spec
        u_raw = await materialize_data_array_var(plot_spec.u_data_array, self.context)
        v_raw = await materialize_data_array_var(plot_spec.v_data_array, self.context)
        u = u_raw.squeeze().load()  # type: ignore[reportUnknownMemberType]
        v = v_raw.squeeze().load()  # type: ignore[reportUnknownMemberType]

        _validate_wind_component(u, "MakeAnimation", "eastward (u)", expected_ndim=3)
        _validate_wind_component(v, "MakeAnimation", "northward (v)", expected_ndim=3)
        if u.shape != v.shape:
            msg = (
                "MakeAnimation: expected u and v wind components to share the same grid, "
                f"but found shapes {u.shape} and {v.shape}."
            )
            raise ValueError(msg)

        self.frame_dim = self.spec.frame_dim
        u = _subsample(u.transpose(self.frame_dim, "lat", "lon"), plot_spec.subsample)
        v = _subsample(v.transpose(self.frame_dim, "lat", "lon"), plot_spec.subsample)
        self.u = u
        self.v = v
        self.frame_vals = u[self.frame_dim].values  # type: ignore[reportUnknownMemberType]

        # Fix the color scale across all frames so a given colorbar value maps to the same color in
        # every frame. Otherwise quiver re-autoscales its norm to each frame's speed range, while
        # the colorbar (created once) stays fixed, making the mapping drift frame to frame.
        speed: Any = np.hypot(u.values, v.values)
        vmin = float(np.nanmin(speed))
        vmax = float(np.nanmax(speed))
        self.speed_clim: tuple[float, float] | None = (
            (vmin, vmax) if np.isfinite(vmin) and np.isfinite(vmax) else None
        )

    @property
    def nframes(self) -> int:
        """Number of frames in the animation."""
        return len(self.frame_vals)

    def init(self) -> Iterable[Any]:
        """Create first frame."""
        plot_spec = self.spec.plot_spec
        geospatial = GeospatialSpec(map_projection=plot_spec.map_projection)
        self.ax = setup_geospatial_axes(  # type: ignore[reportUnknownMemberType]
            self.u.isel({self.frame_dim: 0}), geospatial, self.create_ax_func
        )
        self.artist = _draw_wind_vectors(
            self.ax,
            self.u.isel({self.frame_dim: 0}),
            self.v.isel({self.frame_dim: 0}),
            plot_spec,
        )
        if plot_spec.color_by_speed and not plot_spec.barbs:
            if self.speed_clim is not None:
                self.artist.set_clim(*self.speed_clim)
            _add_speed_colorbar(
                self.ax, cast("Quiver", self.artist), self.u.isel({self.frame_dim: 0})
            )
        annotate_plot(
            self.ax,
            AnnotationSpec(title=plot_spec.title or "Wind"),
            title_postfix=f"\n{self.frame_dim}: {format_coordinate_label(self.frame_vals[0])}",
        )
        return [self.artist]

    @timed_function
    def update(self, frame: int) -> Iterable[Any]:
        """Create i-th frame."""
        uu = self.u.isel({self.frame_dim: frame}).values
        vv = self.v.isel({self.frame_dim: frame}).values
        if self.spec.plot_spec.color_by_speed and not self.spec.plot_spec.barbs:
            self.artist.set_UVC(uu, vv, np.hypot(uu, vv))
            if self.speed_clim is not None:
                self.artist.set_clim(*self.speed_clim)
        else:
            self.artist.set_UVC(uu, vv)
        annotate_plot(
            self.ax,
            AnnotationSpec(title=self.spec.plot_spec.title or "Wind"),
            title_postfix=f"\n{self.frame_dim}: {format_coordinate_label(self.frame_vals[frame])}",
        )
        return [self.artist]


animator_registry.register(WindVectorPlotSpec, WindVectorPlotAnimator)
