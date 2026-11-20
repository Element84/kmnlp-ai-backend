from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import cartopy  # pyright: ignore[reportMissingTypeStubs]
import cartopy.crs  # pyright: ignore[reportMissingTypeStubs]
from matplotlib.axes import Axes

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
    geospatial_cbar_kwargs,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    AnnotationSpec,
    GeospatialSpec,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.contour import ContourMapPlotSpec

if TYPE_CHECKING:
    from matplotlib.contour import QuadContourSet


@make_plot.register(ContourMapPlotSpec)
async def make_contour_map_plot(
    spec: ContourMapPlotSpec,
    context: ExecutionContext,
    create_ax_func: Callable[..., Axes],
) -> "tuple[Axes, QuadContourSet]":
    """Make a 2D geospatial contour map (filled or unfilled iso-lines)."""
    data_array = (await materialize_data_array_var(spec.data_array, context)).squeeze()

    if data_array.ndim != 2:  # noqa: PLR2004
        raise DataArrayShapeError.with_shape(
            func_name=spec.spec_name,
            dims=data_array.dims,  # pyright: ignore[reportArgumentType]
            shape=data_array.shape,
            msg_prefix="Expected a 2D DataArray to contour.",
        )
    if not is_lat_lon_data_array(data_array):
        raise DataArrayShapeError.with_shape(
            func_name=spec.spec_name,
            dims=data_array.dims,  # pyright: ignore[reportArgumentType]
            shape=data_array.shape,
            msg_prefix="Expected DataArray with 'lat' and 'lon' dimensions.",
        )

    # Trim NaN values from the edges of the data array
    data_array = data_array.dropna(dim="lat", how="all").dropna(dim="lon", how="all")

    vmin, vmax = spec.colorbar_min_and_max
    geospatial = GeospatialSpec(map_projection=spec.map_projection)
    ax = setup_geospatial_axes(data_array, geospatial, create_ax_func)  # type: ignore[reportUnknownMemberType]

    plot_func: Any = data_array.plot.contourf if spec.filled else data_array.plot.contour  # type: ignore[reportUnknownMemberType]
    contour_set: Any = plot_func(
        ax=ax,
        levels=spec.levels,
        cmap=spec.cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        transform=cartopy.crs.PlateCarree(),  # type: ignore[reportUnknownMemberType]
        cbar_kwargs=geospatial_cbar_kwargs(data_array),
    )

    if spec.label_contours:
        ax.clabel(contour_set, inline=True, fontsize=8)  # type: ignore[reportUnknownMemberType]

    annotate_plot(ax, AnnotationSpec(title=spec.title or "Contour Map"))
    return ax, contour_set
