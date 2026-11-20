from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from llm_agent.agents.algorithm_agent.algorithm.exec.base import ExecutionContext
from llm_agent.agents.algorithm_agent.algorithm.exec.data_array import (
    DataArrayShapeError,
    materialize_data_array_var,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.plot import (
    annotate_plot,
    make_plot,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.plot_utils import (
    timedelta_to_numeric,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.ensemble import (
    EnsemblePlumePlotSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes

_BAND_PERCENTILES: dict[str, tuple[float, float]] = {
    "minmax": (0.0, 100.0),
    "iqr": (25.0, 75.0),
    "p10_p90": (10.0, 90.0),
}

_BAND_LABELS: dict[str, str] = {
    "minmax": "Min-max spread",
    "iqr": "25th-75th percentile",
    "p10_p90": "10th-90th percentile",
}


def _prepare_x_axis(data_array: xr.DataArray, x_dim: str) -> "tuple[Any, str | None]":
    """Return numeric x-axis values and an optional unit label.

    matplotlib cannot mix timedelta64 x-values with float data (e.g. inside fill_between), which
    is common for forecast 'lead_time' axes, so a timedelta axis is converted to float hours/days.
    """
    if x_dim not in data_array.coords:  # pyright: ignore[reportUnknownMemberType]
        indices: Any = np.arange(data_array.sizes[x_dim])
        return indices, None
    coord: Any = np.asarray(data_array.coords[x_dim].values)  # type: ignore[reportUnknownArgumentType]
    if np.issubdtype(coord.dtype, np.timedelta64):
        return timedelta_to_numeric(coord)
    return coord, None


@make_plot.register(EnsemblePlumePlotSpec)
async def make_ensemble_plume_plot(
    spec: EnsemblePlumePlotSpec,
    context: ExecutionContext,
    create_ax_func: "Callable[..., Axes]",
) -> "Axes":
    """Plot an ensemble forecast as a plume: member lines, mean, and a spread band."""
    data_array = (await materialize_data_array_var(spec.data_array, context)).squeeze()

    if spec.ensemble_dim not in data_array.dims:
        msg = (
            f"{spec.spec_name}: expected DataArray to have an ensemble dimension "
            f"'{spec.ensemble_dim}' but found dims {data_array.dims}."
        )
        raise DataArrayShapeError(msg, func_name=spec.spec_name)
    if data_array.ndim != 2:  # noqa: PLR2004
        raise DataArrayShapeError.with_shape(
            func_name=spec.spec_name,
            dims=data_array.dims,  # pyright: ignore[reportArgumentType]
            shape=data_array.shape,
            msg_prefix=(
                f"Expected a 2D DataArray ('{spec.ensemble_dim}' and one other dimension)."
            ),
        )

    (x_dim,) = (dim for dim in data_array.dims if dim != spec.ensemble_dim)
    # Orient as (x, ensemble) so each column is one member.
    data_array = data_array.transpose(x_dim, spec.ensemble_dim)
    values: Any = np.asarray(data_array.values)  # type: ignore[reportUnknownArgumentType]
    x, x_unit = _prepare_x_axis(data_array, str(x_dim))

    ax = create_ax_func()

    if spec.show_members:
        ax.plot(x, values, color="0.6", lw=0.5, alpha=0.3)  # type: ignore[reportUnknownMemberType]

    if spec.band_type != "none":
        lower_pct, upper_pct = _BAND_PERCENTILES[spec.band_type]
        lower: Any = np.nanpercentile(values, lower_pct, axis=1)
        upper: Any = np.nanpercentile(values, upper_pct, axis=1)
        ax.fill_between(  # type: ignore[reportUnknownMemberType]
            x, lower, upper, alpha=0.25, color="C0", label=_BAND_LABELS[spec.band_type]
        )

    if spec.show_mean:
        mean = np.nanmean(values, axis=1)
        ax.plot(x, mean, color="C0", lw=2, label="Ensemble mean")  # type: ignore[reportUnknownMemberType]

    if spec.band_type != "none" or spec.show_mean:
        ax.legend()  # type: ignore[reportUnknownMemberType]

    annotate_plot(ax, spec.annotation)
    if x_unit is not None and not spec.annotation.xlabel:
        ax.set_xlabel(f"{x_dim} ({x_unit})")  # type: ignore[reportUnknownMemberType]
    return ax
