import logging
from functools import singledispatch
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
from anyio import to_thread

mpl.use("Agg")

import xarray as xr
from matplotlib import pyplot as plt

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    Expression,
    FunctionCall,
    FunctionSet,
    VariableAssignment,
    VariableReference,
)
from llm_agent.agents.algorithm_agent.algorithm.data_array_functions import (
    BinaryOperation,
    DataArrayArgmaxCoords,
    DataArrayArgminCoords,
    DataArrayToList,
    FilterDataArray,
    FilterDataArrayByPolygon,
    GeocodeNaturalLanguageDescription,
    GroupByTimeAndReduceDataArray,
    PlotDataArraysAsRaster,
    PlotDataArraysAsTimeSeries,
    PointCoords,
    ReduceDataArray,
    SelectPointCoordsKey,
    ZeroDimDataArrayToScalar,
)
from llm_agent.agents.algorithm_agent.algorithm.general_plot_functions import (
    PlotBarChart,
)
from llm_agent.agents.scientific_python_agent.utils.agent_code_utils import (
    compute_dask_array,
    filter_data_array,
    geocode_natural_language_description,
    mask_data_array_with_polygon,
    max_coords,
    min_coords,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from shapely.geometry import Polygon


class VariableTypeMismatchError(Exception):
    """Referenced variable is not of the expected type."""


class VariableReferenceError(Exception):
    """Referenced variable not found."""


class DataArrayShapeError(Exception):
    """DataArray is a different shape than expected."""


class ExecutionContext:
    """Execution context that tracks assigned variables."""

    def __init__(self, input_vars: dict[str, Any]) -> None:
        """Construct."""
        self.variables: dict[str, Any] = dict(**input_vars)


async def execute_algorithm(
    algo: Algorithm[FunctionSet], input_vars: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate algorithm."""

    # Useful because `to_thread.run_sync` takes args of (func, *args), which
    # breaks kwargs and can be hard to reason about.
    #
    # Not `async` because `to_thread.run_sync` runs a synchronous function in a
    # worker thread.
    def _exec() -> dict[str, Any]:
        context = ExecutionContext(input_vars)
        for statement in algo.statements:
            eval_expression(statement, context)
        out_var_names = [v.name for v in algo.output_variables]
        out = {v: context.variables.get(v) for v in out_var_names}
        return out

    return await to_thread.run_sync(_exec)


@singledispatch
def eval_expression(expr: Any, context: ExecutionContext) -> Any:  # noqa: ANN401
    """Evaluate expression."""
    raise NotImplementedError


@eval_expression.register(str)
def _(expr: str, context: ExecutionContext) -> str:  # noqa: ARG001
    """Evaluate expression."""
    return expr


@eval_expression.register(int)
def _(expr: int, context: ExecutionContext) -> int:  # noqa: ARG001
    """Evaluate expression."""
    return expr


@eval_expression.register(float)
def _(expr: float, context: ExecutionContext) -> float:  # noqa: ARG001
    """Evaluate expression."""
    return expr


@eval_expression.register(list)
def _(expr: list[Any], context: ExecutionContext) -> list[Any]:
    """Evaluate expression."""
    return [eval_expression(v, context) for v in expr]


@eval_expression.register(Expression)
def _(expr: Expression, context: ExecutionContext) -> Any:  # noqa: ANN401
    """Evaluate expression."""
    return eval_expression(expr, context)


@eval_expression.register(VariableAssignment)
def _(
    expr: VariableAssignment[FunctionSet], context: ExecutionContext
) -> None:
    """Evaluate expression."""
    context.variables[expr.name] = eval_expression(expr.value, context)


@eval_expression.register(VariableReference)
def _(expr: VariableReference, context: ExecutionContext) -> Any:  # noqa: ANN401
    """Evaluate expression."""
    try:
        return context.variables[expr.name]
    except KeyError as e:
        raise VariableReferenceError from e


@eval_expression.register(FunctionCall)
def _(expr: FunctionCall[FunctionSet], context: ExecutionContext) -> Any:  # noqa: ANN401
    """Evaluate expression."""
    return eval_expression(expr.func, context)


@eval_expression.register(BinaryOperation)
def _(
    expr: BinaryOperation, context: ExecutionContext
) -> int | float | xr.DataArray:
    """Evaluate expression."""
    left = expr.left
    if isinstance(left, VariableReference):
        left = eval_expression(left, context)
        if not isinstance(left, int | float | xr.DataArray):
            msg = (
                "Expected left arg to be type `int | float | xr.DataArray`, "
                f"found: {type(left)}."
            )
            raise VariableTypeMismatchError(msg)
    right = expr.right
    if isinstance(right, VariableReference):
        right = eval_expression(right, context)
        if not isinstance(right, int | float | xr.DataArray):
            msg = (
                "Expected right arg to be type `int | float | xr.DataArray`, "
                f"found type: {type(right)}."
            )
            raise VariableTypeMismatchError(msg)
    match expr.op:
        case "add":
            return left + right
        case "sub":
            return left - right
        case "mul":
            return left * right
        case "div":
            return left / right


@eval_expression.register(DataArrayArgmaxCoords)
def _(expr: DataArrayArgmaxCoords, context: ExecutionContext) -> PointCoords:
    """Evaluate expression."""
    data_array: xr.DataArray = eval_expression(expr.data_array, context)
    coords_dict = max_coords(data_array)
    return PointCoords(
        time=coords_dict["time"],  # type: ignore[reportArgumentType]
        lat=coords_dict["lat"],  # type: ignore[reportArgumentType]
        lon=coords_dict["lon"],  # type: ignore[reportArgumentType]
        value=coords_dict["value"],  # type: ignore[reportArgumentType]
    )


@eval_expression.register(DataArrayArgminCoords)
def _(expr: DataArrayArgminCoords, context: ExecutionContext) -> PointCoords:
    """Evaluate expression."""
    data_array: xr.DataArray = eval_expression(expr.data_array, context)
    data_array = compute_dask_array(data_array)
    coords_dict = min_coords(data_array)
    return PointCoords(
        time=coords_dict["time"],  # type: ignore[reportArgumentType]
        lat=coords_dict["lat"],  # type: ignore[reportArgumentType]
        lon=coords_dict["lon"],  # type: ignore[reportArgumentType]
        value=coords_dict["value"],  # type: ignore[reportArgumentType]
    )


@eval_expression.register(DataArrayToList)
def _(expr: DataArrayToList, context: ExecutionContext) -> list[int | float]:
    """Evaluate expression."""
    data_array: xr.DataArray = eval_expression(expr.data_array, context)
    data_array = compute_dask_array(data_array)
    return data_array.values.tolist()  # type: ignore[reportReturnType]


@eval_expression.register(ZeroDimDataArrayToScalar)
def _(
    expr: ZeroDimDataArrayToScalar, context: ExecutionContext
) -> int | float:
    """Evaluate expression."""
    data_array = eval_expression(expr.data_array, context)
    if isinstance(data_array, int | float):
        return data_array
    if not isinstance(data_array, xr.DataArray):
        msg = (
            "Input to ZeroDimDataArrayToScalar is neither a DataArray "
            "nor a scalar."
        )
        raise TypeError(msg)
    data_array = compute_dask_array(data_array)
    return data_array.values.item()  # type: ignore[reportReturnType]


@eval_expression.register(FilterDataArray)
def _(expr: FilterDataArray, context: ExecutionContext) -> xr.DataArray:
    """Evaluate expression."""
    data_array: xr.DataArray = eval_expression(expr.data_array, context)
    time = slice(*expr.time_range) if expr.time_range is not None else None
    lat = slice(*expr.lat_range) if expr.lat_range is not None else None
    lon = slice(*expr.lon_range) if expr.lon_range is not None else None
    out = filter_data_array(data_array, time=time, lat=lat, lon=lon)
    return out


@eval_expression.register(FilterDataArrayByPolygon)
def _(
    expr: FilterDataArrayByPolygon, context: ExecutionContext
) -> xr.DataArray:
    """Evaluate expression."""
    data_array: xr.DataArray = eval_expression(expr.data_array, context)
    polygon: Polygon = eval_expression(expr.polygon, context)
    out = mask_data_array_with_polygon(data_array, polygon=polygon, crop=True)
    return out


@eval_expression.register(GeocodeNaturalLanguageDescription)
def _(
    expr: GeocodeNaturalLanguageDescription,
    context: ExecutionContext,  # noqa: ARG001
) -> "Polygon":
    """Evaluate expression."""
    return geocode_natural_language_description(expr.text)  # type: ignore[reportReturnType]


@eval_expression.register(GroupByTimeAndReduceDataArray)
def _(
    expr: GroupByTimeAndReduceDataArray, context: ExecutionContext
) -> xr.DataArray:
    """Evaluate expression."""
    data_array: xr.DataArray = eval_expression(expr.data_array, context)

    match expr.time_unit:
        case "week":
            group = data_array.time.dt.isocalendar().week
        case "dayofmonth":
            group = "time.day"
        case _:
            group = f"time.{expr.time_unit}"

    data_array_grouped = data_array.groupby(group)
    if expr.time_unit == "day":
        reduce_dims = ("lat", "lon")
    else:
        reduce_dims = ("time", "lat", "lon")
    match expr.op:
        case "sum":
            return data_array_grouped.sum(dim=reduce_dims)
        case "mean":
            return data_array_grouped.mean(dim=reduce_dims)
        case "std":
            return data_array_grouped.std(dim=reduce_dims)
        case "max":
            return data_array_grouped.max(dim=reduce_dims)
        case "min":
            return data_array_grouped.min(dim=reduce_dims)


@eval_expression.register(PlotBarChart)
def _(expr: PlotBarChart, context: ExecutionContext) -> "Figure":
    """Evaluate expression."""
    heights = [eval_expression(h, context) for h in expr.heights]
    heights = [
        compute_dask_array(h) if isinstance(h, xr.DataArray) else h
        for h in heights
    ]
    fig, ax = plt.subplots()  # type: ignore[reportUnknownMemberType]
    ax.bar(x=expr.labels, height=heights)  # type: ignore[reportUnknownMemberType]
    ax.set_title(expr.title)  # type: ignore[reportUnknownMemberType]
    return fig


@eval_expression.register(PlotDataArraysAsTimeSeries)
def _(expr: PlotDataArraysAsTimeSeries, context: ExecutionContext) -> "Figure":
    """Evaluate expression."""
    data_arrays = [
        compute_dask_array(eval_expression(da, context))
        for da in expr.data_arrays
    ]
    for da, da_var in zip(data_arrays, expr.data_arrays, strict=True):
        if da.ndim != 1:
            msg = (
                f"Expected DataArray {da_var.name} to be 1D "
                f"but found shape: {da.shape}."
            )
            raise DataArrayShapeError(msg)

    # The time dim name could be "time" or "dayofyear" or "week" etc. It only
    # makes sense to plot multiple time series in the same plot if they have
    # the same time dim.
    time_dim_names = [da.dims[0] for da in data_arrays]
    same_time_dim_name = len(set(time_dim_names)) == 1
    if not same_time_dim_name:
        msg = (
            "Time series DataArrays do not have the same time dim. "
            f"Found: {time_dim_names}."
        )
        raise ValueError(msg)

    fig, ax = plt.subplots()  # type: ignore[reportUnknownMemberType]
    if not expr.force_overlaid:
        # use xr.DataArray.plot() which get us nice axis labels
        for da, label in zip(data_arrays, expr.labels, strict=False):
            da.plot(ax=ax, label=label)  # type: ignore[reportCallIssue]
    else:
        lengths = [len(da) for da in data_arrays]
        same_lengths = len(set(lengths)) == 1
        if not same_lengths:
            log.warning(
                "Overlaying time series of non-identical lengths: %s",
                lengths,
            )
        # Treat the time series as a list of numbers, ignoring axis info.
        for da, label in zip(data_arrays, expr.labels, strict=False):
            ax.plot(da, label=label)  # type: ignore[reportUnknownMemberType]
    ax.legend()  # type: ignore[reportUnknownMemberType]
    ax.set_title(expr.title)  # type: ignore[reportUnknownMemberType]
    return fig


@eval_expression.register(PlotDataArraysAsRaster)
def _(expr: PlotDataArraysAsRaster, context: ExecutionContext) -> "Figure":
    """Evaluate expression."""
    data_arrays = [
        compute_dask_array(eval_expression(da, context))
        for da in expr.data_arrays
    ]
    fig, axs = plt.subplots(  # type: ignore[reportUnknownMemberType]
        *expr.subplot_grid_shape, squeeze=False, constrained_layout=True
    )

    # Array should be either a 2-dimensional array with x,y coordinates, or
    # 3-dimensional with x,y, and t coordinates.
    for ax, da in zip(axs.flat, data_arrays, strict=True):  # type: ignore[reportUnknownMemberType]
        if da.ndim not in (2, 3):
            # Unexpected number of dimensions
            msg = f"Invalid data array shape: {da.shape}."
            raise ValueError(msg)
        if "time" in da.dims and len(da["time"]) > 0:
            # Time is provided, but we only want to plot lat,lon.
            # Arbitrarily pick the first time.
            log.warning(
                "Multiple time steps found in data array. Found %d",
                len(da["time"]),
            )
            da.isel(time=0).plot(ax=ax)  # type: ignore[reportCallIssue]
        else:
            da.plot(ax=ax)  # type: ignore[reportCallIssue]
    if len(axs.flat) == 1:  # type: ignore[reportUnknownMemberType]
        ax.set_title(expr.title)  # type: ignore[reportUnknownMemberType]
    else:
        fig.suptitle(expr.title)  # type: ignore[reportUnknownMemberType]
    return fig


@eval_expression.register(ReduceDataArray)
def _(expr: ReduceDataArray, context: ExecutionContext) -> xr.DataArray:
    """Evaluate expression."""
    data_array: xr.DataArray = eval_expression(expr.data_array, context)
    match expr.op:
        case "sum":
            return data_array.sum(dim=expr.dim)
        case "mean":
            return data_array.mean(dim=expr.dim)
        case "std":
            return data_array.std(dim=expr.dim)
        case "max":
            return data_array.max(dim=expr.dim)
        case "min":
            return data_array.min(dim=expr.dim)


@eval_expression.register(SelectPointCoordsKey)
def _(expr: SelectPointCoordsKey, context: ExecutionContext) -> str | float:
    """Evaluate expression."""
    point_coords: PointCoords = eval_expression(expr.point_coords, context)
    return getattr(point_coords, expr.key)
