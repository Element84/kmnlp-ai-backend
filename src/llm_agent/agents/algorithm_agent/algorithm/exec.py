from functools import singledispatch
from typing import TYPE_CHECKING, Any

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
    DataArrayToScalar,
    FilterDataArray,
    FilterDataArrayByPolygon,
    GeocodeNaturalLanguageDescription,
    GroupByTimeAndReduceDataArray,
    PlotDataArraysAsRaster,
    PlotDataArraysAsTimeSeries,
    PointCoords,
    ReduceDataArray,
    SelectPointCoordsDim,
)
from llm_agent.agents.algorithm_agent.algorithm.general_plot_functions import (
    PlotBarChart,
)
from llm_agent.agents.scientific_python_agent.utils.agent_code_utils import (
    compute,
    filter_data_array,
    geocode_natural_language_description,
    mask_data_array_with_polygon,
    max_coords,
    min_coords,
)

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


def execute_algorithm(
    algo: Algorithm[FunctionSet], input_vars: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate algorithm."""
    context = ExecutionContext(input_vars)
    for statement in algo.statements:
        eval_expression(statement, context)
    out_var_names = [v.name for v in algo.output_variables]
    out = {v: context.variables.get(v) for v in out_var_names}
    return out


@singledispatch
def eval_expression(expr: Any, context: ExecutionContext) -> Any:  # noqa: ANN401
    """Evaluate expression."""
    raise NotImplementedError


@eval_expression.register(VariableAssignment)
def _(
    expr: VariableAssignment[FunctionSet], context: ExecutionContext
) -> None:
    """Evaluate expression."""
    value = expr.value
    if isinstance(value, Expression):
        value = eval_expression(value, context)
    context.variables[expr.name] = value


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
    )


@eval_expression.register(DataArrayArgminCoords)
def _(expr: DataArrayArgminCoords, context: ExecutionContext) -> PointCoords:
    """Evaluate expression."""
    data_array: xr.DataArray = eval_expression(expr.data_array, context)
    coords_dict = min_coords(data_array)
    return PointCoords(
        time=coords_dict["time"],  # type: ignore[reportArgumentType]
        lat=coords_dict["lat"],  # type: ignore[reportArgumentType]
        lon=coords_dict["lon"],  # type: ignore[reportArgumentType]
    )


@eval_expression.register(DataArrayToList)
def _(expr: DataArrayToList, context: ExecutionContext) -> list[int | float]:
    """Evaluate expression."""
    data_array: xr.DataArray = eval_expression(expr.data_array, context)
    return data_array.values.tolist()  # type: ignore[reportReturnType]


@eval_expression.register(DataArrayToScalar)
def _(expr: DataArrayToScalar, context: ExecutionContext) -> int | float:
    """Evaluate expression."""
    data_array: xr.DataArray = eval_expression(expr.data_array, context)
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
        case "time_dim":
            group = "time"
        case "week":
            group = data_array.time.dt.isocalendar().week
        case "dayofmonth":
            group = "time.day"
        case _:
            group = f"time.{expr.time_unit}"

    data_array_grouped = data_array.groupby(group)

    match expr.op:
        case "sum":
            return data_array_grouped.sum(dim=("time", "lat", "lon"))
        case "mean":
            return data_array_grouped.mean(dim=("time", "lat", "lon"))
        case "std":
            return data_array_grouped.std(dim=("time", "lat", "lon"))
        case "max":
            return data_array_grouped.max(dim=("time", "lat", "lon"))
        case "min":
            return data_array_grouped.min(dim=("time", "lat", "lon"))


@eval_expression.register(PlotBarChart)
def _(expr: PlotBarChart, context: ExecutionContext) -> "Figure":
    """Evaluate expression."""
    heights = [compute(eval_expression(h, context)) for h in expr.heights]
    fig, ax = plt.subplots()  # type: ignore[reportUnknownMemberType]
    ax.bar(x=expr.labels, height=heights)  # type: ignore[reportUnknownMemberType]
    ax.set_title(expr.title)  # type: ignore[reportUnknownMemberType]
    return fig


@eval_expression.register(PlotDataArraysAsTimeSeries)
def _(expr: PlotDataArraysAsTimeSeries, context: ExecutionContext) -> "Figure":
    """Evaluate expression."""
    data_arrays = [
        compute(eval_expression(da, context)) for da in expr.data_arrays
    ]
    for da, da_var in zip(data_arrays, expr.data_arrays, strict=True):
        if da.ndim != 1:
            msg = (
                f"Expected DataArray {da_var.name} to be 1D "
                f"but found shape: {da.shape}."
            )
            raise DataArrayShapeError(msg)
    fig, ax = plt.subplots()  # type: ignore[reportUnknownMemberType]
    for da in data_arrays:
        da.plot(ax=ax)  # type: ignore[reportCallIssue]
    ax.set_title(expr.title)  # type: ignore[reportUnknownMemberType]
    return fig


@eval_expression.register(PlotDataArraysAsRaster)
def _(expr: PlotDataArraysAsRaster, context: ExecutionContext) -> "Figure":
    """Evaluate expression."""
    data_arrays = [
        compute(eval_expression(da, context)) for da in expr.data_arrays
    ]
    fig, axs = plt.subplots(  # type: ignore[reportUnknownMemberType]
        *expr.subplot_grid_shape, squeeze=False, constrained_layout=True
    )
    for ax, da in zip(axs.flat, data_arrays, strict=True):  # type: ignore[reportUnknownMemberType]
        da.isel(time=0).plot(ax=ax)  # type: ignore[reportCallIssue]
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


@eval_expression.register(SelectPointCoordsDim)
def _(expr: SelectPointCoordsDim, context: ExecutionContext) -> str | float:
    """Evaluate expression."""
    point_coords: PointCoords = eval_expression(expr.point_coords, context)
    return getattr(point_coords, expr.dim)
