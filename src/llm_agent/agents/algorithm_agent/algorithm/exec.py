import logging
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Self

import langfuse
import matplotlib as mpl
import shapely
from anyio import to_thread
from langfuse import observe  # pyright: ignore[reportUnknownVariableType]
from shapely.geometry.base import BaseGeometry

from llm_agent.agents.algorithm_agent.algorithm.langfuse import (
    transform_execute_algorithm_input_vars,
)
from llm_agent.context import ToolCallContext
from llm_agent.tools.data_analysis_tool.types import GeoTIFFGenerationError, GeoTIFFUploadError
from llm_agent.transformers.geotiff_transformers import (
    InvalidShapeForGeoTIFFError,
    data_array_to_geotiff_ref,
)
from llm_agent.uploaders.s3_file_uploader import UploadError

mpl.use("Agg")

import xarray as xr
from matplotlib import pyplot as plt
from pydantic import BaseModel
from shapely.geometry import mapping

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    BaseFunctionCall,
    VariableAssignment,
    VariableReference,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
    AssignUserProvidedGeometry,
    BinaryOperation,
    DataArrayArgmaxCoords,
    DataArrayArgminCoords,
    DataArrayToList,
    FilterDataArrayByPolygon,
    GeocodeNaturalLanguageDescription,
    GroupByTimeAndReduceDataArray,
    PlotDataArraysAsRaster,
    PlotDataArraysAsTimeSeries,
    PointCoords,
    ReduceDataArray,
    SelectPointCoordsKey,
    SliceDataArray,
    ZeroDimDataArrayToScalar,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.general_plot_functions import (
    PlotBarChart,
)
from llm_agent.agents.scientific_python_agent.utils.agent_code_utils import (
    compute_dask_array,
    geocode_natural_language_description,
    mask_data_array_with_polygon,
    max_coords,
    min_coords,
    validate_data_array_size_within_limit,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from shapely.geometry import Polygon

    from llm_agent.tools.data_analysis_tool.types import GeoTIFFRef


class AlgorithmExecutionError(Exception):
    """Exception raised when algorithm execution fails."""

    def __init__(self, message: str, func_name: str | None = None) -> None:
        """Constructor."""
        msg = f"{func_name}: {message}" if func_name else message
        super().__init__(msg)


class VariableTypeMismatchError(AlgorithmExecutionError):
    """Referenced variable is not of the expected type."""


class VariableReferenceError(AlgorithmExecutionError):
    """Referenced variable not found."""

    def __init__(self, var_name: str) -> None:
        """Constructor."""
        super().__init__(f"Variable '{var_name}' referenced but not set.")


class DataArrayShapeError(AlgorithmExecutionError):
    """DataArray is a different shape than expected."""

    @classmethod
    def with_shape(
        cls,
        func_name: str,
        dims: tuple[str, ...],
        shape: tuple[int, ...],
        msg_prefix: str | None = None,
    ) -> Self:
        """Create a DataArrayShapeError with the given shape information."""
        shape_dict = dict(zip(dims, shape, strict=False))
        msg = (
            f"{msg_prefix}\nGot shape: {shape_dict}." if msg_prefix else f"Got shape: {shape_dict}."
        )
        return cls(msg, func_name=func_name)


class ZeroLengthDimError(DataArrayShapeError):
    """DataArray has a dimension of size 0."""

    def __init__(
        self, func_name: str, data_array: xr.DataArray, msg_prefix: str | None = None
    ) -> None:
        """Constructor."""
        shape_dict = dict(zip(data_array.dims, data_array.shape, strict=False))
        msg = (
            f"{msg_prefix}\nGot shape: {shape_dict}." if msg_prefix else f"Got shape: {shape_dict}."
        )
        super().__init__(msg, func_name=func_name)


class ExecutionProgressUpdate(BaseModel):
    """Progress update during algorithm execution."""

    message: str = "Progress update."


class ExecutionContext:
    """Execution context that tracks assigned variables."""

    variables: dict[str, Any]
    tool_call_context: ToolCallContext

    def __init__(self, tool_call_context: ToolCallContext, input_vars: dict[str, Any]) -> None:
        """Construct."""
        self.tool_call_context = tool_call_context
        self.variables: dict[str, Any] = dict(**input_vars)


def _validate_data_array_non_zero_dimensions(
    func_name: str, data_array: xr.DataArray, msg_prefix: str | None = None
) -> None:
    """Validate that no dimension of the data array has zero length."""
    if any(size == 0 for size in data_array.sizes.values()):
        raise ZeroLengthDimError(func_name, data_array, msg_prefix=msg_prefix)


@observe(as_type="chain", capture_input=False)
async def execute_algorithm[FunctionCall: BaseFunctionCall](
    tool_call_context: ToolCallContext, algo: Algorithm[FunctionCall], input_vars: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate algorithm."""
    langfuse.get_client().update_current_span(
        input={"algo": algo, "input_vars": transform_execute_algorithm_input_vars(input_vars)}
    )

    # Useful because `to_thread.run_sync` takes args of (func, *args), which
    # breaks kwargs and can be hard to reason about.
    #
    # Not `async` because `to_thread.run_sync` runs a synchronous function in a
    # worker thread.
    @observe(as_type="chain")
    def _exec() -> dict[str, Any]:
        exec_context = ExecutionContext(tool_call_context, input_vars)
        for statement in algo.statements:
            eval_expression(statement, exec_context)
        out_var_names = [v.name for v in algo.output_variables]
        out = {v: exec_context.variables.get(v) for v in out_var_names}
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


@eval_expression.register(VariableAssignment)
def _(expr: VariableAssignment[BaseFunctionCall], context: ExecutionContext) -> None:
    """Evaluate expression."""
    context.variables[expr.name] = eval_expression(expr.value, context)


@eval_expression.register(VariableReference)
def _(expr: VariableReference, context: ExecutionContext) -> Any:  # noqa: ANN401
    """Evaluate expression."""
    try:
        return context.variables[expr.name]
    except KeyError as e:
        raise VariableReferenceError(expr.name) from e


@eval_expression.register(BaseFunctionCall)
def _(expr: BaseFunctionCall, context: ExecutionContext) -> Any:  # noqa: ANN401
    """Evaluate expression."""
    return eval_expression(expr.func, context)


@eval_expression.register(BinaryOperation)
def _(expr: BinaryOperation, context: ExecutionContext) -> int | float | xr.DataArray:
    """Evaluate expression."""
    left = expr.left
    if isinstance(left, VariableReference):
        left = eval_expression(left, context)
        if not isinstance(left, int | float | xr.DataArray):
            msg = f"Expected left arg to be type `int | float | xr.DataArray`, found: {type(left)}."
            raise VariableTypeMismatchError(msg, func_name=f"{expr.func_name}({expr.op})")
    right = expr.right
    if isinstance(right, VariableReference):
        right = eval_expression(right, context)
        if not isinstance(right, int | float | xr.DataArray):
            msg = (
                "Expected right arg to be type `int | float | xr.DataArray`, "
                f"found type: {type(right)}."
            )
            raise VariableTypeMismatchError(msg, func_name=f"{expr.func_name}({expr.op})")
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
    validate_data_array_size_within_limit(data_array)
    coords_dict = max_coords(data_array)
    value = coords_dict.pop("value")
    return PointCoords(
        coords=coords_dict,
        value=value,  # type: ignore[reportArgumentType]
    )


@eval_expression.register(DataArrayArgminCoords)
def _(expr: DataArrayArgminCoords, context: ExecutionContext) -> PointCoords:
    """Evaluate expression."""
    data_array: xr.DataArray = eval_expression(expr.data_array, context)
    validate_data_array_size_within_limit(data_array)
    coords_dict = min_coords(data_array)
    value = coords_dict.pop("value")
    return PointCoords(
        coords=coords_dict,
        value=value,  # type: ignore[reportArgumentType]
    )


@eval_expression.register(DataArrayToList)
def _(expr: DataArrayToList, context: ExecutionContext) -> list[int | float]:
    """Evaluate expression."""
    data_array: xr.DataArray = eval_expression(expr.data_array, context)
    data_array = compute_dask_array(data_array)
    return data_array.values.tolist()  # type: ignore[reportReturnType]


@eval_expression.register(ZeroDimDataArrayToScalar)
def _(expr: ZeroDimDataArrayToScalar, context: ExecutionContext) -> int | float:
    """Evaluate expression."""
    data_array = eval_expression(expr.data_array, context)
    if isinstance(data_array, int | float):
        return data_array
    if not isinstance(data_array, xr.DataArray):
        msg = "Input to ZeroDimDataArrayToScalar is neither a DataArray nor a scalar."
        raise VariableTypeMismatchError(msg, func_name=expr.func_name)
    data_array = compute_dask_array(data_array)
    data_array = data_array.squeeze()
    if data_array.ndim != 0:
        raise DataArrayShapeError.with_shape(
            func_name=expr.func_name,
            dims=data_array.dims,  # pyright: ignore[reportArgumentType]
            shape=data_array.shape,
            msg_prefix="Expected 0D DataArray after squeezing. Consider reducing dims.",
        )
    return data_array.values.item()  # type: ignore[reportReturnType]


@eval_expression.register(SliceDataArray)
def _(expr: SliceDataArray, context: ExecutionContext) -> xr.DataArray:
    """Evaluate expression."""
    data_array: xr.DataArray = eval_expression(expr.data_array, context)
    _validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot slice a DataArray with a zero-length dimension.",
    )
    value_based_indexers = {k: slice(*v) for k, v in expr.dim_slices.items()}
    index_based_indexers = {k: slice(*v) for k, v in expr.dim_index_slices.items()}
    data_array = data_array.sel(value_based_indexers)  # type: ignore[reportUnknownMemberType]
    data_array = data_array.isel(index_based_indexers)  # type: ignore[reportUnknownMemberType]

    _validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Result of slicing has a zero-length dimension.",
    )

    return data_array


@eval_expression.register(FilterDataArrayByPolygon)
def _(expr: FilterDataArrayByPolygon, context: ExecutionContext) -> xr.DataArray:
    """Evaluate expression."""
    data_array: xr.DataArray = eval_expression(expr.data_array, context)
    _validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot filter a DataArray with a zero-length dimension.",
    )

    polygon: Polygon = eval_expression(expr.polygon, context)
    out = mask_data_array_with_polygon(data_array, polygon=polygon)

    _validate_data_array_non_zero_dimensions(
        expr.func_name,
        out,
        msg_prefix="Result of filtering by polygon has a zero-length dimension.",
    )
    return out


class GeocodingSuccessUpdate(BaseModel):
    """Result of geocoding."""

    query: str
    geojson: dict[str, Any]


@eval_expression.register(GeocodeNaturalLanguageDescription)
def _(
    expr: GeocodeNaturalLanguageDescription,
    context: ExecutionContext,
) -> BaseGeometry:
    """Evaluate expression."""
    out = geocode_natural_language_description(
        context.tool_call_context.global_context.place_lookup, expr.text
    )
    context.tool_call_context.update_progress(
        GeocodingSuccessUpdate(query=expr.text, geojson=mapping(out))
    )
    return out


# TODO: (llm-agent#82) Make an update_progress call with the provided AOI?
@eval_expression.register(AssignUserProvidedGeometry)
def _(
    expr: AssignUserProvidedGeometry,
    context: ExecutionContext,  # noqa: ARG001
) -> BaseGeometry:
    """Evaluate expression."""
    geometry = shapely.from_geojson(expr.geometry.model_dump_json())  # type: ignore[reportUnknownMemberType]
    return geometry


def _get_time_coord_name(da: xr.DataArray) -> str:
    for name, coord in da.coords.items():  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if coord.dtype.kind == "M":  # datetime64 # pyright: ignore[reportUnknownMemberType]
            return name  # pyright: ignore[reportReturnType]
    msg = "No datetime coordinate found"
    raise ValueError(msg)


@eval_expression.register(GroupByTimeAndReduceDataArray)
def _(expr: GroupByTimeAndReduceDataArray, context: ExecutionContext) -> xr.DataArray:
    """Evaluate expression."""
    data_array: xr.DataArray = eval_expression(expr.data_array, context)
    _validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot group a DataArray with a zero-length dimension.",
    )
    validate_data_array_size_within_limit(data_array)

    try:
        time_coord = _get_time_coord_name(data_array)
    except ValueError as e:
        msg = f"Failed to find temporal coordinate in DataArray with dims {data_array.dims}."
        raise VariableTypeMismatchError(msg, func_name=expr.func_name) from e

    match expr.time_unit:
        case "week":
            group = data_array[time_coord].dt.isocalendar().week
        case "dayofmonth":
            group = data_array[time_coord].dt.day
        case _:
            group = f"{time_coord}.{expr.time_unit}"

    data_array_grouped = data_array.groupby(group)

    # After grouping, we want to reduce all the non-time dimensions. However,
    # somewhat unintuitively, data_array_grouped.sum(dim=None) does NOT return
    # a 1D array. The correct way to do this is to specify all the dims
    # (including time); i.e.: data_array_grouped.sum(dim=data_array.dims).
    match expr.op:
        case "sum":
            return data_array_grouped.sum(dim=data_array.dims)
        case "mean":
            return data_array_grouped.mean(dim=data_array.dims)
        case "std":
            return data_array_grouped.std(dim=data_array.dims)
        case "max":
            return data_array_grouped.max(dim=data_array.dims)
        case "min":
            return data_array_grouped.min(dim=data_array.dims)


@eval_expression.register(PlotBarChart)
def _(expr: PlotBarChart, context: ExecutionContext) -> "Figure":
    """Evaluate expression."""
    heights = [eval_expression(h, context) for h in expr.heights]
    heights = [compute_dask_array(h) if isinstance(h, xr.DataArray) else h for h in heights]
    fig, ax = plt.subplots()  # type: ignore[reportUnknownMemberType]
    ax.bar(x=expr.labels, height=heights)  # type: ignore[reportUnknownMemberType]
    ax.set_title(expr.title)  # type: ignore[reportUnknownMemberType]
    return fig


@eval_expression.register(PlotDataArraysAsTimeSeries)
def _(expr: PlotDataArraysAsTimeSeries, context: ExecutionContext) -> "Figure":
    """Evaluate expression."""
    data_arrays = [compute_dask_array(eval_expression(da, context)) for da in expr.data_arrays]
    for da, da_var in zip(data_arrays, expr.data_arrays, strict=True):
        if da.ndim != 1:
            raise DataArrayShapeError.with_shape(
                func_name=expr.func_name,
                dims=da.dims,  # pyright: ignore[reportArgumentType]
                shape=da.shape,
                msg_prefix=f"Expected DataArray {da_var.name} to be 1D.",
            )

    # The time dim name could be "time" or "dayofyear" or "week" etc. It only
    # makes sense to plot multiple time series in the same plot if they have
    # the same time dim.
    time_dim_names = [da.dims[0] for da in data_arrays]
    same_time_dim_name = len(set(time_dim_names)) == 1
    if not same_time_dim_name:
        msg = f"Time series DataArrays do not have the same time dim. Found: {time_dim_names}."
        raise DataArrayShapeError(msg, func_name=expr.func_name)

    fig, ax = plt.subplots()  # type: ignore[reportUnknownMemberType]
    if not expr.force_overlaid:
        # use xr.DataArray.plot() which get us nice axis labels
        for da, label in zip(data_arrays, expr.labels, strict=False):
            if len(da) == 1:
                # .plot() throws an error if the array has only 1 item.
                # In such cases, plot a single point instead.
                da.plot.scatter(ax=ax, label=label)  # type: ignore[reportUnknownMemberType]
            else:
                da.plot.line(ax=ax, label=label)
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
            time_dim_name = da.dims[0]
            time_dim = da[time_dim_name]
            if len(da) == 1:
                # calling .plot() w/ a length 1 array plots nothing.
                # In such cases, plot a single point instead.
                ax.scatter(time_dim, da, label=label)  # type: ignore[reportUnknownMemberType]
            else:
                ax.plot(da, label=label)  # type: ignore[reportUnknownMemberType]
            ax.set_ylabel(da.name)  # type: ignore[reportUnknownMemberType]
            ax.set_xlabel(time_dim_name)  # type: ignore[reportUnknownMemberType]
    ax.legend()  # type: ignore[reportUnknownMemberType]
    ax.set_title(expr.title)  # type: ignore[reportUnknownMemberType]
    return fig


def _create_geotiff_ref(data_array: xr.DataArray, context: ExecutionContext) -> "GeoTIFFRef":
    """Evaluate expression."""
    try:
        return data_array_to_geotiff_ref(
            data_array, context.tool_call_context.global_context.s3_client
        )
    except UploadError as e:
        raise GeoTIFFUploadError from e
    except InvalidShapeForGeoTIFFError as e:
        raise GeoTIFFGenerationError(repr(e)) from e
    except Exception as e:
        raise GeoTIFFGenerationError from e


@eval_expression.register(PlotDataArraysAsRaster)
def _(expr: PlotDataArraysAsRaster, context: ExecutionContext) -> "Figure | GeoTIFFRef":
    """Evaluate expression."""
    data_arrays = [compute_dask_array(eval_expression(da, context)) for da in expr.data_arrays]

    def _squeeze(da: xr.DataArray) -> xr.DataArray:
        da_out = da.squeeze(drop=True)
        if da_out.ndim != 2:  # noqa: PLR2004
            raise DataArrayShapeError.with_shape(
                func_name=expr.func_name,
                dims=da.dims,  # pyright: ignore[reportArgumentType]
                shape=da.shape,
                msg_prefix="Expected a DataArray squeezable to 2D.",
            )
        return da_out

    data_arrays = [_squeeze(da) for da in data_arrays]

    if len(data_arrays) == 1 and "lat" in data_arrays[0].dims and "lon" in data_arrays[0].dims:
        _validate_data_array_non_zero_dimensions(
            expr.func_name,
            data_arrays[0],
            msg_prefix="Creating a GeoTIFF is only supported for DataArrays with non-zero lat and "
            "lon dimensions.",
        )
        return _create_geotiff_ref(data_arrays[0], context)

    fig, axs = plt.subplots(  # type: ignore[reportUnknownMemberType]
        *expr.subplot_grid_shape, squeeze=False, constrained_layout=True
    )

    # Array should be either a 2-dimensional array with x, y coordinates, or
    # squeezable to 2D.
    for ax, da in zip(axs.flat, data_arrays, strict=True):  # type: ignore[reportUnknownMemberType]
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
    _validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot reduce a DataArray with a zero-length dimension.",
    )
    validate_data_array_size_within_limit(data_array)
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
    match expr.key:
        case "value":
            return point_coords.value
        case _:
            return point_coords.coords[expr.key]  # type: ignore[reportIndexType]
