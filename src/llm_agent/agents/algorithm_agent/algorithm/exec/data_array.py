# pyright: reportUnknownLambdaType=false
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, cast
from uuid import uuid4

import humanize.filesize
import numpy as np
import xarray as xr
from natural_language_geocoding.errors import GeocodeError
from pydantic import BaseModel
from scipy import stats  # type: ignore[reportMissingTypeStubs]
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon, box, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.exec.base import (
    AlgorithmExecutionError,
    ExecutionContext,
    VariableTypeMismatchError,
    eval_expression,
    summarize_value_for_update,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
    BinaryOperation,
    ComputeAnomaly,
    ComputeClimatology,
    ComputeLinearTrend,
    ComputePercentileRank,
    ComputeTrendPValue,
    CorrelateDataArrays,
    CreateGeoTIFF,
    CumulativeReduce,
    DataArrayArgmaxCoords,
    DataArrayArgminCoords,
    FilterDataArrayByPolygon,
    FilterToTimePeriodAcrossYears,
    GeocodeNaturalLanguageDescription,
    GroupByTimeAndReduceDataArray,
    MaskWhere,
    PointCoords,
    ReduceDataArray,
    RegridToMatch,
    ResampleTime,
    ResampleTimeToMatch,
    RollingReduce,
    SeasonName,
    SelectPointCoordsKey,
    SliceDataArray,
    UnaryOperation,
)
from llm_agent.agents.scientific_python_agent.utils.agent_code_utils import (
    geocode_natural_language_description,
    mask_data_array_with_polygon,
)
from llm_agent.context import DataPersistenceError
from llm_agent.tools.data_analysis_tool.types import GeoTIFFGenerationError, GeoTIFFUploadError
from llm_agent.transformers.geotiff_transformers import (
    InvalidShapeForGeoTIFFError,
    data_array_to_geotiff_ref,
)
from llm_agent.utils.dask import (
    MaterializedDataArray,
    compute_dask_array,
    validate_data_array_size_within_limit,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm_agent.tools.data_analysis_tool.types import GeoTIFFRef

log = logging.getLogger(__name__)

xr.set_options(keep_attrs=True)


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


def validate_data_array_non_zero_dimensions(
    func_name: str, data_array: xr.DataArray, msg_prefix: str | None = None
) -> None:
    """Validate that no dimension of the data array has zero length."""
    if any(size == 0 for size in data_array.sizes.values()):
        raise ZeroLengthDimError(func_name, data_array, msg_prefix=msg_prefix)


class DaskExecutionProgressUpdate(BaseModel):
    """Progress update during Dask array computation."""

    id: str
    label: str | None = None
    completed_tasks: int
    total_tasks: int


async def materialize_data_array(
    data_array: xr.DataArray, context: ExecutionContext, progress_label: str
) -> MaterializedDataArray:
    """Materialize a Dask-backed DataArray, with progress updates."""
    _id = str(uuid4())[:8]

    def _progress_callback(completed: int, total: int) -> None:
        progress_update = DaskExecutionProgressUpdate(
            id=_id,
            completed_tasks=completed,
            total_tasks=total,
            label=progress_label,
        )
        context.tool_call_context.update_progress(progress_update)

    return await compute_dask_array(
        data_array,
        size_limit=context.config.data_array_compute_size_limit,
        progress_callback=_progress_callback,
        progress_poll_interval_sec=context.config.dask_progress_poll_interval_sec,
    )


async def resolve_data_array_var(
    var_ref: VariableReference, context: ExecutionContext
) -> xr.DataArray:
    """Resolve a variable reference to a DataArray, leaving it lazy."""
    value = await eval_expression(var_ref, context)
    if not isinstance(value, xr.DataArray):
        msg = f"Expected variable {var_ref.name} to be a DataArray, found type: {type(value)}."
        raise VariableTypeMismatchError(msg, var_name=var_ref.name, func_name=var_ref.name)
    return value


async def materialize_data_array_var(
    var_ref: VariableReference, context: ExecutionContext
) -> MaterializedDataArray:
    """Helper to materialize a DataArray variable."""
    value = await resolve_data_array_var(var_ref, context)
    return await materialize_data_array(value, context, progress_label=f"{var_ref.name}")


@summarize_value_for_update.register(xr.DataArray)
def _(value: xr.DataArray) -> str | None:
    shape_str = ", ".join(f"{dim}: {size}" for dim, size in value.sizes.items())
    return f"shape: ({shape_str}); size: {humanize.filesize.naturalsize(value.nbytes)}"


@summarize_value_for_update.register(BaseGeometry)
def _(value: BaseGeometry) -> str | None:
    """Helper to create a ComputedVariableUpdate with a value summary."""
    return f"geometry type: {value.geom_type}; bounds: {value.bounds}"


@eval_expression.register(BinaryOperation)
async def eval_binary_operation(
    expr: BinaryOperation, context: ExecutionContext
) -> int | float | xr.DataArray:
    """Apply a binary arithmetic or comparison operation to two operands."""
    left = expr.left
    if isinstance(left, VariableReference):
        left = await eval_expression(left, context)
        if not isinstance(left, int | float | xr.DataArray):
            msg = f"Expected left arg to be type `int | float | xr.DataArray`, found: {type(left)}."
            var_name = expr.left.name if isinstance(expr.left, VariableReference) else "left"
            raise VariableTypeMismatchError(
                msg, var_name=var_name, func_name=f"{expr.func_name}({expr.op})"
            )
    right = expr.right
    if isinstance(right, VariableReference):
        right = await eval_expression(right, context)
        if not isinstance(right, int | float | xr.DataArray):
            msg = (
                "Expected right arg to be type `int | float | xr.DataArray`, "
                f"found type: {type(right)}."
            )
            var_name = expr.right.name if isinstance(expr.right, VariableReference) else "right"
            raise VariableTypeMismatchError(
                msg, var_name=var_name, func_name=f"{expr.func_name}({expr.op})"
            )

    op_func: dict[
        str,
        Callable[
            [int | float | xr.DataArray, int | float | xr.DataArray],
            int | float | xr.DataArray | bool,
        ],
    ] = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
        "eq": lambda x, y: x == y,
        "ne": lambda x, y: x != y,
        "gt": lambda x, y: x > y,
        "lt": lambda x, y: x < y,
        "ge": lambda x, y: x >= y,
        "le": lambda x, y: x <= y,
    }
    return op_func[expr.op](left, right)


@eval_expression.register(UnaryOperation)
async def eval_unary_operation(
    expr: UnaryOperation, context: ExecutionContext
) -> float | xr.DataArray:
    """Apply an element-wise unary math operation to a scalar or DataArray."""
    operand = expr.operand
    if isinstance(operand, VariableReference):
        operand = await eval_expression(operand, context)
        if not isinstance(operand, int | float | xr.DataArray):
            msg = (
                "Expected operand to be type `int | float | xr.DataArray`, "
                f"found type: {type(operand)}."
            )
            var_name = (
                expr.operand.name if isinstance(expr.operand, VariableReference) else "operand"
            )
            raise VariableTypeMismatchError(
                msg, var_name=var_name, func_name=f"{expr.func_name}({expr.op})"
            )

    op_func: dict[str, Callable[[int | float | xr.DataArray], Any]] = {
        "sqrt": np.sqrt,
        "square": np.square,
        "abs": np.abs,
        "exp": np.exp,
        "log": np.log,
        "log10": np.log10,
        "neg": lambda x: -x,
    }
    result = op_func[expr.op](operand)
    if isinstance(result, xr.DataArray):
        return result
    return float(result)


async def _argextremum(
    data_array: xr.DataArray,
    extremum_type: Literal["max", "min"],
    context: ExecutionContext,
    var_name: str,
) -> tuple[dict[str, str | float], float]:
    """Compute argmax or argmin of a DataArray, along with the value at that point."""
    match extremum_type:
        case "max":
            op, arg_op = data_array.max, data_array.argmax
        case "min":
            op, arg_op = data_array.min, data_array.argmin

    extremum_coords_indices = cast("dict[str, xr.DataArray]", arg_op(dim=...))
    extreme_value = cast(
        "float",
        (
            await materialize_data_array(
                op(), context, progress_label=f"{extremum_type}({var_name})"
            )
        ).values.item(),
    )

    extremum_coords_indices_materialized = {
        k: await materialize_data_array(
            v, context, progress_label=f"arg{extremum_type}({var_name})[{k}]"
        )
        for k, v in extremum_coords_indices.items()
    }
    extremum_coords_values = {
        k: (
            await materialize_data_array(
                data_array.coords[k][v.to_numpy()],  # type: ignore[reportAssignmentType]
                context,
                progress_label=f"arg{extremum_type}({var_name})[{k}]",
            )
        ).to_numpy()
        for k, v in extremum_coords_indices_materialized.items()
    }

    coords_dict: dict[str, str | float] = {}
    for k, v in extremum_coords_values.items():
        if v.dtype.kind == "M":  # datetime
            coords_dict[k] = str(np.datetime_as_string(v))
        else:
            coords_dict[k] = round(v.item(), 6)

    return coords_dict, extreme_value


@eval_expression.register(DataArrayArgmaxCoords)
async def eval_data_array_argmax_coords(
    expr: DataArrayArgmaxCoords, context: ExecutionContext
) -> PointCoords:
    """Find the coordinates of the maximum value in a DataArray."""
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_size_within_limit(data_array, context.config.data_array_compute_size_limit)
    coords_dict, value = await _argextremum(data_array, "max", context, expr.data_array.name)
    return PointCoords(coords=coords_dict, value=value)


@eval_expression.register(DataArrayArgminCoords)
async def eval_data_array_argmin_coords(
    expr: DataArrayArgminCoords, context: ExecutionContext
) -> PointCoords:
    """Find the coordinates of the minimum value in a DataArray."""
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_size_within_limit(data_array, context.config.data_array_compute_size_limit)
    coords_dict, value = await _argextremum(data_array, "min", context, expr.data_array.name)
    return PointCoords(coords=coords_dict, value=value)


def _validate_correlation_alignment(
    expr: CorrelateDataArrays, a: xr.DataArray, b: xr.DataArray
) -> None:
    """Guard against correlating inputs whose correlation-axis coordinates don't match.

    xr.corr aligns its inputs with an inner join, so passing two DataArrays with different labels
    along `dim` (e.g. a daily and a monthly series) would silently reduce to only the overlapping
    labels and produce a misleading correlation rather than raising. Require the correlation-axis
    coordinates to be identical when both inputs carry them.
    """
    dim = expr.dim
    if dim in a.coords and dim in b.coords and not a[dim].equals(b[dim]):  # type: ignore[reportUnknownMemberType]
        msg = (
            f"'a' and 'b' have different '{dim}' coordinates (a has {a.sizes.get(dim, '?')} steps, "
            f"b has {b.sizes.get(dim, '?')} steps). Correlating would silently align to only the "
            f"overlapping labels and produce a misleading result. Align the inputs to a common "
            f"'{dim}' axis first (e.g. ResampleTimeToMatch for a time axis, or RegridToMatch for "
            f"lat/lon) so their '{dim}' coordinates match."
        )
        raise AlgorithmExecutionError(msg, func_name=expr.func_name)


@eval_expression.register(CorrelateDataArrays)
async def eval_correlate_data_arrays(
    expr: CorrelateDataArrays, context: ExecutionContext
) -> xr.DataArray | float:
    """Compute the Pearson correlation between two DataArrays along a shared dimension."""
    a: xr.DataArray = await eval_expression(expr.a, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        a,
        msg_prefix="Cannot correlate a DataArray with a zero-length dimension.",
    )
    validate_data_array_size_within_limit(a, context.config.data_array_compute_size_limit)
    b: xr.DataArray = await eval_expression(expr.b, context)

    _validate_correlation_alignment(expr, a, b)

    result = xr.corr(a, b, dim=expr.dim)  # type: ignore[reportUnknownMemberType]
    if result.squeeze().ndim == 0:
        materialized = await materialize_data_array(
            result.squeeze(), context, progress_label=f"corr({expr.a.name}, {expr.b.name})"
        )
        return cast("float", materialized.values.item())
    return result


@eval_expression.register(MaskWhere)
async def eval_mask_where(expr: MaskWhere, context: ExecutionContext) -> xr.DataArray:
    """Mask a DataArray, keeping values only where a boolean condition holds (NaN elsewhere)."""
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot mask a DataArray with a zero-length dimension.",
    )
    condition: xr.DataArray = await eval_expression(expr.condition, context)
    return data_array.where(condition)


class SpatialSlicingUpdate(BaseModel):
    """Progress update when slicing a DataArray spatially (lat/lon)."""

    label: str
    geojson: dict[str, Any]


def _make_spatial_slicing_progress_update(expr: SliceDataArray, context: ExecutionContext) -> None:
    lat_slice = expr.dim_slices.get("lat")
    lon_slice = expr.dim_slices.get("lon")
    match (lat_slice, lon_slice):
        case float(), float():
            point = Point(lon_slice, lat_slice)
            update = SpatialSlicingUpdate(
                label=f"Point: ({lon_slice:.2f}°, {lat_slice:.2f}°)", geojson=mapping(point)
            )
            context.tool_call_context.update_progress(update)
        case (float(lat_start), float(lat_end)), (float(lon_start), float(lon_end)):
            polygon = box(lon_start, lat_start, lon_end, lat_end)
            update = SpatialSlicingUpdate(
                # Format: Bbox: [-80.00°, -40.00°] Lon, [25.00°, 45.00°] Lat
                label=f"Bbox: [{lon_start:.2f}°, {lon_end:.2f}°] Lon, "
                f"[{lat_start:.2f}°, {lat_end:.2f}°] Lat",
                geojson=mapping(polygon),
            )
            context.tool_call_context.update_progress(update)
        case _:
            pass


@eval_expression.register(SliceDataArray)
async def eval_slice_data_array(expr: SliceDataArray, context: ExecutionContext) -> xr.DataArray:
    """Slice a DataArray by value- and index-based dimension ranges."""
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot slice a DataArray with a zero-length dimension.",
    )

    if "lat" in data_array.dims and "lon" in data_array.dims:
        _make_spatial_slicing_progress_update(expr, context)

    single_point_indexers = {k: v for k, v in expr.dim_slices.items() if not isinstance(v, tuple)}
    data_array = data_array.sel(  # type: ignore[reportUnknownMemberType]
        single_point_indexers, method=(None if expr.method == "exact" else expr.method)
    )
    slice_indexers = {k: slice(*v) for k, v in expr.dim_slices.items() if isinstance(v, tuple)}
    data_array = data_array.sel(slice_indexers)  # type: ignore[reportUnknownMemberType]

    index_based_indexers = {
        k: slice(*v) if isinstance(v, tuple) else v for k, v in expr.dim_index_slices.items()
    }
    data_array = data_array.isel(index_based_indexers)  # type: ignore[reportUnknownMemberType]

    validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Result of slicing has a zero-length dimension.",
    )

    return data_array


class FilterByPolygonUpdate(BaseModel):
    """Message indicating the polygon that is being used for analysis."""

    geojson: dict[str, Any]


@eval_expression.register(FilterDataArrayByPolygon)
async def eval_filter_data_array_by_polygon(
    expr: FilterDataArrayByPolygon, context: ExecutionContext
) -> xr.DataArray:
    """Mask a DataArray to the region covered by a polygon."""
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot filter a DataArray with a zero-length dimension.",
    )

    polygon: BaseGeometry = await eval_expression(expr.polygon, context)
    if isinstance(polygon, GeometryCollection):
        polygon = unary_union(polygon.geoms)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
    if not isinstance(polygon, Polygon | MultiPolygon):
        raise VariableTypeMismatchError(
            f"FilterDataArrayByPolygon expected a polygon or multipolygon. Got {polygon.geom_type}",
            var_name=expr.polygon.name,
            func_name=expr.func_name,
        )

    context.tool_call_context.update_progress(FilterByPolygonUpdate(geojson=mapping(polygon)))
    out = mask_data_array_with_polygon(data_array, polygon=polygon)

    validate_data_array_non_zero_dimensions(
        expr.func_name,
        out,
        msg_prefix="Result of filtering by polygon has a zero-length dimension.",
    )
    return out


class GeocodingSuccessUpdate(BaseModel):
    """Result of geocoding."""

    query: str
    geojson: dict[str, Any]


@eval_expression.register(RegridToMatch)
async def eval_regrid_to_match(expr: RegridToMatch, context: ExecutionContext) -> xr.DataArray:
    """Interpolate a DataArray onto another DataArray's lat/lon grid."""
    source: xr.DataArray = await eval_expression(expr.source, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        source,
        msg_prefix="Cannot regrid a DataArray with a zero-length dimension.",
    )
    reference: xr.DataArray = await eval_expression(expr.reference, context)
    return source.interp(  # type: ignore[reportUnknownMemberType]
        lat=reference["lat"], lon=reference["lon"], method=expr.method
    )


@eval_expression.register(GeocodeNaturalLanguageDescription)
async def eval_geocode_natural_language_description(
    expr: GeocodeNaturalLanguageDescription, context: ExecutionContext
) -> BaseGeometry:
    """Geocode a natural language place description into a geometry."""
    try:
        geometry = await geocode_natural_language_description(
            context.tool_call_context.chat_context.global_context.place_lookup,
            context.tool_call_context.chat_context.global_context.bedrock_client,
            expr.text,
        )
    except GeocodeError as e:
        raise AlgorithmExecutionError(
            f"Failed to geocode '{expr.text}': {e!r}", func_name=expr.func_name
        ) from e
    context.tool_call_context.update_progress(
        GeocodingSuccessUpdate(query=expr.text, geojson=mapping(geometry))
    )
    return geometry


@eval_expression.register(GroupByTimeAndReduceDataArray)
async def eval_group_by_time_and_reduce_data_array(
    expr: GroupByTimeAndReduceDataArray, context: ExecutionContext
) -> xr.DataArray:
    """Group a DataArray by a time unit and reduce each group to a single value."""
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot group a DataArray with a zero-length dimension.",
    )
    validate_data_array_size_within_limit(data_array, context.config.data_array_compute_size_limit)

    time_dim = expr.time_dim

    match expr.time_unit:
        case "week":
            group = data_array[time_dim].dt.isocalendar().week
        case "dayofmonth":
            group = data_array[time_dim].dt.day
        case _:
            group = f"{time_dim}.{expr.time_unit}"

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


@eval_expression.register(ComputeClimatology)
async def eval_compute_climatology(
    expr: ComputeClimatology, context: ExecutionContext
) -> xr.DataArray:
    """Compute a per-pixel climatology by grouping over a calendar period, reducing over time."""
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot compute a climatology from a DataArray with a zero-length dimension.",
    )
    validate_data_array_size_within_limit(data_array, context.config.data_array_compute_size_limit)

    time_dim = expr.time_dim
    match expr.period_type:
        case "week":
            group = data_array[time_dim].dt.isocalendar().week
        case _:
            group = f"{time_dim}.{expr.period_type}"

    grouped = data_array.groupby(group)
    # Reduce only the time dim within each group, preserving the other (e.g. spatial) dims so the
    # result is a climatology field indexed by the calendar period.
    return grouped.mean(time_dim) if expr.op == "mean" else grouped.std(time_dim)


@eval_expression.register(ComputeAnomaly)
async def eval_compute_anomaly(expr: ComputeAnomaly, context: ExecutionContext) -> xr.DataArray:
    """Compute a deseasonalized anomaly by subtracting a climatology from each timestep.

    "Climatology" here means the mean value for each calendar period (e.g. week or month) across
    all years in the dataset. The anomaly is the difference between each timestep's value and the
    climatology for that timestep's calendar period. Optionally, the anomaly can be standardized by
    dividing by the climatological standard deviation for that period.
    """
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot compute an anomaly from a DataArray with a zero-length dimension.",
    )
    validate_data_array_size_within_limit(data_array, context.config.data_array_compute_size_limit)
    climatology: xr.DataArray = await eval_expression(expr.climatology, context)

    time_dim = expr.time_dim
    # The climatology's period dim is the one not shared with the data's non-time dims.
    data_other_dims = set(data_array.dims) - {time_dim}
    period_dim = next(str(d) for d in climatology.dims if str(d) not in data_other_dims)

    if period_dim == "week":
        group = data_array[time_dim].dt.isocalendar().week
    else:
        group = f"{time_dim}.{period_dim}"

    # `group` labels each timestep in data_array with its calendar period (e.g. each timestep in
    # January gets the label "1"). groupby(group) - climatology broadcasts the subtraction over
    # each group: xarray aligns the climatology's period coordinate with each timestep's period
    # label and subtracts the matching climatology value from every timestep in that group, so a
    # January timestep has the January climatology subtracted, a February timestep has the
    # February climatology subtracted, etc. The result has the same shape/dims as data_array.
    anomaly = data_array.groupby(group) - climatology
    if expr.climatology_std is None:
        return anomaly

    # Standardized (z-score) anomaly: divide the additive anomaly by the climatological std,
    # again broadcasting per calendar period via the same groupby-and-align mechanism as above.
    climatology_std: xr.DataArray = await eval_expression(expr.climatology_std, context)
    return anomaly.groupby(group) / climatology_std


@eval_expression.register(ComputeLinearTrend)
async def eval_compute_linear_trend(
    expr: ComputeLinearTrend, context: ExecutionContext
) -> xr.DataArray | float:
    """Fit a per-pixel linear trend over time and return the slope per the chosen time unit."""
    # Nanoseconds per time unit to convert polyfit's per-nanosecond slope into a per-unit slope.
    # "day" is an exact duration. "year"/"decade" have no fixed duration on a real calendar (leap
    # years), so we use the standard Julian year convention of 365.25 days/year (the average length
    # over a 4-year leap-year cycle) rather than an arbitrary approximation.
    _ns_per_unit_time: dict[str, float] = {
        "day": 24 * 3600 * 1e9,
        "year": 365.25 * 24 * 3600 * 1e9,
        "decade": 365.25 * 24 * 3600 * 1e9 * 10,
    }
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot fit a trend to a DataArray with a zero-length dimension.",
    )
    validate_data_array_size_within_limit(data_array, context.config.data_array_compute_size_limit)

    # polyfit converts a datetime fit dim to nanoseconds, so the degree-1 coefficient is the slope
    # per nanosecond; scale it to the requested unit.
    fit = data_array.polyfit(dim=expr.time_dim, deg=1)
    coefficients: xr.DataArray = fit["polyfit_coefficients"]
    slope = coefficients.sel(degree=1, drop=True) * _ns_per_unit_time[expr.time_unit]  # type: ignore[reportUnknownMemberType]

    if slope.squeeze().ndim == 0:
        materialized = await materialize_data_array(
            slope.squeeze(), context, progress_label=f"trend({expr.data_array.name})"
        )
        return cast("float", materialized.values.item())
    return slope


def _two_sided_t_pvalue(t_statistic: Any, dof: Any) -> Any:  # noqa: ANN401
    """Two-sided p-value for a t-statistic with the given degrees of freedom."""
    return 2.0 * stats.t.sf(np.abs(t_statistic), dof)  # type: ignore[reportUnknownMemberType]


@eval_expression.register(ComputeTrendPValue)
async def eval_compute_trend_pvalue(
    expr: ComputeTrendPValue, context: ExecutionContext
) -> xr.DataArray | float:
    """Compute the per-pixel two-sided p-value of a linear trend over time (OLS t-test)."""
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot test a trend on a DataArray with a zero-length dimension.",
    )
    validate_data_array_size_within_limit(data_array, context.config.data_array_compute_size_limit)

    # polyfit(cov=True) yields the OLS covariance of the coefficients; coefficients are ordered
    # highest-degree-first, so the slope is index 0. var(slope) gives the standard error, and
    # t = slope / SE follows a t-distribution with (n - 2) degrees of freedom.
    fit = data_array.polyfit(dim=expr.time_dim, deg=1, cov=True)
    slope: xr.DataArray = fit["polyfit_coefficients"].sel(degree=1, drop=True)  # type: ignore[reportUnknownMemberType]
    var_slope: xr.DataArray = fit["polyfit_covariance"].isel(cov_i=0, cov_j=0)  # type: ignore[reportUnknownMemberType]
    t_statistic = slope / np.sqrt(var_slope)
    dof = data_array.notnull().sum(dim=expr.time_dim) - 2

    result = xr.apply_ufunc(  # type: ignore[reportUnknownMemberType]
        _two_sided_t_pvalue,
        t_statistic,
        dof,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    if result.squeeze().ndim == 0:
        materialized = await materialize_data_array(
            result.squeeze(), context, progress_label=f"trend_p({expr.data_array.name})"
        )
        return cast("float", materialized.values.item())
    return result


def _percentile_rank_along_last_axis(values: Any) -> Any:  # noqa: ANN401
    """Rank values along the last axis as 0-100 percentiles, leaving NaN positions as NaN."""
    ranks: Any = stats.rankdata(values, method="average", axis=-1, nan_policy="omit")  # type: ignore[reportUnknownMemberType]
    valid_counts: Any = np.sum(~np.isnan(values), axis=-1, keepdims=True)
    return ranks / np.where(valid_counts == 0, np.nan, valid_counts) * 100.0


@eval_expression.register(ComputePercentileRank)
async def eval_compute_percentile_rank(
    expr: ComputePercentileRank, context: ExecutionContext
) -> xr.DataArray:
    """Rank each timestep's value within its pixel's time series as a 0-100 percentile."""
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot rank a DataArray with a zero-length dimension.",
    )
    if expr.time_dim not in data_array.dims:
        msg = f"expected time dimension '{expr.time_dim}' but found dims {data_array.dims}."
        raise DataArrayShapeError(msg, func_name=expr.func_name)
    validate_data_array_size_within_limit(data_array, context.config.data_array_compute_size_limit)

    # Ranking needs each pixel's whole time series, but nothing beyond it, so rank blockwise over
    # the remaining dims on the cluster rather than pulling the whole array into this process.
    rechunked = data_array if data_array.chunks is None else data_array.chunk({expr.time_dim: -1})  # type: ignore[reportUnknownMemberType]
    ranked = xr.apply_ufunc(  # type: ignore[reportUnknownMemberType]
        _percentile_rank_along_last_axis,
        rechunked,
        input_core_dims=[[expr.time_dim]],
        output_core_dims=[[expr.time_dim]],
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    return ranked.transpose(*data_array.dims)


@eval_expression.register(FilterToTimePeriodAcrossYears)
async def eval_filter_to_time_period_across_years(
    expr: FilterToTimePeriodAcrossYears, context: ExecutionContext
) -> xr.DataArray:
    """Filter a DataArray to a recurring time period (e.g. month or season) across all years."""
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot filter a DataArray with a zero-length dimension.",
    )
    return _filter_to_time_period_across_years(
        data_array=data_array,
        period_type=expr.period_type,
        period_values=expr.period_value,
        time_dim=expr.time_dim,
    )


def _filter_to_time_period_across_years(
    data_array: xr.DataArray,
    period_type: Literal["month", "season", "week", "dayofyear"],
    period_values: int | Sequence[int] | SeasonName | Sequence[SeasonName],
    time_dim: str,
) -> xr.DataArray:
    values = [period_values] if isinstance(period_values, int | str) else period_values

    dt_accessor = data_array[time_dim].dt

    match period_type:
        case "month":
            mask = dt_accessor.month.isin(values)
        case "season":
            mask = dt_accessor.season.isin(values)
        case "week":
            mask = dt_accessor.isocalendar().week.isin(values)
        case "dayofyear":
            mask = dt_accessor.dayofyear.isin(values)

    return data_array.isel({time_dim: mask})


def _dims_being_reduced(dim: str | tuple[str, ...] | None, data_array: xr.DataArray) -> set[str]:
    """Return the set of dim names collapsed by a reduction over ``dim`` (None means all dims)."""
    if dim is None:
        return {str(d) for d in data_array.dims}
    if isinstance(dim, str):
        return {dim}
    return set(dim)


def _lat_area_weights(data_array: xr.DataArray) -> xr.DataArray:
    """Cosine-of-latitude area weights for a DataArray with a degrees-valued 'lat' dim."""
    return cast("xr.DataArray", np.cos(np.deg2rad(data_array["lat"])))


def _apply_streamable_reduce(
    data_array: xr.DataArray,
    op: Literal["sum", "mean", "std", "min", "max"],
    dim: str | tuple[str, ...] | None,
    *,
    area_weight_lat: bool,
) -> xr.DataArray:
    """Apply a reduction that streams over Dask chunks, area-weighting mean/std over latitude."""
    if area_weight_lat and op in ("mean", "std"):
        weighted = data_array.weighted(_lat_area_weights(data_array))
        return weighted.mean(dim=dim) if op == "mean" else weighted.std(dim=dim)
    match op:
        case "sum":
            return data_array.sum(dim=dim)
        case "mean":
            return data_array.mean(dim=dim)
        case "std":
            return data_array.std(dim=dim)
        case "max":
            return data_array.max(dim=dim)
        case "min":
            return data_array.min(dim=dim)


def _apply_holistic_reduce(
    data_array: xr.DataArray,
    dims_to_reduce: set[str],
    func: "Callable[..., Any]",
    **kwargs: float,
) -> xr.DataArray:
    """Apply a reduction that needs its reduction axes whole (median/quantile).

    These cannot be streamed across chunks, but they only need each reduction axis intact, not the
    whole array. Putting just those dims in one chunk and mapping the kernel blockwise keeps the
    work on the cluster, so only the reduced result is gathered instead of the entire input.
    """
    reduce_dims = [str(d) for d in data_array.dims if str(d) in dims_to_reduce]
    rechunked = (
        data_array
        if data_array.chunks is None
        else data_array.chunk(dict.fromkeys(reduce_dims, -1))  # type: ignore[reportUnknownMemberType]
    )

    def _kernel(values: Any) -> Any:  # noqa: ANN401
        # apply_ufunc moves the core dims to the end.
        axis = tuple(range(values.ndim - len(reduce_dims), values.ndim))
        return func(values, axis=axis, **kwargs)

    dtype = data_array.dtype if np.issubdtype(data_array.dtype, np.floating) else np.float64
    return xr.apply_ufunc(  # type: ignore[reportUnknownMemberType]
        _kernel,
        rechunked,
        input_core_dims=[reduce_dims],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[dtype],
    )


@eval_expression.register(ReduceDataArray)
async def eval_reduce_data_array(
    expr: ReduceDataArray, context: ExecutionContext
) -> xr.DataArray | float:
    """Reduce a DataArray along the given dimension(s) using an aggregation op."""
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot reduce a DataArray with a zero-length dimension.",
    )

    dims_to_reduce = _dims_being_reduced(expr.dim, data_array)
    # lat/lon grid cells are not equal-area: cell area scales with cos(latitude). So when a
    # mean/std collapses the latitude dim, area-weight by cos(lat) to avoid over-counting the
    # poles, which a plain average would do.
    area_weight_lat = "lat" in dims_to_reduce and "lat" in data_array.dims

    match expr.op:
        case "sum" | "mean" | "std" | "max" | "min":
            validate_data_array_size_within_limit(
                data_array,
                # Allow a larger size limit for streamable reductions since they can be applied
                # chunk-wise.
                size_limit=context.config.data_array_compute_size_limit * 5,
            )
            out = _apply_streamable_reduce(
                data_array, expr.op, expr.dim, area_weight_lat=area_weight_lat
            )
        case "median":
            validate_data_array_size_within_limit(
                data_array, context.config.data_array_compute_size_limit
            )
            out = _apply_holistic_reduce(data_array, dims_to_reduce, np.nanmedian)
        case "quantile":
            validate_data_array_size_within_limit(
                data_array, context.config.data_array_compute_size_limit
            )
            # `q` is guaranteed non-null for the 'quantile' op by ReduceDataArray's validator.
            out = _apply_holistic_reduce(
                data_array, dims_to_reduce, np.nanquantile, q=cast("float", expr.q)
            )

    # If it is a reduction to a single value, automatically convert to scalar
    if out.squeeze().ndim == 0:
        out = await materialize_data_array(
            out.squeeze(), context, progress_label=f"{expr.op}({expr.data_array.name})"
        )
        out = cast("float", out.values.item())

    return out


@eval_expression.register(RollingReduce)
async def eval_rolling_reduce(expr: RollingReduce, context: ExecutionContext) -> xr.DataArray:
    """Apply a rolling-window reduction along a dimension."""
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot apply a rolling reduction to a DataArray with a zero-length dimension.",
    )
    validate_data_array_size_within_limit(data_array, context.config.data_array_compute_size_limit)

    rolling = data_array.rolling({expr.dim: expr.window}, center=expr.center)
    match expr.op:
        case "mean":
            return rolling.mean()
        case "sum":
            return rolling.sum()
        case "std":
            return rolling.std()
        case "min":
            return rolling.min()
        case "max":
            return rolling.max()


@eval_expression.register(CumulativeReduce)
async def eval_cumulative_reduce(expr: CumulativeReduce, context: ExecutionContext) -> xr.DataArray:
    """Compute a running (cumulative) reduction along a dimension."""
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot accumulate a DataArray with a zero-length dimension.",
    )
    validate_data_array_size_within_limit(data_array, context.config.data_array_compute_size_limit)

    cumulative = data_array.cumulative(expr.dim)
    match expr.op:
        case "sum":
            return cumulative.sum()
        case "max":
            return cumulative.max()
        case "min":
            return cumulative.min()


_RESAMPLE_FREQ: dict[str, str] = {
    "hour": "1h",
    "day": "1D",
    "week": "1W",
    "month": "1MS",
    "quarter": "1QS",
    "year": "1YS",
}


@eval_expression.register(ResampleTime)
async def eval_resample_time(expr: ResampleTime, context: ExecutionContext) -> xr.DataArray:
    """Resample a DataArray onto a coarser calendar cadence, reducing each period."""
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        data_array,
        msg_prefix="Cannot resample a DataArray with a zero-length dimension.",
    )
    validate_data_array_size_within_limit(data_array, context.config.data_array_compute_size_limit)

    if expr.time_dim not in data_array.coords or not np.issubdtype(  # type: ignore[reportUnknownMemberType]
        data_array[expr.time_dim].dtype, np.datetime64
    ):
        msg = (
            f"ResampleTime requires '{expr.time_dim}' to be a datetime axis. Resampling is only "
            f"defined for calendar time, not for e.g. a forecast lead-time (timedelta) axis."
        )
        raise AlgorithmExecutionError(msg, func_name=expr.func_name)

    resampled = data_array.resample({expr.time_dim: _RESAMPLE_FREQ[expr.target_frequency]})  # type: ignore[reportUnknownMemberType]
    match expr.op:
        case "mean":
            return resampled.mean()
        case "sum":
            return resampled.sum()
        case "std":
            return resampled.std()
        case "min":
            return resampled.min()
        case "max":
            return resampled.max()


def _require_datetime_axis(func_name: str, time_dim: str, name: str, da: xr.DataArray) -> None:
    """Raise if `time_dim` on `da` is not a datetime coordinate."""
    if time_dim not in da.coords or not np.issubdtype(da[time_dim].dtype, np.datetime64):  # type: ignore[reportUnknownMemberType]
        msg = (
            f"{func_name} requires '{time_dim}' to be a datetime axis on {name}. Resampling is "
            f"only defined for calendar time, not for e.g. a forecast lead-time (timedelta) axis."
        )
        raise AlgorithmExecutionError(msg, func_name=func_name)


@eval_expression.register(ResampleTimeToMatch)
async def eval_resample_time_to_match(
    expr: ResampleTimeToMatch, context: ExecutionContext
) -> xr.DataArray:
    """Resample `source` onto the reference's inferred cadence and snap to its time labels."""
    source: xr.DataArray = await eval_expression(expr.source, context)
    reference: xr.DataArray = await eval_expression(expr.reference, context)
    validate_data_array_non_zero_dimensions(
        expr.func_name,
        source,
        msg_prefix="Cannot resample a DataArray with a zero-length dimension.",
    )
    validate_data_array_size_within_limit(source, context.config.data_array_compute_size_limit)

    _require_datetime_axis(expr.func_name, expr.time_dim, "source", source)
    _require_datetime_axis(expr.func_name, expr.time_dim, "reference", reference)

    ref_time = reference[expr.time_dim]
    freq = xr.infer_freq(ref_time)  # type: ignore[reportUnknownMemberType]
    if freq is None:
        msg = (
            f"{expr.func_name}: could not infer a regular cadence from the reference's "
            f"'{expr.time_dim}' axis. Use ResampleTime with an explicit target_frequency instead."
        )
        raise AlgorithmExecutionError(msg, func_name=expr.func_name)

    resampled_grouper = source.resample({expr.time_dim: freq})  # type: ignore[reportUnknownMemberType]
    match expr.op:
        case "mean":
            resampled = resampled_grouper.mean()
        case "sum":
            resampled = resampled_grouper.sum()
        case "std":
            resampled = resampled_grouper.std()
        case "min":
            resampled = resampled_grouper.min()
        case "max":
            resampled = resampled_grouper.max()

    # Snap the resampled series onto the reference's exact time labels. A tolerance of the
    # reference's smallest spacing ensures periods the source does not cover become NaN rather
    # than borrowing a distant aggregate.
    ref_values = np.asarray(ref_time.values)  # type: ignore[reportUnknownArgumentType]
    diffs = np.abs(np.diff(ref_values))
    tolerance = diffs.min() if diffs.size else None
    return resampled.reindex({expr.time_dim: ref_values}, method="nearest", tolerance=tolerance)  # type: ignore[reportUnknownMemberType, reportArgumentType]


@eval_expression.register(SelectPointCoordsKey)
async def eval_select_point_coords_key(
    expr: SelectPointCoordsKey, context: ExecutionContext
) -> str | float:
    """Select a single coordinate or the value field from a PointCoords."""
    point_coords: PointCoords = await eval_expression(expr.point_coords, context)
    match expr.key:
        case "value":
            return point_coords.value
        case _:
            return point_coords.coords[expr.key]  # type: ignore[reportIndexType]


@eval_expression.register(CreateGeoTIFF)
async def eval_create_geotiff(expr: CreateGeoTIFF, context: ExecutionContext) -> "GeoTIFFRef":
    """Convert a DataArray into a persisted GeoTIFF reference."""
    data_array: xr.DataArray = await eval_expression(expr.data_array, context)
    data_array = await materialize_data_array_var(expr.data_array, context)
    try:
        return await data_array_to_geotiff_ref(data_array, context.tool_call_context.chat_context)
    except DataPersistenceError as e:
        raise GeoTIFFUploadError from e
    except InvalidShapeForGeoTIFFError as e:
        raise GeoTIFFGenerationError(repr(e)) from e
    except Exception as e:
        raise GeoTIFFGenerationError from e
