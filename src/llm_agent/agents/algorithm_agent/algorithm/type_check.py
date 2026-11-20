# ruff: noqa: ARG001, ANN401, D107

import logging
from functools import singledispatch
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, TypeAdapter

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    BaseFunctionCall,
    Function,
    VariableAssignment,
    VariableReference,
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
    ReduceDataArray,
    RegridToMatch,
    ResampleTime,
    ResampleTimeToMatch,
    RollingReduce,
    SelectPointCoordsKey,
    SliceDataArray,
    UnaryOperation,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    BarPlotSpec,
    GeneralRasterPlotSpec,
    GeospatialRasterPlotSpec,
    HistogramPlotSpec,
    LinePlotSpec,
    MakeAnimation,
    MakePlot,
    MultiLinePlotSpec,
    PlotGridSpec,
    ScatterPlotSpec,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.contour import (
    ContourMapPlotSpec,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.ensemble import (
    EnsemblePlumePlotSpec,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.sst import (
    SSTAnomalyPlotSpec,
    SSTPlotSpec,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.wind import WindVectorPlotSpec
from llm_agent.agents.algorithm_agent.algorithm.utils import TimeCoords

log = logging.getLogger(__name__)


class InferredVariable(BaseModel, frozen=True):
    """Represents a variable whose type is inferred."""

    type: Literal[
        "int",
        "float",
        "str",
        "list",
        "DataArray",
        "PointCoords",
        "geometry",
        "line_plot",
        "raster_plot",
        "bar_plot",
        "histogram",
        "scatter_plot",
        "plot_grid",
        "geotiff",
        "animation",
    ]
    name: str | None = None

    def set_name(self, name: str) -> Self:
        """Return a copy of this variable with the given name."""
        return self.model_copy(update={"name": name})

    def clone(self) -> Self:
        """Return a copy of this variable without the name."""
        return type(self).model_validate(self.model_copy(update={"name": None}).model_dump())

    def transform(self, **kwargs: Any) -> Self:
        """Return a transformed copy of this variable without the name."""
        return type(self).model_validate(
            self.model_copy(update={**kwargs, "name": None}).model_dump()
        )


class InferredListVariable(InferredVariable, frozen=True):
    """Represents a list variable whose type is inferred."""

    type: Literal["list"] = "list"
    items: list[InferredVariable] | None = None
    length: int | None = None


class InferredDataArrayVariable(InferredVariable, frozen=True):
    """Represents a data array variable whose type is inferred."""

    type: Literal["DataArray"] = "DataArray"
    dims: frozenset[str]
    shape: dict[str, int | Literal["unknown"]]


class InferredTemporalDataArrayVariable(InferredDataArrayVariable, frozen=True):
    """Represents a temporal data array variable whose type is inferred."""

    model_config = {"arbitrary_types_allowed": True}
    time_dim: str = "time"
    time_coords: TimeCoords


class InferredResolutionlessTemporalDataArrayVariable(InferredDataArrayVariable, frozen=True):
    """A temporal DataArray with a known time dimension but unknown temporal resolution.

    Produced for inputs that have a time dimension but no declared temporal resolution, so no
    ``TimeCoords`` can be built. It still counts as temporal for validation - time-based operations
    accept it - but any time-based size can only be inferred as ``"unknown"``.
    """

    time_dim: str = "time"


class InferredClimatologyVariable(InferredDataArrayVariable, frozen=True):
    """Represents a climatology DataArray indexed by a calendar period (e.g. month) instead of time.

    It is still a DataArray for downstream purposes, but carries the period metadata so that
    ComputeAnomaly can validate that a climatology matches the data it is applied to.
    """

    period_type: Literal["month", "season", "dayofyear", "week", "quarter"]
    period_dim: str


class InferredPointCoordsVariable(InferredVariable, frozen=True):
    """Represents a point coordinates variable whose type is inferred."""

    type: Literal["PointCoords"] = "PointCoords"
    valid_keys: frozenset[str]


class AlgorithmValidationError(Exception):
    """Exception raised when response validation fails."""

    def __init__(self, message: str, context: "ExecutionContext") -> None:
        inferred_vars_str = self._dump_inferred_variables(context)
        super().__init__(f"{message}\n---\nInferred variables: {inferred_vars_str}")

    def _dump_inferred_variables(self, context: "ExecutionContext") -> str:
        """Dump inferred variables in a human-readable format."""
        return TypeAdapter(dict[str, InferredVariable]).dump_json(context.variables).decode("utf-8")


class VariableTypeMismatchError(AlgorithmValidationError):
    """Referenced variable is not of the expected type."""

    def __init__(
        self,
        *,
        var: InferredVariable,
        expected_type: str,
        func: Function | None = None,
        context: "ExecutionContext",
    ) -> None:
        msg = f"Variable '{var}' is not of type '{expected_type}'"
        if func:
            msg = f"{func.func_name}: {msg}"
        super().__init__(msg, context=context)


class ArgumentTypeMismatchError(AlgorithmValidationError):
    """Function argument is not of the expected type."""

    def __init__(
        self,
        *,
        func: Function | str,
        arg_name: str,
        expected_type: str,
        actual_type: str,
        context: "ExecutionContext",
    ) -> None:
        func_name = func.func_name if isinstance(func, Function) else func
        super().__init__(
            f"{func_name}: Expected argument '{arg_name}' to be of type '{expected_type}', "
            f"got '{actual_type}'",
            context=context,
        )


class InvalidDimNameError(AlgorithmValidationError):
    """Invalid dim name specified in function."""

    def __init__(self, *, dim_name: str, func: Function | str, context: "ExecutionContext") -> None:
        func_name = func.func_name if isinstance(func, Function) else func
        super().__init__(f"{func_name}: Invalid dimension name '{dim_name}'.", context=context)


class BinaryOpDataArrayMisMatchError(AlgorithmValidationError):
    """DataArrays in binary op are incompatible."""

    def __init__(
        self,
        *,
        op: str,
        left: InferredDataArrayVariable,
        right: InferredDataArrayVariable,
        context: "ExecutionContext",
    ) -> None:
        super().__init__(
            f"Incompatible DataArrays for binary op '{op}': left {left!r}, right {right!r}.",
            context=context,
        )


class VariableReferenceError(AlgorithmValidationError):
    """Referenced variable not found."""


class DataArrayShapeError(AlgorithmValidationError):
    """DataArray is a different shape than expected."""


class ExecutionContext:
    """Execution context that tracks assigned variables."""

    variables: dict[str, InferredVariable]
    valid_time_dims: set[str]

    def __init__(
        self, input_vars: dict[str, InferredVariable], *, valid_time_dims: set[str] | None = None
    ) -> None:
        """Construct."""
        self.variables: dict[str, InferredVariable] = dict(**input_vars)
        self.valid_time_dims = valid_time_dims or set()


def _infer_data_array_var(
    vref: VariableReference, expr: Function, context: ExecutionContext
) -> InferredDataArrayVariable:
    var = type_check_expression(vref, context)
    if not isinstance(var, InferredDataArrayVariable):
        raise VariableTypeMismatchError(
            var=var, expected_type="DataArray", func=expr, context=context
        )
    return var


def type_check_algorithm[FunctionCall: BaseFunctionCall](
    algo: Algorithm[FunctionCall],
    input_vars: dict[str, InferredVariable],
    expected_outputs: dict[str, InferredVariable],
    valid_time_dims: set[str] | None = None,
) -> None:
    """Type check algorithm."""
    exec_context = ExecutionContext(input_vars=input_vars, valid_time_dims=valid_time_dims)
    for statement in algo.statements:
        type_check_expression(statement, exec_context)

    for v in algo.output_variables:
        if v.name not in exec_context.variables:
            raise VariableReferenceError(
                f"Output variable '{v.name}' was never assigned.", context=exec_context
            )

    for out_name, out_var in expected_outputs.items():
        if out_name not in {v.name for v in algo.output_variables}:
            raise VariableReferenceError(
                f"Expected output variable '{out_name}' missing from algorithm outputs.",
                context=exec_context,
            )
        inferred_var = exec_context.variables[out_name]
        if inferred_var.type != out_var.type:
            raise VariableTypeMismatchError(
                var=inferred_var, expected_type=out_var.type, func=None, context=exec_context
            )


@singledispatch
def type_check_expression(expr: Any, context: ExecutionContext) -> Any:
    """Type check expression."""
    raise NotImplementedError


@type_check_expression.register(str)
def _(expr: str, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    return InferredVariable(type="str")


@type_check_expression.register(int)
def _(expr: int, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    return InferredVariable(type="int")


@type_check_expression.register(float)
def _(expr: float, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    return InferredVariable(type="float")


@type_check_expression.register(list)
def _(expr: list[Any], context: ExecutionContext) -> InferredListVariable:
    """Type check expression."""
    return InferredListVariable(
        type="list", items=[type_check_expression(v, context) for v in expr]
    )


@type_check_expression.register(VariableAssignment)
def _(expr: VariableAssignment[BaseFunctionCall], context: ExecutionContext) -> None:
    """Type check expression."""
    val = type_check_expression(expr.value, context)
    cast("InferredVariable", val)
    context.variables[expr.name] = val.set_name(expr.name)


@type_check_expression.register(VariableReference)
def _(expr: VariableReference, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    try:
        return context.variables[expr.name]
    except KeyError as e:
        raise VariableReferenceError(f"Variable '{expr.name}' not found.", context=context) from e


@type_check_expression.register(BaseFunctionCall)
def _(expr: BaseFunctionCall, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    return type_check_expression(expr.func, context)


@type_check_expression.register(BinaryOperation)
def _(expr: BinaryOperation, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    left: InferredVariable = type_check_expression(expr.left, context)
    if left.type not in ("int", "float", "DataArray"):
        raise VariableTypeMismatchError(
            var=left, expected_type="int | float | DataArray", func=expr, context=context
        )

    right: InferredVariable = type_check_expression(expr.right, context)
    if right.type not in ("int", "float", "DataArray"):
        raise VariableTypeMismatchError(
            var=right, expected_type="int | float | DataArray", func=expr, context=context
        )

    match (left.type, right.type):
        case ("int", "int"):
            return InferredVariable(type="int")
        case ("float", "float"):
            return InferredVariable(type="float")
        case ("int", "float") | ("float", "int"):
            return InferredVariable(type="float")
        case ("DataArray", "int") | ("DataArray", "float"):
            return cast("InferredDataArrayVariable", left).clone()
        case ("int", "DataArray") | ("float", "DataArray"):
            return cast("InferredDataArrayVariable", right).clone()
        case ("DataArray", "DataArray"):
            left_da = cast("InferredDataArrayVariable", left)
            right_da = cast("InferredDataArrayVariable", right)
            return _validate_data_array_binary_op_compatibility(
                expr, left_da, right_da, context=context
            )


def _validate_data_array_binary_op_compatibility(
    expr: BinaryOperation,
    left: InferredDataArrayVariable,
    right: InferredDataArrayVariable,
    context: ExecutionContext,
) -> InferredDataArrayVariable:
    # xarray broadcasts by dimension NAME: a dim on only one side is broadcast at any size, and a
    # shared dim must have equal sizes or size 1 on one side. The result has the union of dims.
    out_shape: dict[str, int | Literal["unknown"]] = {}
    for dim in left.dims | right.dims:
        in_left, in_right = dim in left.dims, dim in right.dims
        if in_left and not in_right:
            out_shape[dim] = left.shape.get(dim, "unknown")
        elif in_right and not in_left:
            out_shape[dim] = right.shape.get(dim, "unknown")
        else:
            left_size = left.shape.get(dim, "unknown")
            right_size = right.shape.get(dim, "unknown")
            if (
                isinstance(left_size, int)
                and isinstance(right_size, int)
                and left_size != right_size
                and 1 not in (left_size, right_size)
            ):
                raise BinaryOpDataArrayMisMatchError(
                    op=expr.op, left=left, right=right, context=context
                )
            out_shape[dim] = _broadcast_shared_dim_size(left_size, right_size)

    return InferredDataArrayVariable(dims=left.dims | right.dims, shape=out_shape)


def _broadcast_shared_dim_size(
    left_size: int | Literal["unknown"], right_size: int | Literal["unknown"]
) -> int | Literal["unknown"]:
    """Broadcast size of a shared dim: the non-1 side, or unknown when it can't be pinned down."""
    if left_size == 1:
        return right_size
    if right_size == 1:
        return left_size
    if left_size == "unknown" or right_size == "unknown":
        return "unknown"
    return left_size


@type_check_expression.register(UnaryOperation)
def _(expr: UnaryOperation, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    operand: InferredVariable = type_check_expression(expr.operand, context)
    if operand.type not in ("int", "float", "DataArray"):
        raise VariableTypeMismatchError(
            var=operand, expected_type="int | float | DataArray", func=expr, context=context
        )
    if operand.type == "DataArray":
        return cast("InferredDataArrayVariable", operand).clone()
    # Unary math ops (sqrt, log, exp, ...) yield a float even for integer inputs.
    return InferredVariable(type="float")


@type_check_expression.register(DataArrayArgmaxCoords)
def _(expr: DataArrayArgmaxCoords, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    da = _infer_data_array_var(expr.data_array, expr, context)
    return InferredPointCoordsVariable(valid_keys=da.dims | {"value"})


@type_check_expression.register(DataArrayArgminCoords)
def _(expr: DataArrayArgminCoords, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    da = _infer_data_array_var(expr.data_array, expr, context)
    return InferredPointCoordsVariable(valid_keys=da.dims | {"value"})


@type_check_expression.register(CorrelateDataArrays)
def _(expr: CorrelateDataArrays, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    a = _infer_data_array_var(expr.a, expr, context)
    b = _infer_data_array_var(expr.b, expr, context)
    for da in (a, b):
        if expr.dim not in da.dims:
            raise InvalidDimNameError(dim_name=expr.dim, func=expr, context=context)

    # The correlation dim must have matching sizes on both inputs. xr.corr aligns with an inner
    # join, so mismatched sizes (e.g. a daily vs a monthly series) would silently reduce to the
    # overlap instead of erroring. Coordinate-level alignment is enforced at execution time.
    a_dim_size = a.shape.get(expr.dim, "unknown")
    b_dim_size = b.shape.get(expr.dim, "unknown")
    if a_dim_size != "unknown" and b_dim_size != "unknown" and a_dim_size != b_dim_size:  # noqa: PLR1714
        msg = (
            f"CorrelateDataArrays: correlation dim '{expr.dim}' has different sizes on a "
            f"({a_dim_size}) and b ({b_dim_size}). The inputs likely have different temporal "
            f"cadences (e.g. daily vs monthly); resample them to a common cadence first."
        )
        raise DataArrayShapeError(msg, context=context)

    a_other = a.dims - {expr.dim}
    b_other = b.dims - {expr.dim}
    # Shared non-correlation dims, if known, must have matching sizes.
    for dim in a_other & b_other:
        a_size = a.shape.get(dim, "unknown")
        b_size = b.shape.get(dim, "unknown")
        if a_size != "unknown" and b_size != "unknown" and a_size != b_size:  # noqa: PLR1714
            msg = (
                f"CorrelateDataArrays: dim '{dim}' size mismatch between a ({a_size}) and "
                f"b ({b_size})."
            )
            raise DataArrayShapeError(msg, context=context)

    # Correlating over `dim` collapses it; the remaining dims broadcast across the two inputs.
    out_dims = a_other | b_other
    if not out_dims:
        return InferredVariable(type="float")
    out_shape: dict[str, int | Literal["unknown"]] = {}
    for dim in out_dims:
        a_size = a.shape.get(dim, "unknown")
        out_shape[dim] = a_size if a_size != "unknown" else b.shape.get(dim, "unknown")
    return InferredDataArrayVariable(dims=frozenset(out_dims), shape=out_shape)


@type_check_expression.register(MaskWhere)
def _(expr: MaskWhere, context: ExecutionContext) -> InferredDataArrayVariable:
    """Type check expression."""
    data_array = _infer_data_array_var(expr.data_array, expr, context)
    condition = _infer_data_array_var(expr.condition, expr, context)

    # The condition is broadcast onto the data, so its dims must be a subset of the data's dims;
    # otherwise masking would broaden the result, which is almost never intended.
    if not condition.dims <= data_array.dims:
        raise AlgorithmValidationError(
            f"MaskWhere: condition dims {set(condition.dims)} must be a subset of the data's dims "
            f"{set(data_array.dims)}.",
            context=context,
        )
    for dim in condition.dims:
        data_size = data_array.shape.get(dim, "unknown")
        cond_size = condition.shape.get(dim, "unknown")
        if data_size != "unknown" and cond_size != "unknown" and data_size != cond_size:  # noqa: PLR1714
            raise AlgorithmValidationError(
                f"MaskWhere: dim '{dim}' size mismatch between data ({data_size}) and "
                f"condition ({cond_size}).",
                context=context,
            )

    # Masking sets values to NaN but preserves the data's dims, shape, and (if temporal) time axis.
    return data_array.clone()


def _drop_inferred_dim(da: InferredDataArrayVariable, dim: str) -> InferredDataArrayVariable:
    """Return a copy of `da` with `dim` removed, as when selecting a single coordinate/index.

    Removing the time dimension of a temporal variable (or the period dimension of a climatology)
    downgrades it to a plain DataArray, since it no longer has that special axis.
    """
    new_dims = frozenset(da.dims - {dim})
    new_shape: dict[str, int | Literal["unknown"]] = {k: v for k, v in da.shape.items() if k != dim}
    downgrades_to_plain = (
        isinstance(
            da,
            InferredTemporalDataArrayVariable | InferredResolutionlessTemporalDataArrayVariable,
        )
        and dim == da.time_dim
    ) or (isinstance(da, InferredClimatologyVariable) and dim == da.period_dim)
    if downgrades_to_plain:
        return InferredDataArrayVariable(dims=new_dims, shape=new_shape)
    return da.transform(dims=new_dims, shape=new_shape)


@type_check_expression.register(SliceDataArray)
def _(expr: SliceDataArray, context: ExecutionContext) -> InferredDataArrayVariable:  # noqa: C901, PLR0912
    """Type check expression."""
    da = type_check_expression(expr.data_array, context)
    if not isinstance(da, InferredDataArrayVariable):
        raise VariableTypeMismatchError(
            var=da, expected_type="DataArray", func=expr, context=context
        )

    for dim in expr.dim_slices.keys() | expr.dim_index_slices.keys():
        if dim not in da.dims:
            raise InvalidDimNameError(dim_name=dim, func=expr, context=context)

    for dim, dim_slice in expr.dim_slices.items():
        if not isinstance(dim_slice, tuple):
            # A single value selects one label and drops the dimension from the result.
            if (
                isinstance(da, InferredTemporalDataArrayVariable)
                and dim == da.time_dim
                and not isinstance(dim_slice, str)
            ):
                raise ArgumentTypeMismatchError(
                    func=expr,
                    arg_name=f"dim_slices['{da.time_dim}']",
                    expected_type="str",
                    actual_type=str(dim_slice),
                    context=context,
                )
            da = _drop_inferred_dim(da, dim)
            continue

        start, end = dim_slice
        if isinstance(da, InferredTemporalDataArrayVariable) and dim == da.time_dim:
            if not isinstance(start, str | None) or not isinstance(end, str | None):
                raise ArgumentTypeMismatchError(
                    func=expr,
                    arg_name=f"dim_slices['{da.time_dim}']",
                    expected_type="str | None",
                    actual_type=str(expr.dim_slices[da.time_dim]),
                    context=context,
                )
            time_coords_sliced = da.time_coords.slice(start, end)
            da = da.transform(
                time_coords=time_coords_sliced,
                shape={
                    **da.shape,
                    da.time_dim: len(time_coords_sliced),
                },
            )
        else:
            # We cannot infer the resultant shapes for any other kinds of dimensions
            pass

    for dim, dim_index_slice in expr.dim_index_slices.items():
        if not isinstance(dim_index_slice, tuple):
            # A single integer index selects one position and drops the dimension.
            if not (isinstance(da, InferredTemporalDataArrayVariable) and dim == da.time_dim):
                dim_size = da.shape[dim]
                if dim_size != "unknown" and not (0 <= dim_index_slice < dim_size):
                    msg = (
                        f"SliceDataArray: index for dim '{dim}' is out of bounds. "
                        f"index: {dim_index_slice}, dim size: {dim_size}."
                    )
                    raise DataArrayShapeError(msg, context=context)
            da = _drop_inferred_dim(da, dim)
            continue

        start_idx, end_idx = dim_index_slice
        if isinstance(da, InferredTemporalDataArrayVariable) and dim == da.time_dim:
            time_coords_sliced = da.time_coords.slice_by_index(start_idx, end_idx)
            da = da.transform(
                time_coords=time_coords_sliced,
                shape={
                    **da.shape,
                    da.time_dim: len(time_coords_sliced),
                },
            )
        else:
            if start_idx == end_idx:
                msg = (
                    f"SliceDataArray: start and end indices are the same for dim '{dim}', "
                    f"resulting in an empty slice. start_idx: {start_idx}, end_idx: {end_idx}."
                )
                raise DataArrayShapeError(msg, context=context)

            dim_size = da.shape[dim]
            if dim_size == "unknown":
                continue

            if not (0 <= start_idx < dim_size) or not (end_idx and 0 < end_idx <= dim_size):
                msg = (
                    f"SliceDataArray: slice indices for dim '{dim}' are out of bounds. "
                    f"start_idx: {start_idx}, end_idx: {end_idx}, dim size: {dim_size}."
                )
                raise DataArrayShapeError(msg, context=context)

            da = da.transform(shape={**da.shape, dim: end_idx - start_idx})

    return da.clone()


@type_check_expression.register(FilterDataArrayByPolygon)
def _(expr: FilterDataArrayByPolygon, context: ExecutionContext) -> InferredDataArrayVariable:
    """Type check expression."""
    data_array = _infer_data_array_var(expr.data_array, expr, context)

    if not (
        {"lat", "lon"}.issubset(data_array.dims)
        or {"latitude", "longitude"}.issubset(data_array.dims)
    ):
        msg = (
            f"FilterDataArrayByPolygon: expected DataArray with 'lat' and 'lon' dims "
            f"but found dims: {data_array.dims}."
        )
        raise DataArrayShapeError(msg, context=context)

    return data_array.clone()


@type_check_expression.register(GeocodeNaturalLanguageDescription)
def _(expr: GeocodeNaturalLanguageDescription, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    return InferredVariable(type="geometry")


@type_check_expression.register(RegridToMatch)
def _(expr: RegridToMatch, context: ExecutionContext) -> InferredDataArrayVariable:
    """Type check expression."""
    source = _infer_data_array_var(expr.source, expr, context)
    reference = _infer_data_array_var(expr.reference, expr, context)
    for da, name in ((source, "source"), (reference, "reference")):
        if not {"lat", "lon"}.issubset(da.dims):
            msg = f"RegridToMatch: {name} must have 'lat' and 'lon' dims but found: {set(da.dims)}."
            raise DataArrayShapeError(msg, context=context)

    # source keeps its dims and non-spatial shape; only lat/lon are resampled to the reference's.
    new_shape: dict[str, int | Literal["unknown"]] = {
        **source.shape,
        "lat": reference.shape.get("lat", "unknown"),
        "lon": reference.shape.get("lon", "unknown"),
    }
    return source.transform(shape=new_shape)


@type_check_expression.register(GroupByTimeAndReduceDataArray)
def _(expr: GroupByTimeAndReduceDataArray, context: ExecutionContext) -> InferredDataArrayVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)

    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(
            var=data_array, expected_type="DataArray", func=expr, context=context
        )

    if expr.time_dim not in data_array.dims:
        raise InvalidDimNameError(dim_name=expr.time_dim, func=expr, context=context)

    if isinstance(data_array, InferredTemporalDataArrayVariable):
        group_keys = data_array.time_coords.get_groupby_keys(expr.time_unit)
        return InferredDataArrayVariable(
            dims=frozenset({expr.time_unit}), shape={expr.time_unit: len(group_keys)}
        )

    return InferredDataArrayVariable(
        dims=frozenset({expr.time_unit}), shape={expr.time_unit: "unknown"}
    )


@type_check_expression.register(ResampleTime)
def _(expr: ResampleTime, context: ExecutionContext) -> InferredDataArrayVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)

    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(
            var=data_array, expected_type="DataArray", func=expr, context=context
        )

    if expr.time_dim not in data_array.dims:
        raise InvalidDimNameError(dim_name=expr.time_dim, func=expr, context=context)

    # Resampling preserves all dims but changes the number of steps along the time dim.
    new_shape: dict[str, int | Literal["unknown"]] = {
        d: data_array.shape.get(d, "unknown") for d in data_array.dims
    }
    new_shape[expr.time_dim] = "unknown"
    return InferredDataArrayVariable(dims=data_array.dims, shape=new_shape)


@type_check_expression.register(ResampleTimeToMatch)
def _(expr: ResampleTimeToMatch, context: ExecutionContext) -> InferredDataArrayVariable:
    """Type check expression."""
    source = type_check_expression(expr.source, context)
    if not isinstance(source, InferredDataArrayVariable):
        raise VariableTypeMismatchError(
            var=source, expected_type="DataArray", func=expr, context=context
        )
    reference = type_check_expression(expr.reference, context)
    if not isinstance(reference, InferredDataArrayVariable):
        raise VariableTypeMismatchError(
            var=reference, expected_type="DataArray", func=expr, context=context
        )
    if expr.time_dim not in source.dims or expr.time_dim not in reference.dims:
        raise InvalidDimNameError(dim_name=expr.time_dim, func=expr, context=context)

    # Output keeps source's dims; the time dim is snapped onto the reference's time axis.
    new_shape: dict[str, int | Literal["unknown"]] = {
        d: source.shape.get(d, "unknown") for d in source.dims
    }
    new_shape[expr.time_dim] = reference.shape.get(expr.time_dim, "unknown")
    return InferredDataArrayVariable(dims=source.dims, shape=new_shape)


@type_check_expression.register(ComputeClimatology)
def _(expr: ComputeClimatology, context: ExecutionContext) -> InferredClimatologyVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)

    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(
            var=data_array, expected_type="DataArray", func=expr, context=context
        )

    if expr.time_dim not in data_array.dims:
        raise InvalidDimNameError(dim_name=expr.time_dim, func=expr, context=context)

    period_dim = expr.period_type
    other_dims = data_array.dims - {expr.time_dim}

    if isinstance(data_array, InferredTemporalDataArrayVariable):
        period_size: int | Literal["unknown"] = len(
            data_array.time_coords.get_groupby_keys(expr.period_type)
        )
    else:
        period_size = "unknown"

    new_shape: dict[str, int | Literal["unknown"]] = {d: data_array.shape[d] for d in other_dims}
    new_shape[period_dim] = period_size
    return InferredClimatologyVariable(
        dims=frozenset(other_dims | {period_dim}),
        shape=new_shape,
        period_type=expr.period_type,
        period_dim=period_dim,
    )


@type_check_expression.register(ComputeAnomaly)
def _(expr: ComputeAnomaly, context: ExecutionContext) -> InferredDataArrayVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)
    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(
            var=data_array, expected_type="DataArray", func=expr, context=context
        )
    if expr.time_dim not in data_array.dims:
        raise InvalidDimNameError(dim_name=expr.time_dim, func=expr, context=context)

    data_other_dims = data_array.dims - {expr.time_dim}
    climatology = _validate_anomaly_climatology(
        type_check_expression(expr.climatology, context),
        data_array,
        data_other_dims,
        context=context,
        arg_name="climatology",
    )

    if expr.climatology_std is not None:
        std_climatology = _validate_anomaly_climatology(
            type_check_expression(expr.climatology_std, context),
            data_array,
            data_other_dims,
            context=context,
            arg_name="climatology_std",
        )
        if std_climatology.period_type != climatology.period_type:
            raise AlgorithmValidationError(
                f"ComputeAnomaly: climatology_std period '{std_climatology.period_type}' does not "
                f"match climatology period '{climatology.period_type}'.",
                context=context,
            )

    # The anomaly has the same dimensions, shape, and (if temporal) time axis as the input.
    return data_array.clone()


def _validate_anomaly_climatology(
    climatology: InferredVariable,
    data_array: InferredDataArrayVariable,
    data_other_dims: frozenset[str],
    *,
    context: ExecutionContext,
    arg_name: str,
) -> InferredClimatologyVariable:
    """Validate a climatology arg to ComputeAnomaly matches the data's non-time dims."""
    if not isinstance(climatology, InferredClimatologyVariable):
        raise VariableTypeMismatchError(
            var=climatology, expected_type="Climatology", func=None, context=context
        )

    # The climatology's non-period dims must match the data's non-time dims so it can be
    # broadcast back onto each timestep.
    clim_other_dims = climatology.dims - {climatology.period_dim}
    if data_other_dims != clim_other_dims:
        raise AlgorithmValidationError(
            f"ComputeAnomaly: {arg_name} dims {set(clim_other_dims)} do not match the data's "
            f"non-time dims {set(data_other_dims)}.",
            context=context,
        )
    for dim in data_other_dims:
        data_size = data_array.shape.get(dim, "unknown")
        clim_size = climatology.shape.get(dim, "unknown")
        if data_size != "unknown" and clim_size != "unknown" and data_size != clim_size:  # noqa: PLR1714
            raise AlgorithmValidationError(
                f"ComputeAnomaly: {arg_name} dim '{dim}' size mismatch between data ({data_size}) "
                f"and climatology ({clim_size}).",
                context=context,
            )
    return climatology


@type_check_expression.register(ComputeLinearTrend)
def _(expr: ComputeLinearTrend, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)
    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(
            var=data_array, expected_type="DataArray", func=expr, context=context
        )
    if expr.time_dim not in data_array.dims:
        raise InvalidDimNameError(dim_name=expr.time_dim, func=expr, context=context)

    # The fit collapses the time dim, keeping the other (e.g. spatial) dims as a slope field.
    new_dims = data_array.dims - {expr.time_dim}
    if not new_dims:
        return InferredVariable(type="float")
    new_shape: dict[str, int | Literal["unknown"]] = {d: data_array.shape[d] for d in new_dims}
    return InferredDataArrayVariable(dims=frozenset(new_dims), shape=new_shape)


@type_check_expression.register(ComputeTrendPValue)
def _(expr: ComputeTrendPValue, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)
    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(
            var=data_array, expected_type="DataArray", func=expr, context=context
        )
    if expr.time_dim not in data_array.dims:
        raise InvalidDimNameError(dim_name=expr.time_dim, func=expr, context=context)

    # Like the trend, the significance test collapses the time dim, keeping the other dims.
    new_dims = data_array.dims - {expr.time_dim}
    if not new_dims:
        return InferredVariable(type="float")
    new_shape: dict[str, int | Literal["unknown"]] = {d: data_array.shape[d] for d in new_dims}
    return InferredDataArrayVariable(dims=frozenset(new_dims), shape=new_shape)


@type_check_expression.register(ComputePercentileRank)
def _(expr: ComputePercentileRank, context: ExecutionContext) -> InferredDataArrayVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)
    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(
            var=data_array, expected_type="DataArray", func=expr, context=context
        )
    if expr.time_dim not in data_array.dims:
        raise InvalidDimNameError(dim_name=expr.time_dim, func=expr, context=context)

    # Percentile rank replaces each value with its rank in-place: same dims and shape.
    return InferredDataArrayVariable(
        dims=data_array.dims,
        shape={d: data_array.shape.get(d, "unknown") for d in data_array.dims},
    )


@type_check_expression.register(FilterToTimePeriodAcrossYears)
def _(expr: FilterToTimePeriodAcrossYears, context: ExecutionContext) -> InferredDataArrayVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)

    if not isinstance(
        data_array,
        InferredTemporalDataArrayVariable | InferredResolutionlessTemporalDataArrayVariable,
    ):
        raise VariableTypeMismatchError(
            var=data_array, expected_type="TemporalDataArray", func=expr, context=context
        )

    if expr.time_dim not in data_array.dims:
        raise InvalidDimNameError(dim_name=expr.time_dim, func=expr, context=context)

    # Without a resolution the filtered coords can't be computed, so keep the time size unknown.
    if isinstance(data_array, InferredResolutionlessTemporalDataArrayVariable):
        return InferredResolutionlessTemporalDataArrayVariable(
            dims=data_array.dims,
            shape={**data_array.shape, expr.time_dim: "unknown"},
            time_dim=expr.time_dim,
        )

    new_time_coords = data_array.time_coords.filter_to_time_period_across_years(
        period_type=expr.period_type, period_values=expr.period_value
    )
    return InferredTemporalDataArrayVariable(
        dims=data_array.dims,
        shape={**data_array.shape, expr.time_dim: len(new_time_coords)},
        time_dim=expr.time_dim,
        time_coords=new_time_coords,
    )


@type_check_expression.register(ReduceDataArray)
def _(expr: ReduceDataArray, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    data_array = _infer_data_array_var(expr.data_array, expr, context)
    if expr.dim is None:
        # Reducing over all dims results in a scalar
        return InferredVariable(type="float")

    dims_to_reduce = {expr.dim} if isinstance(expr.dim, str) else set(expr.dim)
    if dims_to_reduce == set(data_array.dims):
        # All dims reduced results in a scalar
        return InferredVariable(type="float")

    for dim in dims_to_reduce:
        if dim not in data_array.dims:
            raise InvalidDimNameError(dim_name=dim, func=expr, context=context)

    # Keep dims that are not being reduced
    new_dims = {d for d in data_array.dims if d not in dims_to_reduce}
    new_shape: dict[str, int | Literal["unknown"]] = {d: data_array.shape[d] for d in new_dims}

    if all(size == 1 for size in new_shape.values()):
        return InferredVariable(type="float")

    return InferredDataArrayVariable(dims=frozenset(new_dims), shape=new_shape)


@type_check_expression.register(RollingReduce)
def _(expr: RollingReduce, context: ExecutionContext) -> InferredDataArrayVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)
    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(
            var=data_array, expected_type="DataArray", func=expr, context=context
        )
    if expr.dim not in data_array.dims:
        raise InvalidDimNameError(dim_name=expr.dim, func=expr, context=context)

    dim_size = data_array.shape.get(expr.dim, "unknown")
    if dim_size != "unknown" and expr.window > dim_size:
        msg = (
            f"RollingReduce: window ({expr.window}) exceeds the size of dim '{expr.dim}' "
            f"({dim_size}), which would produce an all-NaN result."
        )
        raise DataArrayShapeError(msg, context=context)

    # A rolling reduction preserves the dims, shape, and (if temporal) time axis of the input.
    return data_array.clone()


@type_check_expression.register(CumulativeReduce)
def _(expr: CumulativeReduce, context: ExecutionContext) -> InferredDataArrayVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)
    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(
            var=data_array, expected_type="DataArray", func=expr, context=context
        )
    if expr.dim not in data_array.dims:
        raise InvalidDimNameError(dim_name=expr.dim, func=expr, context=context)

    # A cumulative reduction preserves the dims, shape, and (if temporal) time axis of the input.
    return data_array.clone()


@type_check_expression.register(SelectPointCoordsKey)
def _(expr: SelectPointCoordsKey, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    point_coords = type_check_expression(expr.point_coords, context)

    if not isinstance(point_coords, InferredPointCoordsVariable):
        raise VariableTypeMismatchError(
            var=point_coords, expected_type="PointCoords", func=expr, context=context
        )

    if expr.key not in point_coords.valid_keys:
        raise AlgorithmValidationError(
            f"Invalid key '{expr.key}' for PointCoords variable. Valid keys are: "
            f"{point_coords.valid_keys}",
            context=context,
        )
    if expr.key in context.valid_time_dims:
        return InferredVariable(type="str")

    return InferredVariable(type="float")


@type_check_expression.register(MakePlot)
def _(expr: MakePlot[Any], context: ExecutionContext) -> InferredVariable:  # noqa: C901, PLR0911, PLR0912
    """Type check expression."""
    match plot_spec := expr.plot_spec:
        case BarPlotSpec():
            return InferredVariable(type="bar_plot")
        case HistogramPlotSpec():
            _ = _infer_data_array_var(plot_spec.data_array, expr, context)
            return InferredVariable(type="histogram")
        case ScatterPlotSpec():
            _ = _infer_data_array_var(plot_spec.x_data_array, expr, context)
            _ = _infer_data_array_var(plot_spec.y_data_array, expr, context)
            return InferredVariable(type="scatter_plot")
        case LinePlotSpec():
            _ = _infer_data_array_var(plot_spec.data_array, expr, context)
            if plot_spec.data_array_error is not None:
                _ = _infer_data_array_var(plot_spec.data_array_error, expr, context)
            return InferredVariable(type="line_plot")
        case MultiLinePlotSpec():
            for line_spec in plot_spec.specs:
                _ = _infer_data_array_var(line_spec.data_array, expr, context)
                if line_spec.data_array_error is not None:
                    _ = _infer_data_array_var(line_spec.data_array_error, expr, context)
            return InferredVariable(type="line_plot")
        case GeneralRasterPlotSpec() | GeospatialRasterPlotSpec():
            _ = _infer_data_array_var(plot_spec.data_array, expr, context)
            return InferredVariable(type="raster_plot")
        case PlotGridSpec():
            return InferredVariable(type="plot_grid")
        case SSTPlotSpec():
            _ = _infer_data_array_var(plot_spec.sst_data_array, expr, context)
            if plot_spec.sea_ice_data_array is not None:
                _ = _infer_data_array_var(plot_spec.sea_ice_data_array, expr, context)
            return InferredVariable(type="raster_plot")
        case SSTAnomalyPlotSpec():
            _ = _infer_data_array_var(plot_spec.ssta_data_array, expr, context)
            return InferredVariable(type="raster_plot")
        case WindVectorPlotSpec():
            _ = _infer_data_array_var(plot_spec.u_data_array, expr, context)
            _ = _infer_data_array_var(plot_spec.v_data_array, expr, context)
            return InferredVariable(type="raster_plot")
        case EnsemblePlumePlotSpec():
            da = _infer_data_array_var(plot_spec.data_array, expr, context)
            if plot_spec.ensemble_dim not in da.dims:
                msg = (
                    f"{plot_spec.spec_name}: expected DataArray to have ensemble dimension "
                    f"'{plot_spec.ensemble_dim}' but found dims {sorted(da.dims)}."
                )
                raise DataArrayShapeError(msg, context=context)
            return InferredVariable(type="line_plot")
        case ContourMapPlotSpec():
            _ = _infer_data_array_var(plot_spec.data_array, expr, context)
            return InferredVariable(type="raster_plot")
        case _:
            raise TypeError(f"Unsupported plot spec type: {type(plot_spec).__name__}")


@type_check_expression.register(MakeAnimation)
def _(expr: MakeAnimation[Any], context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    return InferredVariable(type="animation")


@type_check_expression.register(CreateGeoTIFF)
def _(expr: CreateGeoTIFF, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    data_array = _infer_data_array_var(expr.data_array, expr, context)
    if not (
        {"lat", "lon"}.issubset(data_array.dims)
        or {"latitude", "longitude"}.issubset(data_array.dims)
    ):
        msg = (
            f"CreateGeoTIFF: expected DataArray with 'lat' and 'lon' dims but found dims: "
            f"{data_array.dims}."
        )
        raise DataArrayShapeError(msg, context=context)
    return InferredVariable(type="geotiff")
