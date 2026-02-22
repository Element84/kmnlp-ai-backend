# ruff: noqa: ARG001, ANN401, D107

import logging
from functools import singledispatch
from typing import Any, Literal, Self, cast

from pydantic import BaseModel

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    BaseFunctionCall,
    Function,
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
    ReduceDataArray,
    SelectPointCoordsKey,
    SliceDataArray,
    ZeroDimDataArrayToScalar,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.general_plot_functions import (
    PlotBarChart,
)
from llm_agent.agents.algorithm_agent.algorithm.utils import TimeCoords
from llm_agent.tools.response_shaping_tools.response_shaping_tool import ResponseValidationError

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
        "Geometry",
        "LinePlot",
        "RasterPlot",
        "BarChart",
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


class VariableTypeMismatchError(ResponseValidationError):
    """Referenced variable is not of the expected type."""

    def __init__(
        self, var: InferredVariable, expected_type: str, func: Function | None = None
    ) -> None:
        msg = f"Variable '{var}' is not of type '{expected_type}'"
        if func:
            msg = f"{func.func_name}: {msg}"
        super().__init__(msg)


class ArgumentTypeMismatchError(ResponseValidationError):
    """Function argument is not of the expected type."""

    def __init__(self, func: Function, arg_name: str, expected_type: str, actual_type: str) -> None:
        super().__init__(
            f"{func.func_name}: Expected argument '{arg_name}' to be of type '{expected_type}', "
            f"got '{actual_type}'"
        )


class InvalidDimNameError(ResponseValidationError):
    """Invalid dim name specified in function."""

    def __init__(self, dim_name: str, func: Function) -> None:
        super().__init__(f"{func.func_name}: Invalid dimension name '{dim_name}'.")


class BinaryOpDataArrayMisMatchError(ResponseValidationError):
    """DataArrays in binary op are incompatible."""

    def __init__(
        self, op: str, left: InferredDataArrayVariable, right: InferredDataArrayVariable
    ) -> None:
        super().__init__(
            f"Incompatible DataArrays for binary op '{op}': left {left!r}, right {right!r}."
        )


class VariableReferenceError(ResponseValidationError):
    """Referenced variable not found."""


class DataArrayShapeError(ResponseValidationError):
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
            raise VariableReferenceError(f"Output variable '{v.name}' was never assigned.")

    for out_name, out_var in expected_outputs.items():
        if out_name not in {v.name for v in algo.output_variables}:
            raise VariableReferenceError(
                f"Expected output variable '{out_name}' missing from algorithm outputs."
            )
        inferred_var = exec_context.variables[out_name]
        if inferred_var.type != out_var.type:
            raise VariableTypeMismatchError(inferred_var, out_var.type)


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
        raise VariableReferenceError from e


@type_check_expression.register(BaseFunctionCall)
def _(expr: BaseFunctionCall, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    return type_check_expression(expr.func, context)


@type_check_expression.register(BinaryOperation)
def _(expr: BinaryOperation, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    left: InferredVariable = type_check_expression(expr.left, context)
    if left.type not in ("int", "float", "DataArray"):
        raise VariableTypeMismatchError(left, "int | float | DataArray", expr)

    right: InferredVariable = type_check_expression(expr.right, context)
    if right.type not in ("int", "float", "DataArray"):
        raise VariableTypeMismatchError(right, "int | float | DataArray", expr)

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
            return _validate_data_array_binary_op_compatibility(expr, left_da, right_da)


def _validate_data_array_binary_op_compatibility(
    expr: BinaryOperation, left: InferredDataArrayVariable, right: InferredDataArrayVariable
) -> InferredDataArrayVariable:
    differing_dims = left.dims.symmetric_difference(right.dims)
    differing_dims_sizes = {
        dim: left.shape.get(dim) or right.shape.get(dim) for dim in differing_dims
    }

    # Size = 1 implies broadcasting is possible
    if not all(size == 1 for size in differing_dims_sizes.values()):
        raise BinaryOpDataArrayMisMatchError(expr.op, left, right)

    # Common dims, if known, must have the same size
    common_dims = left.dims.intersection(right.dims)
    for dim in common_dims:
        left_size = left.shape.get(dim, "unknown")
        right_size = right.shape.get(dim, "unknown")
        if left_size != "unknown" and right_size != "unknown" and left_size != right_size:  # noqa: PLR1714
            raise BinaryOpDataArrayMisMatchError(expr.op, left, right)

    # If we reach here, the DataArrays are not provably incompatible.
    # Conservatively infer the output shape.
    return InferredDataArrayVariable(
        dims=left.dims,
        shape={
            dim: "unknown"
            if "unknown" in (left.shape.get(dim, "unknown"), right.shape.get(dim, "unknown"))
            else left.shape[dim]
            for dim in left.dims.union(right.dims)
        },
    )


@type_check_expression.register(DataArrayArgmaxCoords)
def _(expr: DataArrayArgmaxCoords, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)
    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(data_array, "DataArray", expr)
    return InferredVariable(type="PointCoords")


@type_check_expression.register(DataArrayArgminCoords)
def _(expr: DataArrayArgminCoords, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)
    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(data_array, "DataArray", expr)
    return InferredVariable(type="PointCoords")


@type_check_expression.register(DataArrayToList)
def _(expr: DataArrayToList, context: ExecutionContext) -> InferredListVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)
    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(data_array, "DataArray", expr)
    if len(data_array.dims) != 1:
        msg = f"DataArrayToList: expected 1D DataArray but found dims: {data_array.dims}."
        raise DataArrayShapeError(msg)
    return InferredListVariable()


@type_check_expression.register(ZeroDimDataArrayToScalar)
def _(expr: ZeroDimDataArrayToScalar, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)
    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(data_array, "DataArray", expr)
    if len(data_array.dims) > 0:
        squeezable = all(data_array.shape[dim] == 1 for dim in data_array.dims)
        if not squeezable:
            msg = (
                f"ZeroDimDataArrayToScalar: input DataArray is not provably squeezable. "
                f"Consider reducing dims. Inferred dims: {data_array.dims}. "
                f"Inferred shape: {data_array.shape}."
            )
            raise DataArrayShapeError(msg)
    return InferredVariable(type="float")


@type_check_expression.register(SliceDataArray)
def _(expr: SliceDataArray, context: ExecutionContext) -> InferredDataArrayVariable:  # noqa: C901
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)
    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(data_array, "DataArray", expr)

    for dim in expr.dim_slices:
        if dim not in data_array.dims:
            raise InvalidDimNameError(dim, expr)

    if isinstance(data_array, InferredTemporalDataArrayVariable):
        if data_array.time_dim in expr.dim_slices:
            start, end = expr.dim_slices[data_array.time_dim]
            if not isinstance(start, str | None) or not isinstance(end, str | None):
                raise ArgumentTypeMismatchError(
                    expr,
                    f"dim_slices['{data_array.time_dim}']",
                    "str | None",
                    str(expr.dim_slices[data_array.time_dim]),
                )
            time_coords_sliced = data_array.time_coords.slice(start, end)
            return data_array.transform(
                time_coords=time_coords_sliced,
                shape={
                    **data_array.shape,
                    data_array.time_dim: len(time_coords_sliced),
                },
            )
        if data_array.time_dim in expr.dim_index_slices:
            start, end = expr.dim_index_slices[data_array.time_dim]
            time_coords_sliced = data_array.time_coords.slice_by_index(start, end)
            return data_array.transform(
                time_coords=time_coords_sliced,
                shape={
                    **data_array.shape,
                    data_array.time_dim: len(time_coords_sliced),
                },
            )

    if expr.dim_index_slices:
        new_shape = dict(data_array.shape)
        for dim, (start_idx, end_idx) in expr.dim_index_slices.items():
            if start_idx == end_idx:
                msg = (
                    f"SliceDataArray: start and end indices are the same for dim '{dim}', "
                    f"resulting in an empty slice. start_idx: {start_idx}, end_idx: {end_idx}."
                )
                raise DataArrayShapeError(msg)

            dim_size = data_array.shape[dim]
            if dim_size == "unknown":
                continue

            if not (0 <= start_idx < dim_size) or not (end_idx and 0 < end_idx <= dim_size):
                msg = (
                    f"SliceDataArray: slice indices for dim '{dim}' are out of bounds. "
                    f"start_idx: {start_idx}, end_idx: {end_idx}, dim size: {dim_size}."
                )
                raise DataArrayShapeError(msg)

            new_shape[dim] = end_idx - start_idx

        return data_array.clone().transform(shape=new_shape)

    return data_array.clone()


@type_check_expression.register(FilterDataArrayByPolygon)
def _(expr: FilterDataArrayByPolygon, context: ExecutionContext) -> InferredDataArrayVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)
    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(data_array, "DataArray", expr)

    if not (
        {"lat", "lon"}.issubset(data_array.dims)
        or {"latitude", "longitude"}.issubset(data_array.dims)
    ):
        msg = (
            f"FilterDataArrayByPolygon: expected DataArray with 'lat' and 'lon' dims "
            f"but found dims: {data_array.dims}."
        )
        raise DataArrayShapeError(msg)

    return data_array.clone()


@type_check_expression.register(GeocodeNaturalLanguageDescription)
def _(expr: GeocodeNaturalLanguageDescription, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    return InferredVariable(type="Geometry")


@type_check_expression.register(AssignUserProvidedGeometry)
def _(expr: AssignUserProvidedGeometry, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    return InferredVariable(type="Geometry")


@type_check_expression.register(GroupByTimeAndReduceDataArray)
def _(expr: GroupByTimeAndReduceDataArray, context: ExecutionContext) -> InferredDataArrayVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)

    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(data_array, "DataArray", expr)

    if isinstance(data_array, InferredTemporalDataArrayVariable):
        group_keys = data_array.time_coords.get_groupby_keys(expr.time_unit)
        return InferredDataArrayVariable(
            dims=frozenset({expr.time_unit}), shape={expr.time_unit: len(group_keys)}
        )

    return InferredDataArrayVariable(
        dims=frozenset({expr.time_unit}), shape={expr.time_unit: "unknown"}
    )


@type_check_expression.register(PlotBarChart)
def _(expr: PlotBarChart, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    heights = [type_check_expression(h, context) for h in expr.heights]
    for i, h in enumerate(heights):
        if h.type not in ("int", "float"):
            raise ArgumentTypeMismatchError(expr, f"heights[{i}]", "int | float", h.type)
    return InferredVariable(type="BarChart")


@type_check_expression.register(PlotDataArraysAsTimeSeries)
def _(expr: PlotDataArraysAsTimeSeries, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    data_arrays = [type_check_expression(da, context) for da in expr.data_arrays]
    for da in data_arrays:
        if not isinstance(da, InferredDataArrayVariable):
            raise VariableTypeMismatchError(da, "DataArray", expr)
        if len(da.dims) != 1:
            msg = f"PlotDataArraysAsTimeSeries: expected 1D DataArray but found dims: {da.dims}."
            raise DataArrayShapeError(msg)
    return InferredVariable(type="LinePlot")


@type_check_expression.register(PlotDataArraysAsRaster)
def _(expr: PlotDataArraysAsRaster, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    data_arrays = [type_check_expression(da, context) for da in expr.data_arrays]

    expected_plots = expr.subplot_grid_shape[0] * expr.subplot_grid_shape[1]
    if len(data_arrays) != expected_plots:
        raise ArgumentTypeMismatchError(
            expr,
            "data_arrays",
            f"list of length {expected_plots} (from subplot grid shape)",
            f"list of length {len(data_arrays)}",
        )

    for da in data_arrays:
        if not isinstance(da, InferredDataArrayVariable):
            raise VariableTypeMismatchError(da, "DataArray", expr)
        if len(da.dims) < 2:  # noqa: PLR2004
            msg = f"PlotDataArraysAsRaster: expected at least 2 dims but found dims: {da.dims}."
            raise DataArrayShapeError(msg)
        known_unsqueezable_dims = {dim for dim in da.dims if da.shape[dim] not in (1, "unknown")}
        if len(known_unsqueezable_dims) > 2:  # noqa: PLR2004
            msg = (
                "PlotDataArraysAsRaster: input array not squeezable to 2D. "
                f"Inferred shape: {da.shape}."
            )
            raise DataArrayShapeError(msg)
    return InferredVariable(type="RasterPlot")


@type_check_expression.register(ReduceDataArray)
def _(expr: ReduceDataArray, context: ExecutionContext) -> InferredDataArrayVariable:
    """Type check expression."""
    data_array = type_check_expression(expr.data_array, context)
    if not isinstance(data_array, InferredDataArrayVariable):
        raise VariableTypeMismatchError(data_array, "DataArray", expr)
    if expr.dim is None:
        # Reducing over all dims results in a 0D DataArray
        return InferredDataArrayVariable(dims=frozenset(), shape={})

    rdims = {expr.dim} if isinstance(expr.dim, str) else set(expr.dim)
    if rdims == set(data_array.dims):
        # All dims reduced results in a 0D DataArray
        return InferredDataArrayVariable(dims=frozenset(), shape={})

    for dim in rdims:
        if dim not in data_array.dims:
            raise InvalidDimNameError(dim, expr)

    # Keep dims that are not being reduced
    new_dims = {d for d in data_array.dims if d not in rdims}
    new_shape: dict[str, int | Literal["unknown"]] = {d: data_array.shape[d] for d in new_dims}
    return InferredDataArrayVariable(dims=frozenset(new_dims), shape=new_shape)


@type_check_expression.register(SelectPointCoordsKey)
def _(expr: SelectPointCoordsKey, context: ExecutionContext) -> InferredVariable:
    """Type check expression."""
    point_coords = type_check_expression(expr.point_coords, context)
    if point_coords.type != "PointCoords":
        raise VariableTypeMismatchError(point_coords, "PointCoords", expr)
    match expr.key:
        case _ if expr.key in context.valid_time_dims:
            return InferredVariable(type="str")
        case _:
            return InferredVariable(type="float")
