# pyright: reportUnknownLambdaType=false
# ruff: noqa: PLR0913
from collections.abc import Callable
from itertools import product
from typing import Literal, cast
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    box,
)
from shapely.geometry.base import BaseGeometry
from xarray.core.groupby import DataArrayGroupBy

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.exec.base import (
    AlgorithmExecutionError,
    ExecutionConfig,
    ExecutionContext,
    VariableTypeMismatchError,
    eval_expression,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.data_array import GeocodingSuccessUpdate
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
    BinaryOperation,
    ComputeAnomaly,
    ComputeClimatology,
    ComputeLinearTrend,
    ComputePercentileRank,
    ComputeTrendPValue,
    CorrelateDataArrays,
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
from llm_agent.context import ToolCallContext, ToolProgressUpdate
from llm_agent.utils.dask import (
    DataSizeTooLargeError,
)
from tests.unit.agents.algorithm_agent.utils import (
    DimCoords,
    make_test_data_array,
    make_test_data_array_spatiotemporal,
)
from tests.utils.mocks.context import MockChatContext

type DimsT = str | tuple[str, ...] | None
type GroupByFuncT = Callable[[xr.DataArray], DataArrayGroupBy]
type ReductionOpT = Callable[[xr.DataArray | DataArrayGroupBy, DimsT], xr.DataArray]


def _reduce_to_scalar_if_possible(da: xr.DataArray) -> xr.DataArray:
    if da.squeeze().ndim == 0:
        return da.squeeze().values.item()
    return da


REDUCTION_OPS: dict[str, ReductionOpT] = {
    "sum": lambda da, dim: _reduce_to_scalar_if_possible(da.sum(dim=dim)),
    "mean": lambda da, dim: _reduce_to_scalar_if_possible(da.mean(dim=dim)),
    "std": lambda da, dim: _reduce_to_scalar_if_possible(da.std(dim=dim)),
    "min": lambda da, dim: _reduce_to_scalar_if_possible(da.min(dim=dim)),
    "max": lambda da, dim: _reduce_to_scalar_if_possible(da.max(dim=dim)),
}
GROUPBY_REDUCTION_OPS: dict[str, ReductionOpT] = {
    "sum": lambda da, dim: da.sum(dim=dim),
    "mean": lambda da, dim: da.mean(dim=dim),
    "std": lambda da, dim: da.std(dim=dim),
    "min": lambda da, dim: da.min(dim=dim),
    "max": lambda da, dim: da.max(dim=dim),
}
TIME_GROUPBY_OPS: dict[str, GroupByFuncT] = {
    "date": lambda da: da.groupby("time.date"),
    "year": lambda da: da.groupby("time.year"),
    "quarter": lambda da: da.groupby("time.quarter"),
    "month": lambda da: da.groupby("time.month"),
    "week": lambda da: da.groupby(da.time.dt.isocalendar().week),
    "dayofmonth": lambda da: da.groupby("time.day"),
    "dayofyear": lambda da: da.groupby("time.dayofyear"),
    "dayofweek": lambda da: da.groupby("time.dayofweek"),
}
TIME_GROUPBY_OUT_LEN: dict[str, int] = {
    "date": 366,
    "year": 1,
    "quarter": 4,
    "month": 12,
    "week": 52,
    "dayofmonth": 31,
    "dayofyear": 366,
    "dayofweek": 7,
}

default_exec_config = ExecutionConfig()


@pytest.fixture
def context(chat_context: MockChatContext):
    return ToolCallContext(tool_call_id="", chat_context=chat_context)


@pytest.mark.parametrize(
    ("op", "expected_func", "expected_func_reverse_order"),
    [
        ("add", lambda da, val: da + val, lambda da, val: val + da),
        ("sub", lambda da, val: da - val, lambda da, val: val - da),
        ("mul", lambda da, val: da * val, lambda da, val: val * da),
        ("div", lambda da, val: da / val, lambda da, val: val / da),
        ("eq", lambda da, val: da == val, lambda da, val: val == da),
        ("ne", lambda da, val: da != val, lambda da, val: val != da),
        ("gt", lambda da, val: da > val, lambda da, val: val > da),
        ("lt", lambda da, val: da < val, lambda da, val: val < da),
        ("ge", lambda da, val: da >= val, lambda da, val: val >= da),
        ("le", lambda da, val: da <= val, lambda da, val: val <= da),
    ],  # type: ignore[reportUnknownArgumentType]
)
async def test_binary_operation_with_scalar(
    context: ToolCallContext,
    op: Literal["add", "sub", "mul", "div", "eq", "ne", "gt", "lt", "ge", "le"],
    expected_func: Callable[[xr.DataArray, float], xr.DataArray],
    expected_func_reverse_order: Callable[[xr.DataArray, float], xr.DataArray],
):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    expr1 = BinaryOperation(
        func_name="BinaryOperation",
        left=VariableReference(name="da"),
        right=2,
        op=op,
    )
    out1 = await eval_expression(expr1, exec_context)
    xr.testing.assert_allclose(out1, expected_func(da, 2))  # type: ignore[reportUnknownMemberType]

    expr2 = BinaryOperation(
        func_name="BinaryOperation",
        left=0.5,
        right=VariableReference(name="da"),
        op=op,
    )
    out2 = await eval_expression(expr2, exec_context)
    xr.testing.assert_allclose(out2, expected_func_reverse_order(da, 0.5))  # type: ignore[reportUnknownMemberType]


@pytest.mark.parametrize(
    ("op", "expected_func"),
    [
        ("add", lambda da1, da2: da1 + da2),
        ("sub", lambda da1, da2: da1 - da2),
        ("mul", lambda da1, da2: da1 * da2),
        ("div", lambda da1, da2: da1 / da2),
    ],  # type: ignore[reportUnknownArgumentType]
)
async def test_binary_operation_with_data_arrays(
    context: ToolCallContext,
    op: Literal["add", "sub", "mul", "div"],
    expected_func: Callable[[xr.DataArray, xr.DataArray], xr.DataArray],
):
    da1 = make_test_data_array_spatiotemporal() * 2
    da2 = make_test_data_array_spatiotemporal() * 3
    exec_context = ExecutionContext(
        context, input_vars={"da1": da1, "da2": da2}, config=default_exec_config
    )

    expr = BinaryOperation(
        func_name="BinaryOperation",
        left=VariableReference(name="da1"),
        right=VariableReference(name="da2"),
        op=op,
    )
    out = await eval_expression(expr, exec_context)
    xr.testing.assert_allclose(out, expected_func(da1, da2))  # type: ignore[reportUnknownMemberType]


@pytest.mark.parametrize("op", ["sqrt", "square", "abs", "exp", "log", "log10", "neg"])
async def test_unary_operation_with_data_array(
    context: ToolCallContext,
    op: Literal["sqrt", "square", "abs", "exp", "log", "log10", "neg"],
):
    # Use a strictly-positive range so log/sqrt are well-defined everywhere.
    da = make_test_data_array(
        name="var",
        dims=[
            ("lat", DimCoords(start=-90, end=90, num=8, freq=None)),
            ("lon", DimCoords(start=-180, end=180, num=12, freq=None)),
        ],
        val_range=(1.0, 50.0),
    )
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    expr = UnaryOperation(
        func_name="UnaryOperation",
        operand=VariableReference(name="da"),
        op=op,
    )
    out = await eval_expression(expr, exec_context)
    assert isinstance(out, xr.DataArray)

    # numpy ufuncs dispatch to xarray and return a DataArray at runtime, even though the
    # numpy stubs annotate the result as an ndarray.
    match op:
        case "sqrt":
            expected = np.sqrt(da)
        case "square":
            expected = np.square(da)
        case "abs":
            expected = np.abs(da)
        case "exp":
            expected = np.exp(da)
        case "log":
            expected = np.log(da)
        case "log10":
            expected = np.log10(da)
        case "neg":
            expected = -da
    xr.testing.assert_allclose(out, expected)  # type: ignore[reportUnknownMemberType]


@pytest.mark.parametrize(
    ("op", "value", "expected"),
    [
        ("sqrt", 9.0, 3.0),
        ("square", 4.0, 16.0),
        ("abs", -7.0, 7.0),
        ("neg", 3.0, -3.0),
    ],
)
async def test_unary_operation_with_scalar(
    context: ToolCallContext,
    op: Literal["sqrt", "square", "abs", "neg"],
    value: float,
    expected: float,
):
    exec_context = ExecutionContext(context, input_vars={"x": value}, config=default_exec_config)

    expr = UnaryOperation(
        func_name="UnaryOperation",
        operand=VariableReference(name="x"),
        op=op,
    )
    out = await eval_expression(expr, exec_context)
    assert isinstance(out, float)
    assert out == expected


async def test_unary_operation_wind_speed_from_components(context: ToolCallContext):
    """sqrt(u**2 + v**2) should recover wind speed magnitude."""
    u = make_test_data_array(
        name="u",
        dims=[("lat", DimCoords(start=-90, end=90, num=6, freq=None))],
        val_range=(1.0, 10.0),
        random_seed=1,
    )
    v = make_test_data_array(
        name="v",
        dims=[("lat", DimCoords(start=-90, end=90, num=6, freq=None))],
        val_range=(1.0, 10.0),
        random_seed=2,
    )
    exec_context = ExecutionContext(
        context, input_vars={"u": u, "v": v}, config=default_exec_config
    )

    u_sq = await eval_expression(
        UnaryOperation(
            func_name="UnaryOperation", operand=VariableReference(name="u"), op="square"
        ),
        exec_context,
    )
    exec_context.variables["u_sq"] = u_sq
    v_sq = await eval_expression(
        UnaryOperation(
            func_name="UnaryOperation", operand=VariableReference(name="v"), op="square"
        ),
        exec_context,
    )
    exec_context.variables["v_sq"] = v_sq
    sum_sq = await eval_expression(
        BinaryOperation(
            func_name="BinaryOperation",
            left=VariableReference(name="u_sq"),
            right=VariableReference(name="v_sq"),
            op="add",
        ),
        exec_context,
    )
    exec_context.variables["sum_sq"] = sum_sq
    speed = await eval_expression(
        UnaryOperation(
            func_name="UnaryOperation", operand=VariableReference(name="sum_sq"), op="sqrt"
        ),
        exec_context,
    )
    xr.testing.assert_allclose(speed, np.sqrt(u**2 + v**2))  # type: ignore[reportUnknownMemberType]


async def test_data_array_argmax_coords(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    da[{"time": 1, "lat": 5, "lon": 10}] = 1000
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    expr = DataArrayArgmaxCoords(
        func_name="DataArrayArgmaxCoords",
        data_array=VariableReference(name="da"),
    )
    out: PointCoords = await eval_expression(expr, exec_context)
    out_expected = PointCoords(
        coords={
            "time": str(np.datetime_as_string(da.coords["time"].values[1])),  # type: ignore[reportUnknownMemberType]
            "lat": round(da.coords["lat"].values[5].item(), 6),  # type: ignore[reportUnknownMemberType]
            "lon": round(da.coords["lon"].values[10].item(), 6),  # type: ignore[reportUnknownMemberType]
        },
        value=1000,
    )
    assert out.model_dump() == out_expected.model_dump()


async def test_data_array_argmax_coords_size_lim(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(
        context,
        input_vars={"da": da},
        config=ExecutionConfig(data_array_compute_size_limit=10**3),  # 1 KB
    )

    expr = DataArrayArgmaxCoords(
        func_name="DataArrayArgmaxCoords",
        data_array=VariableReference(name="da"),
    )
    with pytest.raises(DataSizeTooLargeError):
        _ = await eval_expression(expr, exec_context)


async def test_data_array_argmin_coords(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    da[{"time": 1, "lat": 5, "lon": 10}] = -1000
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    expr = DataArrayArgminCoords(
        func_name="DataArrayArgminCoords",
        data_array=VariableReference(name="da"),
    )
    out: PointCoords = await eval_expression(expr, exec_context)
    out_expected = PointCoords(
        coords={
            "time": str(np.datetime_as_string(da.coords["time"].values[1])),  # type: ignore[reportUnknownMemberType]
            "lat": round(da.coords["lat"].values[5].item(), 6),  # type: ignore[reportUnknownMemberType]
            "lon": round(da.coords["lon"].values[10].item(), 6),  # type: ignore[reportUnknownMemberType]
        },
        value=-1000,
    )
    assert out.model_dump() == out_expected.model_dump()


async def test_data_array_argmin_coords_size_lim(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(
        context,
        input_vars={"da": da},
        config=ExecutionConfig(data_array_compute_size_limit=10**3),  # 1 KB
    )

    expr = DataArrayArgminCoords(
        func_name="DataArrayArgminCoords",
        data_array=VariableReference(name="da"),
    )
    with pytest.raises(DataSizeTooLargeError):
        _ = await eval_expression(expr, exec_context)


async def test_slice_data_array_spatiotemporal_time_slice(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    SliceDataArray_LatLonTime = SliceDataArray.with_constrained_dims(  # noqa: N806
        ["time", "lat", "lon"]
    )

    expr = SliceDataArray_LatLonTime(
        func_name="SliceDataArray_LatLonTime",
        data_array=VariableReference(name="da"),
        dim_slices={"time": ("2024-01-03", "2024-01-05")},
    )
    out: xr.DataArray = await eval_expression(expr, exec_context)
    out_dates = [
        str(np.datetime_as_string(t, unit="D"))  # type: ignore[reportUnknownArgumentType]
        for t in out.time.data
    ]
    assert len(out.time) == 3
    assert out_dates == ["2024-01-03", "2024-01-04", "2024-01-05"]


async def test_slice_data_array_spatiotemporal_time_slice_same_start_end(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    SliceDataArray_LatLonTime = SliceDataArray.with_constrained_dims(  # noqa: N806
        ["time", "lat", "lon"]
    )
    expr = SliceDataArray_LatLonTime(
        func_name="SliceDataArray_LatLonTime",
        data_array=VariableReference(name="da"),
        dim_slices={"time": ("2024-01-03", "2024-01-03")},
    )
    out: xr.DataArray = await eval_expression(expr, exec_context)
    # A range (tuple) with equal bounds keeps the dimension with length 1.
    assert out.dims == ("time", "lat", "lon")
    assert len(out.time) == 1


async def test_slice_data_array_spatiotemporal_time_single_value(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    SliceDataArray_LatLonTime = SliceDataArray.with_constrained_dims(  # noqa: N806
        ["time", "lat", "lon"]
    )
    expr = SliceDataArray_LatLonTime(
        func_name="SliceDataArray_LatLonTime",
        data_array=VariableReference(name="da"),
        dim_slices={"time": "2024-01-03"},
    )
    out: xr.DataArray = await eval_expression(expr, exec_context)
    # A single value selects one label and drops the dimension.
    assert out.dims == ("lat", "lon")


async def test_slice_data_array_spatiotemporal_single_index(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    SliceDataArray_LatLonTime = SliceDataArray.with_constrained_dims(  # noqa: N806
        ["time", "lat", "lon"]
    )
    expr = SliceDataArray_LatLonTime(
        func_name="SliceDataArray_LatLonTime",
        data_array=VariableReference(name="da"),
        dim_index_slices={"time": 0},
    )
    out: xr.DataArray = await eval_expression(expr, exec_context)
    # A single integer index also drops the dimension.
    assert out.dims == ("lat", "lon")


async def test_slice_data_array_spatiotemporal_spatial_slice(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    SliceDataArray_LatLonTime = SliceDataArray.with_constrained_dims(  # noqa: N806
        ["time", "lat", "lon"]
    )
    expr = SliceDataArray_LatLonTime(
        func_name="SliceDataArray_LatLonTime",
        data_array=VariableReference(name="da"),
        dim_slices={
            "lat": (-10, 10),
            "lon": (-10, 10),
        },
    )
    out: xr.DataArray = await eval_expression(expr, exec_context)
    assert out.shape == (len(da.time), 40, 40)


async def test_slice_data_array_general(context: ToolCallContext):
    da = make_test_data_array(
        name="analysed_sst",
        dims=[
            ("x", DimCoords(start=0, end=10, num=11, freq=None)),
            ("y", DimCoords(start=0, end=10, num=11, freq=None)),
        ],
        val_range=(0, 1),
    )
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    SliceDataArray_XY = SliceDataArray.with_constrained_dims(  # noqa: N806
        ["x", "y"]
    )
    expr = SliceDataArray_XY(
        func_name="SliceDataArray_XY",
        data_array=VariableReference(name="da"),
        dim_slices={
            "x": (2.5, 4.5),
            "y": (6.5, 9.5),
        },
    )
    out: xr.DataArray = await eval_expression(expr, exec_context)
    assert out.shape == (2, 3)


@pytest.mark.parametrize(
    "geom",
    [
        Point(0, 0),
        LineString([(0, 0), (1, 1)]),
        MultiPoint([(0, 0), (1, 1)]),
        MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
        LinearRing([(0, 0), (1, 1), (1, 0), (0, 0)]),
        GeometryCollection([Point(0, 0), LineString([(0, 0), (1, 1)])]),
    ],
    ids=[
        "Point",
        "LineString",
        "MultiPoint",
        "MultiLineString",
        "LinearRing",
        "GeometryCollection",
    ],
)
async def test_filter_data_array_by_polygon_rejects_non_polygon(
    context: ToolCallContext, geom: BaseGeometry
):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(
        context, input_vars={"da": da, "geom": geom}, config=default_exec_config
    )
    expr = FilterDataArrayByPolygon(
        func_name="FilterDataArrayByPolygon",
        data_array=VariableReference(name="da"),
        polygon=VariableReference(name="geom"),
    )
    with pytest.raises(VariableTypeMismatchError):
        _ = await eval_expression(expr, exec_context)


@pytest.mark.parametrize(
    "geom",
    [
        Polygon.from_bounds(-10, -10, 10, 10),
        GeometryCollection(
            [Polygon.from_bounds(-10, -10, 10, 10), Polygon.from_bounds(-5, -5, 5, 5)]
        ),
    ],
    ids=[
        "Polygon",
        "GeometryCollection",
    ],
)
async def test_filter_data_array_by_polygon(context: ToolCallContext, geom: BaseGeometry):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(
        context, input_vars={"da": da, "geom": geom}, config=default_exec_config
    )

    expr = FilterDataArrayByPolygon(
        func_name="FilterDataArrayByPolygon",
        data_array=VariableReference(name="da"),
        polygon=VariableReference(name="geom"),
    )
    out: xr.DataArray = await eval_expression(expr, exec_context)
    assert out.shape == da.shape

    out_cropped = out.dropna(dim="lat", how="all").dropna(dim="lon", how="all")
    assert out_cropped.shape == (len(da.time), 40, 40)


async def test_filter_data_array_by_polygon_2d(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    polygon = Polygon.from_bounds(-10, -10, 10, 10)
    exec_context = ExecutionContext(
        context, input_vars={"da": da.isel(time=0), "polygon": polygon}, config=default_exec_config
    )

    expr = FilterDataArrayByPolygon(
        func_name="FilterDataArrayByPolygon",
        data_array=VariableReference(name="da"),
        polygon=VariableReference(name="polygon"),
    )
    out: xr.DataArray = await eval_expression(expr, exec_context)
    assert out.shape == da.isel(time=0).shape

    out_cropped = out.dropna(dim="lat", how="all").dropna(dim="lon", how="all")
    assert out_cropped.shape == (40, 40)


# Generate all combinations of (time_unit, op, expected_func)
groupby_reduce_param_combinations: list[
    tuple[int, str, str, Callable[[xr.DataArray], xr.DataArray], int]
] = [
    (
        366,
        time_unit_name,
        op,
        (
            lambda da, g=groupby_func, r=reduce_func: r(
                g(da),
                cast("tuple[str, ...]", da.dims),
            )
        ),
        TIME_GROUPBY_OUT_LEN[time_unit_name],
    )
    for (time_unit_name, groupby_func), (op, reduce_func) in product(
        TIME_GROUPBY_OPS.items(), GROUPBY_REDUCTION_OPS.items()
    )
]


@pytest.mark.parametrize(
    ("in_time_steps", "time_unit", "op", "expected_func", "out_time_steps"),
    groupby_reduce_param_combinations,
)
async def test_group_by_time_and_reduce_data_array(
    context: ToolCallContext,
    in_time_steps: int,
    time_unit: Literal[
        "date",
        "week",
        "month",
        "quarter",
        "year",
        "dayofweek",
        "dayofmonth",
        "dayofyear",
    ],
    op: Literal["sum", "mean", "std", "min", "max"],
    expected_func: Callable[[xr.DataArray], xr.DataArray],
    out_time_steps: int,
):
    # 2024-01-01 to 2024-02-29
    da = make_test_data_array_spatiotemporal(
        start_date="2024-01-01",
        num_time_steps=in_time_steps,
        lat_shape=10,
        lon_shape=20,
    )
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    expr = GroupByTimeAndReduceDataArray(
        func_name="GroupByTimeAndReduceDataArray",
        data_array=VariableReference(name="da"),
        time_unit=time_unit,
        op=op,
    )
    out: xr.DataArray = await eval_expression(expr, exec_context)
    assert out.ndim == 1
    assert len(out) == out_time_steps
    xr.testing.assert_allclose(out, expected_func(da))  # type: ignore[reportUnknownMemberType]


@pytest.mark.parametrize(
    ("period_type", "period_value"),
    [
        ("month", 1),  # January
        ("month", [1, 3]),  # January and March
        ("season", "JJA"),  # June-July-August
        ("week", 20),  # First week of the year
        ("dayofyear", 100),  # 100th day of the year
    ],
)
async def test_filter_to_time_period_across_years(
    context: ToolCallContext,
    period_type: Literal["month", "season", "week", "dayofyear"],
    period_value: int | list[int] | SeasonName | list[SeasonName],
):
    da = make_test_data_array_spatiotemporal(
        start_date="2020-01-01",
        num_time_steps=365 * 3,  # 3 years of daily data
        lat_shape=10,
        lon_shape=20,
    )
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    expr = FilterToTimePeriodAcrossYears(
        func_name="FilterToTimePeriodAcrossYears",
        data_array=VariableReference(name="da"),
        period_type=period_type,
        period_value=period_value,
    )
    out: xr.DataArray = await eval_expression(expr, exec_context)
    assert out.ndim == 3

    match period_type:
        case "month":
            if isinstance(period_value, int):
                assert len(out.time) == 3 * 31  # 3 years of January data
                assert set(out.time.dt.month.values) == {1}
            else:
                assert len(out.time) == 3 * 31 * 2  # 3 years of data for each month in the list
                assert set(out.time.dt.month.values) == {1, 3}
        case "season":
            assert len(out.time) == 3 * (30 + 31 + 31)
            assert set(out.time.dt.season.values) == {"JJA"}
        case "week":
            assert len(out.time) == 3 * 7
            assert set(out.time.dt.isocalendar().week.values) == {20}
        case "dayofyear":
            assert len(out.time) == 3
            assert set(out.time.dt.dayofyear.values) == {100}


@pytest.mark.parametrize(
    ("in_time_steps", "time_unit", "op", "expected_func", "out_time_steps"),
    groupby_reduce_param_combinations,
)
async def test_group_by_time_and_reduce_data_array_nonspatial(
    context: ToolCallContext,
    in_time_steps: int,
    time_unit: Literal[
        "date",
        "week",
        "month",
        "quarter",
        "year",
        "dayofweek",
        "dayofmonth",
        "dayofyear",
    ],
    op: Literal["sum", "mean", "std", "min", "max"],
    expected_func: Callable[[xr.DataArray], xr.DataArray],
    out_time_steps: int,
):
    da = make_test_data_array(
        name="SSI",
        dims=[
            ("time", DimCoords(start="2024-01-01", end=None, num=in_time_steps, freq="D")),
            # values from noaa-cdr-solar-spectral-irradiance-pds_daily
            ("wavelength", DimCoords(start=115.5, end=99975.0, num=10, freq=None)),
        ],
        val_range=(0, 1),
    )
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    expr = GroupByTimeAndReduceDataArray(
        func_name="GroupByTimeAndReduceDataArray",
        data_array=VariableReference(name="da"),
        time_unit=time_unit,
        op=op,
    )
    out: xr.DataArray = await eval_expression(expr, exec_context)
    assert out.ndim == 1
    assert len(out) == out_time_steps
    xr.testing.assert_allclose(out, expected_func(da))  # type: ignore[reportUnknownMemberType]

    # TODO: test whether output is compatible with line plot


async def test_group_by_time_and_reduce_data_array_size_lim(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal(num_time_steps=60)
    exec_context = ExecutionContext(
        context,
        input_vars={"da": da},
        config=ExecutionConfig(data_array_compute_size_limit=10**3),  # 1 KB
    )

    expr = GroupByTimeAndReduceDataArray(
        func_name="GroupByTimeAndReduceDataArray",
        data_array=VariableReference(name="da"),
        time_unit="date",
        op="max",
    )
    with pytest.raises(DataSizeTooLargeError):
        _ = await eval_expression(expr, exec_context)


reduce_param_combinations: list[
    tuple[str, DimsT, Callable[[xr.DataArray, ReductionOpT, DimsT], xr.DataArray]]
] = [
    (
        op,
        dim,
        lambda da, r=reduce_func, dim=dim: r(da, dim),
    )
    for (op, reduce_func), dim in product(REDUCTION_OPS.items(), [None, "time", ("lat", "lon")])
]


@pytest.mark.parametrize(("op", "dim", "expected_func"), reduce_param_combinations)
async def test_reduce_data_array_spatiotemporal(
    context: ToolCallContext,
    op: Literal["sum", "mean", "std", "min", "max"],
    dim: str | tuple[str, ...] | None,
    expected_func: Callable[[xr.DataArray], xr.DataArray | float],
):
    da = make_test_data_array_spatiotemporal(num_time_steps=60)
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    ReduceDataArray_LatLonTime = ReduceDataArray.with_constrained_dims(  # noqa: N806
        ["time", "lat", "lon"]
    )

    expr = ReduceDataArray_LatLonTime(
        func_name="ReduceDataArray_LatLonTime",
        data_array=VariableReference(name="da"),
        op=op,
        dim=dim,
    )
    out = await eval_expression(expr, exec_context)

    reduced_dims = set(da.dims) if dim is None else {dim} if isinstance(dim, str) else set(dim)
    if op in ("mean", "std") and "lat" in reduced_dims:
        # mean/std are area-weighted by cos(latitude) when the latitude dim is collapsed.
        weighted = da.weighted(np.cos(np.deg2rad(da["lat"])))  # type: ignore[reportUnknownMemberType]
        raw = weighted.mean(dim=dim) if op == "mean" else weighted.std(dim=dim)
        expected_out = _reduce_to_scalar_if_possible(raw)
    else:
        expected_out = expected_func(da)  # type: ignore[reportUnknownType]

    if isinstance(expected_out, xr.DataArray):
        xr.testing.assert_allclose(out, expected_out)  # type: ignore[reportUnknownMemberType]
    else:
        assert out == expected_out


@pytest.mark.parametrize(
    ("op", "dim", "expected_func"),
    [
        ("max", "x", REDUCTION_OPS["max"]),
        ("mean", ("x", "y"), REDUCTION_OPS["mean"]),
    ],
)
async def test_reduce_data_array_general(
    context: ToolCallContext,
    op: Literal["sum", "mean", "std", "min", "max"],
    dim: str | tuple[str, ...] | None,
    expected_func: Callable[[xr.DataArray, str | tuple[str, ...] | None], xr.DataArray | float],
):
    da = make_test_data_array(
        name="analysed_sst",
        dims=[
            ("x", DimCoords(start=0, end=10, num=11, freq=None)),
            ("y", DimCoords(start=0, end=10, num=11, freq=None)),
            ("z", DimCoords(start=0, end=10, num=11, freq=None)),
        ],
        val_range=(0, 1),
    )
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    ReduceDataArray_XYZ = ReduceDataArray.with_constrained_dims(  # noqa: N806
        ["x", "y", "z"]
    )
    dim = "x"
    expr = ReduceDataArray_XYZ(
        func_name="ReduceDataArray_XYZ",
        data_array=VariableReference(name="da"),
        op=op,
        dim=dim,
    )
    out = await eval_expression(expr, exec_context)
    expected_out = expected_func(da, dim)  # type: ignore[reportUnknownType]
    if isinstance(expected_out, xr.DataArray):
        xr.testing.assert_equal(out, expected_out)  # type: ignore[reportUnknownMemberType]
    else:
        assert out == expected_out


@pytest.mark.parametrize("dim", [None, "time", ("lat", "lon")])
async def test_reduce_data_array_median(
    context: ToolCallContext,
    dim: str | tuple[str, ...] | None,
):
    da = make_test_data_array_spatiotemporal(num_time_steps=60)
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    reduce_cls = ReduceDataArray.with_constrained_dims(["time", "lat", "lon"])
    expr = reduce_cls(
        func_name="ReduceDataArray_LatLonTime",  # type: ignore[reportArgumentType]
        data_array=VariableReference(name="da"),
        op="median",
        dim=dim,
    )
    out = await eval_expression(expr, exec_context)
    expected_out = cast("xr.DataArray | float", _reduce_to_scalar_if_possible(da.median(dim=dim)))  # type: ignore[reportUnknownMemberType]
    if isinstance(expected_out, xr.DataArray):
        xr.testing.assert_equal(out, expected_out)  # type: ignore[reportUnknownMemberType]
    else:
        assert out == expected_out


@pytest.mark.parametrize("dim", [None, "time", ("lat", "lon")])
async def test_reduce_data_array_quantile(
    context: ToolCallContext,
    dim: str | tuple[str, ...] | None,
):
    da = make_test_data_array_spatiotemporal(num_time_steps=60)
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    reduce_cls = ReduceDataArray.with_constrained_dims(["time", "lat", "lon"])
    expr = reduce_cls(
        func_name="ReduceDataArray_LatLonTime",  # type: ignore[reportArgumentType]
        data_array=VariableReference(name="da"),
        op="quantile",
        dim=dim,
        q=0.9,
    )
    out = await eval_expression(expr, exec_context)
    expected = da.quantile(0.9, dim=dim).drop_vars("quantile", errors="ignore")  # type: ignore[reportUnknownMemberType]
    expected_out = cast("xr.DataArray | float", _reduce_to_scalar_if_possible(expected))
    if isinstance(expected_out, xr.DataArray):
        # The exec path drops the scalar `quantile` coordinate that xarray attaches.
        assert "quantile" not in out.coords
        # Unlike xarray's .quantile(), the exec path keeps non-index coords such as the CRS, so
        # downstream steps like CreateGeoTIFF still work.
        assert "spatial_ref" in out.coords
        xr.testing.assert_equal(out.drop_vars("spatial_ref"), expected_out)  # type: ignore[reportUnknownMemberType]
    else:
        assert out == expected_out


@pytest.mark.parametrize("op", ["median", "quantile"])
@pytest.mark.parametrize("dim", ["time", ("lat", "lon")])
async def test_reduce_data_array_holistic_chunked_stays_lazy(
    context: ToolCallContext,
    op: Literal["median", "quantile"],
    dim: str | tuple[str, ...],
):
    # Chunked along the reduction dims: the reduce must stay lazy so only the (small) result is
    # gathered, rather than pulling the whole input into this process.
    chunked = make_test_data_array_spatiotemporal(num_time_steps=60).chunk(  # type: ignore[reportUnknownMemberType]
        {"time": 10, "lat": 5, "lon": 5}
    )
    reduce_cls = ReduceDataArray.with_constrained_dims(["time", "lat", "lon"])
    expr = reduce_cls(
        func_name="ReduceDataArray_LatLonTime",  # type: ignore[reportArgumentType]
        data_array=VariableReference(name="da"),
        op=op,
        dim=dim,
        q=0.9 if op == "quantile" else None,
    )

    out = await eval_expression(
        expr, ExecutionContext(context, input_vars={"da": chunked}, config=default_exec_config)
    )
    assert out.chunks is not None

    eager_out = await eval_expression(
        expr,
        ExecutionContext(context, input_vars={"da": chunked.compute()}, config=default_exec_config),  # type: ignore[reportUnknownMemberType]
    )
    xr.testing.assert_allclose(out.compute(), eager_out)  # type: ignore[reportUnknownMemberType]


@pytest.mark.parametrize("op", ["median", "quantile"])
async def test_reduce_data_array_holistic_chunked_full_reduction(
    context: ToolCallContext,
    op: Literal["median", "quantile"],
):
    # Reducing over every dim of a chunked array: dask cannot parallelize this, so it must be
    # rechunked and run on the cluster rather than raising or gathering the whole input.
    chunked = make_test_data_array_spatiotemporal(num_time_steps=60).chunk(  # type: ignore[reportUnknownMemberType]
        {"time": 10, "lat": 5, "lon": 5}
    )
    reduce_cls = ReduceDataArray.with_constrained_dims(["time", "lat", "lon"])
    expr = reduce_cls(
        func_name="ReduceDataArray_LatLonTime",  # type: ignore[reportArgumentType]
        data_array=VariableReference(name="da"),
        op=op,
        dim=None,
        q=0.9 if op == "quantile" else None,
    )

    out = await eval_expression(
        expr, ExecutionContext(context, input_vars={"da": chunked}, config=default_exec_config)
    )
    eager = chunked.compute()  # type: ignore[reportUnknownMemberType]
    expected = float(eager.median()) if op == "median" else float(eager.quantile(0.9))  # type: ignore[reportUnknownMemberType]
    assert isinstance(out, float)
    assert np.isclose(out, expected)


async def test_reduce_data_array_mean_area_weighted_deterministic(context: ToolCallContext):
    # A field that varies only with latitude, on a symmetric lat grid, so the cos-weighted mean
    # is provably different from the plain mean and can be computed by hand.
    lat = [-60.0, 0.0, 60.0]
    lon = [0.0, 10.0]
    values = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]])  # value depends only on lat
    da = xr.DataArray(values, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="v")
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    reduce_cls = ReduceDataArray.with_constrained_dims(["lat", "lon"])
    expr = reduce_cls(
        func_name="ReduceDataArray_LatLon",  # type: ignore[reportArgumentType]
        data_array=VariableReference(name="da"),
        op="mean",
        dim=("lat", "lon"),
    )
    out = await eval_expression(expr, exec_context)

    # cos weights are cos([-60, 0, 60] deg) = [0.5, 1.0, 0.5]:
    # weighted mean = (0.5*0 + 1.0*0 + 0.5*3) / (0.5 + 1.0 + 0.5) = 0.75.
    # The unweighted mean would be (0 + 0 + 3) / 3 = 1.0.
    assert np.isclose(out, 0.75)
    assert not np.isclose(out, 1.0)


@pytest.mark.parametrize("dim", [("lat", "lon"), "lat", None])
async def test_reduce_data_array_mean_weighted_matches_xarray(
    context: ToolCallContext,
    dim: str | tuple[str, ...] | None,
):
    da = make_test_data_array_spatiotemporal(num_time_steps=12)
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    reduce_cls = ReduceDataArray.with_constrained_dims(["time", "lat", "lon"])
    expr = reduce_cls(
        func_name="ReduceDataArray_LatLonTime",  # type: ignore[reportArgumentType]
        data_array=VariableReference(name="da"),
        op="mean",
        dim=dim,
    )
    out = await eval_expression(expr, exec_context)

    weights = np.cos(np.deg2rad(da["lat"]))
    expected = da.weighted(weights).mean(dim=dim)  # type: ignore[reportUnknownMemberType]
    if isinstance(out, float):
        assert np.isclose(out, float(expected))
    else:
        xr.testing.assert_allclose(out, expected)  # type: ignore[reportUnknownMemberType]


@pytest.mark.parametrize("dim", ["time", "lon"])
async def test_reduce_data_array_mean_unweighted_without_lat(
    context: ToolCallContext,
    dim: str,
):
    da = make_test_data_array_spatiotemporal(num_time_steps=12)
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    reduce_cls = ReduceDataArray.with_constrained_dims(["time", "lat", "lon"])
    expr = reduce_cls(
        func_name="ReduceDataArray_LatLonTime",  # type: ignore[reportArgumentType]
        data_array=VariableReference(name="da"),
        op="mean",
        dim=dim,
    )
    out = await eval_expression(expr, exec_context)

    # lat is preserved (not collapsed), so no area weighting is applied.
    xr.testing.assert_allclose(out, da.mean(dim=dim))  # type: ignore[reportUnknownMemberType]


async def test_reduce_data_array_std_weighted_over_lat(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal(num_time_steps=12)
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    reduce_cls = ReduceDataArray.with_constrained_dims(["time", "lat", "lon"])
    expr = reduce_cls(
        func_name="ReduceDataArray_LatLonTime",  # type: ignore[reportArgumentType]
        data_array=VariableReference(name="da"),
        op="std",
        dim=("lat", "lon"),
    )
    out = await eval_expression(expr, exec_context)

    weights = np.cos(np.deg2rad(da["lat"]))
    expected = da.weighted(weights).std(dim=("lat", "lon"))  # type: ignore[reportUnknownMemberType]
    xr.testing.assert_allclose(out, expected)  # type: ignore[reportUnknownMemberType]


async def test_reduce_data_array_size_lim(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal(num_time_steps=60)
    exec_context = ExecutionContext(
        context,
        input_vars={"da": da},
        config=ExecutionConfig(data_array_compute_size_limit=10**3),  # 1 KB
    )

    expr = ReduceDataArray(
        func_name="ReduceDataArray",
        data_array=VariableReference(name="da"),
        op="max",
        dim=None,
    )
    with pytest.raises(DataSizeTooLargeError):
        _ = await eval_expression(expr, exec_context)


async def test_select_point_coords_dim(context: ToolCallContext):
    point_coords = PointCoords(coords={"time": "2024-01-01", "lat": 50, "lon": -10}, value=0.1)
    exec_context = ExecutionContext(
        context, input_vars={"point_coords": point_coords}, config=default_exec_config
    )

    expr = SelectPointCoordsKey(
        func_name="SelectPointCoordsKey",
        point_coords=VariableReference(name="point_coords"),
        key="time",
    )
    assert await eval_expression(expr, exec_context) == point_coords.coords["time"]

    expr = SelectPointCoordsKey(
        func_name="SelectPointCoordsKey",
        point_coords=VariableReference(name="point_coords"),
        key="lat",
    )
    assert await eval_expression(expr, exec_context) == point_coords.coords["lat"]

    expr = SelectPointCoordsKey(
        func_name="SelectPointCoordsKey",
        point_coords=VariableReference(name="point_coords"),
        key="lon",
    )
    assert await eval_expression(expr, exec_context) == point_coords.coords["lon"]

    expr = SelectPointCoordsKey(
        func_name="SelectPointCoordsKey",
        point_coords=VariableReference(name="point_coords"),
        key="value",
    )
    assert await eval_expression(expr, exec_context) == point_coords.value


async def test_geocode_natural_language_description(chat_context: MockChatContext):
    progress_updates: list[ToolProgressUpdate[GeocodingSuccessUpdate]] = []

    def progress_callback(update: ToolProgressUpdate[GeocodingSuccessUpdate]) -> None:
        progress_updates.append(update)

    context = ExecutionContext(
        ToolCallContext(
            tool_call_id="",
            chat_context=chat_context,
            progress_callback=progress_callback,
        ),
        input_vars={},
        config=default_exec_config,
    )

    expr = GeocodeNaturalLanguageDescription(
        func_name="GeocodeNaturalLanguageDescription",
        text="Seattle, WA",
    )
    with patch(
        # the path is to the module where it's used, not where it's defined
        "llm_agent.agents.algorithm_agent.algorithm.exec.data_array.geocode_natural_language_description",
        return_value=box(-10, -10, 10, 10),
    ):
        _ = await eval_expression(expr, context)

    assert len(progress_updates) == 1
    assert isinstance(progress_updates[0].update, GeocodingSuccessUpdate)
    assert progress_updates[0].update.geojson == {
        "type": "Polygon",
        "coordinates": (
            (
                (10.0, -10.0),
                (10.0, 10.0),
                (-10.0, 10.0),
                (-10.0, -10.0),
                (10.0, -10.0),
            ),
        ),
    }


async def test_compute_climatology_recovers_monthly_means(context: ToolCallContext):
    # ~2 years of daily data whose value at every pixel equals that day's month number, so the
    # monthly climatology must recover the month number at every pixel.
    da = make_test_data_array_spatiotemporal(num_time_steps=731, lat_shape=2, lon_shape=3)
    da = xr.zeros_like(da) + da["time"].dt.month
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    expr = ComputeClimatology(
        func_name="ComputeClimatology",
        data_array=VariableReference(name="da"),
        period_type="month",
        op="mean",
    )
    out = await eval_expression(expr, exec_context)

    assert set(out.dims) == {"month", "lat", "lon"}
    assert out.sizes["month"] == 12
    for month in range(1, 13):
        np.testing.assert_allclose(out.sel(month=month).to_numpy(), float(month))


@pytest.mark.parametrize("period", ["month", "season", "dayofyear", "week", "quarter"])
@pytest.mark.parametrize("op", ["mean", "std"])
async def test_compute_climatology_matches_xarray(
    context: ToolCallContext,
    period: Literal["month", "season", "dayofyear", "week", "quarter"],
    op: Literal["mean", "std"],
):
    da = make_test_data_array_spatiotemporal(num_time_steps=731, lat_shape=2, lon_shape=3)
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    expr = ComputeClimatology(
        func_name="ComputeClimatology",
        data_array=VariableReference(name="da"),
        period_type=period,
        op=op,
    )
    out = await eval_expression(expr, exec_context)

    group = da["time"].dt.isocalendar().week if period == "week" else f"time.{period}"
    grouped = da.groupby(group)  # type: ignore[reportUnknownMemberType]
    expected = grouped.mean("time") if op == "mean" else grouped.std("time")

    # Spatial dims are preserved (per-pixel climatology, not a collapsed series).
    assert set(out.dims) == {period, "lat", "lon"}
    xr.testing.assert_allclose(out, expected)  # type: ignore[reportUnknownMemberType]


async def test_compute_anomaly_deseasonalizes_to_zero(context: ToolCallContext):
    # If every day's value equals its month number, the monthly climatology equals the month
    # number, so the deseasonalized anomaly must be zero everywhere.
    da = make_test_data_array_spatiotemporal(num_time_steps=731, lat_shape=2, lon_shape=3)
    da = xr.zeros_like(da) + da["time"].dt.month
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    clim = await eval_expression(
        ComputeClimatology(
            func_name="ComputeClimatology",
            data_array=VariableReference(name="da"),
            period_type="month",
        ),
        exec_context,
    )
    exec_context.variables["clim"] = clim

    anom = await eval_expression(
        ComputeAnomaly(
            func_name="ComputeAnomaly",
            data_array=VariableReference(name="da"),
            climatology=VariableReference(name="clim"),
        ),
        exec_context,
    )

    assert set(anom.dims) == {"time", "lat", "lon"}
    assert anom.sizes["time"] == da.sizes["time"]
    np.testing.assert_allclose(anom.to_numpy(), 0.0, atol=1e-9)


@pytest.mark.parametrize("period", ["month", "season", "week"])
async def test_compute_anomaly_matches_xarray(
    context: ToolCallContext,
    period: Literal["month", "season", "week"],
):
    da = make_test_data_array_spatiotemporal(num_time_steps=731, lat_shape=2, lon_shape=3)
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    clim = await eval_expression(
        ComputeClimatology(
            func_name="ComputeClimatology",
            data_array=VariableReference(name="da"),
            period_type=period,
        ),
        exec_context,
    )
    exec_context.variables["clim"] = clim

    anom = await eval_expression(
        ComputeAnomaly(
            func_name="ComputeAnomaly",
            data_array=VariableReference(name="da"),
            climatology=VariableReference(name="clim"),
        ),
        exec_context,
    )

    group = da["time"].dt.isocalendar().week if period == "week" else f"time.{period}"
    expected = da.groupby(group) - clim  # type: ignore[reportUnknownMemberType]

    assert set(anom.dims) == {"time", "lat", "lon"}
    xr.testing.assert_allclose(anom, expected)  # type: ignore[reportUnknownMemberType]

    # The anomaly is deseasonalized: the mean over each period group is ~0.
    regrouped_mean = anom.groupby(group).mean("time")
    np.testing.assert_allclose(regrouped_mean.to_numpy(), 0.0, atol=1e-9)


@pytest.mark.parametrize("period", ["month", "season"])
async def test_compute_anomaly_standardized(
    context: ToolCallContext,
    period: Literal["month", "season"],
):
    da = make_test_data_array_spatiotemporal(num_time_steps=731, lat_shape=2, lon_shape=3)
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    clim_mean = await eval_expression(
        ComputeClimatology(
            func_name="ComputeClimatology",
            data_array=VariableReference(name="da"),
            period_type=period,
            op="mean",
        ),
        exec_context,
    )
    clim_std = await eval_expression(
        ComputeClimatology(
            func_name="ComputeClimatology",
            data_array=VariableReference(name="da"),
            period_type=period,
            op="std",
        ),
        exec_context,
    )
    exec_context.variables["cmean"] = clim_mean
    exec_context.variables["cstd"] = clim_std

    z = await eval_expression(
        ComputeAnomaly(
            func_name="ComputeAnomaly",
            data_array=VariableReference(name="da"),
            climatology=VariableReference(name="cmean"),
            climatology_std=VariableReference(name="cstd"),
        ),
        exec_context,
    )

    group = f"time.{period}"
    expected = (da.groupby(group) - clim_mean).groupby(group) / clim_std  # type: ignore[reportUnknownMemberType]

    assert set(z.dims) == {"time", "lat", "lon"}
    xr.testing.assert_allclose(z, expected)  # type: ignore[reportUnknownMemberType]

    # Standardized: the std over each period group is ~1.
    np.testing.assert_allclose(z.groupby(group).std("time").to_numpy(), 1.0, atol=1e-6)


async def test_mask_where_with_comparison_condition(context: ToolCallContext):
    da = make_test_data_array(
        name="v",
        dims=[
            ("lat", DimCoords(start=-90, end=90, num=8, freq=None)),
            ("lon", DimCoords(start=-180, end=180, num=12, freq=None)),
        ],
        val_range=(0.0, 1.0),
    )
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    # condition: da > 0.5, built with BinaryOperation
    cond = await eval_expression(
        BinaryOperation(
            func_name="BinaryOperation",
            left=VariableReference(name="da"),
            right=0.5,
            op="gt",
        ),
        exec_context,
    )
    exec_context.variables["cond"] = cond

    out = await eval_expression(
        MaskWhere(
            func_name="MaskWhere",
            data_array=VariableReference(name="da"),
            condition=VariableReference(name="cond"),
        ),
        exec_context,
    )

    xr.testing.assert_equal(out, da.where(da > 0.5))  # type: ignore[reportUnknownMemberType]
    # Some points are masked (NaN) and the kept points are unchanged.
    assert bool(out.isnull().any())
    assert bool((out > 0.5).sum() == (da > 0.5).sum())


async def test_mask_where_broadcasts_lower_dim_condition(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal(num_time_steps=5, lat_shape=4, lon_shape=6)
    spatial_mask = da.isel(time=0) > 0  # (lat, lon) boolean condition
    exec_context = ExecutionContext(
        context, input_vars={"da": da, "cond": spatial_mask}, config=default_exec_config
    )

    out = await eval_expression(
        MaskWhere(
            func_name="MaskWhere",
            data_array=VariableReference(name="da"),
            condition=VariableReference(name="cond"),
        ),
        exec_context,
    )

    assert set(out.dims) == {"time", "lat", "lon"}
    xr.testing.assert_equal(out, da.where(spatial_mask))  # type: ignore[reportUnknownMemberType]


@pytest.mark.parametrize(
    ("time_unit", "expected_slope"),
    [("year", 2.0), ("decade", 20.0), ("day", 2.0 / 365.25)],
)
async def test_compute_linear_trend_recovers_known_slope(
    context: ToolCallContext,
    time_unit: Literal["day", "year", "decade"],
    expected_slope: float,
):
    # ~10 years of daily data with an exact linear trend of 2.0 units/year at every pixel.
    da = make_test_data_array_spatiotemporal(num_time_steps=3653, lat_shape=2, lon_shape=3)
    years = (da["time"] - da["time"][0]) / np.timedelta64(1, "D") / 365.25
    da = xr.zeros_like(da) + 2.0 * years
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    out = await eval_expression(
        ComputeLinearTrend(
            func_name="ComputeLinearTrend",
            data_array=VariableReference(name="da"),
            time_unit=time_unit,
        ),
        exec_context,
    )

    assert set(out.dims) == {"lat", "lon"}
    np.testing.assert_allclose(out.to_numpy(), expected_slope, rtol=1e-6)


async def test_compute_linear_trend_pure_time_series_is_scalar(context: ToolCallContext):
    da = make_test_data_array(
        name="v",
        dims=[("time", DimCoords(start="2000-01-01", end=None, num=3653, freq="D"))],
        val_range=(0.0, 1.0),
    )
    years = (da["time"] - da["time"][0]) / np.timedelta64(1, "D") / 365.25
    da = xr.zeros_like(da) + 3.0 * years
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    out = await eval_expression(
        ComputeLinearTrend(
            func_name="ComputeLinearTrend",
            data_array=VariableReference(name="da"),
            time_unit="year",
        ),
        exec_context,
    )

    assert isinstance(out, float)
    assert np.isclose(out, 3.0, rtol=1e-6)


@pytest.mark.parametrize("op", ["mean", "sum", "std", "min", "max"])
@pytest.mark.parametrize("center", [False, True])
async def test_rolling_reduce_matches_xarray(
    context: ToolCallContext,
    op: Literal["mean", "sum", "std", "min", "max"],
    center: bool,  # noqa: FBT001
):
    da = make_test_data_array_spatiotemporal(num_time_steps=30, lat_shape=2, lon_shape=3)
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    reduce_cls = RollingReduce.with_constrained_dims(["time", "lat", "lon"])
    expr = reduce_cls(
        func_name="RollingReduce_LatLonTime",  # type: ignore[reportArgumentType]
        data_array=VariableReference(name="da"),
        dim="time",
        window=5,
        op=op,
        center=center,
    )
    out = await eval_expression(expr, exec_context)

    rolling = da.rolling({"time": 5}, center=center)
    expected = getattr(rolling, op)()

    assert out.shape == da.shape
    xr.testing.assert_allclose(out, expected)  # type: ignore[reportUnknownMemberType]


async def test_rolling_reduce_trailing_edges_are_nan(context: ToolCallContext):
    # A trailing (non-centered) window leaves the first (window-1) positions as NaN.
    da = make_test_data_array_spatiotemporal(num_time_steps=10, lat_shape=1, lon_shape=1)
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    reduce_cls = RollingReduce.with_constrained_dims(["time", "lat", "lon"])
    expr = reduce_cls(
        func_name="RollingReduce_LatLonTime",  # type: ignore[reportArgumentType]
        data_array=VariableReference(name="da"),
        dim="time",
        window=3,
        op="sum",
    )
    out = await eval_expression(expr, exec_context)

    assert bool(out.isel(time=slice(0, 2)).isnull().all())
    assert not bool(out.isel(time=slice(2, None)).isnull().any())


@pytest.mark.parametrize("op", ["sum", "max", "min"])
async def test_cumulative_reduce_matches_xarray(
    context: ToolCallContext,
    op: Literal["sum", "max", "min"],
):
    da = make_test_data_array_spatiotemporal(num_time_steps=15, lat_shape=2, lon_shape=3)
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    cumulative_cls = CumulativeReduce.with_constrained_dims(["time", "lat", "lon"])
    expr = cumulative_cls(
        func_name="CumulativeReduce_LatLonTime",  # type: ignore[reportArgumentType]
        data_array=VariableReference(name="da"),
        dim="time",
        op=op,
    )
    out = await eval_expression(expr, exec_context)

    expected = getattr(da.cumulative("time"), op)()

    assert out.shape == da.shape
    xr.testing.assert_allclose(out, expected)  # type: ignore[reportUnknownMemberType]


async def test_cumulative_reduce_running_total(context: ToolCallContext):
    # A running sum is monotonically non-decreasing for non-negative data.
    da = make_test_data_array(
        name="precip",
        dims=[("time", DimCoords(start="2020-01-01", end=None, num=8, freq="D"))],
        val_range=(0.0, 5.0),
    )
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    cumulative_cls = CumulativeReduce.with_constrained_dims(["time"])
    expr = cumulative_cls(
        func_name="CumulativeReduce_Time",  # type: ignore[reportArgumentType]
        data_array=VariableReference(name="da"),
        dim="time",
        op="sum",
    )
    out = await eval_expression(expr, exec_context)

    np.testing.assert_allclose(out.to_numpy(), np.cumsum(da.to_numpy()))
    assert np.isclose(float(out.isel(time=-1)), float(da.sum()))


@pytest.mark.parametrize("method", ["linear", "nearest"])
async def test_regrid_to_match(
    context: ToolCallContext,
    method: Literal["linear", "nearest"],
):
    source = make_test_data_array_spatiotemporal(num_time_steps=3, lat_shape=8, lon_shape=10)
    reference = make_test_data_array(
        name="ref",
        dims=[
            ("lat", DimCoords(start=-80, end=80, num=5, freq=None)),
            ("lon", DimCoords(start=-170, end=170, num=6, freq=None)),
        ],
        val_range=(0.0, 1.0),
    )
    exec_context = ExecutionContext(
        context, input_vars={"src": source, "ref": reference}, config=default_exec_config
    )

    expr = RegridToMatch(
        func_name="RegridToMatch",
        source=VariableReference(name="src"),
        reference=VariableReference(name="ref"),
        method=method,
    )
    out = await eval_expression(expr, exec_context)

    # Non-spatial dims preserved; lat/lon taken from the reference grid.
    assert out.sizes["time"] == source.sizes["time"]
    assert out.sizes["lat"] == reference.sizes["lat"]
    assert out.sizes["lon"] == reference.sizes["lon"]
    expected = source.interp(lat=reference["lat"], lon=reference["lon"], method=method)
    xr.testing.assert_allclose(out, expected)  # type: ignore[reportUnknownMemberType]


async def test_regrid_to_match_enables_binary_op_across_grids(context: ToolCallContext):
    # After regridding, a source on a different grid can be combined with the reference.
    source = make_test_data_array_spatiotemporal(num_time_steps=1, lat_shape=8, lon_shape=10)
    reference = make_test_data_array_spatiotemporal(num_time_steps=1, lat_shape=4, lon_shape=5)
    exec_context = ExecutionContext(
        context, input_vars={"src": source, "ref": reference}, config=default_exec_config
    )

    regridded = await eval_expression(
        RegridToMatch(
            func_name="RegridToMatch",
            source=VariableReference(name="src"),
            reference=VariableReference(name="ref"),
        ),
        exec_context,
    )
    assert regridded.sizes["lat"] == reference.sizes["lat"]
    assert regridded.sizes["lon"] == reference.sizes["lon"]


async def test_correlate_data_arrays_map_matches_xarray(context: ToolCallContext):
    a = make_test_data_array_spatiotemporal(num_time_steps=40, lat_shape=2, lon_shape=3)
    b = make_test_data_array_spatiotemporal(
        num_time_steps=40, lat_shape=2, lon_shape=3, random_seed=7
    )
    exec_context = ExecutionContext(
        context, input_vars={"a": a, "b": b}, config=default_exec_config
    )

    correlate_cls = CorrelateDataArrays.with_constrained_dims(["time", "lat", "lon"])
    expr = correlate_cls(
        func_name="CorrelateDataArrays_LatLonTime",  # type: ignore[reportArgumentType]
        a=VariableReference(name="a"),
        b=VariableReference(name="b"),
        dim="time",
    )
    out = await eval_expression(expr, exec_context)

    assert set(out.dims) == {"lat", "lon"}
    xr.testing.assert_allclose(out, xr.corr(a, b, dim="time"))  # type: ignore[reportUnknownMemberType]


async def test_correlate_data_arrays_scalar_and_perfect_correlation(context: ToolCallContext):
    # Two 1-D series → scalar; a series perfectly correlates with itself (r == 1).
    a = make_test_data_array(
        name="a",
        dims=[("time", DimCoords(start="2020-01-01", end=None, num=30, freq="D"))],
        val_range=(0.0, 10.0),
    )
    exec_context = ExecutionContext(context, input_vars={"a": a}, config=default_exec_config)

    correlate_cls = CorrelateDataArrays.with_constrained_dims(["time"])
    expr = correlate_cls(
        func_name="CorrelateDataArrays_Time",  # type: ignore[reportArgumentType]
        a=VariableReference(name="a"),
        b=VariableReference(name="a"),
        dim="time",
    )
    out = await eval_expression(expr, exec_context)

    assert isinstance(out, float)
    assert np.isclose(out, 1.0)


async def test_correlate_data_arrays_mismatched_time_axis_raises(context: ToolCallContext):
    # A daily and a monthly series share no common cadence; xr.corr would silently inner-join to
    # the overlapping labels, so the guard must reject them.
    daily = make_test_data_array(
        name="daily",
        dims=[("time", DimCoords(start="2020-01-01", end=None, num=90, freq="D"))],
        val_range=(0.0, 10.0),
    )
    monthly = make_test_data_array(
        name="monthly",
        dims=[("time", DimCoords(start="2020-01-01", end=None, num=3, freq="MS"))],
        val_range=(0.0, 10.0),
    )
    exec_context = ExecutionContext(
        context, input_vars={"a": daily, "b": monthly}, config=default_exec_config
    )

    correlate_cls = CorrelateDataArrays.with_constrained_dims(["time"])
    expr = correlate_cls(
        func_name="CorrelateDataArrays_Time",  # type: ignore[reportArgumentType]
        a=VariableReference(name="a"),
        b=VariableReference(name="b"),
        dim="time",
    )
    with pytest.raises(AlgorithmExecutionError, match="different 'time' coordinates"):
        await eval_expression(expr, exec_context)


async def test_resample_time_daily_to_monthly_mean(context: ToolCallContext):
    da = make_test_data_array(
        name="da",
        dims=[("time", DimCoords(start="2020-01-01", end=None, num=90, freq="D"))],
        val_range=(0.0, 10.0),
    )
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = ResampleTime(
        func_name="ResampleTime",
        data_array=VariableReference(name="da"),
        target_frequency="month",
        op="mean",
    )
    out = await eval_expression(expr, exec_context)

    assert isinstance(out, xr.DataArray)
    assert out.sizes["time"] == 3  # 90 days from Jan 1 span Jan, Feb, Mar
    xr.testing.assert_allclose(out, da.resample(time="1MS").mean())  # type: ignore[reportUnknownMemberType]


async def test_resample_time_preserves_spatial_dims(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal(num_time_steps=75, lat_shape=2, lon_shape=3)
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = ResampleTime(
        func_name="ResampleTime",
        data_array=VariableReference(name="da"),
        target_frequency="month",
        op="sum",
    )
    out = await eval_expression(expr, exec_context)

    assert set(out.dims) == {"time", "lat", "lon"}
    assert out.sizes["lat"] == 2
    assert out.sizes["lon"] == 3
    assert out.sizes["time"] < da.sizes["time"]
    xr.testing.assert_allclose(out, da.resample(time="1MS").sum())  # type: ignore[reportUnknownMemberType]


async def test_resample_time_non_datetime_axis_raises(context: ToolCallContext):
    # A non-datetime 'time' axis (e.g. an integer or timedelta lead time) cannot be resampled.
    da = xr.DataArray(
        np.arange(11.0, dtype=float),
        dims=["time"],
        coords={"time": np.arange(11)},
    )
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = ResampleTime(
        func_name="ResampleTime",
        data_array=VariableReference(name="da"),
        target_frequency="month",
        op="mean",
    )
    with pytest.raises(AlgorithmExecutionError, match="datetime"):
        await eval_expression(expr, exec_context)


async def test_resample_time_to_match_daily_onto_monthly(context: ToolCallContext):
    daily = make_test_data_array(
        name="daily",
        dims=[("time", DimCoords(start="2020-01-01", end=None, num=90, freq="D"))],
        val_range=(0.0, 10.0),
    )
    monthly = make_test_data_array(
        name="monthly",
        dims=[("time", DimCoords(start="2020-01-01", end=None, num=3, freq="MS"))],
        val_range=(0.0, 10.0),
    )
    exec_context = ExecutionContext(
        context, input_vars={"daily": daily, "monthly": monthly}, config=default_exec_config
    )
    expr = ResampleTimeToMatch(
        func_name="ResampleTimeToMatch",
        source=VariableReference(name="daily"),
        reference=VariableReference(name="monthly"),
        op="mean",
    )
    out = await eval_expression(expr, exec_context)

    # Output is snapped onto the reference's exact time axis.
    assert out.sizes["time"] == monthly.sizes["time"]
    np.testing.assert_array_equal(out["time"].to_numpy(), monthly["time"].to_numpy())
    # Each output value is the mean of the daily values in that calendar month.
    assert np.isclose(out.isel(time=0).item(), daily.sel(time="2020-01").mean().item())  # type: ignore[reportUnknownMemberType]
    assert np.isclose(out.isel(time=1).item(), daily.sel(time="2020-02").mean().item())  # type: ignore[reportUnknownMemberType]


async def test_resample_time_to_match_irregular_reference_raises(context: ToolCallContext):
    daily = make_test_data_array(
        name="daily",
        dims=[("time", DimCoords(start="2020-01-01", end=None, num=90, freq="D"))],
        val_range=(0.0, 10.0),
    )
    irregular = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=["time"],
        coords={
            "time": np.array(["2020-01-01", "2020-01-05", "2020-03-01"], dtype="datetime64[ns]")
        },
    )
    exec_context = ExecutionContext(
        context, input_vars={"daily": daily, "ref": irregular}, config=default_exec_config
    )
    expr = ResampleTimeToMatch(
        func_name="ResampleTimeToMatch",
        source=VariableReference(name="daily"),
        reference=VariableReference(name="ref"),
        op="mean",
    )
    with pytest.raises(AlgorithmExecutionError, match="cadence"):
        await eval_expression(expr, exec_context)


async def test_compute_percentile_rank(context: ToolCallContext):
    # Two pixels with known series; ranks are per-pixel and independent.
    values = np.array(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [2.0, 5.0],
            [5.0, 5.0],
            [4.0, 30.0],
        ]
    )  # (5 time, 2 lon)
    da = xr.DataArray(
        values[:, None, :],
        dims=["time", "lat", "lon"],
        coords={"time": np.arange(5), "lat": [0.0], "lon": [0.0, 1.0]},
    )
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = ComputePercentileRank(
        func_name="ComputePercentileRank", data_array=VariableReference(name="da")
    )
    out = await eval_expression(expr, exec_context)

    assert set(out.dims) == {"time", "lat", "lon"}
    np.testing.assert_allclose(out.isel(lat=0, lon=0).to_numpy(), [20.0, 60.0, 40.0, 100.0, 80.0])
    np.testing.assert_allclose(out.isel(lat=0, lon=1).to_numpy(), [60.0, 80.0, 30.0, 30.0, 100.0])


async def test_compute_percentile_rank_ignores_nan(context: ToolCallContext):
    da = xr.DataArray(
        np.array([1.0, 3.0, 2.0, np.nan, 5.0]).reshape(5, 1, 1),
        dims=["time", "lat", "lon"],
        coords={"time": np.arange(5), "lat": [0.0], "lon": [0.0]},
    )
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = ComputePercentileRank(
        func_name="ComputePercentileRank", data_array=VariableReference(name="da")
    )
    out = await eval_expression(expr, exec_context)

    vals = out.isel(lat=0, lon=0).to_numpy()
    # Only 4 valid values: ranks [1,3,2,-,4] over 4 -> [25,75,50,-,100]; NaN stays NaN.
    assert np.isnan(vals[3])
    np.testing.assert_allclose(vals[[0, 1, 2, 4]], [25.0, 75.0, 50.0, 100.0])


async def test_compute_percentile_rank_chunked_stays_lazy(context: ToolCallContext):
    # Ranking only needs each pixel's time series, so a chunked input must stay lazy and rank
    # blockwise over lat/lon instead of being gathered whole.
    chunked = make_test_data_array_spatiotemporal(num_time_steps=30).chunk(  # type: ignore[reportUnknownMemberType]
        {"time": 10, "lat": 5, "lon": 5}
    )
    expr = ComputePercentileRank(
        func_name="ComputePercentileRank", data_array=VariableReference(name="da")
    )

    out = await eval_expression(
        expr, ExecutionContext(context, input_vars={"da": chunked}, config=default_exec_config)
    )
    assert out.chunks is not None
    assert out.dims == chunked.dims

    eager_out = await eval_expression(
        expr,
        ExecutionContext(context, input_vars={"da": chunked.compute()}, config=default_exec_config),  # type: ignore[reportUnknownMemberType]
    )
    xr.testing.assert_allclose(out.compute(), eager_out)  # type: ignore[reportUnknownMemberType]


async def test_compute_trend_pvalue_chunked_stays_lazy(context: ToolCallContext):
    # The t-test is elementwise over the time-reduced grid, so a chunked input must stay lazy
    # rather than gathering the t and dof maps into this process.
    chunked = make_test_data_array_spatiotemporal(num_time_steps=40).chunk(  # type: ignore[reportUnknownMemberType]
        {"time": -1, "lat": 5, "lon": 5}
    )
    expr = ComputeTrendPValue(
        func_name="ComputeTrendPValue", data_array=VariableReference(name="da")
    )

    out = await eval_expression(
        expr, ExecutionContext(context, input_vars={"da": chunked}, config=default_exec_config)
    )
    assert out.chunks is not None

    eager_out = await eval_expression(
        expr,
        ExecutionContext(context, input_vars={"da": chunked.compute()}, config=default_exec_config),  # type: ignore[reportUnknownMemberType]
    )
    xr.testing.assert_allclose(out.compute(), eager_out)  # type: ignore[reportUnknownMemberType]


async def test_compute_trend_pvalue_flags_significance(context: ToolCallContext):
    # Pixel 0: a strong linear trend (significant). Pixel 1: an oscillation with no linear trend
    # (not significant). Deterministic so the significance verdicts are stable.
    base = make_test_data_array(
        name="v",
        dims=[("time", DimCoords(start="2000-01-01", end=None, num=60, freq="D"))],
        val_range=(0.0, 1.0),
    )
    idx = np.arange(base.sizes["time"], dtype=float)
    trend_vals = 2.0 * idx + 0.5 * np.sin(idx)
    oscillation_vals = np.sin(2 * np.pi * idx / 12.0)
    data = np.stack([trend_vals, oscillation_vals], axis=1)[:, None, :]  # (time, lat=1, lon=2)
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": base["time"], "lat": [0.0], "lon": [0.0, 1.0]},
    )
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    out = await eval_expression(
        ComputeTrendPValue(func_name="ComputeTrendPValue", data_array=VariableReference(name="da")),
        exec_context,
    )

    assert set(out.dims) == {"lat", "lon"}
    # p-values are valid probabilities; the trend pixel is significant, the oscillation is not.
    assert bool(((out >= 0) & (out <= 1)).all())
    assert float(out.isel(lat=0, lon=0)) < 0.05
    assert float(out.isel(lat=0, lon=1)) > 0.05


async def test_compute_trend_pvalue_pure_time_series_is_scalar(context: ToolCallContext):
    da = make_test_data_array(
        name="v",
        dims=[("time", DimCoords(start="2000-01-01", end=None, num=60, freq="D"))],
        val_range=(0.0, 1.0),
    )
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    out = await eval_expression(
        ComputeTrendPValue(func_name="ComputeTrendPValue", data_array=VariableReference(name="da")),
        exec_context,
    )

    assert isinstance(out, float)
    assert 0.0 <= out <= 1.0
    da = make_test_data_array(
        name="v",
        dims=[("time", DimCoords(start="2000-01-01", end=None, num=60, freq="D"))],
        val_range=(0.0, 1.0),
    )
    exec_context = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)

    out = await eval_expression(
        ComputeTrendPValue(func_name="ComputeTrendPValue", data_array=VariableReference(name="da")),
        exec_context,
    )

    assert isinstance(out, float)
    assert 0.0 <= out <= 1.0
