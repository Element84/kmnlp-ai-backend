# ruff: noqa: PLR0913
# pyright: reportUnknownLambdaType=false
from collections.abc import Callable
from itertools import product
from typing import Literal, cast
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from shapely.geometry import Polygon, box
from xarray.core.groupby import DataArrayGroupBy

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.exec import (
    DataArrayShapeError,
    ExecutionContext,
    GeocodingSuccessUpdate,
    ZeroLengthDimError,
    eval_expression,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
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
from llm_agent.agents.scientific_python_agent.utils.agent_code_utils import (
    DataSizeTooLargeError,
)
from llm_agent.context import LLMAgentContext, ToolCallContext, ToolProgressUpdate
from tests.unit.agents.algorithm_agent.utils import (
    DimCoords,
    make_test_data_array,
    make_test_data_array_spatiotemporal,
)

type DimsT = str | tuple[str, ...] | None
type GroupByFuncT = Callable[[xr.DataArray], DataArrayGroupBy]
type ReductionOpT = Callable[[xr.DataArray | DataArrayGroupBy, DimsT], xr.DataArray]

REDUCTION_OPS: dict[str, ReductionOpT] = {
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


@pytest.fixture
def context(global_context: Callable[..., LLMAgentContext]):
    return ToolCallContext(tool_call_id="", global_context=global_context())


@pytest.mark.parametrize(
    ("op", "expected_func", "expected_func_reverse_order"),
    [
        ("add", lambda da, val: da + val, lambda da, val: val + da),
        ("sub", lambda da, val: da - val, lambda da, val: val - da),
        ("mul", lambda da, val: da * val, lambda da, val: val * da),
        ("div", lambda da, val: da / val, lambda da, val: val / da),
    ],  # type: ignore[reportUnknownArgumentType]
)
def test_binary_operation_with_scalar(
    context: ToolCallContext,
    op: Literal["add", "sub", "mul", "div"],
    expected_func: Callable[[xr.DataArray, float], xr.DataArray],
    expected_func_reverse_order: Callable[[xr.DataArray, float], xr.DataArray],
):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(context, {"da": da})

    expr1 = BinaryOperation(
        func_name="BinaryOperation",
        left=VariableReference(name="da"),
        right=2,
        op=op,
    )
    out1 = eval_expression(expr1, exec_context)
    xr.testing.assert_allclose(out1, expected_func(da, 2))  # type: ignore[reportUnknownMemberType]

    expr2 = BinaryOperation(
        func_name="BinaryOperation",
        left=0.5,
        right=VariableReference(name="da"),
        op=op,
    )
    out2 = eval_expression(expr2, exec_context)
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
def test_binary_operation_with_data_arrays(
    context: ToolCallContext,
    op: Literal["add", "sub", "mul", "div"],
    expected_func: Callable[[xr.DataArray, xr.DataArray], xr.DataArray],
):
    da1 = make_test_data_array_spatiotemporal() * 2
    da2 = make_test_data_array_spatiotemporal() * 3
    exec_context = ExecutionContext(context, {"da1": da1, "da2": da2})

    expr = BinaryOperation(
        func_name="BinaryOperation",
        left=VariableReference(name="da1"),
        right=VariableReference(name="da2"),
        op=op,
    )
    out = eval_expression(expr, exec_context)
    xr.testing.assert_allclose(out, expected_func(da1, da2))  # type: ignore[reportUnknownMemberType]


def test_data_array_argmax_coords(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    da[{"time": 1, "lat": 5, "lon": 10}] = 1000
    exec_context = ExecutionContext(context, {"da": da})

    expr = DataArrayArgmaxCoords(
        func_name="DataArrayArgmaxCoords",
        data_array=VariableReference(name="da"),
    )
    out: PointCoords = eval_expression(expr, exec_context)
    out_expected = PointCoords(
        coords={
            "time": str(np.datetime_as_string(da.coords["time"].values[1])),  # type: ignore[reportUnknownMemberType]
            "lat": round(da.coords["lat"].values[5].item(), 6),  # type: ignore[reportUnknownMemberType]
            "lon": round(da.coords["lon"].values[10].item(), 6),  # type: ignore[reportUnknownMemberType]
        },
        value=1000,
    )
    assert out.model_dump() == out_expected.model_dump()


@patch(
    "llm_agent.agents.scientific_python_agent.utils.agent_code_utils.MAX_ALLOWED_COMPUTE_SIZE",
    10**3,  # 1 KB
)
def test_data_array_argmax_coords_size_lim(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(context, {"da": da})

    expr = DataArrayArgmaxCoords(
        func_name="DataArrayArgmaxCoords",
        data_array=VariableReference(name="da"),
    )
    with pytest.raises(DataSizeTooLargeError):
        _ = eval_expression(expr, exec_context)


def test_data_array_argmin_coords(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    da[{"time": 1, "lat": 5, "lon": 10}] = -1000
    exec_context = ExecutionContext(context, {"da": da})

    expr = DataArrayArgminCoords(
        func_name="DataArrayArgminCoords",
        data_array=VariableReference(name="da"),
    )
    out: PointCoords = eval_expression(expr, exec_context)
    out_expected = PointCoords(
        coords={
            "time": str(np.datetime_as_string(da.coords["time"].values[1])),  # type: ignore[reportUnknownMemberType]
            "lat": round(da.coords["lat"].values[5].item(), 6),  # type: ignore[reportUnknownMemberType]
            "lon": round(da.coords["lon"].values[10].item(), 6),  # type: ignore[reportUnknownMemberType]
        },
        value=-1000,
    )
    assert out.model_dump() == out_expected.model_dump()


@patch(
    "llm_agent.agents.scientific_python_agent.utils.agent_code_utils.MAX_ALLOWED_COMPUTE_SIZE",
    10**3,  # 1 KB
)
def test_data_array_argmin_coords_size_lim(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(context, {"da": da})

    expr = DataArrayArgminCoords(
        func_name="DataArrayArgminCoords",
        data_array=VariableReference(name="da"),
    )
    with pytest.raises(DataSizeTooLargeError):
        _ = eval_expression(expr, exec_context)


def test_data_array_to_list(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    da = da.mean(dim=("lat", "lon")) * 0
    exec_context = ExecutionContext(context, {"da": da})

    expr = DataArrayToList(
        func_name="DataArrayToList",
        data_array=VariableReference(name="da"),
    )
    out = eval_expression(expr, exec_context)
    out_expected = [0] * len(da)
    assert_allclose(out, out_expected)


def test_data_array_to_scalar(context: ToolCallContext):
    da = xr.DataArray(np.ones((1,)))
    exec_context = ExecutionContext(context, {"da": da})

    expr = ZeroDimDataArrayToScalar(
        func_name="ZeroDimDataArrayToScalar",
        data_array=VariableReference(name="da"),
    )
    out = eval_expression(expr, exec_context)
    assert isinstance(out, float)
    assert out == 1


def test_data_array_to_scalar_allows_squeezable(context: ToolCallContext):
    da = xr.DataArray(np.ones((1, 1, 1)))
    exec_context = ExecutionContext(context, {"da": da})

    expr = ZeroDimDataArrayToScalar(
        func_name="ZeroDimDataArrayToScalar",
        data_array=VariableReference(name="da"),
    )
    out = eval_expression(expr, exec_context)
    assert isinstance(out, float)
    assert out == 1


def test_data_array_to_scalar_error_if_not_squeezable(context: ToolCallContext):
    da = xr.DataArray(np.ones((1, 2, 1)))
    exec_context = ExecutionContext(context, {"da": da})

    expr = ZeroDimDataArrayToScalar(
        func_name="ZeroDimDataArrayToScalar",
        data_array=VariableReference(name="da"),
    )
    with pytest.raises(DataArrayShapeError, match="Expected 0D DataArray after squeezing"):
        _ = eval_expression(expr, exec_context)


def test_slice_data_array_spatiotemporal_time_slice(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(context, {"da": da})

    SliceDataArray_LatLonTime = SliceDataArray.with_constrained_dims(  # noqa: N806
        ["time", "lat", "lon"]
    )

    expr = SliceDataArray_LatLonTime(
        func_name="SliceDataArray_LatLonTime",
        data_array=VariableReference(name="da"),
        dim_slices={"time": ("2024-01-03", "2024-01-05")},
    )
    out: xr.DataArray = eval_expression(expr, exec_context)
    out_dates = [
        str(np.datetime_as_string(t, unit="D"))  # type: ignore[reportUnknownArgumentType]
        for t in out.time.data
    ]
    assert len(out.time) == 3
    assert out_dates == ["2024-01-03", "2024-01-04", "2024-01-05"]


def test_slice_data_array_spatiotemporal_time_slice_same_start_end(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(context, {"da": da})

    SliceDataArray_LatLonTime = SliceDataArray.with_constrained_dims(  # noqa: N806
        ["time", "lat", "lon"]
    )
    expr = SliceDataArray_LatLonTime(
        func_name="SliceDataArray_LatLonTime",
        data_array=VariableReference(name="da"),
        dim_slices={"time": ("2024-01-03", "2024-01-03")},
    )
    out: xr.DataArray = eval_expression(expr, exec_context)
    out_dates = [
        str(np.datetime_as_string(t, unit="D"))  # type: ignore[reportUnknownArgumentType]
        for t in out.time.data
    ]
    assert len(out.time) == 1
    assert out_dates == ["2024-01-03"]


def test_slice_data_array_spatiotemporal_error_on_zero_dim_result(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(context, {"da": da})

    SliceDataArray_LatLonTime = SliceDataArray.with_constrained_dims(  # noqa: N806
        ["time", "lat", "lon"]
    )
    expr = SliceDataArray_LatLonTime(
        func_name="SliceDataArray_LatLonTime",
        data_array=VariableReference(name="da"),
        dim_slices={"time": ("2024-01-03T01:00:00", "2024-01-03T01:00:00")},
    )
    with pytest.raises(ZeroLengthDimError, match="Result of slicing has a zero-length dimension"):
        _ = eval_expression(expr, exec_context)


def test_slice_data_array_spatiotemporal_spatial_slice(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    exec_context = ExecutionContext(context, {"da": da})

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
    out: xr.DataArray = eval_expression(expr, exec_context)
    assert out.shape == (len(da.time), 40, 40)


def test_slice_data_array_general(context: ToolCallContext):
    da = make_test_data_array(
        name="analysed_sst",
        dims=[
            ("x", DimCoords(start=0, end=10, num=11, freq=None)),
            ("y", DimCoords(start=0, end=10, num=11, freq=None)),
        ],
        val_range=(0, 1),
    )
    exec_context = ExecutionContext(context, {"da": da})

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
    out: xr.DataArray = eval_expression(expr, exec_context)
    assert out.shape == (2, 3)


def test_filter_data_array_by_polygon(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    polygon = Polygon.from_bounds(-10, -10, 10, 10)
    exec_context = ExecutionContext(context, {"da": da, "polygon": polygon})

    expr = FilterDataArrayByPolygon(
        func_name="FilterDataArrayByPolygon",
        data_array=VariableReference(name="da"),
        polygon=VariableReference(name="polygon"),
    )
    out: xr.DataArray = eval_expression(expr, exec_context)
    assert out.shape == (len(da.time), 40, 40)


def test_filter_data_array_by_polygon_2d(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal()
    polygon = Polygon.from_bounds(-10, -10, 10, 10)
    exec_context = ExecutionContext(context, {"da": da.isel(time=0), "polygon": polygon})

    expr = FilterDataArrayByPolygon(
        func_name="FilterDataArrayByPolygon",
        data_array=VariableReference(name="da"),
        polygon=VariableReference(name="polygon"),
    )
    out: xr.DataArray = eval_expression(expr, exec_context)
    assert out.shape == (40, 40)


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
        TIME_GROUPBY_OPS.items(), REDUCTION_OPS.items()
    )
]


@pytest.mark.parametrize(
    ("in_time_steps", "time_unit", "op", "expected_func", "out_time_steps"),
    groupby_reduce_param_combinations,
)
def test_group_by_time_and_reduce_data_array(
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
    exec_context = ExecutionContext(context, {"da": da})

    expr = GroupByTimeAndReduceDataArray(
        func_name="GroupByTimeAndReduceDataArray",
        data_array=VariableReference(name="da"),
        time_unit=time_unit,
        op=op,
    )
    out: xr.DataArray = eval_expression(expr, exec_context)
    assert out.ndim == 1
    assert len(out) == out_time_steps
    xr.testing.assert_allclose(out, expected_func(da))  # type: ignore[reportUnknownMemberType]

    # test whether output is compatible with PlotDataArraysAsTimeSeries
    exec_context = ExecutionContext(context, {"out": out})
    expr = PlotDataArraysAsTimeSeries(
        func_name="PlotDataArraysAsTimeSeries",
        data_arrays=[VariableReference(name="out")],
        labels=["out"],
        title="test",
    )
    eval_expression(expr, exec_context)


@pytest.mark.parametrize(
    ("in_time_steps", "time_unit", "op", "expected_func", "out_time_steps"),
    groupby_reduce_param_combinations,
)
def test_group_by_time_and_reduce_data_array_nonspatial(
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
    exec_context = ExecutionContext(context, {"da": da})

    expr = GroupByTimeAndReduceDataArray(
        func_name="GroupByTimeAndReduceDataArray",
        data_array=VariableReference(name="da"),
        time_unit=time_unit,
        op=op,
    )
    out: xr.DataArray = eval_expression(expr, exec_context)
    assert out.ndim == 1
    assert len(out) == out_time_steps
    xr.testing.assert_allclose(out, expected_func(da))  # type: ignore[reportUnknownMemberType]

    # test whether output is compatible with PlotDataArraysAsTimeSeries
    exec_context = ExecutionContext(context, {"out": out})
    expr = PlotDataArraysAsTimeSeries(
        func_name="PlotDataArraysAsTimeSeries",
        data_arrays=[VariableReference(name="out")],
        labels=["out"],
        title="test",
    )
    eval_expression(expr, exec_context)


@patch(
    "llm_agent.agents.scientific_python_agent.utils.agent_code_utils.MAX_ALLOWED_COMPUTE_SIZE",
    10**3,  # 1 KB
)
def test_group_by_time_and_reduce_data_array_size_lim(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal(num_time_steps=60)
    exec_context = ExecutionContext(context, {"da": da})

    expr = GroupByTimeAndReduceDataArray(
        func_name="GroupByTimeAndReduceDataArray",
        data_array=VariableReference(name="da"),
        time_unit="date",
        op="max",
    )
    with pytest.raises(DataSizeTooLargeError):
        _ = eval_expression(expr, exec_context)


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
def test_reduce_data_array_spatiotemporal(
    context: ToolCallContext,
    op: Literal["sum", "mean", "std", "min", "max"],
    dim: str | tuple[str, ...] | None,
    expected_func: Callable[[xr.DataArray], xr.DataArray],
):
    da = make_test_data_array_spatiotemporal(num_time_steps=60)
    exec_context = ExecutionContext(context, {"da": da})

    ReduceDataArray_LatLonTime = ReduceDataArray.with_constrained_dims(  # noqa: N806
        ["time", "lat", "lon"]
    )

    expr = ReduceDataArray_LatLonTime(
        func_name="ReduceDataArray_LatLonTime",
        data_array=VariableReference(name="da"),
        op=op,
        dim=dim,
    )
    out: xr.DataArray = eval_expression(expr, exec_context)
    xr.testing.assert_equal(out, expected_func(da))  # type: ignore[reportUnknownMemberType]


@pytest.mark.parametrize(
    ("op", "dim", "expected_func"),
    [
        ("max", "x", REDUCTION_OPS["max"]),
        ("mean", ("x", "y"), REDUCTION_OPS["mean"]),
    ],
)
def test_reduce_data_array_general(
    context: ToolCallContext,
    op: Literal["sum", "mean", "std", "min", "max"],
    dim: str | tuple[str, ...] | None,
    expected_func: Callable[[xr.DataArray], xr.DataArray],
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
    exec_context = ExecutionContext(context, {"da": da})

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
    out: xr.DataArray = eval_expression(expr, exec_context)
    xr.testing.assert_equal(out, expected_func(da, dim=dim))  # type: ignore[reportUnknownMemberType]


@patch(
    "llm_agent.agents.scientific_python_agent.utils.agent_code_utils.MAX_ALLOWED_COMPUTE_SIZE",
    10**3,  # 1 KB
)
def test_reduce_data_array_size_lim(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal(num_time_steps=60)
    exec_context = ExecutionContext(context, {"da": da})

    expr = ReduceDataArray(
        func_name="ReduceDataArray",
        data_array=VariableReference(name="da"),
        op="max",
        dim=None,
    )
    with pytest.raises(DataSizeTooLargeError):
        _ = eval_expression(expr, exec_context)


def test_select_point_coords_dim(context: ToolCallContext):
    point_coords = PointCoords(coords={"time": "2024-01-01", "lat": 50, "lon": -10}, value=0.1)
    exec_context = ExecutionContext(context, {"point_coords": point_coords})

    expr = SelectPointCoordsKey(
        func_name="SelectPointCoordsKey",
        point_coords=VariableReference(name="point_coords"),
        key="time",
    )
    assert eval_expression(expr, exec_context) == point_coords.coords["time"]

    expr = SelectPointCoordsKey(
        func_name="SelectPointCoordsKey",
        point_coords=VariableReference(name="point_coords"),
        key="lat",
    )
    assert eval_expression(expr, exec_context) == point_coords.coords["lat"]

    expr = SelectPointCoordsKey(
        func_name="SelectPointCoordsKey",
        point_coords=VariableReference(name="point_coords"),
        key="lon",
    )
    assert eval_expression(expr, exec_context) == point_coords.coords["lon"]

    expr = SelectPointCoordsKey(
        func_name="SelectPointCoordsKey",
        point_coords=VariableReference(name="point_coords"),
        key="value",
    )
    assert eval_expression(expr, exec_context) == point_coords.value


def test_plot_data_arrays_as_raster(context: ToolCallContext):
    da1 = make_test_data_array_spatiotemporal(num_time_steps=1)
    da2 = make_test_data_array_spatiotemporal(num_time_steps=1, random_seed=0)
    exec_context = ExecutionContext(context, {"da1": da1, "da2": da2})

    expr = PlotDataArraysAsRaster(
        func_name="PlotDataArraysAsRaster",
        data_arrays=[
            VariableReference(name="da1"),
            VariableReference(name="da2"),
        ],
        subplot_grid_shape=(2, 1),
        title="test",
    )
    eval_expression(expr, exec_context)


def test_plot_data_arrays_as_raster_shape_error(context: ToolCallContext):
    da = make_test_data_array_spatiotemporal(num_time_steps=2)
    exec_context = ExecutionContext(context, {"da": da})

    expr = PlotDataArraysAsRaster(
        func_name="PlotDataArraysAsRaster",
        data_arrays=[VariableReference(name="da")],
        subplot_grid_shape=(1, 1),
        title="test",
    )
    with pytest.raises(DataArrayShapeError, match="Expected a DataArray squeezable to 2D"):
        eval_expression(expr, exec_context)


def test_plot_data_arrays_as_time_series(context: ToolCallContext):
    da1 = make_test_data_array_spatiotemporal().groupby("time").mean(dim=("time", "lat", "lon"))
    da2 = (
        make_test_data_array_spatiotemporal(random_seed=0)
        .groupby("time")
        .mean(dim=("time", "lat", "lon"))
    )
    exec_context = ExecutionContext(context, {"da1": da1, "da2": da2})

    expr = PlotDataArraysAsTimeSeries(
        func_name="PlotDataArraysAsTimeSeries",
        data_arrays=[
            VariableReference(name="da1"),
            VariableReference(name="da2"),
        ],
        labels=["da1", "da2"],
        title="test",
    )
    eval_expression(expr, exec_context)


def test_plot_data_arrays_as_time_series_nonidentical_dim_error(context: ToolCallContext):
    da1 = (
        make_test_data_array_spatiotemporal().groupby("time.month").mean(dim=("time", "lat", "lon"))
    )
    da2 = (
        make_test_data_array_spatiotemporal(random_seed=0)
        .groupby("time")
        .mean(dim=("time", "lat", "lon"))
    )
    exec_context = ExecutionContext(context, {"da1": da1, "da2": da2})

    expr = PlotDataArraysAsTimeSeries(
        func_name="PlotDataArraysAsTimeSeries",
        data_arrays=[
            VariableReference(name="da1"),
            VariableReference(name="da2"),
        ],
        labels=["da1", "da2"],
        title="test",
    )
    with pytest.raises(
        DataArrayShapeError, match="Time series DataArrays do not have the same time dim"
    ):
        eval_expression(expr, exec_context)


def test_geocode_natural_language_description(global_context: Callable[..., LLMAgentContext]):
    progress_updates: list[ToolProgressUpdate[GeocodingSuccessUpdate]] = []

    def progress_callback(update: ToolProgressUpdate[GeocodingSuccessUpdate]) -> None:
        progress_updates.append(update)

    context = ExecutionContext(
        ToolCallContext(
            tool_call_id="",
            global_context=global_context(),
            progress_callback=progress_callback,
        ),
        input_vars={},
    )

    expr = GeocodeNaturalLanguageDescription(
        func_name="GeocodeNaturalLanguageDescription",
        text="Seattle, WA",
    )
    with patch(
        # the path is to the module where it's used, not where it's defined
        "llm_agent.agents.algorithm_agent.algorithm.exec.geocode_natural_language_description",
        return_value=box(-10, -10, 10, 10),
    ):
        _ = eval_expression(expr, context)

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
