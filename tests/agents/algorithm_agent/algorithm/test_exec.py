from collections.abc import Callable
from itertools import product
from typing import Literal

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from shapely.geometry import Polygon

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.data_array_functions import (
    BinaryOperation,
    DataArrayArgmaxCoords,
    DataArrayArgminCoords,
    DataArrayToList,
    DataArrayToScalar,
    FilterDataArray,
    FilterDataArrayByPolygon,
    GroupByTimeAndReduceDataArray,
    PlotDataArraysAsRaster,
    PlotDataArraysAsTimeSeries,
    PointCoords,
    ReduceDataArray,
    SelectPointCoordsDim,
)
from llm_agent.agents.algorithm_agent.algorithm.exec import (
    ExecutionContext,
    eval_expression,
)
from tests.agents.algorithm_agent.utils import make_dummy_data_array

REDUCTION_OPS = {
    "sum": lambda da, dim: da.sum(dim=dim),
    "mean": lambda da, dim: da.mean(dim=dim),
    "std": lambda da, dim: da.std(dim=dim),
    "min": lambda da, dim: da.min(dim=dim),
    "max": lambda da, dim: da.max(dim=dim),
}
TIME_GROUPBY_OPS = {
    "time_dim": lambda da: da.groupby("time"),
    "year": lambda da: da.groupby("time.year"),
    "quarter": lambda da: da.groupby("time.quarter"),
    "month": lambda da: da.groupby("time.month"),
    "week": lambda da: da.groupby(da.time.dt.isocalendar().week),
    "dayofmonth": lambda da: da.groupby("time.day"),
    "dayofyear": lambda da: da.groupby("time.dayofyear"),
    "dayofweek": lambda da: da.groupby("time.dayofweek"),
}


@pytest.mark.parametrize(
    ("op", "expected_func", "expected_func_reverse_order"),
    [
        ("add", lambda da, val: da + val, lambda da, val: val + da),
        ("sub", lambda da, val: da - val, lambda da, val: val - da),
        ("mul", lambda da, val: da * val, lambda da, val: val * da),
        ("div", lambda da, val: da / val, lambda da, val: val / da),
    ],
)
def test_binary_operation_with_scalar(
    op: Literal["add", "sub", "mul", "div"],
    expected_func: Callable[[xr.DataArray, float], xr.DataArray],
    expected_func_reverse_order: Callable[[xr.DataArray, float], xr.DataArray],
):
    da = make_dummy_data_array()
    context = ExecutionContext({"da": da})

    expr1 = BinaryOperation(
        func_name="BinaryOperation",
        left=VariableReference(name="da"),
        right=2,
        op=op,
    )
    out1 = eval_expression(expr1, context)
    xr.testing.assert_allclose(out1, expected_func(da, 2))

    expr2 = BinaryOperation(
        func_name="BinaryOperation",
        left=0.5,
        right=VariableReference(name="da"),
        op=op,
    )
    out2 = eval_expression(expr2, context)
    xr.testing.assert_allclose(out2, expected_func_reverse_order(da, 0.5))


@pytest.mark.parametrize(
    ("op", "expected_func"),
    [
        ("add", lambda da1, da2: da1 + da2),
        ("sub", lambda da1, da2: da1 - da2),
        ("mul", lambda da1, da2: da1 * da2),
        ("div", lambda da1, da2: da1 / da2),
    ],
)
def test_binary_operation_with_data_arrays(
    op: Literal["add", "sub", "mul", "div"],
    expected_func: Callable[[xr.DataArray, xr.DataArray], xr.DataArray],
):
    da1 = make_dummy_data_array() * 2
    da2 = make_dummy_data_array() * 3
    context = ExecutionContext({"da1": da1, "da2": da2})

    expr = BinaryOperation(
        func_name="BinaryOperation",
        left=VariableReference(name="da1"),
        right=VariableReference(name="da2"),
        op=op,
    )
    out = eval_expression(expr, context)
    xr.testing.assert_allclose(out, expected_func(da1, da2))


def test_data_array_argmax_coords():
    da = make_dummy_data_array()
    da[{"time": 1, "lat": 5, "lon": 10}] = 1000
    context = ExecutionContext({"da": da})

    expr = DataArrayArgmaxCoords(
        func_name="DataArrayArgmaxCoords",
        data_array=VariableReference(name="da"),
    )
    out: PointCoords = eval_expression(expr, context)
    out_expected = PointCoords(
        time=str(np.datetime_as_string(da.coords["time"].values[1], unit="D")),
        lat=round(da.coords["lat"].values[5].item(), 6),
        lon=round(da.coords["lon"].values[10].item(), 6),
    )
    assert out.model_dump() == out_expected.model_dump()


def test_data_array_argmin_coords():
    da = make_dummy_data_array()
    da[{"time": 1, "lat": 5, "lon": 10}] = -1000
    context = ExecutionContext({"da": da})

    expr = DataArrayArgminCoords(
        func_name="DataArrayArgminCoords",
        data_array=VariableReference(name="da"),
    )
    out: PointCoords = eval_expression(expr, context)
    out_expected = PointCoords(
        time=str(np.datetime_as_string(da.coords["time"].values[1], unit="D")),
        lat=round(da.coords["lat"].values[5].item(), 6),
        lon=round(da.coords["lon"].values[10].item(), 6),
    )
    assert out.model_dump() == out_expected.model_dump()


def test_data_array_to_list():
    da = make_dummy_data_array()
    da = da.mean(dim=("lat", "lon")) * 0
    context = ExecutionContext({"da": da})

    expr = DataArrayToList(
        func_name="DataArrayToList",
        data_array=VariableReference(name="da"),
    )
    out = eval_expression(expr, context)
    out_expected = [0] * len(da)
    assert_allclose(out, out_expected)


def test_data_array_to_scalar():
    da = xr.DataArray(np.ones((1,)))
    context = ExecutionContext({"da": da})

    expr = DataArrayToScalar(
        func_name="DataArrayToScalar",
        data_array=VariableReference(name="da"),
    )
    out = eval_expression(expr, context)
    assert isinstance(out, float)
    assert out == 1


def test_filter_data_array_time():
    da = make_dummy_data_array()
    context = ExecutionContext({"da": da})

    expr = FilterDataArray(
        func_name="FilterDataArray",
        data_array=VariableReference(name="da"),
        time_range=("2024-01-03", "2024-01-05"),
    )
    out: xr.DataArray = eval_expression(expr, context)
    out_dates = [
        str(np.datetime_as_string(t, unit="D")) for t in out.time.data
    ]
    assert len(out.time) == 3
    assert out_dates == ["2024-01-03", "2024-01-04", "2024-01-05"]

    expr = FilterDataArray(
        func_name="FilterDataArray",
        data_array=VariableReference(name="da"),
        time_range=("2024-01-03", "2024-01-03"),
    )
    out: xr.DataArray = eval_expression(expr, context)
    out_dates = [
        str(np.datetime_as_string(t, unit="D")) for t in out.time.data
    ]
    assert len(out.time) == 1
    assert out_dates == ["2024-01-03"]


def test_filter_data_array_spatial():
    da = make_dummy_data_array()
    context = ExecutionContext({"da": da})

    expr = FilterDataArray(
        func_name="FilterDataArray",
        data_array=VariableReference(name="da"),
        lat_range=(-10, 10),
        lon_range=(-10, 10),
    )
    out: xr.DataArray = eval_expression(expr, context)
    assert out.shape == (len(da.time), 40, 40)


def test_filter_data_array_by_polygon():
    da = make_dummy_data_array()
    polygon = Polygon.from_bounds(-10, -10, 10, 10)
    context = ExecutionContext({"da": da, "polygon": polygon})

    expr = FilterDataArrayByPolygon(
        func_name="FilterDataArrayByPolygon",
        data_array=VariableReference(name="da"),
        polygon=VariableReference(name="polygon"),
    )
    out: xr.DataArray = eval_expression(expr, context)
    assert out.shape == (len(da.time), 40, 40)


# Generate all combinations of (time_unit, op, expected_func)
groupby_reduce_param_combinations = [
    (
        time_unit_name,
        op,
        lambda da, g=groupby_func, f=reduce_func: f(
            g(da), dim=("time", "lat", "lon")
        ),
    )
    for (time_unit_name, groupby_func), (op, reduce_func) in product(
        TIME_GROUPBY_OPS.items(), REDUCTION_OPS.items()
    )
]


@pytest.mark.parametrize(
    ("time_unit", "op", "expected_func"), groupby_reduce_param_combinations
)
def test_group_by_time_and_reduce_data_array(
    time_unit: Literal[
        "time_dim",
        "year",
        "quarter",
        "month",
        "week",
        "dayofmonth",
        "dayofyear",
        "dayofweek",
    ],
    op: Literal["sum", "mean", "std", "min", "max"],
    expected_func: Callable[[xr.DataArray], xr.DataArray],
):
    da = make_dummy_data_array(num_time_steps=60)
    context = ExecutionContext({"da": da})

    expr = GroupByTimeAndReduceDataArray(
        func_name="GroupByTimeAndReduceDataArray",
        data_array=VariableReference(name="da"),
        time_unit=time_unit,
        op=op,
    )
    out: xr.DataArray = eval_expression(expr, context)
    xr.testing.assert_allclose(out, expected_func(da))


reduce_param_combinations = [
    (
        op,
        dim,
        lambda da, f=reduce_func, dim=dim: f(da, dim=dim),
    )
    for (op, reduce_func), dim in product(
        REDUCTION_OPS.items(), [None, "time", ("lat", "lon")]
    )
]


@pytest.mark.parametrize(
    ("op", "dim", "expected_func"), reduce_param_combinations
)
def test_reduce_data_array(
    op: Literal["sum", "mean", "std", "min", "max"],
    dim: str | tuple[str, ...] | None,
    expected_func: Callable[[xr.DataArray], xr.DataArray],
):
    da = make_dummy_data_array(num_time_steps=60)
    context = ExecutionContext({"da": da})

    expr = ReduceDataArray(
        func_name="ReduceDataArray",
        data_array=VariableReference(name="da"),
        op=op,
        dim=dim,
    )
    out: xr.DataArray = eval_expression(expr, context)
    xr.testing.assert_equal(out, expected_func(da))


def test_select_point_coords_dim():
    point_coords = PointCoords(time="2024-01-01", lat=50, lon=-10)
    context = ExecutionContext({"point_coords": point_coords})

    expr = SelectPointCoordsDim(
        func_name="SelectPointCoordsDim",
        point_coords=VariableReference(name="point_coords"),
        dim="time",
    )
    assert eval_expression(expr, context) == point_coords.time

    expr = SelectPointCoordsDim(
        func_name="SelectPointCoordsDim",
        point_coords=VariableReference(name="point_coords"),
        dim="lat",
    )
    assert eval_expression(expr, context) == point_coords.lat

    expr = SelectPointCoordsDim(
        func_name="SelectPointCoordsDim",
        point_coords=VariableReference(name="point_coords"),
        dim="lon",
    )
    assert eval_expression(expr, context) == point_coords.lon


def test_plot_data_arrays_as_raster():
    da1 = make_dummy_data_array()
    da2 = make_dummy_data_array(random_seed=0)
    context = ExecutionContext({"da1": da1, "da2": da2})

    expr = PlotDataArraysAsRaster(
        func_name="PlotDataArraysAsRaster",
        data_arrays=[
            VariableReference(name="da1"),
            VariableReference(name="da2"),
        ],
        subplot_grid_shape=(2, 1),
        title="test",
    )
    eval_expression(expr, context)


def test_plot_data_arrays_as_time_series():
    da1 = (
        make_dummy_data_array()
        .groupby("time")
        .mean(dim=("time", "lat", "lon"))
    )
    da2 = (
        make_dummy_data_array(random_seed=0)
        .groupby("time")
        .mean(dim=("time", "lat", "lon"))
    )
    context = ExecutionContext({"da1": da1, "da2": da2})

    expr = PlotDataArraysAsTimeSeries(
        func_name="PlotDataArraysAsTimeSeries",
        data_arrays=[
            VariableReference(name="da1"),
            VariableReference(name="da2"),
        ],
        title="test",
    )
    eval_expression(expr, context)
