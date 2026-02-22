import json
from contextlib import nullcontext as does_not_raise
from datetime import UTC, datetime, timedelta
from typing import Literal

import pytest
from geojson_pydantic.geometries import parse_geometry_obj

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
    AssignUserProvidedGeometry,
    DataArrayArgmaxCoords,
    DataArrayArgminCoords,
    DataArrayToList,
    FilterDataArrayByPolygon,
    GeocodeNaturalLanguageDescription,
    GroupByTimeAndReduceDataArray,
    ReduceDataArray,
    SelectPointCoordsKey,
    SliceDataArray,
    ZeroDimDataArrayToScalar,
)
from llm_agent.agents.algorithm_agent.algorithm.type_check import (
    DataArrayShapeError,
    ExecutionContext,
    InferredDataArrayVariable,
    InferredListVariable,
    InferredTemporalDataArrayVariable,
    InferredVariable,
    InvalidDimNameError,
    VariableTypeMismatchError,
    type_check_expression,
)
from llm_agent.agents.algorithm_agent.algorithm.utils import TimeCoords


class TestDataArrayFunctions:
    """Test DataArray-specific functions."""

    def test_data_array_argmax_coords(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        data_array = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        context.variables["da"] = data_array

        expr = DataArrayArgmaxCoords(
            func_name="DataArrayArgmaxCoords",
            data_array=VariableReference(name="da"),
        )
        result = type_check_expression(expr, context)

        assert result == InferredVariable(type="PointCoords")

    def test_data_array_argmax_coords_invalid_type(self):
        context = ExecutionContext({"x": InferredVariable(type="int")})

        expr = DataArrayArgmaxCoords(
            func_name="DataArrayArgmaxCoords",
            data_array=VariableReference(name="x"),
        )

        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)

    def test_data_array_argmin_coords(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        data_array = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        context.variables["da"] = data_array

        expr = DataArrayArgminCoords(
            func_name="DataArrayArgminCoords",
            data_array=VariableReference(name="da"),
        )
        result = type_check_expression(expr, context)

        assert result == InferredVariable(type="PointCoords")

    def test_data_array_to_list_valid(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        data_array = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time"}),
            shape={"time": 10},
        )
        context.variables["da"] = data_array

        expr = DataArrayToList(
            func_name="DataArrayToList",
            data_array=VariableReference(name="da"),
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredListVariable)

    def test_data_array_to_list_invalid_dims(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        data_array = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat"}),
            shape={"time": 10, "lat": 20},
        )
        context.variables["da"] = data_array

        expr = DataArrayToList(
            func_name="DataArrayToList",
            data_array=VariableReference(name="da"),
        )

        with pytest.raises(DataArrayShapeError, match="expected 1D DataArray"):
            type_check_expression(expr, context)

    def test_zero_dim_data_array_to_scalar_valid(self):
        context = ExecutionContext({})
        data_array = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset(),
            shape={},
        )
        context.variables["da"] = data_array

        expr = ZeroDimDataArrayToScalar(
            func_name="ZeroDimDataArrayToScalar",
            data_array=VariableReference(name="da"),
        )
        result = type_check_expression(expr, context)

        assert result == InferredVariable(type="float")

    def test_zero_dim_data_array_to_scalar_invalid_dims(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        data_array = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time"}),
            shape={"time": 2},
        )
        context.variables["da"] = data_array

        expr = ZeroDimDataArrayToScalar(
            func_name="ZeroDimDataArrayToScalar",
            data_array=VariableReference(name="da"),
        )

        with pytest.raises(DataArrayShapeError, match="input DataArray is not provably squeezable"):
            type_check_expression(expr, context)

    def test_zero_dim_data_array_to_scalar_allows_squeezable(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        data_array = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "wavelength"}),
            shape={"time": 1, "wavelength": 1},
        )
        context.variables["da"] = data_array

        expr = ZeroDimDataArrayToScalar(
            func_name="ZeroDimDataArrayToScalar",
            data_array=VariableReference(name="da"),
        )

        with does_not_raise():
            type_check_expression(expr, context)


class TestSliceDataArray:
    """Test SliceDataArray function."""

    def test_slice_data_array_valid_dims(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        data_array = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        context.variables["da"] = data_array

        expr = SliceDataArray(
            func_name="SliceDataArray",
            data_array=VariableReference(name="da"),
            dim_slices={"lat": (-10, 10), "lon": (-20, 20)},
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})
        assert result.name is None

    def test_slice_data_array_invalid_dim(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        data_array = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        context.variables["da"] = data_array

        expr = SliceDataArray(
            func_name="SliceDataArray",
            data_array=VariableReference(name="da"),
            dim_slices={"invalid_dim": (0, 10)},
        )

        with pytest.raises(InvalidDimNameError):
            type_check_expression(expr, context)

    def test_slice_temporal_data_array(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        time_coords = TimeCoords(
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 3, tzinfo=UTC),
            freq=timedelta(days=1),
        )
        data_array = InferredTemporalDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 3, "lat": 20, "lon": 30},
            time_coords=time_coords,
        )
        context.variables["da"] = data_array

        expr = SliceDataArray(
            func_name="SliceDataArray",
            data_array=VariableReference(name="da"),
            dim_slices={"time": ("2024-01-01", "2024-01-02")},
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredTemporalDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})


class TestFilterDataArrayByPolygon:
    """Test FilterDataArrayByPolygon function."""

    def test_filter_valid_spatial_array(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        data_array = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        polygon = InferredVariable(type="Geometry")
        context.variables.update({"da": data_array, "poly": polygon})

        expr = FilterDataArrayByPolygon(
            func_name="FilterDataArrayByPolygon",
            data_array=VariableReference(name="da"),
            polygon=VariableReference(name="poly"),
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})
        assert result.name is None

    def test_filter_missing_spatial_dims(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        data_array = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "x", "y"}),  # No lat/lon
            shape={"time": 10, "x": 20, "y": 30},
        )
        polygon = InferredVariable(type="Geometry")
        context.variables.update({"da": data_array, "poly": polygon})

        expr = FilterDataArrayByPolygon(
            func_name="FilterDataArrayByPolygon",
            data_array=VariableReference(name="da"),
            polygon=VariableReference(name="poly"),
        )

        with pytest.raises(DataArrayShapeError, match="expected DataArray with 'lat' and 'lon'"):
            type_check_expression(expr, context)


class TestGeocodeNaturalLanguageDescription:
    """Test GeocodeNaturalLanguageDescription function."""

    def test_geocode_returns_geometry(self):
        context = ExecutionContext({}, valid_time_dims={"time"})

        expr = GeocodeNaturalLanguageDescription(
            func_name="GeocodeNaturalLanguageDescription",
            text="Seattle, WA",
        )
        result = type_check_expression(expr, context)

        assert result == InferredVariable(type="Geometry")


class TestAssignUserProvidedGeometry:
    """Test AssignUserProvidedGeometry function."""

    def test_assign_user_provided_geometry(self, gulf_of_mexico_geojson: str):
        context = ExecutionContext({}, valid_time_dims={"time"})

        geometry = parse_geometry_obj(json.loads(gulf_of_mexico_geojson))
        expr = AssignUserProvidedGeometry(func_name="AssignUserProvidedGeometry", geometry=geometry)
        result = type_check_expression(expr, context)

        assert result == InferredVariable(type="Geometry")


class TestGroupByTimeAndReduceDataArray:
    """Test GroupByTimeAndReduceDataArray function."""

    def test_group_by_time_valid_temporal_array(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        time_coords = TimeCoords(
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 2, tzinfo=UTC),
            freq=timedelta(days=1),
        )
        data_array = InferredTemporalDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 2, "lat": 20, "lon": 30},
            time_coords=time_coords,
        )
        context.variables["da"] = data_array

        expr = GroupByTimeAndReduceDataArray(
            func_name="GroupByTimeAndReduceDataArray",
            data_array=VariableReference(name="da"),
            time_unit="month",
            op="mean",
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"month"})


class TestReduceDataArray:
    """Test ReduceDataArray function."""

    def test_reduce_all_dims(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        context.variables["da"] = da

        expr = ReduceDataArray(
            func_name="ReduceDataArray",
            data_array=VariableReference(name="da"),
            op="mean",
            dim=None,
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset()
        assert result.shape == {}

    def test_reduce_single_dim(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        context.variables["da"] = da

        expr = ReduceDataArray(
            func_name="ReduceDataArray",
            data_array=VariableReference(name="da"),
            op="mean",
            dim="time",
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"lat", "lon"})
        assert result.shape == {"lat": 20, "lon": 30}

    def test_reduce_multiple_dims(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        context.variables["da"] = da

        expr = ReduceDataArray(
            func_name="ReduceDataArray",
            data_array=VariableReference(name="da"),
            op="mean",
            dim=("lat", "lon"),
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"time"})
        assert result.shape == {"time": 10}

    def test_reduce_all_dims_explicit(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat"}),
            shape={"time": 10, "lat": 20},
        )
        context.variables["da"] = da

        expr = ReduceDataArray(
            func_name="ReduceDataArray",
            data_array=VariableReference(name="da"),
            op="mean",
            dim=("time", "lat"),
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset()
        assert result.shape == {}

    def test_reduce_invalid_dim(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        context.variables["da"] = da

        expr = ReduceDataArray(
            func_name="ReduceDataArray",
            data_array=VariableReference(name="da"),
            op="mean",
            dim="invalid_dim",
        )

        with pytest.raises(InvalidDimNameError):
            type_check_expression(expr, context)


class TestSelectPointCoordsKey:
    """Test SelectPointCoordsKey function."""

    def test_select_time_key(self):
        context = ExecutionContext(
            {"pc": InferredVariable(type="PointCoords")}, valid_time_dims={"time"}
        )

        expr = SelectPointCoordsKey(
            func_name="SelectPointCoordsKey",
            point_coords=VariableReference(name="pc"),
            key="time",
        )
        result = type_check_expression(expr, context)

        assert result == InferredVariable(type="str")

    @pytest.mark.parametrize("key", ["lat", "lon", "value"])
    def test_select_spatial_keys(self, key: Literal["lat", "lon", "value"]) -> None:
        context = ExecutionContext(
            {"pc": InferredVariable(type="PointCoords")}, valid_time_dims={"time"}
        )

        expr = SelectPointCoordsKey(
            func_name="SelectPointCoordsKey",
            point_coords=VariableReference(name="pc"),
            key=key,
        )
        result = type_check_expression(expr, context)

        assert result == InferredVariable(type="float")

    def test_select_invalid_variable_type(self):
        context = ExecutionContext({"x": InferredVariable(type="int")}, valid_time_dims={"time"})

        expr = SelectPointCoordsKey(
            func_name="SelectPointCoordsKey",
            point_coords=VariableReference(name="x"),
            key="lat",
        )

        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)
