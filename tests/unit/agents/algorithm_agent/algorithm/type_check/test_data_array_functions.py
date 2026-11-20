from datetime import UTC, datetime, timedelta
from typing import Literal

import pytest

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
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
    ReduceDataArray,
    RegridToMatch,
    ResampleTime,
    ResampleTimeToMatch,
    RollingReduce,
    SeasonName,
    SelectPointCoordsKey,
    SliceDataArray,
)
from llm_agent.agents.algorithm_agent.algorithm.type_check import (
    AlgorithmValidationError,
    ArgumentTypeMismatchError,
    DataArrayShapeError,
    ExecutionContext,
    InferredClimatologyVariable,
    InferredDataArrayVariable,
    InferredPointCoordsVariable,
    InferredResolutionlessTemporalDataArrayVariable,
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

        assert result == InferredPointCoordsVariable(
            valid_keys=frozenset({"time", "lat", "lon", "value"})
        )

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

        assert result == InferredPointCoordsVariable(
            valid_keys=frozenset({"time", "lat", "lon", "value"})
        )


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

    def test_slice_single_value_drops_dim(self):
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
            dim_slices={"lat": 5.0},
        )
        result = type_check_expression(expr, context)

        assert result.dims == frozenset({"time", "lon"})
        assert "lat" not in result.shape

    def test_slice_single_index_drops_dim(self):
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
            dim_index_slices={"lon": 3},
        )
        result = type_check_expression(expr, context)

        assert result.dims == frozenset({"time", "lat"})
        assert "lon" not in result.shape

    def test_slice_single_index_out_of_bounds(self):
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
            dim_index_slices={"lat": 25},
        )

        with pytest.raises(DataArrayShapeError, match="out of bounds"):
            type_check_expression(expr, context)

    def test_slice_temporal_single_value_drops_time_dim(self):
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
            dim_slices={"time": "2024-01-02"},
        )
        result = type_check_expression(expr, context)

        # Dropping the time dim downgrades the variable to a plain (non-temporal) DataArray.
        assert isinstance(result, InferredDataArrayVariable)
        assert not isinstance(result, InferredTemporalDataArrayVariable)
        assert result.dims == frozenset({"lat", "lon"})

    def test_slice_temporal_single_value_non_str_raises(self):
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
            dim_slices={"time": 5.0},
        )

        with pytest.raises(ArgumentTypeMismatchError):
            type_check_expression(expr, context)


class TestFilterDataArrayByPolygon:
    """Test FilterDataArrayByPolygon function."""

    def test_filter_valid_spatial_array(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        data_array = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        polygon = InferredVariable(type="geometry")
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
        polygon = InferredVariable(type="geometry")
        context.variables.update({"da": data_array, "poly": polygon})

        expr = FilterDataArrayByPolygon(
            func_name="FilterDataArrayByPolygon",
            data_array=VariableReference(name="da"),
            polygon=VariableReference(name="poly"),
        )

        with pytest.raises(DataArrayShapeError, match="expected DataArray with 'lat' and 'lon'"):
            type_check_expression(expr, context)


class TestRegridToMatch:
    """Test RegridToMatch function."""

    @staticmethod
    def _spatial_da(*, lat: int, lon: int, temporal: bool = False) -> InferredDataArrayVariable:
        if temporal:
            time_coords = TimeCoords(
                start=datetime(2020, 1, 1, tzinfo=UTC),
                end=datetime(2020, 1, 10, tzinfo=UTC),
                freq=timedelta(days=1),
            )
            return InferredTemporalDataArrayVariable(
                type="DataArray",
                dims=frozenset({"time", "lat", "lon"}),
                shape={"time": len(time_coords), "lat": lat, "lon": lon},
                time_coords=time_coords,
            )
        return InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"lat", "lon"}),
            shape={"lat": lat, "lon": lon},
        )

    def test_takes_reference_lat_lon_and_keeps_source_dims(self):
        source = self._spatial_da(lat=180, lon=360, temporal=True)
        reference = self._spatial_da(lat=90, lon=180)
        context = ExecutionContext({"src": source, "ref": reference}, valid_time_dims={"time"})
        expr = RegridToMatch(
            func_name="RegridToMatch",
            source=VariableReference(name="src"),
            reference=VariableReference(name="ref"),
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredTemporalDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})
        assert result.shape == {"time": 10, "lat": 90, "lon": 180}

    def test_source_without_lat_lon_raises(self):
        source = InferredDataArrayVariable(
            type="DataArray", dims=frozenset({"x", "y"}), shape={"x": 10, "y": 10}
        )
        reference = self._spatial_da(lat=90, lon=180)
        context = ExecutionContext({"src": source, "ref": reference})
        expr = RegridToMatch(
            func_name="RegridToMatch",
            source=VariableReference(name="src"),
            reference=VariableReference(name="ref"),
        )
        with pytest.raises(DataArrayShapeError):
            type_check_expression(expr, context)

    def test_reference_without_lat_lon_raises(self):
        source = self._spatial_da(lat=180, lon=360)
        reference = InferredDataArrayVariable(
            type="DataArray", dims=frozenset({"x", "y"}), shape={"x": 10, "y": 10}
        )
        context = ExecutionContext({"src": source, "ref": reference})
        expr = RegridToMatch(
            func_name="RegridToMatch",
            source=VariableReference(name="src"),
            reference=VariableReference(name="ref"),
        )
        with pytest.raises(DataArrayShapeError):
            type_check_expression(expr, context)

    def test_non_data_array_source_raises(self):
        context = ExecutionContext(
            {"src": InferredVariable(type="int"), "ref": self._spatial_da(lat=90, lon=180)}
        )
        expr = RegridToMatch(
            func_name="RegridToMatch",
            source=VariableReference(name="src"),
            reference=VariableReference(name="ref"),
        )
        with pytest.raises(VariableTypeMismatchError):
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

        assert result == InferredVariable(type="geometry")


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


class TestComputeClimatology:
    """Test ComputeClimatology function."""

    @staticmethod
    def _full_year_temporal_da() -> InferredTemporalDataArrayVariable:
        time_coords = TimeCoords(
            start=datetime(2020, 1, 1, tzinfo=UTC),
            end=datetime(2020, 12, 31, tzinfo=UTC),
            freq=timedelta(days=1),
        )
        return InferredTemporalDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": len(time_coords), "lat": 20, "lon": 30},
            time_coords=time_coords,
        )

    def test_month_climatology_preserves_spatial_dims(self):
        context = ExecutionContext({"da": self._full_year_temporal_da()}, valid_time_dims={"time"})
        expr = ComputeClimatology(
            func_name="ComputeClimatology",
            data_array=VariableReference(name="da"),
            period_type="month",
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredClimatologyVariable)
        assert result.dims == frozenset({"month", "lat", "lon"})
        assert result.shape == {"month": 12, "lat": 20, "lon": 30}
        assert result.period_type == "month"
        assert result.period_dim == "month"

    def test_season_climatology_has_four_groups(self):
        context = ExecutionContext({"da": self._full_year_temporal_da()}, valid_time_dims={"time"})
        expr = ComputeClimatology(
            func_name="ComputeClimatology",
            data_array=VariableReference(name="da"),
            period_type="season",
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredClimatologyVariable)
        assert result.dims == frozenset({"season", "lat", "lon"})
        assert result.shape == {"season": 4, "lat": 20, "lon": 30}

    def test_non_temporal_array_yields_unknown_period_size(self):
        # A DataArray with a time dim but no TimeCoords (temporal resolution unknown).
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 100, "lat": 20, "lon": 30},
        )
        context = ExecutionContext({"da": da}, valid_time_dims={"time"})
        expr = ComputeClimatology(
            func_name="ComputeClimatology",
            data_array=VariableReference(name="da"),
            period_type="month",
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredClimatologyVariable)
        assert result.shape == {"month": "unknown", "lat": 20, "lon": 30}

    def test_invalid_time_dim(self):
        context = ExecutionContext({"da": self._full_year_temporal_da()}, valid_time_dims={"time"})
        expr = ComputeClimatology(
            func_name="ComputeClimatology",
            data_array=VariableReference(name="da"),
            period_type="month",
            time_dim="not_a_dim",
        )
        with pytest.raises(InvalidDimNameError):
            type_check_expression(expr, context)

    def test_non_data_array_input(self):
        context = ExecutionContext({"x": InferredVariable(type="int")})
        expr = ComputeClimatology(
            func_name="ComputeClimatology",
            data_array=VariableReference(name="x"),
            period_type="month",
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)


class TestComputeAnomaly:
    """Test ComputeAnomaly function."""

    @staticmethod
    def _temporal_da() -> InferredTemporalDataArrayVariable:
        time_coords = TimeCoords(
            start=datetime(2020, 1, 1, tzinfo=UTC),
            end=datetime(2021, 12, 31, tzinfo=UTC),
            freq=timedelta(days=1),
        )
        return InferredTemporalDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": len(time_coords), "lat": 20, "lon": 30},
            time_coords=time_coords,
        )

    @staticmethod
    def _month_clim(*, lat: int = 20, lon: int = 30) -> InferredClimatologyVariable:
        return InferredClimatologyVariable(
            dims=frozenset({"month", "lat", "lon"}),
            shape={"month": 12, "lat": lat, "lon": lon},
            period_type="month",
            period_dim="month",
        )

    def test_anomaly_preserves_input_temporal_type(self):
        context = ExecutionContext(
            {"da": self._temporal_da(), "clim": self._month_clim()}, valid_time_dims={"time"}
        )
        expr = ComputeAnomaly(
            func_name="ComputeAnomaly",
            data_array=VariableReference(name="da"),
            climatology=VariableReference(name="clim"),
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredTemporalDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})
        assert result.shape == {"time": 731, "lat": 20, "lon": 30}

    def test_dim_mismatch_raises(self):
        clim = InferredClimatologyVariable(
            dims=frozenset({"month", "lat"}),  # missing lon
            shape={"month": 12, "lat": 20},
            period_type="month",
            period_dim="month",
        )
        context = ExecutionContext(
            {"da": self._temporal_da(), "clim": clim}, valid_time_dims={"time"}
        )
        expr = ComputeAnomaly(
            func_name="ComputeAnomaly",
            data_array=VariableReference(name="da"),
            climatology=VariableReference(name="clim"),
        )
        with pytest.raises(AlgorithmValidationError):
            type_check_expression(expr, context)

    def test_size_mismatch_raises(self):
        context = ExecutionContext(
            {"da": self._temporal_da(), "clim": self._month_clim(lat=99)},
            valid_time_dims={"time"},
        )
        expr = ComputeAnomaly(
            func_name="ComputeAnomaly",
            data_array=VariableReference(name="da"),
            climatology=VariableReference(name="clim"),
        )
        with pytest.raises(AlgorithmValidationError):
            type_check_expression(expr, context)

    def test_climatology_arg_must_be_a_climatology(self):
        plain_da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"month", "lat", "lon"}),
            shape={"month": 12, "lat": 20, "lon": 30},
        )
        context = ExecutionContext(
            {"da": self._temporal_da(), "clim": plain_da}, valid_time_dims={"time"}
        )
        expr = ComputeAnomaly(
            func_name="ComputeAnomaly",
            data_array=VariableReference(name="da"),
            climatology=VariableReference(name="clim"),
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)

    def test_non_data_array_input(self):
        context = ExecutionContext(
            {"x": InferredVariable(type="int"), "clim": self._month_clim()},
            valid_time_dims={"time"},
        )
        expr = ComputeAnomaly(
            func_name="ComputeAnomaly",
            data_array=VariableReference(name="x"),
            climatology=VariableReference(name="clim"),
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)

    def test_standardized_anomaly_valid(self):
        context = ExecutionContext(
            {
                "da": self._temporal_da(),
                "cmean": self._month_clim(),
                "cstd": self._month_clim(),
            },
            valid_time_dims={"time"},
        )
        expr = ComputeAnomaly(
            func_name="ComputeAnomaly",
            data_array=VariableReference(name="da"),
            climatology=VariableReference(name="cmean"),
            climatology_std=VariableReference(name="cstd"),
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredTemporalDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})

    def test_std_climatology_period_mismatch_raises(self):
        season_clim = InferredClimatologyVariable(
            dims=frozenset({"season", "lat", "lon"}),
            shape={"season": 4, "lat": 20, "lon": 30},
            period_type="season",
            period_dim="season",
        )
        context = ExecutionContext(
            {
                "da": self._temporal_da(),
                "cmean": self._month_clim(),
                "cstd": season_clim,
            },
            valid_time_dims={"time"},
        )
        expr = ComputeAnomaly(
            func_name="ComputeAnomaly",
            data_array=VariableReference(name="da"),
            climatology=VariableReference(name="cmean"),
            climatology_std=VariableReference(name="cstd"),
        )
        with pytest.raises(AlgorithmValidationError):
            type_check_expression(expr, context)


class TestComputeLinearTrend:
    """Test ComputeLinearTrend function."""

    def test_slope_map_removes_time_dim(self):
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 100, "lat": 20, "lon": 30},
        )
        context = ExecutionContext({"da": da}, valid_time_dims={"time"})
        expr = ComputeLinearTrend(
            func_name="ComputeLinearTrend",
            data_array=VariableReference(name="da"),
            time_unit="decade",
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"lat", "lon"})
        assert result.shape == {"lat": 20, "lon": 30}

    def test_pure_time_series_yields_scalar(self):
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time"}),
            shape={"time": 100},
        )
        context = ExecutionContext({"da": da}, valid_time_dims={"time"})
        expr = ComputeLinearTrend(
            func_name="ComputeLinearTrend",
            data_array=VariableReference(name="da"),
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredVariable)
        assert result.type == "float"

    def test_invalid_time_dim(self):
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 100, "lat": 20, "lon": 30},
        )
        context = ExecutionContext({"da": da}, valid_time_dims={"time"})
        expr = ComputeLinearTrend(
            func_name="ComputeLinearTrend",
            data_array=VariableReference(name="da"),
            time_dim="not_a_dim",
        )
        with pytest.raises(InvalidDimNameError):
            type_check_expression(expr, context)

    def test_non_data_array_input(self):
        context = ExecutionContext({"x": InferredVariable(type="int")})
        expr = ComputeLinearTrend(
            func_name="ComputeLinearTrend",
            data_array=VariableReference(name="x"),
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)


class TestComputeTrendPValue:
    """Test ComputeTrendPValue function."""

    def test_pvalue_map_removes_time_dim(self):
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 100, "lat": 20, "lon": 30},
        )
        context = ExecutionContext({"da": da}, valid_time_dims={"time"})
        expr = ComputeTrendPValue(
            func_name="ComputeTrendPValue", data_array=VariableReference(name="da")
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"lat", "lon"})
        assert result.shape == {"lat": 20, "lon": 30}

    def test_pure_time_series_yields_scalar(self):
        da = InferredDataArrayVariable(
            type="DataArray", dims=frozenset({"time"}), shape={"time": 100}
        )
        context = ExecutionContext({"da": da}, valid_time_dims={"time"})
        expr = ComputeTrendPValue(
            func_name="ComputeTrendPValue", data_array=VariableReference(name="da")
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredVariable)
        assert result.type == "float"

    def test_invalid_time_dim(self):
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 100, "lat": 20, "lon": 30},
        )
        context = ExecutionContext({"da": da}, valid_time_dims={"time"})
        expr = ComputeTrendPValue(
            func_name="ComputeTrendPValue",
            data_array=VariableReference(name="da"),
            time_dim="not_a_dim",
        )
        with pytest.raises(InvalidDimNameError):
            type_check_expression(expr, context)

    def test_non_data_array_input(self):
        context = ExecutionContext({"x": InferredVariable(type="int")})
        expr = ComputeTrendPValue(
            func_name="ComputeTrendPValue", data_array=VariableReference(name="x")
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)


class TestMaskWhere:
    """Test MaskWhere function."""

    @staticmethod
    def _temporal_da() -> InferredTemporalDataArrayVariable:
        time_coords = TimeCoords(
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 5, tzinfo=UTC),
            freq=timedelta(days=1),
        )
        return InferredTemporalDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": len(time_coords), "lat": 20, "lon": 30},
            time_coords=time_coords,
        )

    def test_preserves_temporal_type_and_shape(self):
        cond = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 5, "lat": 20, "lon": 30},
        )
        context = ExecutionContext(
            {"da": self._temporal_da(), "cond": cond}, valid_time_dims={"time"}
        )
        expr = MaskWhere(
            func_name="MaskWhere",
            data_array=VariableReference(name="da"),
            condition=VariableReference(name="cond"),
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredTemporalDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})
        assert result.shape == {"time": 5, "lat": 20, "lon": 30}

    def test_lower_dim_condition_is_allowed(self):
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 5, "lat": 20, "lon": 30},
        )
        cond = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"lat", "lon"}),
            shape={"lat": 20, "lon": 30},
        )
        context = ExecutionContext({"da": da, "cond": cond})
        expr = MaskWhere(
            func_name="MaskWhere",
            data_array=VariableReference(name="da"),
            condition=VariableReference(name="cond"),
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})

    def test_condition_with_extra_dim_raises(self):
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"lat", "lon"}),
            shape={"lat": 20, "lon": 30},
        )
        cond = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"lat", "lon", "depth"}),
            shape={"lat": 20, "lon": 30, "depth": 5},
        )
        context = ExecutionContext({"da": da, "cond": cond})
        expr = MaskWhere(
            func_name="MaskWhere",
            data_array=VariableReference(name="da"),
            condition=VariableReference(name="cond"),
        )
        with pytest.raises(AlgorithmValidationError):
            type_check_expression(expr, context)

    def test_condition_size_mismatch_raises(self):
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"lat", "lon"}),
            shape={"lat": 20, "lon": 30},
        )
        cond = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"lat", "lon"}),
            shape={"lat": 10, "lon": 30},  # lat size differs
        )
        context = ExecutionContext({"da": da, "cond": cond})
        expr = MaskWhere(
            func_name="MaskWhere",
            data_array=VariableReference(name="da"),
            condition=VariableReference(name="cond"),
        )
        with pytest.raises(AlgorithmValidationError):
            type_check_expression(expr, context)

    def test_non_data_array_condition_raises(self):
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"lat", "lon"}),
            shape={"lat": 20, "lon": 30},
        )
        context = ExecutionContext({"da": da, "cond": InferredVariable(type="int")})
        expr = MaskWhere(
            func_name="MaskWhere",
            data_array=VariableReference(name="da"),
            condition=VariableReference(name="cond"),
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)


class TestFilterToTimePeriodAcrossYears:
    """Test FilterToTimePeriodAcrossYears function."""

    @classmethod
    def setup_class(cls) -> None:
        time_coords = TimeCoords(
            start=datetime(2020, 1, 1, tzinfo=UTC),
            end=datetime(2024, 12, 31, tzinfo=UTC),
            freq=timedelta(days=1),
        )
        data_array = InferredTemporalDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": len(time_coords), "lat": 20, "lon": 30},
            time_coords=time_coords,
        )
        cls.context = ExecutionContext({"da": data_array}, valid_time_dims={"time"})

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
    def test_filter_to_time_period_valid_temporal_array(
        self,
        period_type: Literal["month", "season", "week", "dayofyear"],
        period_value: int | list[int] | SeasonName | list[SeasonName],
    ):
        expr = FilterToTimePeriodAcrossYears(
            func_name="FilterToTimePeriodAcrossYears",
            data_array=VariableReference(name="da"),
            time_dim="time",
            period_type=period_type,
            period_value=period_value,
        )
        result = type_check_expression(expr, self.context)

        assert isinstance(result, InferredTemporalDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})

    def test_resolutionless_temporal_array_passes_with_unknown_time_size(self):
        # A temporal array with no resolution is accepted, but the time size stays unknown.
        da = InferredResolutionlessTemporalDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": "unknown", "lat": 20, "lon": 30},
        )
        context = ExecutionContext({"da": da}, valid_time_dims={"time"})
        expr = FilterToTimePeriodAcrossYears(
            func_name="FilterToTimePeriodAcrossYears",
            data_array=VariableReference(name="da"),
            time_dim="time",
            period_type="season",
            period_value="JJA",
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredResolutionlessTemporalDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})
        assert result.shape == {"time": "unknown", "lat": 20, "lon": 30}

    def test_plain_data_array_still_rejected(self):
        # A plain DataArray with no temporal marker at all is still rejected.
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 100, "lat": 20, "lon": 30},
        )
        context = ExecutionContext({"da": da}, valid_time_dims={"time"})
        expr = FilterToTimePeriodAcrossYears(
            func_name="FilterToTimePeriodAcrossYears",
            data_array=VariableReference(name="da"),
            time_dim="time",
            period_type="month",
            period_value=1,
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)


class TestResolutionlessTemporalDataArray:
    """A temporal DataArray with unknown resolution passes temporal checks with unknown sizes."""

    @staticmethod
    def _da() -> InferredResolutionlessTemporalDataArrayVariable:
        return InferredResolutionlessTemporalDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": "unknown", "lat": 20, "lon": 30},
        )

    def test_climatology_yields_unknown_period_size(self):
        context = ExecutionContext({"da": self._da()}, valid_time_dims={"time"})
        expr = ComputeClimatology(
            func_name="ComputeClimatology",
            data_array=VariableReference(name="da"),
            period_type="month",
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredClimatologyVariable)
        assert result.shape == {"month": "unknown", "lat": 20, "lon": 30}

    def test_groupby_yields_unknown_size(self):
        context = ExecutionContext({"da": self._da()}, valid_time_dims={"time"})
        expr = GroupByTimeAndReduceDataArray(
            func_name="GroupByTimeAndReduceDataArray",
            data_array=VariableReference(name="da"),
            op="mean",
            time_unit="month",
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert result.shape == {"month": "unknown"}

    def test_slice_single_time_value_drops_to_plain(self):
        context = ExecutionContext({"da": self._da()}, valid_time_dims={"time"})
        expr = SliceDataArray(
            func_name="SliceDataArray",
            data_array=VariableReference(name="da"),
            dim_slices={"time": "2024-01-01"},
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert not isinstance(result, InferredResolutionlessTemporalDataArrayVariable)
        assert result.dims == frozenset({"lat", "lon"})

    def test_slice_time_range_preserves_resolutionless(self):
        context = ExecutionContext({"da": self._da()}, valid_time_dims={"time"})
        expr = SliceDataArray(
            func_name="SliceDataArray",
            data_array=VariableReference(name="da"),
            dim_slices={"time": ("2024-01-01", "2024-12-31")},
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredResolutionlessTemporalDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})


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

        assert isinstance(result, InferredVariable)
        assert result.type == "float"

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

        assert isinstance(result, InferredVariable)
        assert result.type == "float"

    def test_returns_scalar_if_all_remaining_dims_squeezable(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 1, "lat": 4, "lon": 4},
        )
        context.variables["da"] = da

        expr = ReduceDataArray(
            func_name="ReduceDataArray",
            data_array=VariableReference(name="da"),
            op="mean",
            dim=("lat", "lon"),
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredVariable)
        assert result.type == "float"

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

    def test_reduce_median_single_dim(self):
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
            op="median",
            dim="time",
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"lat", "lon"})
        assert result.shape == {"lat": 20, "lon": 30}

    def test_reduce_quantile_all_dims_is_scalar(self):
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
            op="quantile",
            dim=None,
            q=0.9,
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredVariable)
        assert result.type == "float"


class TestRollingReduce:
    """Test RollingReduce function."""

    @staticmethod
    def _temporal_da() -> InferredTemporalDataArrayVariable:
        time_coords = TimeCoords(
            start=datetime(2020, 1, 1, tzinfo=UTC),
            end=datetime(2020, 1, 20, tzinfo=UTC),
            freq=timedelta(days=1),
        )
        return InferredTemporalDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": len(time_coords), "lat": 20, "lon": 30},
            time_coords=time_coords,
        )

    def test_preserves_temporal_type_and_shape(self):
        context = ExecutionContext({"da": self._temporal_da()}, valid_time_dims={"time"})
        expr = RollingReduce(
            func_name="RollingReduce",
            data_array=VariableReference(name="da"),
            dim="time",
            window=3,
            op="mean",
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredTemporalDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})
        assert result.shape == {"time": 20, "lat": 20, "lon": 30}

    def test_window_exceeding_dim_size_raises(self):
        context = ExecutionContext({"da": self._temporal_da()}, valid_time_dims={"time"})
        expr = RollingReduce(
            func_name="RollingReduce",
            data_array=VariableReference(name="da"),
            dim="time",
            window=100,
            op="mean",
        )
        with pytest.raises(DataArrayShapeError):
            type_check_expression(expr, context)

    def test_invalid_dim(self):
        context = ExecutionContext({"da": self._temporal_da()}, valid_time_dims={"time"})
        expr = RollingReduce(
            func_name="RollingReduce",
            data_array=VariableReference(name="da"),
            dim="not_a_dim",
            window=3,
            op="mean",
        )
        with pytest.raises(InvalidDimNameError):
            type_check_expression(expr, context)

    def test_non_data_array_input(self):
        context = ExecutionContext({"x": InferredVariable(type="int")})
        expr = RollingReduce(
            func_name="RollingReduce",
            data_array=VariableReference(name="x"),
            dim="time",
            window=3,
            op="mean",
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)


class TestCumulativeReduce:
    """Test CumulativeReduce function."""

    @staticmethod
    def _temporal_da() -> InferredTemporalDataArrayVariable:
        time_coords = TimeCoords(
            start=datetime(2020, 1, 1, tzinfo=UTC),
            end=datetime(2020, 1, 20, tzinfo=UTC),
            freq=timedelta(days=1),
        )
        return InferredTemporalDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": len(time_coords), "lat": 20, "lon": 30},
            time_coords=time_coords,
        )

    def test_preserves_temporal_type_and_shape(self):
        context = ExecutionContext({"da": self._temporal_da()}, valid_time_dims={"time"})
        expr = CumulativeReduce(
            func_name="CumulativeReduce",
            data_array=VariableReference(name="da"),
            dim="time",
            op="sum",
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredTemporalDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})
        assert result.shape == {"time": 20, "lat": 20, "lon": 30}

    def test_invalid_dim(self):
        context = ExecutionContext({"da": self._temporal_da()}, valid_time_dims={"time"})
        expr = CumulativeReduce(
            func_name="CumulativeReduce",
            data_array=VariableReference(name="da"),
            dim="not_a_dim",
            op="sum",
        )
        with pytest.raises(InvalidDimNameError):
            type_check_expression(expr, context)

    def test_non_data_array_input(self):
        context = ExecutionContext({"x": InferredVariable(type="int")})
        expr = CumulativeReduce(
            func_name="CumulativeReduce",
            data_array=VariableReference(name="x"),
            dim="time",
            op="sum",
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)


class TestCorrelateDataArrays:
    """Test CorrelateDataArrays function."""

    @staticmethod
    def _da(
        dims: set[str], shape: dict[str, int | Literal["unknown"]]
    ) -> InferredDataArrayVariable:
        return InferredDataArrayVariable(type="DataArray", dims=frozenset(dims), shape=shape)

    def test_two_fields_yield_map(self):
        a = self._da({"time", "lat", "lon"}, {"time": 10, "lat": 4, "lon": 5})
        b = self._da({"time", "lat", "lon"}, {"time": 10, "lat": 4, "lon": 5})
        context = ExecutionContext({"a": a, "b": b}, valid_time_dims={"time"})
        expr = CorrelateDataArrays(
            func_name="CorrelateDataArrays",
            a=VariableReference(name="a"),
            b=VariableReference(name="b"),
            dim="time",
        )
        result = type_check_expression(expr, context)
        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"lat", "lon"})
        assert result.shape == {"lat": 4, "lon": 5}

    def test_field_vs_index_broadcasts(self):
        a = self._da({"time", "lat", "lon"}, {"time": 10, "lat": 4, "lon": 5})
        index = self._da({"time"}, {"time": 10})
        context = ExecutionContext({"a": a, "idx": index}, valid_time_dims={"time"})
        expr = CorrelateDataArrays(
            func_name="CorrelateDataArrays",
            a=VariableReference(name="a"),
            b=VariableReference(name="idx"),
            dim="time",
        )
        result = type_check_expression(expr, context)
        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"lat", "lon"})

    def test_two_series_yield_scalar(self):
        a = self._da({"time"}, {"time": 10})
        b = self._da({"time"}, {"time": 10})
        context = ExecutionContext({"a": a, "b": b}, valid_time_dims={"time"})
        expr = CorrelateDataArrays(
            func_name="CorrelateDataArrays",
            a=VariableReference(name="a"),
            b=VariableReference(name="b"),
            dim="time",
        )
        result = type_check_expression(expr, context)
        assert isinstance(result, InferredVariable)
        assert result.type == "float"

    def test_missing_dim_raises(self):
        a = self._da({"time", "lat", "lon"}, {"time": 10, "lat": 4, "lon": 5})
        b = self._da({"lat", "lon"}, {"lat": 4, "lon": 5})
        context = ExecutionContext({"a": a, "b": b}, valid_time_dims={"time"})
        expr = CorrelateDataArrays(
            func_name="CorrelateDataArrays",
            a=VariableReference(name="a"),
            b=VariableReference(name="b"),
            dim="time",
        )
        with pytest.raises(InvalidDimNameError):
            type_check_expression(expr, context)

    def test_shared_dim_size_mismatch_raises(self):
        a = self._da({"time", "lat", "lon"}, {"time": 10, "lat": 4, "lon": 5})
        b = self._da({"time", "lat", "lon"}, {"time": 10, "lat": 8, "lon": 5})
        context = ExecutionContext({"a": a, "b": b}, valid_time_dims={"time"})
        expr = CorrelateDataArrays(
            func_name="CorrelateDataArrays",
            a=VariableReference(name="a"),
            b=VariableReference(name="b"),
            dim="time",
        )
        with pytest.raises(DataArrayShapeError):
            type_check_expression(expr, context)

    def test_correlation_dim_size_mismatch_raises(self):
        # Daily (many steps) vs monthly (few steps) over the same period -> different 'time' sizes.
        a = self._da({"time", "lat", "lon"}, {"time": 365, "lat": 4, "lon": 5})
        b = self._da({"time", "lat", "lon"}, {"time": 12, "lat": 4, "lon": 5})
        context = ExecutionContext({"a": a, "b": b}, valid_time_dims={"time"})
        expr = CorrelateDataArrays(
            func_name="CorrelateDataArrays",
            a=VariableReference(name="a"),
            b=VariableReference(name="b"),
            dim="time",
        )
        with pytest.raises(DataArrayShapeError, match="different sizes"):
            type_check_expression(expr, context)

    def test_non_data_array_raises(self):
        a = self._da({"time"}, {"time": 10})
        context = ExecutionContext(
            {"a": a, "b": InferredVariable(type="int")}, valid_time_dims={"time"}
        )
        expr = CorrelateDataArrays(
            func_name="CorrelateDataArrays",
            a=VariableReference(name="a"),
            b=VariableReference(name="b"),
            dim="time",
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)


class TestComputePercentileRank:
    """Test ComputePercentileRank function."""

    @staticmethod
    def _da(
        dims: set[str], shape: dict[str, int | Literal["unknown"]]
    ) -> InferredDataArrayVariable:
        return InferredDataArrayVariable(type="DataArray", dims=frozenset(dims), shape=shape)

    def test_preserves_dims_and_shape(self):
        da = self._da({"time", "lat", "lon"}, {"time": 44, "lat": 4, "lon": 5})
        context = ExecutionContext({"da": da}, valid_time_dims={"time"})
        expr = ComputePercentileRank(
            func_name="ComputePercentileRank", data_array=VariableReference(name="da")
        )
        result = type_check_expression(expr, context)
        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})
        assert result.shape == {"time": 44, "lat": 4, "lon": 5}

    def test_missing_time_dim_raises(self):
        da = self._da({"lat", "lon"}, {"lat": 4, "lon": 5})
        context = ExecutionContext({"da": da}, valid_time_dims={"time"})
        expr = ComputePercentileRank(
            func_name="ComputePercentileRank", data_array=VariableReference(name="da")
        )
        with pytest.raises(InvalidDimNameError):
            type_check_expression(expr, context)

    def test_non_data_array_raises(self):
        context = ExecutionContext({"da": InferredVariable(type="int")}, valid_time_dims={"time"})
        expr = ComputePercentileRank(
            func_name="ComputePercentileRank", data_array=VariableReference(name="da")
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)


class TestResampleTime:
    """Test ResampleTime function."""

    @staticmethod
    def _da(
        dims: set[str], shape: dict[str, int | Literal["unknown"]]
    ) -> InferredDataArrayVariable:
        return InferredDataArrayVariable(type="DataArray", dims=frozenset(dims), shape=shape)

    def test_preserves_dims_with_unknown_time_size(self):
        da = self._da({"time", "lat", "lon"}, {"time": 365, "lat": 4, "lon": 5})
        context = ExecutionContext({"da": da}, valid_time_dims={"time"})
        expr = ResampleTime(
            func_name="ResampleTime",
            data_array=VariableReference(name="da"),
            target_frequency="month",
            op="mean",
        )
        result = type_check_expression(expr, context)
        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})
        assert result.shape == {"time": "unknown", "lat": 4, "lon": 5}

    def test_missing_time_dim_raises(self):
        da = self._da({"lat", "lon"}, {"lat": 4, "lon": 5})
        context = ExecutionContext({"da": da}, valid_time_dims={"time"})
        expr = ResampleTime(
            func_name="ResampleTime",
            data_array=VariableReference(name="da"),
            target_frequency="month",
            op="mean",
        )
        with pytest.raises(InvalidDimNameError):
            type_check_expression(expr, context)

    def test_non_data_array_raises(self):
        context = ExecutionContext({"da": InferredVariable(type="int")}, valid_time_dims={"time"})
        expr = ResampleTime(
            func_name="ResampleTime",
            data_array=VariableReference(name="da"),
            target_frequency="month",
            op="mean",
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)


class TestResampleTimeToMatch:
    """Test ResampleTimeToMatch function."""

    @staticmethod
    def _da(
        dims: set[str], shape: dict[str, int | Literal["unknown"]]
    ) -> InferredDataArrayVariable:
        return InferredDataArrayVariable(type="DataArray", dims=frozenset(dims), shape=shape)

    def test_output_takes_reference_time_size(self):
        source = self._da({"time", "lat", "lon"}, {"time": 365, "lat": 4, "lon": 5})
        reference = self._da({"time"}, {"time": 12})
        context = ExecutionContext({"s": source, "r": reference}, valid_time_dims={"time"})
        expr = ResampleTimeToMatch(
            func_name="ResampleTimeToMatch",
            source=VariableReference(name="s"),
            reference=VariableReference(name="r"),
            op="mean",
        )
        result = type_check_expression(expr, context)
        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})
        assert result.shape == {"time": 12, "lat": 4, "lon": 5}

    def test_source_missing_time_raises(self):
        source = self._da({"lat", "lon"}, {"lat": 4, "lon": 5})
        reference = self._da({"time"}, {"time": 12})
        context = ExecutionContext({"s": source, "r": reference}, valid_time_dims={"time"})
        expr = ResampleTimeToMatch(
            func_name="ResampleTimeToMatch",
            source=VariableReference(name="s"),
            reference=VariableReference(name="r"),
            op="mean",
        )
        with pytest.raises(InvalidDimNameError):
            type_check_expression(expr, context)

    def test_non_data_array_raises(self):
        reference = self._da({"time"}, {"time": 12})
        context = ExecutionContext(
            {"s": InferredVariable(type="int"), "r": reference}, valid_time_dims={"time"}
        )
        expr = ResampleTimeToMatch(
            func_name="ResampleTimeToMatch",
            source=VariableReference(name="s"),
            reference=VariableReference(name="r"),
            op="mean",
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)


class TestSelectPointCoordsKey:
    """Test SelectPointCoordsKey function."""

    def test_select_time_key(self):
        context = ExecutionContext(
            {
                "pc": InferredPointCoordsVariable(
                    valid_keys=frozenset({"time", "lat", "lon", "value"})
                )
            },
            valid_time_dims={"time"},
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
            {
                "pc": InferredPointCoordsVariable(
                    valid_keys=frozenset({"time", "lat", "lon", "value"})
                )
            },
            valid_time_dims={"time"},
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

    def test_select_invalid_key(self):
        context = ExecutionContext(
            {"pc": InferredPointCoordsVariable(valid_keys=frozenset({"lat", "lon", "value"}))},
            valid_time_dims={"time"},
        )

        expr = SelectPointCoordsKey(
            func_name="SelectPointCoordsKey",
            point_coords=VariableReference(name="pc"),
            key="invalid_key",
        )

        with pytest.raises(
            AlgorithmValidationError, match="Invalid key 'invalid_key' for PointCoords variable"
        ):
            type_check_expression(expr, context)
