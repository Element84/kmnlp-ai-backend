import pytest

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
    PlotDataArraysAsRaster,
    PlotDataArraysAsTimeSeries,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.general_plot_functions import (
    PlotBarChart,
)
from llm_agent.agents.algorithm_agent.algorithm.type_check import (
    ArgumentTypeMismatchError,
    DataArrayShapeError,
    ExecutionContext,
    InferredDataArrayVariable,
    InferredVariable,
    type_check_expression,
)


class TestPlotFunctions:
    """Test plotting functions."""

    def test_plot_bar_chart_valid(self):
        context = ExecutionContext(
            {
                "h1": InferredVariable(type="int"),
                "h2": InferredVariable(type="float"),
            }
        )

        expr = PlotBarChart(
            func_name="PlotBarChart",
            labels=["A", "B"],
            heights=[VariableReference(name="h1"), VariableReference(name="h2")],
            title="Test Chart",
        )
        result = type_check_expression(expr, context)

        assert result == InferredVariable(type="BarChart")

    def test_plot_bar_chart_invalid_height_type(self):
        context = ExecutionContext({"s": InferredVariable(type="str")})

        expr = PlotBarChart(
            func_name="PlotBarChart",
            labels=["A"],
            heights=[VariableReference(name="s")],
            title="Test Chart",
        )

        with pytest.raises(ArgumentTypeMismatchError):
            type_check_expression(expr, context)

    def test_plot_data_arrays_as_time_series_valid(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        da1 = InferredDataArrayVariable(
            type="DataArray", dims=frozenset({"time"}), shape={"time": 10}
        )
        da2 = InferredDataArrayVariable(
            type="DataArray", dims=frozenset({"time"}), shape={"time": 10}
        )
        context.variables.update({"da1": da1, "da2": da2})

        expr = PlotDataArraysAsTimeSeries(
            func_name="PlotDataArraysAsTimeSeries",
            data_arrays=[VariableReference(name="da1"), VariableReference(name="da2")],
            labels=["Series 1", "Series 2"],
            title="Time Series",
        )
        result = type_check_expression(expr, context)

        assert result == InferredVariable(type="LinePlot")

    def test_plot_data_arrays_as_time_series_invalid_dims(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        da = InferredDataArrayVariable(
            type="DataArray", dims=frozenset({"time", "lat"}), shape={"time": 10, "lat": 20}
        )
        context.variables["da"] = da

        expr = PlotDataArraysAsTimeSeries(
            func_name="PlotDataArraysAsTimeSeries",
            data_arrays=[VariableReference(name="da")],
            labels=["Series 1"],
            title="Time Series",
        )

        with pytest.raises(DataArrayShapeError, match="expected 1D DataArray"):
            type_check_expression(expr, context)

    def test_plot_data_arrays_as_raster_valid_grid(self):
        context = ExecutionContext({})
        da1 = InferredDataArrayVariable(
            type="DataArray", dims=frozenset({"lat", "lon"}), shape={"lat": 20, "lon": 30}
        )
        da2 = InferredDataArrayVariable(
            type="DataArray", dims=frozenset({"lat", "lon"}), shape={"lat": 20, "lon": 30}
        )
        context.variables.update({"da1": da1, "da2": da2})

        expr = PlotDataArraysAsRaster(
            func_name="PlotDataArraysAsRaster",
            data_arrays=[VariableReference(name="da1"), VariableReference(name="da2")],
            subplot_grid_shape=(1, 2),  # 1 row, 2 cols = 2 plots total
            title="Raster Plots",
        )
        result = type_check_expression(expr, context)

        assert result == InferredVariable(type="RasterPlot")

    def test_plot_data_arrays_as_raster_mismatched_grid_size(self):
        context = ExecutionContext({})
        da = InferredDataArrayVariable(
            type="DataArray", dims=frozenset({"lat", "lon"}), shape={"lat": 20, "lon": 30}
        )
        context.variables["da"] = da

        expr = PlotDataArraysAsRaster(
            func_name="PlotDataArraysAsRaster",
            data_arrays=[VariableReference(name="da")],  # 1 array
            subplot_grid_shape=(2, 2),  # 2x2 = 4 plots expected
            title="Raster Plots",
        )

        with pytest.raises(ArgumentTypeMismatchError, match="list of length 4"):
            type_check_expression(expr, context)
