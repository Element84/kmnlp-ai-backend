import pytest

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    HistogramPlotSpec,
    MakePlot,
    ScatterPlotSpec,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.contour import (
    ContourMapPlotSpec,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.ensemble import (
    EnsemblePlumePlotSpec,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.wind import WindVectorPlotSpec
from llm_agent.agents.algorithm_agent.algorithm.type_check import (
    ExecutionContext,
    InferredDataArrayVariable,
    InferredListVariable,
    InferredVariable,
    VariableTypeMismatchError,
    type_check_expression,
)


class TestBasicTypeInference:
    """Test basic type inference for primitive types."""

    def test_str_literal(self):
        context = ExecutionContext({})
        result = type_check_expression("hello", context)
        assert result == InferredVariable(type="str")

    def test_int_literal(self):
        context = ExecutionContext({})
        result = type_check_expression(42, context)
        assert result == InferredVariable(type="int")

    def test_float_literal(self):
        context = ExecutionContext({})
        result = type_check_expression(3.14, context)
        assert result == InferredVariable(type="float")

    def test_list_literal(self):
        context = ExecutionContext({})
        result = type_check_expression([1, 2, 3], context)
        expected = InferredListVariable(
            type="list",
            items=[
                InferredVariable(type="int"),
                InferredVariable(type="int"),
                InferredVariable(type="int"),
            ],
        )
        assert result == expected

    def test_mixed_list_literal(self):
        context = ExecutionContext({})
        result = type_check_expression([1, "hello", 3.14], context)
        expected = InferredListVariable(
            type="list",
            items=[
                InferredVariable(type="int"),
                InferredVariable(type="str"),
                InferredVariable(type="float"),
            ],
        )
        assert result == expected


class TestPlotTypeInference:
    """Test type inference for plot specs."""

    def test_histogram_plot_infers_histogram_type(self):
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"lat", "lon"}),
            shape={"lat": 4, "lon": 4},
        )
        context = ExecutionContext({"da": da})
        expr = MakePlot(
            func_name="MakePlot",
            plot_spec=HistogramPlotSpec(data_array=VariableReference(name="da")),
        )
        result = type_check_expression(expr, context)
        assert result == InferredVariable(type="histogram")

    def test_histogram_requires_a_data_array(self):
        context = ExecutionContext({"x": InferredVariable(type="int")})
        expr = MakePlot(
            func_name="MakePlot",
            plot_spec=HistogramPlotSpec(data_array=VariableReference(name="x")),
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)

    def test_scatter_plot_infers_scatter_type(self):
        x = InferredDataArrayVariable(
            type="DataArray", dims=frozenset({"time"}), shape={"time": 10}
        )
        y = InferredDataArrayVariable(
            type="DataArray", dims=frozenset({"time"}), shape={"time": 10}
        )
        context = ExecutionContext({"x": x, "y": y})
        expr = MakePlot(
            func_name="MakePlot",
            plot_spec=ScatterPlotSpec(
                x_data_array=VariableReference(name="x"),
                y_data_array=VariableReference(name="y"),
            ),
        )
        result = type_check_expression(expr, context)
        assert result == InferredVariable(type="scatter_plot")

    def test_scatter_plot_requires_data_arrays(self):
        context = ExecutionContext(
            {
                "x": InferredDataArrayVariable(
                    type="DataArray", dims=frozenset({"time"}), shape={"time": 10}
                ),
                "y": InferredVariable(type="int"),
            }
        )
        expr = MakePlot(
            func_name="MakePlot",
            plot_spec=ScatterPlotSpec(
                x_data_array=VariableReference(name="x"),
                y_data_array=VariableReference(name="y"),
            ),
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)

    def test_wind_vector_plot_infers_raster_type(self):
        u = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"lat", "lon"}),
            shape={"lat": 4, "lon": 4},
        )
        v = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"lat", "lon"}),
            shape={"lat": 4, "lon": 4},
        )
        context = ExecutionContext({"u": u, "v": v})
        expr = MakePlot(
            func_name="MakePlot",
            plot_spec=WindVectorPlotSpec(
                u_data_array=VariableReference(name="u"),
                v_data_array=VariableReference(name="v"),
            ),
        )
        result = type_check_expression(expr, context)
        assert result == InferredVariable(type="raster_plot")

    def test_wind_vector_plot_requires_data_arrays(self):
        context = ExecutionContext(
            {
                "u": InferredDataArrayVariable(
                    type="DataArray", dims=frozenset({"lat", "lon"}), shape={"lat": 4, "lon": 4}
                ),
                "v": InferredVariable(type="int"),
            }
        )
        expr = MakePlot(
            func_name="MakePlot",
            plot_spec=WindVectorPlotSpec(
                u_data_array=VariableReference(name="u"),
                v_data_array=VariableReference(name="v"),
            ),
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)

    def test_ensemble_plume_plot_infers_line_type(self):
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"ensemble_member", "lead_time"}),
            shape={"ensemble_member": 15, "lead_time": 12},
        )
        context = ExecutionContext({"da": da})
        expr = MakePlot(
            func_name="MakePlot",
            plot_spec=EnsemblePlumePlotSpec(data_array=VariableReference(name="da")),
        )
        result = type_check_expression(expr, context)
        assert result == InferredVariable(type="line_plot")

    def test_ensemble_plume_plot_requires_ensemble_dim(self):
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lead_time"}),
            shape={"time": 15, "lead_time": 12},
        )
        context = ExecutionContext({"da": da})
        expr = MakePlot(
            func_name="MakePlot",
            plot_spec=EnsemblePlumePlotSpec(data_array=VariableReference(name="da")),
        )
        with pytest.raises(Exception, match="ensemble"):
            type_check_expression(expr, context)

    def test_contour_map_plot_infers_raster_type(self):
        da = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"lat", "lon"}),
            shape={"lat": 8, "lon": 8},
        )
        context = ExecutionContext({"da": da})
        expr = MakePlot(
            func_name="MakePlot",
            plot_spec=ContourMapPlotSpec(data_array=VariableReference(name="da")),
        )
        result = type_check_expression(expr, context)
        assert result == InferredVariable(type="raster_plot")
