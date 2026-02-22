from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    VariableAssignment,
    VariableReference,
    make_function_call_model,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets import (
    SpatioTemporalDataArrayFunctionSet,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
    BinaryOperation,
    DataArrayToList,
    SliceDataArray,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.general_plot_functions import (
    PlotBarChart,
)
from llm_agent.agents.algorithm_agent.algorithm.type_check import (
    ArgumentTypeMismatchError,
    BinaryOpDataArrayMisMatchError,
    ExecutionContext,
    InferredDataArrayVariable,
    InferredListVariable,
    InferredVariable,
    InvalidDimNameError,
    VariableTypeMismatchError,
    type_check_algorithm,
    type_check_expression,
)

default_function_call_model = make_function_call_model(
    SpatioTemporalDataArrayFunctionSet().functions
)


class TestAlgorithmTypeChecking:
    """Test full algorithm type checking."""

    def test_simple_algorithm(self):
        # Create a simple algorithm that adds two numbers
        statements = [
            VariableAssignment[default_function_call_model](name="result", value=42),
        ]

        algorithm = Algorithm[default_function_call_model](
            statements=statements,
            output_variables=[VariableReference(name="result")],
        )
        input_vars: dict[str, InferredVariable] = {}
        expected_outputs: dict[str, InferredVariable] = {
            "result": InferredVariable(type="int"),
        }

        # Should not raise any exceptions
        type_check_algorithm(algorithm, input_vars, expected_outputs)


class TestErrorMessages:
    """Test that error messages are informative."""

    def test_variable_type_mismatch_error_message(self):
        var = InferredVariable(type="str", name="my_var")
        func = DataArrayToList(
            func_name="DataArrayToList",
            data_array=VariableReference(name="my_var"),
        )

        error = VariableTypeMismatchError(var, "DataArray", func)
        assert "DataArrayToList" in str(error)
        assert "my_var" in str(error)
        assert "DataArray" in str(error)

    def test_argument_type_mismatch_error_message(self):
        func = PlotBarChart(
            func_name="PlotBarChart",
            labels=["A"],
            heights=[VariableReference(name="bad_height")],
            title="Test",
        )

        error = ArgumentTypeMismatchError(func, "heights[0]", "int | float", "str")
        assert "PlotBarChart" in str(error)
        assert "heights[0]" in str(error)
        assert "int | float" in str(error)
        assert "str" in str(error)

    def test_invalid_dim_name_error_message(self):
        func = SliceDataArray(
            func_name="SliceDataArray",
            data_array=VariableReference(name="data"),
            dim_slices={"bad_dim": (0, 10)},
        )

        error = InvalidDimNameError("bad_dim", func)
        assert "SliceDataArray" in str(error)
        assert "bad_dim" in str(error)

    def test_binary_op_mismatch_error_message(self):
        left = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        right = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"lat", "lon"}),
            shape={"lat": 20, "lon": 30},
        )

        error = BinaryOpDataArrayMisMatchError("add", left, right)
        assert "add" in str(error)
        assert "Incompatible DataArrays" in str(error)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_list(self):
        context = ExecutionContext({})
        result = type_check_expression([], context)

        expected = InferredListVariable(type="list", items=[])
        assert result == expected

    def test_dataarray_with_none_shape_values(self):
        """Test DataArray with unknown dimension sizes."""
        context = ExecutionContext({}, valid_time_dims={"time"})
        da1 = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": "unknown", "lat": 20, "lon": "unknown"},  # Some unknown sizes
        )
        da2 = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        context.variables.update({"da1": da1, "da2": da2})

        expr = BinaryOperation(
            func_name="BinaryOperation",
            left=VariableReference(name="da1"),
            right=VariableReference(name="da2"),
            op="add",
        )
        result = type_check_expression(expr, context)

        # Should merge the shapes, being conservative with unknowns
        assert isinstance(result, InferredDataArrayVariable)
        assert result.shape == {"time": "unknown", "lat": 20, "lon": "unknown"}
