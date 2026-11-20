from typing import Literal

import pytest

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
    BinaryOperation,
)
from llm_agent.agents.algorithm_agent.algorithm.type_check import (
    BinaryOpDataArrayMisMatchError,
    ExecutionContext,
    InferredDataArrayVariable,
    InferredVariable,
    VariableTypeMismatchError,
    type_check_expression,
)


class TestBinaryOperations:
    """Test binary operation type checking."""

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_int_int_operations(self, op: Literal["add", "sub", "mul", "div"]) -> None:
        context = ExecutionContext({})
        expr = BinaryOperation(func_name="BinaryOperation", left=5, right=3, op=op)
        result = type_check_expression(expr, context)
        assert result == InferredVariable(type="int")

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_float_float_operations(self, op: Literal["add", "sub", "mul", "div"]) -> None:
        context = ExecutionContext({})
        expr = BinaryOperation(func_name="BinaryOperation", left=5.0, right=3.0, op=op)
        result = type_check_expression(expr, context)
        assert result == InferredVariable(type="float")

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (5, 3.0),  # int + float
            (5.0, 3),  # float + int
        ],
    )
    def test_int_float_operations(
        self, op: Literal["add", "sub", "mul", "div"], left: float, right: float
    ) -> None:
        context = ExecutionContext({})
        expr = BinaryOperation(func_name="BinaryOperation", left=left, right=right, op=op)
        result = type_check_expression(expr, context)
        assert result == InferredVariable(type="float")

    def test_dataarray_scalar_operations(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        data_array = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        context.variables["da"] = data_array

        # DataArray + scalar
        expr1 = BinaryOperation(
            func_name="BinaryOperation",
            left=VariableReference(name="da"),
            right=5,
            op="add",
        )
        result1 = type_check_expression(expr1, context)
        assert isinstance(result1, InferredDataArrayVariable)
        assert result1.dims == frozenset({"time", "lat", "lon"})
        assert result1.name is None

        # scalar + DataArray
        expr2 = BinaryOperation(
            func_name="BinaryOperation",
            left=5.0,
            right=VariableReference(name="da"),
            op="add",
        )
        result2 = type_check_expression(expr2, context)
        assert isinstance(result2, InferredDataArrayVariable)
        assert result2.dims == frozenset({"time", "lat", "lon"})

    def test_dataarray_dataarray_compatible(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        da1 = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
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

        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})

    def test_dataarray_dataarray_broadcasts_subset_dims(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        da1 = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        da2 = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"lat", "lon"}),
            shape={"lat": 20, "lon": 30},
        )
        context.variables.update({"da1": da1, "da2": da2})

        expr = BinaryOperation(
            func_name="BinaryOperation",
            left=VariableReference(name="da1"),
            right=VariableReference(name="da2"),
            op="add",
        )

        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})
        assert result.shape == {"time": 10, "lat": 20, "lon": 30}

    def test_dataarray_dataarray_broadcasts_size_one_dim(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        da1 = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"lat", "lon"}),
            shape={"lat": 1, "lon": 30},
        )
        da2 = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"lat", "lon"}),
            shape={"lat": 20, "lon": 30},
        )
        context.variables.update({"da1": da1, "da2": da2})

        expr = BinaryOperation(
            func_name="BinaryOperation",
            left=VariableReference(name="da1"),
            right=VariableReference(name="da2"),
            op="add",
        )

        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"lat", "lon"})
        assert result.shape == {"lat": 20, "lon": 30}

    def test_dataarray_dataarray_incompatible_shapes(self):
        context = ExecutionContext({}, valid_time_dims={"time"})
        da1 = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        da2 = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 25, "lon": 30},  # Different lat size
        )
        context.variables.update({"da1": da1, "da2": da2})

        expr = BinaryOperation(
            func_name="BinaryOperation",
            left=VariableReference(name="da1"),
            right=VariableReference(name="da2"),
            op="add",
        )

        with pytest.raises(BinaryOpDataArrayMisMatchError):
            type_check_expression(expr, context)

    def test_invalid_operand_types(self):
        context = ExecutionContext({"s": InferredVariable(type="str")})

        expr = BinaryOperation(
            func_name="BinaryOperation",
            left=VariableReference(name="s"),
            right=5,
            op="add",
        )

        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)
