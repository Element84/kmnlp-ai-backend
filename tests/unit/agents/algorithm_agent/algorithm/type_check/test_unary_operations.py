from typing import Literal

import pytest

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
    UnaryOperation,
)
from llm_agent.agents.algorithm_agent.algorithm.type_check import (
    ExecutionContext,
    InferredDataArrayVariable,
    InferredVariable,
    VariableTypeMismatchError,
    type_check_expression,
)

UnaryOpT = Literal["sqrt", "square", "abs", "exp", "log", "log10", "neg"]

ALL_OPS: list[UnaryOpT] = ["sqrt", "square", "abs", "exp", "log", "log10", "neg"]


class TestUnaryOperations:
    """Test unary operation type checking."""

    @pytest.mark.parametrize("op", ALL_OPS)
    def test_int_operand_yields_float(self, op: UnaryOpT) -> None:
        context = ExecutionContext({})
        expr = UnaryOperation(func_name="UnaryOperation", operand=5, op=op)
        result = type_check_expression(expr, context)
        assert result == InferredVariable(type="float")

    @pytest.mark.parametrize("op", ALL_OPS)
    def test_float_operand_yields_float(self, op: UnaryOpT) -> None:
        context = ExecutionContext({})
        expr = UnaryOperation(func_name="UnaryOperation", operand=5.0, op=op)
        result = type_check_expression(expr, context)
        assert result == InferredVariable(type="float")

    @pytest.mark.parametrize("op", ALL_OPS)
    def test_dataarray_operand_preserves_dims_and_shape(self, op: UnaryOpT) -> None:
        context = ExecutionContext({})
        data_array = InferredDataArrayVariable(
            type="DataArray",
            dims=frozenset({"time", "lat", "lon"}),
            shape={"time": 10, "lat": 20, "lon": 30},
        )
        context.variables["da"] = data_array

        expr = UnaryOperation(
            func_name="UnaryOperation",
            operand=VariableReference(name="da"),
            op=op,
        )
        result = type_check_expression(expr, context)

        assert isinstance(result, InferredDataArrayVariable)
        assert result.dims == frozenset({"time", "lat", "lon"})
        assert result.shape == {"time": 10, "lat": 20, "lon": 30}
        assert result.name is None

    def test_invalid_operand_type(self) -> None:
        context = ExecutionContext({"s": InferredVariable(type="str")})
        expr = UnaryOperation(
            func_name="UnaryOperation",
            operand=VariableReference(name="s"),
            op="sqrt",
        )
        with pytest.raises(VariableTypeMismatchError):
            type_check_expression(expr, context)
