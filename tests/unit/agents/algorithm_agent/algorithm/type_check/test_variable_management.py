import pytest

from llm_agent.agents.algorithm_agent.algorithm.base import (
    VariableAssignment,
    VariableReference,
    make_function_call_model,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets import (
    SpatioTemporalDataArrayFunctionSet,
)
from llm_agent.agents.algorithm_agent.algorithm.type_check import (
    ExecutionContext,
    InferredVariable,
    VariableReferenceError,
    type_check_expression,
)

default_function_call_model = make_function_call_model(
    SpatioTemporalDataArrayFunctionSet().functions
)


class TestVariableManagement:
    """Test variable assignment and reference."""

    def test_variable_assignment(self):
        context = ExecutionContext({})
        assignment = VariableAssignment[default_function_call_model](name="x", value=42)

        type_check_expression(assignment, context)

        assert "x" in context.variables
        assert context.variables["x"] == InferredVariable(type="int", name="x")

    def test_variable_reference(self):
        context = ExecutionContext({"x": InferredVariable(type="int", name="x")})
        reference = VariableReference(name="x")

        result = type_check_expression(reference, context)

        assert result == InferredVariable(type="int", name="x")

    def test_variable_reference_not_found(self):
        context = ExecutionContext({})
        reference = VariableReference(name="nonexistent")

        with pytest.raises(VariableReferenceError):
            type_check_expression(reference, context)
