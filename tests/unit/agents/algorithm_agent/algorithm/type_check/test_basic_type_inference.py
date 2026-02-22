from llm_agent.agents.algorithm_agent.algorithm.type_check import (
    ExecutionContext,
    InferredListVariable,
    InferredVariable,
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
