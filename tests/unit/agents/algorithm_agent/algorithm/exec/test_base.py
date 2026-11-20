import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    BaseFunctionCall,
    Function,
    VariableAssignment,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.base import (
    ExecutionConfig,
    eval_expression,
    execute_algorithm,
)
from llm_agent.context import ToolCallContext

# Global list to track execution order
_execution_order: list[str] = []


class SlowFunction(Function, frozen=True):
    """Mock function that simulates slow work."""

    func_name: str = "SlowFunction"
    value: str


@eval_expression.register(SlowFunction)
async def _eval_slow_function(expr: SlowFunction, context: Any) -> str:  # noqa: ARG001, ANN401 # type: ignore[reportUnusedFunction]
    """Evaluate slow function - simulates work and tracks execution."""
    _execution_order.append(expr.value)
    # Simulate some async work
    await asyncio.sleep(0.01)
    return expr.value


@pytest.mark.asyncio
async def test_execute_algorithm_cancellation_between_statements():
    """Test that execute_algorithm can be cancelled between statements."""
    # Reset execution order
    _execution_order.clear()

    # Create an algorithm with 3 statements
    algo = Algorithm[BaseFunctionCall](
        statements=[
            VariableAssignment(
                name="var1",
                value=BaseFunctionCall(func=SlowFunction(value="first")),
                description="First statement",
            ),
            VariableAssignment(
                name="var2",
                value=BaseFunctionCall(func=SlowFunction(value="second")),
                description="Second statement",
            ),
            VariableAssignment(
                name="var3",
                value=BaseFunctionCall(func=SlowFunction(value="third")),
                description="Third statement",
            ),
        ],
        output_variables=[],
    )

    # Create mock tool call context
    tool_call_context = MagicMock(spec=ToolCallContext)
    tool_call_context.progress_callback = MagicMock()

    # Start execution in a task
    task = asyncio.create_task(
        execute_algorithm(tool_call_context, algo, input_vars={}, exec_config=ExecutionConfig())
    )

    # Give it time to start and execute first statement
    await asyncio.sleep(0.02)

    # Cancel the task (should be between first and second statement)
    task.cancel()

    # Verify that cancellation happened
    with pytest.raises(asyncio.CancelledError):
        await task

    # Verify that not all statements were executed
    # First statement should have executed, but not all three
    assert len(_execution_order) < 3, (
        f"Expected cancellation to stop execution, but got: {_execution_order}"
    )
    assert len(_execution_order) >= 1, "Expected at least first statement to execute"


@pytest.mark.asyncio
async def test_execute_algorithm_completes_without_cancellation():
    """Test that execute_algorithm works normally when not cancelled."""
    # Reset execution order
    _execution_order.clear()

    # Create a simple algorithm
    algo = Algorithm[BaseFunctionCall](
        statements=[
            VariableAssignment(
                name="var1",
                value=BaseFunctionCall(func=SlowFunction(value="only")),
                description="Simple assignment",
            ),
        ],
        output_variables=[],
    )

    # Create mock tool call context
    tool_call_context = MagicMock(spec=ToolCallContext)
    tool_call_context.progress_callback = MagicMock()

    # Execute without cancellation
    result = await execute_algorithm(
        tool_call_context, algo, input_vars={}, exec_config=ExecutionConfig()
    )

    # Should complete successfully
    assert result == {}
    # Statement should have executed
    assert len(_execution_order) == 1
    assert _execution_order[0] == "only"
