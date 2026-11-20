import logging
from functools import singledispatch
from typing import Any

import langfuse
from langfuse import observe  # pyright: ignore[reportUnknownVariableType]
from pydantic import BaseModel

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    BaseFunctionCall,
    VariableAssignment,
    VariableReference,
)
from llm_agent.agents.algorithm_agent.algorithm.langfuse import (
    transform_execute_algorithm_input_vars,
)
from llm_agent.context import DataRef, ToolCallContext
from llm_agent.utils.dask import MAX_ALLOWED_COMPUTE_SIZE

log = logging.getLogger(__name__)


class AlgorithmExecutionError(Exception):
    """Exception raised when algorithm execution fails."""

    def __init__(self, message: str, func_name: str | None = None) -> None:
        """Constructor."""
        msg = f"{func_name}: {message}" if func_name else message
        super().__init__(msg)


class VariableTypeMismatchError(AlgorithmExecutionError):
    """Referenced variable is not of the expected type."""

    def __init__(self, message: str, var_name: str, func_name: str | None = None) -> None:
        """Constructor."""
        if func_name:
            message = f"{func_name}: variable: {var_name}: {message}"
        else:
            message = f"variable: {var_name}: {message}"
        super().__init__(message)


class VariableReferenceError(AlgorithmExecutionError):
    """Referenced variable not found."""

    def __init__(self, var_name: str) -> None:
        """Constructor."""
        super().__init__(f"Variable '{var_name}' referenced but not set.")


class ExecutionConfig(BaseModel):
    """Configuration for algorithm execution."""

    data_array_compute_size_limit: int = MAX_ALLOWED_COMPUTE_SIZE
    dask_progress_poll_interval_sec: float = 5.0


class ExecutionContext:
    """Execution context that tracks assigned variables."""

    variables: dict[str, Any]
    tool_call_context: ToolCallContext
    config: ExecutionConfig

    def __init__(
        self,
        tool_call_context: ToolCallContext,
        input_vars: dict[str, Any],
        config: ExecutionConfig,
    ) -> None:
        """Construct."""
        self.tool_call_context = tool_call_context
        self.variables: dict[str, Any] = dict(**input_vars)
        self.config = config


class ComputingVariableUpdate(BaseModel):
    """Represents a progress update indicating that a variable is being computed."""

    step_num: int
    name: str


class ComputedVariableUpdate(BaseModel):
    """Represents a progress update indicating that a variable has been computed."""

    step_num: int
    name: str
    value_summary: str | None = None


@observe(as_type="chain", capture_input=False)
async def execute_algorithm[FunctionCall: BaseFunctionCall](
    tool_call_context: ToolCallContext,
    algo: Algorithm[FunctionCall],
    input_vars: dict[str, Any],
    exec_config: ExecutionConfig,
) -> dict[str, Any]:
    """Evaluate algorithm."""
    langfuse_client = langfuse.get_client()
    langfuse_client.update_current_span(
        input={"algo": algo, "input_vars": transform_execute_algorithm_input_vars(input_vars)}
    )

    # Useful because `to_thread.run_sync` takes args of (func, *args), which
    # breaks kwargs and can be hard to reason about.
    #
    # Not `async` because `to_thread.run_sync` runs a synchronous function in a
    # worker thread.
    exec_context = ExecutionContext(tool_call_context, input_vars, config=exec_config)
    for i, statement in enumerate(algo.statements):
        tool_call_context.update_progress(ComputingVariableUpdate(step_num=i, name=statement.name))

        with langfuse_client.start_as_current_observation(
            name=_statement_span_name(i, statement), as_type="chain", input=statement
        ) as span:
            await eval_expression(statement, exec_context)
            value_summary = summarize_value_for_update(exec_context.variables[statement.name])
            span.update(output=value_summary)

        tool_call_context.update_progress(
            ComputedVariableUpdate(step_num=i, name=statement.name, value_summary=value_summary)
        )
    out_var_names = [v.name for v in algo.output_variables]
    out = {v: exec_context.variables.get(v) for v in out_var_names}
    return out


def _statement_span_name(step_num: int, statement: VariableAssignment[BaseFunctionCall]) -> str:
    match statement.value:
        case BaseFunctionCall(func=func):
            return f"Step {step_num + 1}: {statement.name} = {func.func_name}()"
        case _:
            return f"Step {step_num + 1}: {statement.name}"


@singledispatch
def summarize_value_for_update(value: Any) -> str | None:  # noqa: ANN401
    """Summarize a value for inclusion in a progress update."""
    return repr(value)


@summarize_value_for_update.register(DataRef)
def _(value: DataRef) -> str | None:  # noqa: ARG001
    return None


@singledispatch
async def eval_expression(expr: Any, context: ExecutionContext) -> Any:  # noqa: ANN401
    """Evaluate expression."""
    raise NotImplementedError(f"No eval_expression implementation for type {type(expr)}")


@eval_expression.register(str)
@eval_expression.register(int)
@eval_expression.register(float)
async def eval_literal[T: str | int | float](expr: T, context: ExecutionContext) -> T:  # noqa: ARG001
    """Evaluate literal expression."""
    return expr


@eval_expression.register(list)
async def eval_list(expr: list[Any], context: ExecutionContext) -> list[Any]:
    """Evaluate expression."""
    return [await eval_expression(v, context) for v in expr]


@eval_expression.register(VariableAssignment)
async def eval_variable_assignment(
    expr: VariableAssignment[BaseFunctionCall], context: ExecutionContext
) -> None:
    """Evaluate expression."""
    context.variables[expr.name] = await eval_expression(expr.value, context)


@eval_expression.register(VariableReference)
async def eval_variable_reference(expr: VariableReference, context: ExecutionContext) -> Any:  # noqa: ANN401
    """Evaluate expression."""
    try:
        return context.variables[expr.name]
    except KeyError as e:
        raise VariableReferenceError(expr.name) from e


@eval_expression.register(BaseFunctionCall)
async def eval_function_call(expr: BaseFunctionCall, context: ExecutionContext) -> Any:  # noqa: ANN401
    """Evaluate expression."""
    return await eval_expression(expr.func, context)
