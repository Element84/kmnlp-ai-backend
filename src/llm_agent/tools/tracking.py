from typing import Any

from pydantic import BaseModel
from pydantic_ai import RunContext

from llm_agent.context import AgentDeps
from llm_agent.tools.agent_tool import AgentTool


class TrackingInfo(BaseModel):
    """Links a tool call to a predefined plan and step."""

    plan_id: str
    step_id: str


class ToolInputWithTracking[InputT: BaseModel](BaseModel):
    """Wraps a tool's input with plan/step tracking info."""

    tracking: TrackingInfo
    tool_input: InputT


class TrackingTool[InputT: BaseModel, OutputT](AgentTool[ToolInputWithTracking[InputT], OutputT]):
    """Wraps an :class:`AgentTool` to add plan/step tracking fields to its input.

    The wrapped tool is exposed under ``<name>WithTracking`` with a flattened
    ``{tracking, tool_input}`` schema, where ``tool_input`` keeps the wrapped
    tool's own input schema. The ``tracking`` info is not used by the tool's
    execution; it rides on the tool-call event stream so the caller can link the
    call back to a plan/step.
    """

    def __init__(self, wrapped: AgentTool[InputT, OutputT]) -> None:
        """Wrap an existing ``AgentTool``."""
        self._wrapped = wrapped
        self.name = f"{wrapped.name}WithTracking"
        self.description = wrapped.description
        # `wrapped.input_model` is concrete only at runtime; keep the subscript
        # untyped so the dynamic parameterization does not trip the type checker.
        wrapped_input_model: Any = wrapped.input_model
        self.input_model = ToolInputWithTracking[wrapped_input_model]
        self.output_type = wrapped.output_type

    async def __call__(
        self, ctx: RunContext[AgentDeps], tool_input: ToolInputWithTracking[InputT], /
    ) -> OutputT:
        """Delegate to the wrapped tool, dropping the tracking envelope."""
        return await self._wrapped(ctx, tool_input.tool_input)
