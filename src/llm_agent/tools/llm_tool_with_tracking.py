import logging

from pydantic import BaseModel

from llm_agent.context import ToolCallContext
from llm_agent.tools.llm_tool_typed import ErrorResponse, LLMToolTyped, ToolOutput

log = logging.getLogger(__name__)


class TrackingInfo(BaseModel):
    """Tool call tracking info."""

    plan_id: str
    step_id: str


class ToolInputWithTracking[InputT: BaseModel](BaseModel):
    """Tool input model with tracking info."""

    tracking: TrackingInfo
    tool_input: InputT


class LLMToolWithTracking[InputT: BaseModel, UserOutputT: BaseModel, LLMOutputT: BaseModel](
    LLMToolTyped[ToolInputWithTracking[InputT], UserOutputT, LLMOutputT]
):
    """Tool with extra input fields to link it to predefined plan and step."""

    def __init__(self, tool: LLMToolTyped[InputT, UserOutputT, LLMOutputT]) -> None:
        """Constructor."""
        super().__init__(
            name=f"{tool.tool_spec.name}WithTracking",
            description=tool.tool_spec.description,
            input_model=ToolInputWithTracking[tool.input_model],
            user_output_model=tool.user_output_model,
            llm_output_model=tool.llm_output_model,
        )
        self._wrapped_tool = tool
        self._wrap_output = tool._wrap_output  # noqa: SLF001

    def _execute_typed(
        self, context: ToolCallContext, tool_input: ToolInputWithTracking[InputT]
    ) -> ToolOutput[UserOutputT, LLMOutputT] | ErrorResponse:
        return self._wrapped_tool._execute_typed(context, tool_input.tool_input)  # noqa: SLF001
