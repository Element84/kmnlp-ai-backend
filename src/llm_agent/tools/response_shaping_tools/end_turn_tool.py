from pydantic import BaseModel

from llm_agent.tools.response_shaping_tools.response_shaping_tool import ResponseShapingTool


class EmptyInput(BaseModel):
    """Empty input."""


end_turn_tool = ResponseShapingTool(
    name="EndTurnTool",
    description="Tool to use to end the turn. Can be included in parallel tool calls, if "
    "responses to the other tool calls are not required.",
    response_model=EmptyInput,
)
