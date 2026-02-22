from pydantic import BaseModel

from llm_agent.tools.response_shaping_tools.response_shaping_tool import (
    ResponseShapingTool,
)


class TextResponse(BaseModel, strict=True, extra="forbid"):
    """Text response."""

    text: str


text_response_tool = ResponseShapingTool(
    name="TextResponseTool",
    description="Tool to use to output free-form text.",
    response_model=TextResponse,
)
