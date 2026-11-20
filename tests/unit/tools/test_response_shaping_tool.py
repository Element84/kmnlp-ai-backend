from collections.abc import Callable

from pydantic import BaseModel
from pydantic_ai import RunContext

from llm_agent.context import AgentDeps
from llm_agent.tools.response_shaping_tool import ResponseShapingTool, ResponseShapingToolAck
from tests.utils.mocks.context import MockChatContext


class _Response(BaseModel):
    answer: str
    score: int


class TestResponseShapingTool:
    def _make_tool(self) -> ResponseShapingTool[_Response]:
        return ResponseShapingTool(
            name="ShapeResponse",
            description="Shape the structured response.",
            response_model=_Response,
        )

    def test_name_is_set(self):
        assert self._make_tool().name == "ShapeResponse"

    def test_description_includes_empty_output_note(self):
        tool = self._make_tool()
        assert tool.description.startswith("Shape the structured response.")
        assert "simple acknowledgement on success" in tool.description

    def test_schema_flattens_to_response_model_fields(self):
        tool = self._make_tool()
        schema = tool.as_pydantic_tool().function_schema.json_schema
        assert set(schema["properties"]) == {"answer", "score"}

    async def test_call_returns_ack(
        self,
        chat_context: MockChatContext,
        make_run_context: Callable[..., RunContext[AgentDeps]],
    ):
        tool = self._make_tool()
        ctx = make_run_context(chat_context)

        result = await tool(ctx, _Response(answer="42", score=1))

        assert result == ResponseShapingToolAck(success=True)
