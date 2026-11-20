from collections.abc import Callable

from pydantic import BaseModel
from pydantic_ai import RunContext

from llm_agent.context import AgentDeps
from llm_agent.tools.agent_tool import AgentTool
from llm_agent.tools.tracking import ToolInputWithTracking, TrackingInfo, TrackingTool
from tests.utils.mocks.context import MockChatContext


class _DummyInput(BaseModel):
    value: str


class _DummyTool(AgentTool[_DummyInput, _DummyInput]):
    name = "DummyTool"
    description = "A dummy tool."
    input_model = _DummyInput
    output_type = _DummyInput

    async def __call__(self, ctx: RunContext[AgentDeps], tool_input: _DummyInput, /) -> _DummyInput:  # noqa: ARG002
        return tool_input


class TestTrackingTool:
    def test_name_is_suffixed(self):
        tool = TrackingTool(_DummyTool())
        assert tool.name == "DummyToolWithTracking"

    def test_description_is_inherited(self):
        tool = TrackingTool(_DummyTool())
        assert tool.description == "A dummy tool."

    def test_schema_flattens_to_tracking_and_tool_input(self):
        tool = TrackingTool(_DummyTool())
        schema = tool.as_pydantic_tool().function_schema.json_schema
        assert set(schema["properties"]) == {"tracking", "tool_input"}

    async def test_call_delegates_and_drops_tracking_envelope(
        self,
        chat_context: MockChatContext,
        make_run_context: Callable[..., RunContext[AgentDeps]],
    ):
        tool = TrackingTool(_DummyTool())
        ctx = make_run_context(chat_context)
        wrapped_input = ToolInputWithTracking[_DummyInput](
            tracking=TrackingInfo(plan_id="plan-1", step_id="step-1"),
            tool_input=_DummyInput(value="hello"),
        )

        result = await tool(ctx, wrapped_input)

        assert result == _DummyInput(value="hello")
