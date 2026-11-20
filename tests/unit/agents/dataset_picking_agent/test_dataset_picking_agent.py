from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel

from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import (
    DatasetPickingAgent,
    DatasetSearchQuery,
    DatasetSearchResponse,
)
from llm_agent.context import AgentDeps
from llm_agent.tools.agent_tool import AgentTool
from llm_agent.tools.fetch_datasets_tool import FetchDatasetsTool
from tests.utils.mocks.context import MockChatContext


class _StubToolInput(BaseModel):
    pass


class _StubTool(AgentTool[_StubToolInput, str]):
    """An ``AgentTool`` that records how many times it is called."""

    input_model = _StubToolInput
    output_type = str

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "A stub tool."
        self.call_count = 0

    async def __call__(self, ctx: RunContext[AgentDeps], tool_input: _StubToolInput, /) -> str:  # noqa: ARG002
        self.call_count += 1
        return "stub result"


class TestDatasetPickingAgent:
    def test_default_tool_is_fetch_datasets(self):
        agent = DatasetPickingAgent(TestModel())
        tools = agent._tools  # pyright: ignore[reportPrivateUsage]
        assert len(tools) == 1
        assert isinstance(tools[0], FetchDatasetsTool)

    async def test_response_with_no_tool_call(self, chat_context: MockChatContext):
        refusal = DatasetSearchResponse(
            explanation="I'm sorry, Dave. I'm afraid I can't do that.",
            result=None,
            is_refusal=True,
        )
        model = TestModel(call_tools=[], custom_output_args=refusal.model_dump())
        agent = DatasetPickingAgent(model)

        response = await agent.run(
            chat_context, DatasetSearchQuery(query="Open the pod bay doors, Hal.")
        )

        assert response == refusal

    async def test_response_with_tool_call(self, chat_context: MockChatContext):
        called_tool = _StubTool("FetchDatasetsTool")
        other_tool = _StubTool("SomeOtherTool")
        final_response = DatasetSearchResponse(
            explanation="I looked it up, and the answer is all of it!",
            result=None,
            is_refusal=False,
        )
        model = TestModel(
            call_tools=["FetchDatasetsTool"], custom_output_args=final_response.model_dump()
        )
        agent = DatasetPickingAgent(model, tools=[called_tool, other_tool])

        response = await agent.run(
            chat_context, DatasetSearchQuery(query="What data do you know about?")
        )

        assert response == final_response
        assert called_tool.call_count == 1
        assert other_tool.call_count == 0
