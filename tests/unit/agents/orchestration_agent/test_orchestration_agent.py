from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from llm_agent.agents.orchestration_agent.interface import UserMessage
from llm_agent.agents.orchestration_agent.orchestration_agent import OrchestrationAgent
from llm_agent.context import AgentDeps
from llm_agent.tools.agent_tool import AgentTool
from llm_agent.tools.data_analysis_tool.data_analysis_tool import DataAnalysisToolOutput
from llm_agent.tools.data_search_tool import DataSearchToolResult
from tests.utils.mocks.context import MockChatContext


class _DataSearchInput(BaseModel):
    natural_language_query: str


class _StubDataSearchTool(AgentTool[_DataSearchInput, DataSearchToolResult]):
    """Stub ``DataSearchTool`` that records calls and returns a canned result."""

    name = "DataSearchTool"
    description = "Search for relevant datasets."
    input_model = _DataSearchInput
    output_type = DataSearchToolResult

    def __init__(self) -> None:
        self.call_count = 0

    async def __call__(
        self,
        ctx: RunContext[AgentDeps],  # noqa: ARG002
        tool_input: _DataSearchInput,  # noqa: ARG002
        /,
    ) -> DataSearchToolResult:
        self.call_count += 1
        return DataSearchToolResult(explanation="Found one dataset.", datasets=[])


class _DataAnalysisInput(BaseModel):
    problem_statement: str


class _StubDataAnalysisTool(AgentTool[_DataAnalysisInput, DataAnalysisToolOutput]):
    """Stub ``DataAnalysisTool`` that records calls and returns a canned result."""

    name = "DataAnalysisTool"
    description = "Analyze data."
    input_model = _DataAnalysisInput
    output_type = DataAnalysisToolOutput

    def __init__(self) -> None:
        self.call_count = 0

    async def __call__(
        self,
        ctx: RunContext[AgentDeps],  # noqa: ARG002
        tool_input: _DataAnalysisInput,  # noqa: ARG002
        /,
    ) -> DataAnalysisToolOutput:
        self.call_count += 1
        return DataAnalysisToolOutput(
            outputs={"sst_mean": 20.04}, solution_pseudocode="__pseudocode__"
        )


def _count_tool_returns(messages: list[ModelMessage]) -> int:
    return sum(
        isinstance(part, ToolReturnPart)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


def _orchestrate(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # noqa: ARG001
    """Drive a deterministic search -> analyze -> respond workflow.

    The number of tool results seen so far determines the next step, so the model
    walks through the tools in order before producing its final text reply.
    """
    match _count_tool_returns(messages):
        case 0:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="DataSearchTool",
                        args={"natural_language_query": "Sea surface temperature"},
                    )
                ]
            )
        case 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="DataAnalysisTool",
                        args={
                            "problem_statement": (
                                "Compute the mean SST in the Black Sea on June 01, 2024."
                            )
                        },
                    )
                ]
            )
        case _:
            return ModelResponse(parts=[TextPart(content="The mean SST was 20.04 deg C.")])


async def _orchestrate_stream(
    messages: list[ModelMessage],
    info: AgentInfo,  # noqa: ARG001
) -> AsyncIterator[str | DeltaToolCalls]:
    """Streaming variant of :func:`_orchestrate`.

    Tool turns yield a single tool-call delta; the final turn streams the text
    reply across multiple deltas so the test can observe text-delta events.
    """
    match _count_tool_returns(messages):
        case 0:
            yield {
                0: DeltaToolCall(
                    name="DataSearchTool",
                    json_args='{"natural_language_query": "Sea surface temperature"}',
                )
            }
        case 1:
            yield {
                0: DeltaToolCall(
                    name="DataAnalysisTool",
                    json_args=(
                        '{"problem_statement": "Compute the mean SST in the Black Sea '
                        'on June 01, 2024."}'
                    ),
                )
            }
        case _:
            yield "The mean SST "
            yield "was 20.04 deg C."


@pytest.mark.asyncio
async def test_orchestration_agent(chat_context: MockChatContext) -> None:
    data_search_tool = _StubDataSearchTool()
    data_analysis_tool = _StubDataAnalysisTool()
    agent = OrchestrationAgent(
        FunctionModel(_orchestrate), tools=[data_search_tool, data_analysis_tool]
    )

    response = await agent.run(
        chat_context,
        UserMessage(
            user_message=(
                "What was the mean sea surface temperature in the Black Sea on June 01, 2024?"
            )
        ),
    )

    assert "20.04" in response
    assert data_search_tool.call_count == 1
    assert data_analysis_tool.call_count == 1


@pytest.mark.asyncio
async def test_orchestration_agent_run_stream(chat_context: MockChatContext) -> None:
    data_search_tool = _StubDataSearchTool()
    data_analysis_tool = _StubDataAnalysisTool()
    agent = OrchestrationAgent(
        FunctionModel(stream_function=_orchestrate_stream),
        tools=[data_search_tool, data_analysis_tool],
    )

    events = [
        event
        async for event in agent.run_stream(
            chat_context,
            UserMessage(
                user_message=(
                    "What was the mean sea surface temperature in the Black Sea on June 01, 2024?"
                )
            ),
        )
    ]

    tool_calls = [event for event in events if isinstance(event, FunctionToolCallEvent)]
    tool_results = [event for event in events if isinstance(event, FunctionToolResultEvent)]
    text_deltas = [
        event.delta.content_delta
        for event in events
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta)
    ]
    streamed_text = "".join(
        [
            *(
                event.part.content
                for event in events
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart)
            ),
            *text_deltas,
        ]
    )

    # Both tools were invoked, in order, and produced results.
    assert [call.part.tool_name for call in tool_calls] == ["DataSearchTool", "DataAnalysisTool"]
    assert [result.part.tool_name for result in tool_results] == [
        "DataSearchTool",
        "DataAnalysisTool",
    ]
    # The final text-only reply streamed incrementally and ended the turn.
    assert len(text_deltas) >= 1
    assert streamed_text == "The mean SST was 20.04 deg C."
    assert data_search_tool.call_count == 1
    assert data_analysis_tool.call_count == 1
