import asyncio
from collections.abc import Callable

import pytest
from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from llm_agent.agents.orchestration_agent.interface import UserMessage
from llm_agent.agents.orchestration_agent.orchestration_agent import OrchestrationAgent
from llm_agent.context import AgentDeps
from llm_agent.tools.agent_tool import AgentTool
from tests.utils.mocks.context import MockChatContext


class _ToolInput(BaseModel):
    text: str


class _ToolOutput(BaseModel):
    result: str


class CancellableTool(AgentTool[_ToolInput, _ToolOutput]):
    """Tool that raises ``CancelledError`` during execution."""

    name = "CancellableTool"
    description = "A tool that gets cancelled during execution."
    input_model = _ToolInput
    output_type = _ToolOutput

    async def __call__(self, ctx: RunContext[AgentDeps], tool_input: _ToolInput, /) -> _ToolOutput:  # noqa: ARG002
        raise asyncio.CancelledError


class SlowTool(AgentTool[_ToolInput, _ToolOutput]):
    """Tool that sleeps for a long time, allowing external cancellation."""

    name = "SlowTool"
    description = "A slow tool that can be cancelled externally."
    input_model = _ToolInput
    output_type = _ToolOutput

    async def __call__(self, ctx: RunContext[AgentDeps], tool_input: _ToolInput, /) -> _ToolOutput:  # noqa: ARG002
        await asyncio.sleep(5)
        raise RuntimeError("tools should be cancelled before returning")


def _call_tool_once(tool_name: str) -> Callable[[list[ModelMessage], AgentInfo], ModelResponse]:
    """Build a model function that calls ``tool_name`` on its first turn.

    Each test's tool aborts the run, so the model is only ever asked once; the
    text fallback exists purely to keep the function total.
    """

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # noqa: ARG001
        if any(isinstance(message, ModelResponse) for message in messages):
            return ModelResponse(parts=[TextPart(content="done")])
        return ModelResponse(
            parts=[ToolCallPart(tool_name=tool_name, args={"text": "some output"})]
        )

    return model_function


class TestOrchestrationAgentCancellation:
    async def test_cancellation_from_tool(self, chat_context: MockChatContext) -> None:
        """A tool that raises ``CancelledError`` aborts the run rather than being swallowed."""
        agent = OrchestrationAgent(
            FunctionModel(_call_tool_once("CancellableTool")), tools=[CancellableTool()]
        )

        with pytest.raises(asyncio.CancelledError):
            await agent.run(chat_context, UserMessage(user_message="Test message"))

    async def test_cancellation_from_external(self, chat_context: MockChatContext) -> None:
        """Cancelling the task while a slow tool runs propagates ``CancelledError``."""
        agent = OrchestrationAgent(FunctionModel(_call_tool_once("SlowTool")), tools=[SlowTool()])

        task = asyncio.create_task(
            agent.run(chat_context, UserMessage(user_message="Test message"))
        )

        # Give the run time to reach the slow tool, then cancel (simulates disconnect).
        await asyncio.sleep(0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
