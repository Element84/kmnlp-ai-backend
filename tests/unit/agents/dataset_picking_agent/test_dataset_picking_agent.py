from collections.abc import Callable
from typing import cast
from unittest.mock import Mock

import pytest
from e84_geoai_common.llm.core import (
    ExecutableLLMTool,
    JSONContent,
    LLMTool,
    LLMToolResultContent,
    LLMToolUseContent,
)
from e84_geoai_common.llm.models.claude import BedrockClaudeLLM
from e84_geoai_common.llm.tests.mock_bedrock_runtime import (
    claude_response_with_content,
    make_test_bedrock_runtime_client,
)
from pydantic import BaseModel

from llm_agent import DATASET_PICKING_AGENT_DEFAULT_LLM_MODEL_ID
from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import (
    DatasetPickingAgent,
    DatasetResponseTextContent,
)
from llm_agent.context import LLMAgentContext, ToolCallContext
from llm_agent.utils.prompt_as_service import PromptServiceAdapter


class TestDatasetPickingAgent:
    def test_default_tool_dict(self):
        agent = DatasetPickingAgent(prompt_service=Mock())

        assert len(agent.tools) == 1
        assert agent.tools[0].tool_spec.name == "FetchDatasetsTool"

    @pytest.mark.asyncio
    async def test_response_with_no_tool_call(self, global_context: Callable[..., LLMAgentContext]):
        # HACK: Because of the "{" prefix in the inference config, this string
        # needs to not have a leading `{`.
        stub_response = claude_response_with_content(
            """
    "explanation": "I'm sorry, Dave. I'm afraid I can't do that.",
    "is_refusal": false
}"""
        )
        client = make_test_bedrock_runtime_client([stub_response])
        llm = BedrockClaudeLLM(
            model_id=DATASET_PICKING_AGENT_DEFAULT_LLM_MODEL_ID,
            client=client,
        )
        agent = DatasetPickingAgent(PromptServiceAdapter(llm.prompt))
        assistant_messages = [
            message async for message in agent.run(global_context(), "Open the pod bay doors, Hal.")
        ]

        assert len(assistant_messages) == 1
        message = assistant_messages[0]
        assert len(message.parsed_content) == 1

        parsed_response = message.parsed_content[0]
        parsed_response = cast("DatasetResponseTextContent", parsed_response)

        assert parsed_response.result is None

        assert parsed_response.explanation == "I'm sorry, Dave. I'm afraid I can't do that."

    # FUTURE: This test doesn't really test the flow right.
    # It would be better if it validated:
    #   - the tool call happened in between in between 2 calls to the LLM
    #   - the content of the messages that were sent
    @pytest.mark.asyncio
    async def test_response_with_tool_call(self, global_context: Callable[..., LLMAgentContext]):
        # Chain of responses from the LLM
        call_tool_response = claude_response_with_content(
            [
                {"text": """ "explanation": "Let me look that up.", "is_refusal": false }"""},
                {"id": "some-id", "name": "FetchDatasetsTool", "input": {}},
            ]
        )
        final_response = claude_response_with_content(
            [
                {
                    "text": """ "explanation": "I looked it up, and the answer is all of it!", "is_refusal": false }"""  # noqa: E501
                }
            ]
        )

        # Mock the provided tools
        mock_called_tool = MockTool(
            tool_spec=LLMTool(
                name="FetchDatasetsTool",
                description="d",
                input_model=_StubToolQuery,
                output_model=_StubToolResponse,
            )
        )
        mock_other_tool = MockTool(
            tool_spec=LLMTool(
                name="SomeOtherTool",
                description="d",
                input_model=_StubToolQuery,
                output_model=_StubToolResponse,
            )
        )
        tools = [mock_called_tool, mock_other_tool]

        # Mock the Becrock Client, LLM, and initialize the agent
        client = make_test_bedrock_runtime_client([call_tool_response, final_response])
        llm = BedrockClaudeLLM(
            model_id=DATASET_PICKING_AGENT_DEFAULT_LLM_MODEL_ID,
            client=client,
        )
        agent = DatasetPickingAgent(PromptServiceAdapter(llm.prompt), tools)  # type: ignore[reportArgumentType]

        # Send the query
        assistant_messages = [
            message async for message in agent.run(global_context(), "What data do you know about?")
        ]

        # Should get 2 messages back
        assert len(assistant_messages) == 2

        # Assertions
        first_response_contents = assistant_messages[0].parsed_content
        assert len(first_response_contents) == 2
        assert first_response_contents[0] == DatasetResponseTextContent(
            explanation="Let me look that up.", result=None, is_refusal=False
        )
        assert first_response_contents[1] == LLMToolUseContent(
            id="some-id",
            name="FetchDatasetsTool",
            input={},
        )

        final_response = assistant_messages[1].parsed_content
        assert len(final_response) == 1
        assert (
            cast("DatasetResponseTextContent", final_response[0]).explanation
            == "I looked it up, and the answer is all of it!"
        )

        assert mock_called_tool.execute_calls == 1
        assert mock_other_tool.execute_calls == 0


class _StubToolQuery(BaseModel):
    pass


class _StubToolResponse(BaseModel):
    response: str


class MockTool(ExecutableLLMTool[ToolCallContext]):
    execute_calls: int = 0

    def execute(
        self,
        context: ToolCallContext,  # noqa: ARG002
        tool_use_request: LLMToolUseContent,  # noqa: ARG002
    ) -> LLMToolResultContent:
        self.execute_calls += 1
        return LLMToolResultContent(
            id="some-id",
            content=[
                JSONContent(data=_StubToolResponse(response="some string").model_dump(mode="json"))
            ],
            status="success",
        )

    def _execute(
        self,
        context: ToolCallContext,  # noqa: ARG002
        tool_use_request: LLMToolUseContent,  # noqa: ARG002
    ) -> LLMToolResultContent:
        return LLMToolResultContent(id="fake", content=[])
