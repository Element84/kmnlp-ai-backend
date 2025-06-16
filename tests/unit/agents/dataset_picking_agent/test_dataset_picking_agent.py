from typing import cast

import pytest
from e84_geoai_common.llm.core import (
    JSONContent,
    LLMTool,
    LLMToolResultContent,
    LLMToolUseContent,
)
from e84_geoai_common.llm.models.claude import (
    CLAUDE_BEDROCK_MODEL_IDS,
    BedrockClaudeLLM,
)
from e84_geoai_common.llm.tests.mock_bedrock import (
    claude_response_with_content,
    make_test_bedrock_client,
)
from pydantic import BaseModel

from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import (
    DatasetPickingAgent,
    DatasetResponseTextContent,
)
from llm_agent.tools.data_search_tool import DEFAULT_MODEL_NAME
from llm_agent.tools.fetch_datasets_tool import fetch_datasets_tool


class TestDatasetPickingAgent:
    def test_default_tool_dict(self):
        llm = BedrockClaudeLLM()
        agent = DatasetPickingAgent(llm)

        assert agent.tools == {
            "FetchDatasetsTool": fetch_datasets_tool,
        }

    @pytest.mark.asyncio
    async def test_response_with_no_tool_call(self):
        # HACK: Because of the "{" prefix in the inference config, this string
        # needs to not have a leading `{`.
        stub_response = claude_response_with_content(
            """
    "explanation": "I'm sorry, Dave. I'm afraid I can't do that.",
    "is_refusal": false
}"""
        )
        client = make_test_bedrock_client([stub_response])
        llm = BedrockClaudeLLM(
            model_id=CLAUDE_BEDROCK_MODEL_IDS[DEFAULT_MODEL_NAME],
            client=client,
        )
        agent = DatasetPickingAgent(llm)
        assistant_messages = [
            message
            async for message in agent.run("Open the pod bay doors, Hal.")
        ]

        assert len(assistant_messages) == 1
        message = assistant_messages[0]
        assert len(message.parsed_content) == 1

        parsed_response = message.parsed_content[0]
        parsed_response = cast("DatasetResponseTextContent", parsed_response)

        assert parsed_response.result is None

        assert (
            parsed_response.explanation
            == "I'm sorry, Dave. I'm afraid I can't do that."
        )

    # TODO: This test doesn't really test the flow right.
    # It would be better if it validated:
    #   - the tool call happened in between in between 2 calls to the LLM
    #   - the content of the messages that were sent
    @pytest.mark.asyncio
    async def test_response_with_tool_call(self):
        # Chain of responses from the LLM
        call_tool_response = claude_response_with_content(
            [
                {
                    "text": """ "explanation": "Let me look that up.", "is_refusal": false }"""  # noqa: E501
                },
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
            name="FetchDatasetsTool",
            description="d",
            input_model=_StubToolQuery,
            output_model=_StubToolResponse,
        )
        mock_other_tool = MockTool(
            name="SomeOtherTool",
            description="d",
            input_model=_StubToolQuery,
            output_model=_StubToolResponse,
        )
        tools = {
            "FetchDatasetsTool": mock_called_tool,
            "SomeOtherTool": mock_other_tool,
        }

        # Mock the Becrock Client, LLM, and initialize the agent
        client = make_test_bedrock_client([call_tool_response, final_response])
        llm = BedrockClaudeLLM(
            model_id=CLAUDE_BEDROCK_MODEL_IDS[DEFAULT_MODEL_NAME],
            client=client,
        )
        agent = DatasetPickingAgent(llm, tools)  # type: ignore[reportArgumentType]

        # Send the query
        assistant_messages = [
            message
            async for message in agent.run("What data do you know about?")
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


class MockTool(LLMTool):
    execute_calls: int = 0

    def execute(
        self,
        tool_use_request: LLMToolUseContent,  # noqa: ARG002
    ) -> LLMToolResultContent:
        self.execute_calls += 1
        return LLMToolResultContent(
            id="some-id",
            content=[
                JSONContent(
                    data=_StubToolResponse(response="some string").model_dump(
                        mode="json"
                    )
                )
            ],
            status="success",
        )
