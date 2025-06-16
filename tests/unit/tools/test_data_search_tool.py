from collections.abc import AsyncIterator, Sequence
from json import JSONDecodeError
from typing import Self, TypeVar
from unittest.mock import Mock

import pytest
from anyio import to_thread
from e84_geoai_common.llm.core import (
    Base64ImageContent,
    JSONContent,
    LLMMessage,
    LLMToolUseContent,
    TextContent,
)
from pydantic import ValidationError

from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import (
    DatasetPickingAgentMessage,
    DatasetResponseTextContent,
    MaxPromptTurnsExceededError,
    UnexpectedResponseTypeError,
)
from llm_agent.tools.data_search_tool import DataSearchTool
from llm_agent.utils import EnrichedValidationError
from tests.unit.dao.test_knowledge_graph_dao import CRW_SST_DATASET_JSON

# `MockAsyncIterator` is potentially generally useful, but it is currently only
# used here. Move it out if there is something else that can take advantage of
# it.
A = TypeVar("A")


class MockAsyncIterator(AsyncIterator[A]):
    """Convenience class for building an `AsyncIterator` from a `Sequence`."""

    def __init__(self, seq: Sequence[A]) -> None:
        """Initializer. Takes a `Sequence`."""
        self.iter = iter(seq)

    def __aiter__(self) -> Self:
        """Return the `AsyncIterator`: self."""
        return self

    async def __anext__(self) -> A:
        """Return the next element in the sequence."""
        try:
            return next(self.iter)
        except StopIteration as e:
            raise StopAsyncIteration from e


class TestDataSearchTool:
    @pytest.mark.asyncio
    async def test_data_search_tool_fetches_dataset_by_variable_iri(self):
        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run.return_value = MockAsyncIterator(
            [
                DatasetPickingAgentMessage(
                    parsed_content=[
                        DatasetResponseTextContent(
                            explanation="",
                            result=[
                                "http://kmnlp-demo.noaa.gov/crw_daily_5km"
                            ],
                            is_refusal=False,
                        )
                    ]
                )
            ]
        )

        tool_use_request = LLMToolUseContent(
            id="abc123",
            name="DataAnalysisTool",
            input={"natural_language_query": "sea surface temperature"},
        )
        data_search_tool = DataSearchTool(mock_dataset_picking_agent)

        result = await to_thread.run_sync(
            data_search_tool.execute, tool_use_request
        )

        assert result.id == "abc123"
        assert result.status == "success"
        assert len(result.content) == 1
        assert result.content[0].model_dump(mode="json") == JSONContent(
            data={
                "datasets": [CRW_SST_DATASET_JSON],
                "error_message": None,
            }
        ).model_dump(mode="json")

    @pytest.mark.asyncio
    async def test_data_search_tool_ignores_tool_call_messages(self):
        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run.return_value = MockAsyncIterator(
            [
                DatasetPickingAgentMessage(
                    parsed_content=[
                        DatasetResponseTextContent(
                            explanation="I will make a tool call for that.",
                            result=None,
                            is_refusal=False,
                        ),
                        LLMToolUseContent(
                            id="some-id",
                            name="some-name",
                            input={},
                        ),
                    ]
                ),
                DatasetPickingAgentMessage(
                    parsed_content=[
                        DatasetResponseTextContent(
                            explanation="",
                            result=[
                                "http://kmnlp-demo.noaa.gov/crw_daily_5km"
                            ],
                            is_refusal=False,
                        )
                    ]
                ),
            ]
        )

        tool_use_request = LLMToolUseContent(
            id="abc123",
            name="DataAnalysisTool",
            input={"natural_language_query": "sea surface temperature"},
        )
        data_search_tool = DataSearchTool(mock_dataset_picking_agent)

        result = await to_thread.run_sync(
            data_search_tool.execute, tool_use_request
        )

        assert result.id == "abc123"
        assert result.status == "success"
        assert len(result.content) == 1
        assert result.content[0].model_dump(mode="json") == JSONContent(
            data={
                "datasets": [CRW_SST_DATASET_JSON],
                "error_message": None,
            }
        ).model_dump(mode="json")

    @pytest.mark.asyncio
    async def test_data_search_tool_handles_json_decode_error(self):
        tool_use_request = LLMToolUseContent(
            id="abc123",
            name="DataAnalysisTool",
            input={"natural_language_query": "sea surface temperature"},
        )

        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run.side_effect = JSONDecodeError(
            "invalid json", "{some invalid json", 0
        )

        data_search_tool = DataSearchTool(mock_dataset_picking_agent)
        result = await to_thread.run_sync(
            data_search_tool.execute, tool_use_request
        )

        assert result.id == "abc123"
        assert result.status == "error"
        assert result.content == [
            TextContent(text="Error parsing JSON response")
        ]

    @pytest.mark.asyncio
    async def test_data_search_tool_handle_pydantic_validation_error(self):
        tool_use_request = LLMToolUseContent(
            id="abc123",
            name="DataAnalysisTool",
            input={"natural_language_query": "sea surface temperature"},
        )

        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run.side_effect = EnrichedValidationError(
            error=ValidationError("some title", []), data="some data"
        )

        data_search_tool = DataSearchTool(mock_dataset_picking_agent)
        result = await to_thread.run_sync(
            data_search_tool.execute, tool_use_request
        )

        assert result.id == "abc123"
        assert result.status == "error"
        assert result.content == [
            TextContent(text="Validation error parsing dataset")
        ]

    @pytest.mark.asyncio
    async def test_data_search_tool_handle_too_many_turns_exception(self):
        tool_use_request = LLMToolUseContent(
            id="abc123",
            name="DataAnalysisTool",
            input={"natural_language_query": "sea surface temperature"},
        )

        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run.side_effect = (
            MaxPromptTurnsExceededError(
                [
                    LLMMessage(
                        role="user",
                        content="some query",
                    )
                ]
            )
        )

        data_search_tool = DataSearchTool(mock_dataset_picking_agent)
        result = await to_thread.run_sync(
            data_search_tool.execute, tool_use_request
        )

        assert result.id == "abc123"
        assert result.status == "error"
        assert result.content == [
            TextContent(text="Max tool use turns in DatasetPickingAgent")
        ]

    @pytest.mark.asyncio
    async def test_data_search_tool_handle_unexpected_response_type_error(
        self,
    ):
        tool_use_request = LLMToolUseContent(
            id="abc123",
            name="DataAnalysisTool",
            input={"natural_language_query": "sea surface temperature"},
        )

        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run.side_effect = (
            UnexpectedResponseTypeError(
                Base64ImageContent(media_type="image/jpeg", data="")
            )
        )

        data_search_tool = DataSearchTool(mock_dataset_picking_agent)
        result = await to_thread.run_sync(
            data_search_tool.execute, tool_use_request
        )

        assert result.id == "abc123"
        assert result.status == "error"
        assert result.content == [
            TextContent(text="Unexpected response content type")
        ]

    @pytest.mark.asyncio
    async def test_data_search_tool_unhandled_error_returns_message(self):
        tool_use_request = LLMToolUseContent(
            id="abc123",
            name="DataAnalysisTool",
            input={"natural_language_query": "sea surface temperature"},
        )

        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run.side_effect = ZeroDivisionError

        data_search_tool = DataSearchTool(mock_dataset_picking_agent)
        result = await to_thread.run_sync(
            data_search_tool.execute, tool_use_request
        )

        assert result.id == "abc123"
        assert result.status == "error"
        assert result.content == [
            TextContent(
                text="An unhandled exception was encountered. That exception "
                "was: ZeroDivisionError()"
            )
        ]
