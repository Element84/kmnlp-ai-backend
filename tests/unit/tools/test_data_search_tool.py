from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import nullcontext as does_not_raise
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast
from unittest.mock import Mock

import pytest
from anyio import to_thread
from e84_geoai_common.llm.core import (
    Base64ImageContent,
    JSONContent,
    LLMToolUseContent,
    LLMUserMessage,
)
from pydantic import ValidationError

from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import (
    DatasetPickingAgentMessage,
    DatasetResponseTextContent,
    MaxPromptTurnsExceededError,
    UnableToExtractJSONFromStringError,
    UnexpectedResponseTypeError,
)
from llm_agent.context import LLMAgentContext, ToolCallContext, ToolProgressUpdate
from llm_agent.tools.data_search_tool import (
    DataSearchTool,
    DataSearchToolResult,
    SearchInitiatedUpdate,
)
from llm_agent.utils.utils import EnrichedLLMToolResultContent, EnrichedValidationError
from tests.unit.dao.test_knowledge_graph_dao import CRW_SST_DATASET_JSON

if TYPE_CHECKING:
    from pydantic import RootModel

    from llm_agent.tools.llm_tool_typed import ErrorResponse


@pytest.fixture
def context(global_context: Callable[..., LLMAgentContext]):
    return ToolCallContext(tool_call_id="", global_context=global_context())


data_search_tool = DataSearchTool(dataset_picking_agent=Mock())

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
    @classmethod
    def setup_class(cls) -> None:
        cls.tool = data_search_tool
        cls.tool_cls = DataSearchTool
        cls.tool_input = {"natural_language_query": "sea surface temperature"}

    def test_name(self):
        tool = DataSearchTool(Mock())
        assert tool.tool_spec.name == "DataSearchTool"

    def test_no_error_on_repr(self):
        with does_not_raise():
            _ = repr(self.tool)

    def test_output_model(self):
        assert self.tool.tool_spec.output_model is not None
        assert (
            self.tool.tool_spec.output_model.model_json_schema()["title"]
            == "DataSearchToolResponse"
        )
        success_response = {
            "datasets": [
                {
                    "id": "id1",
                    "source": {
                        "format": "zarr",
                        "uri": "source1",
                    },
                    "label": "label1",
                    "description": "description1",
                    "comment": "comment1",
                    "temporal_coverage": [
                        "2020-01-01T00:00:00Z",
                        "2024-11-30T12:00:00Z",
                    ],
                    "temporal_resolution": "P1D",
                    "spatial_coverage": "global",
                    "variables": [],
                },
            ]
        }

        with does_not_raise():
            res = cast(
                "RootModel[DataSearchToolResult | ErrorResponse]",
                self.tool.tool_spec.output_model.model_validate(success_response),
            )
        assert isinstance(res.root, self.tool.user_output_model)

        error_response = {"message": "Error"}
        with does_not_raise():
            err = cast(
                "RootModel[DataSearchToolResult | ErrorResponse]",
                self.tool.tool_spec.output_model.model_validate(error_response),
            )
        assert isinstance(err.root, self.tool.error_model)  # pyright: ignore[reportPrivateUsage]

    async def test_data_search_tool_fetches_dataset_by_variable_id(
        self, global_context: Callable[..., LLMAgentContext]
    ):
        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run.return_value = MockAsyncIterator(
            [
                DatasetPickingAgentMessage(
                    parsed_content=[
                        DatasetResponseTextContent(
                            explanation="",
                            result=["crw_daily_5km"],
                            is_refusal=False,
                        )
                    ]
                )
            ]
        )
        data_search_tool = self.tool_cls(mock_dataset_picking_agent)

        progress_updates: list[ToolProgressUpdate[Any]] = []

        def progress_callback(update: ToolProgressUpdate[Any]) -> None:
            progress_updates.append(update)

        tool_use_request = LLMToolUseContent(
            id="abc123",
            name="DataAnalysisTool",
            input={"natural_language_query": "sea surface temperature"},
        )

        tool_call_context = ToolCallContext(
            tool_call_id=tool_use_request.id,
            global_context=global_context(),
            progress_callback=progress_callback,
        )

        result = await to_thread.run_sync(
            data_search_tool.execute, tool_call_context, tool_use_request
        )

        assert isinstance(result, EnrichedLLMToolResultContent)
        assert result.id == "abc123"
        assert result.status == "success"
        assert len(result.content) == 1
        assert result.content[0].model_dump(mode="json") == {
            "data": {
                "datasets": [CRW_SST_DATASET_JSON],
                "truncated_dataset_ids": None,
            }
        }

        assert len(progress_updates) == 1
        assert isinstance(progress_updates[0].update, SearchInitiatedUpdate)
        assert progress_updates[0].update.query == "sea surface temperature"

    async def test_data_search_tool_ignores_tool_call_messages(self, context: ToolCallContext):
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
                            result=["crw_daily_5km"],
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
        data_search_tool = self.tool_cls(mock_dataset_picking_agent)

        result = await to_thread.run_sync(data_search_tool.execute, context, tool_use_request)

        assert isinstance(result, EnrichedLLMToolResultContent)
        assert result.id == "abc123"
        assert result.status == "success"
        assert len(result.content) == 1
        assert result.content[0].model_dump(mode="json") == {
            "data": {
                "datasets": [CRW_SST_DATASET_JSON],
                "truncated_dataset_ids": None,
            }
        }

    async def test_data_search_tool_handles_json_decode_error(self, context: ToolCallContext):
        tool_use_request = LLMToolUseContent(
            id="abc123",
            name="DataAnalysisTool",
            input={"natural_language_query": "sea surface temperature"},
        )

        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run.side_effect = JSONDecodeError(
            "invalid json", "{some invalid json", 0
        )

        data_search_tool = self.tool_cls(mock_dataset_picking_agent)
        result = await to_thread.run_sync(data_search_tool.execute, context, tool_use_request)

        assert result.id == "abc123"
        assert result.status == "error"
        assert result.content == [JSONContent(data={"message": "Error parsing JSON response"})]

    async def test_data_search_tool_handle_pydantic_validation_error(
        self, context: ToolCallContext
    ):
        tool_use_request = LLMToolUseContent(
            id="abc123",
            name="DataAnalysisTool",
            input={"natural_language_query": "sea surface temperature"},
        )

        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run.side_effect = EnrichedValidationError(
            error=ValidationError("some title", []), data="some data"
        )

        data_search_tool = self.tool_cls(mock_dataset_picking_agent)
        result = await to_thread.run_sync(data_search_tool.execute, context, tool_use_request)

        assert result.id == "abc123"
        assert result.status == "error"
        assert result.content == [JSONContent(data={"message": "Validation error parsing dataset"})]

    async def test_data_search_tool_handle_too_many_turns_exception(self, context: ToolCallContext):
        tool_use_request = LLMToolUseContent(
            id="abc123",
            name="DataAnalysisTool",
            input={"natural_language_query": "sea surface temperature"},
        )

        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run.side_effect = MaxPromptTurnsExceededError(
            [LLMUserMessage(content="some query")]
        )

        data_search_tool = self.tool_cls(mock_dataset_picking_agent)
        result = await to_thread.run_sync(data_search_tool.execute, context, tool_use_request)

        assert result.id == "abc123"
        assert result.status == "error"
        assert result.content == [
            JSONContent(data={"message": "Max tool use turns in DatasetPickingAgent"})
        ]

    async def test_data_search_tool_handle_unexpected_response_type_error(
        self, context: ToolCallContext
    ):
        tool_use_request = LLMToolUseContent(
            id="abc123",
            name="DataAnalysisTool",
            input={"natural_language_query": "sea surface temperature"},
        )

        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run.side_effect = UnexpectedResponseTypeError(
            Base64ImageContent(media_type="image/jpeg", data="")
        )

        data_search_tool = self.tool_cls(mock_dataset_picking_agent)
        result = await to_thread.run_sync(data_search_tool.execute, context, tool_use_request)

        assert result.id == "abc123"
        assert result.status == "error"
        assert result.content == [JSONContent(data={"message": "Unexpected response content type"})]

    async def test_data_search_tool_handles_value_error(self, context: ToolCallContext):
        tool_use_request = LLMToolUseContent(
            id="abc123",
            name="DataAnalysisTool",
            input={"natural_language_query": "sea surface temperature"},
        )

        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run.side_effect = UnableToExtractJSONFromStringError(
            "string with no json"
        )

        data_search_tool = self.tool_cls(mock_dataset_picking_agent)
        result = await to_thread.run_sync(data_search_tool.execute, context, tool_use_request)

        assert result.id == "abc123"
        assert result.status == "error"
        assert result.content == [
            JSONContent(data={"message": "Error extracting JSON from response"})
        ]

    async def test_data_search_tool_unhandled_error_returns_message(self, context: ToolCallContext):
        tool_use_request = LLMToolUseContent(
            id="abc123",
            name="DataAnalysisTool",
            input={"natural_language_query": "sea surface temperature"},
        )

        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run.side_effect = ZeroDivisionError

        data_search_tool = self.tool_cls(mock_dataset_picking_agent)
        result = await to_thread.run_sync(data_search_tool.execute, context, tool_use_request)

        assert result.id == "abc123"
        assert result.status == "error"
        assert result.content == [
            JSONContent(
                data={"message": ("An unhandled exception was encountered: `ZeroDivisionError`")}
            )
        ]


class TestDataSearchToolWithTracking(TestDataSearchTool):
    @classmethod
    def setup_class(cls) -> None:
        cls.tool = data_search_tool
        cls.tool_cls = DataSearchTool
        cls.tool_input = {
            "tracking": {
                "plan_id": "plan-1",
                "step_id": "step-1",
            },
            "tool_input": {"natural_language_query": "sea surface temperature"},
        }
