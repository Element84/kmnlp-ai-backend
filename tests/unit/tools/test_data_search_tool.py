from collections.abc import Awaitable, Callable
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest.mock import AsyncMock, Mock

from pydantic_ai import RunContext

from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import DatasetSearchResponse
from llm_agent.context import AgentDeps, ToolProgressUpdate
from llm_agent.tools.agent_tool import ErrorResponse
from llm_agent.tools.data_search_tool import (
    DataSearchTool,
    SearchCompletedUpdate,
    SearchInitiatedUpdate,
    _DataSearchToolQuery,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.dao.test_knowledge_graph_dao import CRW_SST_DATASET_JSON
from tests.utils.mocks.context import MockChatContext

data_search_tool = DataSearchTool(dataset_picking_agent=Mock())


class TestDataSearchTool:
    def test_name(self):
        tool = DataSearchTool(Mock())
        assert tool.name == "DataSearchTool"

    def test_no_error_on_repr(self):
        with does_not_raise():
            _ = repr(data_search_tool)

    async def test_data_search_tool_fetches_dataset_by_variable_id(
        self,
        chat_context: MockChatContext,
        make_run_context: Callable[..., RunContext[AgentDeps]],
    ):
        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run = AsyncMock(
            return_value=DatasetSearchResponse(
                explanation="",
                result=["crw_5km_sst_daily"],
                is_refusal=False,
            )
        )
        tool = DataSearchTool(mock_dataset_picking_agent)

        progress_updates: list[ToolProgressUpdate[Any]] = []
        ctx = make_run_context(chat_context, progress_callback=progress_updates.append)

        result = await tool(
            ctx, _DataSearchToolQuery(natural_language_query="sea surface temperature")
        )

        assert not isinstance(result, ErrorResponse)
        assert result.model_dump(mode="json") == {
            "datasets": [CRW_SST_DATASET_JSON],
            "truncated_dataset_ids": None,
            "explanation": "",
        }

        assert {type(update.update) for update in progress_updates} == {
            SearchInitiatedUpdate,
            SearchCompletedUpdate,
        }
        assert progress_updates[0].update.query == "sea surface temperature"
        assert progress_updates[1].update.query == "sea surface temperature"
        assert progress_updates[1].update.num_datasets == 1

    async def test_data_search_tool_unhandled_error_returns_message(
        self,
        chat_context: MockChatContext,
        make_run_context: Callable[..., RunContext[AgentDeps]],
        invoke_tool_safely: Callable[..., Awaitable[Any]],
    ):
        mock_dataset_picking_agent = Mock()
        mock_dataset_picking_agent.run = AsyncMock(side_effect=ZeroDivisionError)

        tool = DataSearchTool(mock_dataset_picking_agent)
        ctx = make_run_context(chat_context)

        result = await invoke_tool_safely(
            tool, ctx, _DataSearchToolQuery(natural_language_query="sea surface temperature")
        )

        assert isinstance(result, ErrorResponse)
        assert result.message == "An unhandled exception was encountered: `ZeroDivisionError`"
