from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic_ai import RunContext

from llm_agent.context import AgentDeps
from llm_agent.tools.agent_tool import ErrorResponse
from llm_agent.tools.execute_sparql_query_tool import (
    ExecuteSparqlQueryTool,
    _ExecuteSparqlQueryToolQuery,  # pyright: ignore[reportPrivateUsage]
)
from tests.utils.mocks.context import MockChatContext


class TestExecuteSparqlQueryTool:
    def test_name(self):
        tool = ExecuteSparqlQueryTool()
        assert tool.name == "ExecuteSparqlQueryTool"

    def test_definition_contains_shacl(self):
        tool = ExecuteSparqlQueryTool()
        assert ":DatasetShape a sh:NodeShape" in tool.description

    async def test_happy_path(
        self,
        make_chat_context: Callable[..., MockChatContext],
        make_run_context: Callable[..., RunContext[AgentDeps]],
    ):
        mock_dao = Mock()
        mock_dao.execute_read_only_query.return_value = ["some", "stuff"]
        ctx = make_run_context(make_chat_context(knowledge_graph_dao=mock_dao))
        tool = ExecuteSparqlQueryTool()

        result = await tool(ctx, _ExecuteSparqlQueryToolQuery(sparql_query="DESCRIBE ?x"))

        assert result.result == ["some", "stuff"]
        mock_dao.execute_read_only_query.assert_called_once_with("DESCRIBE ?x")

    async def test_unhandled_error_returns_error_response(
        self,
        make_chat_context: Callable[..., MockChatContext],
        make_run_context: Callable[..., RunContext[AgentDeps]],
        invoke_tool_safely: Callable[..., Awaitable[Any]],
    ):
        """An unexpected exception is caught by the safety net and returned as an error."""
        mock_dao = Mock()
        mock_dao.execute_read_only_query.side_effect = ZeroDivisionError
        ctx = make_run_context(make_chat_context(knowledge_graph_dao=mock_dao))
        tool = ExecuteSparqlQueryTool()

        result = await invoke_tool_safely(
            tool, ctx, _ExecuteSparqlQueryToolQuery(sparql_query="DESCRIBE ?x")
        )

        assert isinstance(result, ErrorResponse)
        assert result.message == "An unhandled exception was encountered: `ZeroDivisionError`"

    async def test_call_propagates_unexpected_errors(
        self,
        make_chat_context: Callable[..., MockChatContext],
        make_run_context: Callable[..., RunContext[AgentDeps]],
    ):
        """The pure ``__call__`` does not swallow exceptions (the wrapper does)."""
        mock_dao = Mock()
        mock_dao.execute_read_only_query.side_effect = ZeroDivisionError
        ctx = make_run_context(make_chat_context(knowledge_graph_dao=mock_dao))
        tool = ExecuteSparqlQueryTool()

        with pytest.raises(ZeroDivisionError):
            await tool(ctx, _ExecuteSparqlQueryToolQuery(sparql_query="DESCRIBE ?x"))
