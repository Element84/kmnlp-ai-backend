from collections.abc import Callable
from unittest.mock import Mock

from e84_geoai_common.llm.core import LLMToolUseContent

from llm_agent.context import LLMAgentContext, ToolCallContext
from llm_agent.tools.execute_sparql_query_tool import ExecuteSparqlQueryTool
from llm_agent.utils.utils import EnrichedLLMToolResultContent


class TestExecuteSparqlQueryTool:
    def test_name(self):
        tool = ExecuteSparqlQueryTool()
        assert tool.tool_spec.name == "ExecuteSparqlQueryTool"

    def test_definition_contains_shacl(self):
        tool = ExecuteSparqlQueryTool()
        assert ":DatasetShape a sh:NodeShape" in tool.tool_spec.description

    def test_happy_path(self, global_context: Callable[..., LLMAgentContext]):
        mock_dao = Mock()
        mock_dao.execute_read_only_query.return_value = ["some", "stuff"]
        context = ToolCallContext(
            tool_call_id="",
            global_context=global_context(knowledge_graph_dao=mock_dao),
        )
        tool = ExecuteSparqlQueryTool()

        result = tool.execute(
            context,
            LLMToolUseContent(
                id="some-id",
                name="some-name",
                input={"sparql_query": "DESCRIBE ?x"},
            ),
        )

        assert isinstance(result, EnrichedLLMToolResultContent)
        assert result.id == "some-id"
        assert result.status == "success"
        assert len(result.content) == 1
        assert result.content[0].model_dump(mode="json") == {"data": {"result": ["some", "stuff"]}}

    def test_handles_unexpected_errors(self, global_context: Callable[..., LLMAgentContext]):
        mock_dao = Mock()
        mock_dao.execute_read_only_query.side_effect = ZeroDivisionError
        context = ToolCallContext(
            tool_call_id="",
            global_context=global_context(knowledge_graph_dao=mock_dao),
        )
        tool = ExecuteSparqlQueryTool()

        result = tool.execute(
            context,
            LLMToolUseContent(
                id="some-id",
                name="some-name",
                input={"sparql_query": "DESCRIBE ?x"},
            ),
        )

        assert result.id == "some-id"
        assert result.status == "error"
        assert result.content[0].model_dump(mode="json") == {
            "data": {"message": ("An unhandled exception was encountered: `ZeroDivisionError`")}
        }
