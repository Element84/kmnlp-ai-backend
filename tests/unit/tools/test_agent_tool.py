from typing import Any
from unittest.mock import Mock

import pytest

from llm_agent.model_factory import make_bedrock_model
from llm_agent.tools.agent_tool import AgentTool
from llm_agent.tools.data_analysis_tool.data_analysis_tool import DataAnalysisTool
from llm_agent.tools.data_search_tool import DataSearchTool
from llm_agent.tools.execute_sparql_query_tool import ExecuteSparqlQueryTool
from llm_agent.tools.fetch_datasets_tool import FetchDatasetsTool
from llm_agent.tools.resolve_data_ref_tool import ResolveDataRefTool
from llm_agent.tools.response_shaping_tool import ResponseShapingTool, ResponseShapingToolAck
from llm_agent.tools.tracking import TrackingTool

# One instance of every concrete ``AgentTool`` paired with the name of the model
# that its ``output_type`` should surface in the return schema. For union output
# types (``... | ErrorResponse``) the success-result model name is used.
_TOOL_CASES: list[tuple[AgentTool[Any, Any], str]] = [
    (ResolveDataRefTool(), "BinaryContent"),
    (FetchDatasetsTool(), "FetchDatasetsToolResult"),
    (ExecuteSparqlQueryTool(), "ExecuteSparqlQueryToolResult"),
    (DataSearchTool(dataset_picking_agent=Mock()), "DataSearchToolResult"),
    (DataAnalysisTool(make_bedrock_model(Mock(), "__model_id__")), "DataAnalysisToolOutput"),
    (
        ResponseShapingTool(name="Shape", description="d", response_model=ResponseShapingToolAck),
        "ResponseShapingToolAck",
    ),
    # Wrapper tool: must propagate the wrapped tool's return schema.
    (TrackingTool(FetchDatasetsTool()), "FetchDatasetsToolResult"),
]

_CASE_IDS = [type(tool).__name__ for tool, _ in _TOOL_CASES]


def _schema_model_names(schema: dict[str, Any]) -> set[str]:
    """Collect the Pydantic model names declared in a JSON schema.

    A single-model schema carries its name in the top-level ``title``; union
    schemas declare each member under ``$defs``.
    """
    names: set[str] = set()
    if title := schema.get("title"):
        names.add(title)
    names.update(schema.get("$defs", {}).keys())
    return names


class TestAgentToolReturnSchema:
    """Verify ``include_return_schema=True`` produces a correct return schema."""

    @pytest.mark.parametrize("tool", [tool for tool, _ in _TOOL_CASES], ids=_CASE_IDS)
    def test_tool_def_includes_return_schema(self, tool: AgentTool[Any, Any]):
        """Every tool exposes a non-empty return schema in its tool definition.

        ``build_safe_tool`` sets ``include_return_schema=True``; this confirms the
        flag actually takes effect by checking the ``ToolDefinition`` carries both
        the flag and a concrete (non-empty) return schema derived from the tool's
        ``output_type``.
        """
        tool_def = tool.as_pydantic_tool().tool_def

        assert tool_def.include_return_schema is True
        assert tool_def.return_schema, (
            f"{type(tool).__name__} produced an empty return schema; "
            "include_return_schema has no effect"
        )

    @pytest.mark.parametrize(("tool", "expected_model_name"), _TOOL_CASES, ids=_CASE_IDS)
    def test_return_schema_references_output_model(
        self, tool: AgentTool[Any, Any], expected_model_name: str
    ):
        """The return schema is built from the tool's declared ``output_type``.

        Checks that the output model's name appears in the generated schema, so a
        tool whose ``output_type`` drifts from its actual return type is caught.
        """
        schema = tool.as_pydantic_tool().tool_def.return_schema or {}

        assert expected_model_name in _schema_model_names(schema)
