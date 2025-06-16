from e84_geoai_common.llm.core import (
    JSONContent,
    LLMToolResultContent,
    LLMToolUseContent,
)
from pydantic import BaseModel, ConfigDict

from llm_agent.dao import KnowledgeGraphDAO, knowledge_graph_dao
from llm_agent.model.dataset import Dataset
from llm_agent.tools.base_tool import BaseTool


class _FetchDatasetsToolQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _FetchDatasetsToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    datasets: list[Dataset]


TOOL_DESCRIPTION = (
    "Query the Knowledge Graph for its datasets. "
    "Each dataset will have one or more associated data variables."
)


class FetchDatasetsTool(BaseTool):
    """A tool to fetch datasets from the knowledge graph."""

    _knowledge_graph_dao: KnowledgeGraphDAO

    def __init__(self, knowledge_graph_dao: KnowledgeGraphDAO) -> None:
        """Initializer."""
        super().__init__(
            description=TOOL_DESCRIPTION,
            input_model=_FetchDatasetsToolQuery,
            output_model=_FetchDatasetsToolResult,
            execution_func=self._execute_fetch_datasets_tool,
        )
        self._knowledge_graph_dao = knowledge_graph_dao

    def _execute_fetch_datasets_tool(
        self, tool_use_request: LLMToolUseContent
    ) -> LLMToolResultContent:
        result: list[Dataset] = self._knowledge_graph_dao.fetch_variables()
        shaped = _FetchDatasetsToolResult(datasets=result)
        return LLMToolResultContent(
            id=tool_use_request.id,
            content=[JSONContent(data=shaped.model_dump(mode="json"))],
            status="success",
        )


fetch_datasets_tool = FetchDatasetsTool(knowledge_graph_dao)
