import logging
from functools import partial
from typing import TYPE_CHECKING, Self

from anyio.to_thread import run_sync
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import RunContext

from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import (
    DatasetPickingAgent,
    DatasetSearchQuery,
    DatasetSearchResponse,
)
from llm_agent.context import AgentDeps
from llm_agent.model.dataset import Dataset
from llm_agent.tools.agent_tool import AgentTool, ErrorResponse, ToolInput

if TYPE_CHECKING:
    from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO

log = logging.getLogger(__name__)

MAX_NUM_RESULTS = 10

TOOL_DESCRIPTION = f"""\
A tool that searches for relevant climate-related datasets and variables within those datasets, \
given a natural language query.

Results are capped at {MAX_NUM_RESULTS} datasets per call. Call this tool again with the truncated \
dataset IDs to retrieve more results, if needed.
"""


class _DataSearchToolQuery(ToolInput):
    model_config = ConfigDict(
        # Title in the JSON schema, w/o leading underscore to reduce confusion.
        title="DataSearchToolQuery",
        extra="forbid",
        frozen=True,
    )

    natural_language_query: str | None = Field(
        default=None,
        description="A natural language description of what you are looking for. This should be "
        "in full sentences, not a list of keywords.",
    )
    keywords: list[str] | None = Field(
        default=None,
        description="Input for a keyword-based search as an alternative to a natural "
        "language query.",
    )
    dataset_ids: list[str] | None = Field(
        default=None,
        description="Specific dataset IDs to restrict the search to, if known. These can be used "
        "to specify datasets returned in a previous truncated search result. If specified, "
        "natural_language_query and keywords will be ignored.",
        max_length=MAX_NUM_RESULTS,
    )

    @model_validator(mode="after")
    def check_not_all_none(self) -> Self:
        if (
            self.natural_language_query is None
            and (self.keywords is None or len(self.keywords) == 0)
            and (self.dataset_ids is None or len(self.dataset_ids) == 0)
        ):
            msg = (
                "At least one of natural_language_query, keywords, or dataset_ids must be provided."
            )
            raise ValueError(msg)
        return self


class DataSearchToolResult(BaseModel):
    """DataSearchTool output type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    explanation: str | None = Field(
        default=None,
        description="Explanation of the search results from the sub-agent.",
    )
    datasets: list[Dataset] = Field(
        description=f"List of datasets matching the search query, capped at {MAX_NUM_RESULTS}.",
        max_length=MAX_NUM_RESULTS,
    )
    truncated_dataset_ids: list[str] | None = Field(
        default=None,
        description="Dataset IDs in the search results that were truncated due to the max results "
        "limit. These can be requested in a follow-up call.",
    )


class SearchInitiatedUpdate(BaseModel):
    """Event indicating a search has been initiated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str


class SearchCompletedUpdate(BaseModel):
    """Event indicating a search has been completed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    num_datasets: int


class DataSearchTool(AgentTool[_DataSearchToolQuery, DataSearchToolResult | ErrorResponse]):
    """Tool that queries the dataset picking agent and returns its responses."""

    name = "DataSearchTool"
    description = TOOL_DESCRIPTION
    input_model = _DataSearchToolQuery
    output_type = DataSearchToolResult | ErrorResponse

    def __init__(self, dataset_picking_agent: DatasetPickingAgent) -> None:
        """Initializer."""
        self._dataset_picking_agent = dataset_picking_agent

    async def __call__(
        self, ctx: RunContext[AgentDeps], tool_input: _DataSearchToolQuery, /
    ) -> DataSearchToolResult | ErrorResponse:
        """Search for relevant datasets, delegating to the dataset picking agent."""
        knowledge_graph_dao = ctx.deps.chat_context.global_context.knowledge_graph_dao
        if tool_input.dataset_ids:
            # If dataset IDs are provided, fetch those specific datasets directly.
            dataset_ids = tool_input.dataset_ids
            included_ids, truncated_ids = (
                dataset_ids[:MAX_NUM_RESULTS],
                dataset_ids[MAX_NUM_RESULTS:],
            )
            datasets = await run_sync(partial(knowledge_graph_dao.fetch_datasets, ids=included_ids))
            return DataSearchToolResult(
                explanation=None, datasets=datasets, truncated_dataset_ids=truncated_ids or None
            )

        query = tool_input.natural_language_query or ", ".join(tool_input.keywords or [])
        ctx.deps.update_progress(ctx, SearchInitiatedUpdate(query=query))

        response = await self._dataset_picking_agent.run(
            ctx.deps.chat_context,
            DatasetSearchQuery(query=query),
            progress_callback=ctx.deps.make_progress_callback_for_nested_tool(ctx),
        )

        tool_result = await _dataset_picking_agent_response_to_tool_result(
            response, knowledge_graph_dao
        )

        ctx.deps.update_progress(
            ctx, SearchCompletedUpdate(query=query, num_datasets=len(tool_result.datasets))
        )

        return tool_result


async def _dataset_picking_agent_response_to_tool_result(
    response: DatasetSearchResponse, knowledge_graph_dao: "KnowledgeGraphDAO"
) -> DataSearchToolResult:
    dataset_ids = response.result or []
    included_ids, truncated_ids = dataset_ids[:MAX_NUM_RESULTS], dataset_ids[MAX_NUM_RESULTS:]
    datasets = await run_sync(partial(knowledge_graph_dao.fetch_datasets, ids=included_ids))
    return DataSearchToolResult(
        explanation=response.explanation,
        datasets=datasets,
        truncated_dataset_ids=truncated_ids or None,
    )
