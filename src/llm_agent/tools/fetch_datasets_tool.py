import logging
from datetime import UTC, datetime
from functools import partial
from time import perf_counter
from typing import Self

from anyio.to_thread import run_sync
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_ai import RunContext

from llm_agent.context import AgentDeps
from llm_agent.model.dataset import Dataset, DatasetSummaryView
from llm_agent.tools.agent_tool import AgentTool, ToolInput

log = logging.getLogger(__name__)


TOOL_DESCRIPTION = """
Query a knowledge graph containing metadata about various climate datasets, and return a list of \
datasets matching the provided filters. Each dataset will include one or more variables, and each \
variable will include one or more dimensions.
"""

MAX_DATASETS_FOR_DETAILED_VIEW = 10


class _TemporalCoverageFilter(BaseModel):
    """Temporal coverage filter for filtering datasets by time range."""

    model_config = ConfigDict(
        # Title in the JSON schema, w/o leading underscore to reduce confusion.
        title="TemporalCoverageFilter",
        extra="forbid",
        frozen=True,
    )

    start: datetime | None = Field(
        description="Start of the time range, inclusive. Can be None to indicate no lower bound. "
        "Must be UTC."
    )
    end: datetime | None = Field(
        description="End of the time range, inclusive. Can be None to indicate no upper bound. "
        "Must be UTC."
    )

    @model_validator(mode="after")
    def _validate_end_after_start(self) -> Self:
        if self.start and self.end and self.start > self.end:
            msg = f"Start ({self.start}) must be less than or equal to end ({self.end})"
            raise ValueError(msg)
        return self

    @field_validator("start", "end")
    @classmethod
    def _ensure_utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        return v.replace(tzinfo=UTC) if v.tzinfo is None else v.astimezone(UTC)


class _FetchDatasetsToolQuery(ToolInput):
    """Query arguments for FetchDatasetsTool.

    Filter the returned datasets to those that match the provided filters.
    """

    model_config = ConfigDict(
        # Title in the JSON schema, w/o leading underscore to reduce confusion.
        title="FetchDatasetsToolQuery",
        extra="forbid",
        frozen=True,
    )

    id_filter: list[str] | None = Field(
        default=None,
        description="Optionally filter datasets to those whose dataset ID is in this list.",
    )

    temporal_coverage_filter: _TemporalCoverageFilter | None = Field(
        default=None,
        description=(
            "Filter datasets to those whose temporal coverage intersects with the "
            "provided filter. Both of the times in the time range are "
            "inclusive. No additional logic is applied to the range. If the "
            "value is None, no temporal filtering will be applied."
        ),
    )

    regex_filter: str | None = Field(
        default=None,
        description="Optionally filter datasets to those whose metadata matches this regex."
        "The regex will always be applied in a case-insensitive manner. Avoid making the regex "
        "overly restrictive; e.g., by making strong assumptions about the order of words. "
        "Consider relaxing the regex if you get no results.",
        examples=[r"(?:sea surface temperature)", r"(?:coral reef)|(?:air temperature)"],
    )

    detailed_view: bool = Field(
        default=False,
        description="If True, output will include some extra info plus details of all variables "
        "and dimensions. If False, only variable and dimension names will be included. This can "
        "be used as a second step to more thoroughly inspect datasets once they have been "
        "narrowed down to a few candidates.",
    )


class _FetchDatasetsToolResult(BaseModel):
    model_config = ConfigDict(
        # Title in the JSON schema, w/o leading underscore to reduce confusion.
        title="FetchDatasetsToolResult",
        extra="forbid",
        frozen=True,
    )
    datasets: list[Dataset] | list[DatasetSummaryView] = Field(
        description="The datasets matching the provided filters. The datasets will be in summary "
        "view if the query's detailed_view is False, and in full view otherwise. If the search "
        "returns more than 10 datasets, summary view will be used regardless of the "
        "detailed_view value."
    )


class KnowledgeGraphQueryUpdate(BaseModel):
    """Event indicating a knowledge graph query has been initiated."""

    id_filter: list[str] | None
    temporal_filter: tuple[str | None, str | None] | None
    regex_filter: str | None


class QueryResultsUpdate(BaseModel):
    """Event indicating results have been retrieved for a knowledge graph query."""

    num_datasets: int
    execution_time_sec: float
    detailed_view: bool


class FetchDatasetsTool(AgentTool[_FetchDatasetsToolQuery, _FetchDatasetsToolResult]):
    """A Pydantic AI tool to fetch datasets from the knowledge graph."""

    name = "FetchDatasetsTool"
    description = TOOL_DESCRIPTION
    input_model = _FetchDatasetsToolQuery
    output_type = _FetchDatasetsToolResult

    async def __call__(
        self,
        ctx: RunContext[AgentDeps],
        query: _FetchDatasetsToolQuery,
        /,
    ) -> _FetchDatasetsToolResult:
        """Fetch datasets matching the provided filters from the knowledge graph."""
        if query.temporal_coverage_filter:
            min_start_time = query.temporal_coverage_filter.start
            max_end_time = query.temporal_coverage_filter.end
        else:
            min_start_time = None
            max_end_time = None

        update = KnowledgeGraphQueryUpdate(
            id_filter=query.id_filter,
            temporal_filter=(
                (min_start_time.isoformat() if min_start_time else None),
                (max_end_time.isoformat() if max_end_time else None),
            )
            if query.temporal_coverage_filter
            else None,
            regex_filter=query.regex_filter,
        )
        log.info("FetchDatasetsTool: KnowledgeGraphQueryUpdate: %s", update)
        ctx.deps.update_progress(ctx, update)

        knowledge_graph_dao = ctx.deps.chat_context.global_context.knowledge_graph_dao
        tic = perf_counter()
        # The knowledge graph query is blocking; run it off the event loop.
        result = await run_sync(
            partial(
                knowledge_graph_dao.fetch_datasets,
                ids=query.id_filter,
                start_time=min_start_time,
                end_time=max_end_time,
                regex_pattern=query.regex_filter,
            )
        )
        toc = perf_counter()

        results_update = QueryResultsUpdate(
            num_datasets=len(result),
            execution_time_sec=toc - tic,
            detailed_view=query.detailed_view,
        )
        log.info("FetchDatasetsTool: QueryResultsUpdate: %s", results_update)
        ctx.deps.update_progress(ctx, results_update)

        if not query.detailed_view or len(result) > MAX_DATASETS_FOR_DETAILED_VIEW:
            # convert to summary view to save tokens
            return _FetchDatasetsToolResult(datasets=[ds.summary_view() for ds in result])

        return _FetchDatasetsToolResult(datasets=result)
