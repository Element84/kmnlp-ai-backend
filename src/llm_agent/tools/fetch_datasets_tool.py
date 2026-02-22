import logging
from datetime import datetime
from typing import Self

from e84_geoai_common.util import timed_function
from langfuse import observe  # type: ignore[reportUnknownVariableType]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_agent.context import ToolCallContext
from llm_agent.model.dataset import Dataset, DatasetSummaryView
from llm_agent.tools.llm_tool_input import LLMToolInput
from llm_agent.tools.llm_tool_typed import LLMToolTyped, ToolOutput

log = logging.getLogger(__name__)


TOOL_DESCRIPTION = """
Query a knowledge graph containing metadata about various climate datasets, and return a list of \
datasets matching the provided filters. Each dataset will include one or more variables, and each \
variable will include one or more dimensions.
"""

MAX_DATASETS_FOR_DETAILED_VIEW = 10


class _TemporalCoverageFilter(LLMToolInput):
    """Temporal coverage filter for filtering datasets by time range."""

    model_config = ConfigDict(
        # Title in the JSON schema, w/o leading underscore to reduce confusion.
        title="TemporalCoverageFilter",
        extra="forbid",
        frozen=True,
    )

    start: datetime | None = Field(
        description="Start of the time range, inclusive. Can be None to indicate no lower bound."
    )
    end: datetime | None = Field(
        description="End of the time range, inclusive. Can be None to indicate no upper bound."
    )

    @model_validator(mode="after")
    def _validate_end_after_start(self) -> Self:
        if self.start and self.end and self.start > self.end:
            msg = f"Start ({self.start}) must be less than or equal to end ({self.end})"
            raise ValueError(msg)
        return self


class _FetchDatasetsToolQuery(LLMToolInput):
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
        "The regex will always be applied in a case-insensitive manner.",
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


class FetchDatasetsTool(
    LLMToolTyped[_FetchDatasetsToolQuery, _FetchDatasetsToolResult, _FetchDatasetsToolResult]
):
    """A tool to fetch datasets from the knowledge graph."""

    def __init__(self) -> None:
        """Initializer."""
        super().__init__(
            name="FetchDatasetsTool",
            description=TOOL_DESCRIPTION,
            input_model=_FetchDatasetsToolQuery,
            user_output_model=_FetchDatasetsToolResult,
            llm_output_model=_FetchDatasetsToolResult,
        )

    @observe(as_type="tool", name=__name__)
    @timed_function()
    def _execute_typed(
        self,
        context: ToolCallContext,
        tool_input: _FetchDatasetsToolQuery,
    ) -> ToolOutput[_FetchDatasetsToolResult, _FetchDatasetsToolResult]:
        if tool_input.temporal_coverage_filter:
            min_start_time = tool_input.temporal_coverage_filter.start
            max_end_time = tool_input.temporal_coverage_filter.end
        else:
            min_start_time = None
            max_end_time = None

        result = context.global_context.knowledge_graph_dao.fetch_datasets(
            ids=tool_input.id_filter,
            start_time=min_start_time,
            end_time=max_end_time,
            regex_pattern=tool_input.regex_filter,
        )

        if not tool_input.detailed_view or len(result) > MAX_DATASETS_FOR_DETAILED_VIEW:
            # convert to summary view to save tokens
            result = [ds.summary_view() for ds in result]

        out = _FetchDatasetsToolResult(datasets=result)
        return ToolOutput(user=out, llm=out)
