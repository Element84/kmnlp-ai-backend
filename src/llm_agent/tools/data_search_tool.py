import logging
from json import JSONDecodeError

from anyio import from_thread
from e84_geoai_common.util import timed_function
from langfuse import observe  # type: ignore[reportUnknownVariableType]
from pydantic import BaseModel, ConfigDict, Field

from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import (
    DatasetPickingAgent,
    DatasetResponseTextContent,
    MaxPromptTurnsExceededError,
    UnableToExtractJSONFromStringError,
    UnexpectedResponseTypeError,
)
from llm_agent.context import ToolCallContext
from llm_agent.model.dataset import Dataset
from llm_agent.tools.llm_tool_input import LLMToolInput
from llm_agent.tools.llm_tool_typed import ErrorResponse, LLMToolTyped, ToolOutput
from llm_agent.utils.utils import EnrichedValidationError

log = logging.getLogger(__name__)

MAX_NUM_RESULTS = 10

TOOL_DESCRIPTION = f"""\
A tool that searches for relevant climate-related datasets and variables within those datasets, \
given a natural language query.

Results are capped at {MAX_NUM_RESULTS} datasets per call. Call this tool again with the truncated \
dataset IDs to retrieve more results, if needed.
"""


class _DataSearchToolQuery(LLMToolInput):
    model_config = ConfigDict(
        # Title in the JSON schema, w/o leading underscore to reduce confusion.
        title="DataSearchToolQuery",
        extra="forbid",
        frozen=True,
    )

    natural_language_query: str
    dataset_ids: list[str] | None = Field(
        default=None,
        description="Specific dataset IDs to restrict the search to, if known. These can be used "
        "to specify datasets returned in a previous truncated search result. If specified, "
        "natural_language_query will be ignored.",
        max_length=MAX_NUM_RESULTS,
    )


class DataSearchToolResult(BaseModel):
    """DataSearchTool output type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

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


class DataSearchTool(
    LLMToolTyped[_DataSearchToolQuery, DataSearchToolResult, DataSearchToolResult],
):
    """Tool to query to the dataset picking agent and return its responses."""

    def __init__(self, dataset_picking_agent: DatasetPickingAgent) -> None:
        """Initializer."""
        super().__init__(
            name="DataSearchTool",
            description=TOOL_DESCRIPTION,
            input_model=_DataSearchToolQuery,
            user_output_model=DataSearchToolResult,
            llm_output_model=DataSearchToolResult,
        )
        self._dataset_picking_agent = dataset_picking_agent

    @observe(as_type="tool", name=__name__)
    @timed_function()
    def _execute_typed(
        self, context: ToolCallContext, tool_input: _DataSearchToolQuery
    ) -> ToolOutput[DataSearchToolResult, DataSearchToolResult] | ErrorResponse:
        context.update_progress(SearchInitiatedUpdate(query=tool_input.natural_language_query))
        # `from_thread.run` is used to call an asynchronous function from a
        # synchronous function in a worker thread.
        #
        # From the perspective of the function, it effectively makes the
        # asynchronous call synchronous. However, it does still respect `await`
        # (relinquishing the CPU).
        #
        # DO NOT use `from_thread.run_sync` for this purpose. That method will
        # run the relevant code on the main event loop thread.
        out = from_thread.run(self._a_execute_data_search_tool, context, tool_input)
        if isinstance(out, ErrorResponse):
            return out
        return ToolOutput(user=out, llm=out)

    async def _a_execute_data_search_tool(
        self, context: ToolCallContext, query_params: _DataSearchToolQuery
    ) -> DataSearchToolResult | ErrorResponse:
        """Execute Data Search Tool asynchronlously."""
        knowledge_graph_dao = context.global_context.knowledge_graph_dao
        if query_params.dataset_ids:
            # If dataset IDs are provided, fetch those specific datasets directly
            dataset_ids = query_params.dataset_ids
            included_ids, truncated_ids = (
                dataset_ids[:MAX_NUM_RESULTS],
                dataset_ids[MAX_NUM_RESULTS:],
            )
            datasets = knowledge_graph_dao.fetch_datasets(ids=included_ids)
            return DataSearchToolResult(datasets=datasets)

        try:
            parsed_responses = [
                content
                async for assistant_message in self._dataset_picking_agent.run(
                    context.global_context, query_params.natural_language_query
                )
                for content in assistant_message.parsed_content
                if (isinstance(content, DatasetResponseTextContent))
            ]

        except (
            JSONDecodeError,
            EnrichedValidationError,
            UnexpectedResponseTypeError,
            MaxPromptTurnsExceededError,
            UnableToExtractJSONFromStringError,
        ) as e:
            message: str
            match e:
                case JSONDecodeError():
                    message = "Error parsing JSON response"

                case EnrichedValidationError():
                    message = "Validation error parsing dataset"

                case UnexpectedResponseTypeError():
                    message = "Unexpected response content type"

                case MaxPromptTurnsExceededError():
                    message = "Max tool use turns in DatasetPickingAgent"

                case UnableToExtractJSONFromStringError():
                    message = "Error extracting JSON from response"

            return ErrorResponse(message=message)

        dataset_ids = [id_ for response in parsed_responses for id_ in response.result or []]
        included_ids, truncated_ids = dataset_ids[:MAX_NUM_RESULTS], dataset_ids[MAX_NUM_RESULTS:]
        datasets = knowledge_graph_dao.fetch_datasets(ids=included_ids)
        return DataSearchToolResult(
            datasets=datasets,
            truncated_dataset_ids=truncated_ids or None,
        )
