import logging
from json import JSONDecodeError

from anyio import from_thread
from e84_geoai_common.llm.core import (
    JSONContent,
    LLMToolResultContent,
    LLMToolUseContent,
    TextContent,
)
from e84_geoai_common.llm.models import (
    CLAUDE_BEDROCK_MODEL_IDS,
    BedrockClaudeLLM,
)
from pydantic import BaseModel, ConfigDict

from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import (
    DatasetPickingAgent,
    DatasetResponseTextContent,
    MaxPromptTurnsExceededError,
    UnexpectedResponseTypeError,
)
from llm_agent.dao import KnowledgeGraphDAO, knowledge_graph_dao
from llm_agent.model.dataset import Dataset
from llm_agent.tools.base_tool import BaseTool
from llm_agent.utils import EnrichedValidationError

log = logging.getLogger(__name__)

TOOL_DESCRIPTION = "A tool that searches for relevant climate-related "
"datasets and variables within those datasets, given a natural language query."

# Not upgrading to Claude 3.7 because it seems to consistently output extra
# stuff besides the JSON, resulting in JSON decoding errors.
DEFAULT_MODEL_NAME = "Claude 3.5 Sonnet"


class _DataSearchToolQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    natural_language_query: str


class DataSearchToolResult(BaseModel):
    """DataSearchTool output type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    datasets: list[Dataset] | None
    error_message: str | None = None


class DataSearchTool(BaseTool):
    """Tool to query to the dataset picking agent and return its responses."""

    _dataset_picking_agent: DatasetPickingAgent

    def __init__(
        self,
        dataset_picking_agent: DatasetPickingAgent,
        knowledge_graph_dao: KnowledgeGraphDAO = knowledge_graph_dao,
    ) -> None:
        """Initializer."""
        super().__init__(
            description=TOOL_DESCRIPTION,
            input_model=_DataSearchToolQuery,
            output_model=DataSearchToolResult,
            execution_func=self._execute_data_search_tool,
        )

        self._dataset_picking_agent = dataset_picking_agent
        self._knowledge_graph_dao = knowledge_graph_dao

    def _execute_data_search_tool(
        self, tool_use_request: LLMToolUseContent
    ) -> LLMToolResultContent:
        # `from_thread.run` is used to call an asynchronous function from a
        # synchronous function in a worker thread.
        #
        # From the perspective of the function, it effectively makes the
        # asynchronous call synchronous. However, it does still respect `await`
        # (relinquishing the CPU).
        #
        # DO NOT use `from_thread.run_sync` for this purpose. That method will
        # run the relevant code on the main event loop thread.
        return from_thread.run(
            self._a_execute_data_search_tool, tool_use_request
        )

    async def _a_execute_data_search_tool(
        self, tool_use_request: LLMToolUseContent
    ) -> LLMToolResultContent:
        """Execute Data Search Tool asynchronlously."""
        query_params = _DataSearchToolQuery.model_validate(
            tool_use_request.input
        )

        try:
            parsed_responses = [
                content
                async for assistant_message in self._dataset_picking_agent.run(
                    query_params.natural_language_query
                )
                for content in assistant_message.parsed_content
                if (isinstance(content, DatasetResponseTextContent))
            ]

        except (
            JSONDecodeError,
            EnrichedValidationError,
            UnexpectedResponseTypeError,
            MaxPromptTurnsExceededError,
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

            return LLMToolResultContent(
                id=tool_use_request.id,
                content=[TextContent(text=message)],
                status="error",
            )

        dataset_iris = [
            iri
            for response in parsed_responses
            for iri in response.result or []
        ]
        datasets = self._knowledge_graph_dao.get_datasets_by_iris(dataset_iris)
        return LLMToolResultContent(
            id=tool_use_request.id,
            content=[
                JSONContent(
                    data=DataSearchToolResult(datasets=datasets).model_dump(
                        mode="json"
                    )
                )
            ],
            status="success",
        )


data_search_tool = DataSearchTool(
    DatasetPickingAgent(
        llm=BedrockClaudeLLM(
            model_id=CLAUDE_BEDROCK_MODEL_IDS[DEFAULT_MODEL_NAME]
        )
    )
)
