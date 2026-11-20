from langfuse import observe  # type: ignore[reportUnknownVariableType]
from openai import BaseModel
from pydantic import ConfigDict, Field
from pydantic_ai import RunContext

from llm_agent.context import AgentDeps
from llm_agent.model.dataset import Dataset
from llm_agent.tools.agent_tool import AgentTool, ToolInput

TOOL_DESCRIPTION = """
Query a vector store containing embeddings of the descriptions and labels of various climate \
datasets, and return a list of datasets that mostly closely match the provided user query. Each \
dataset will include one or more vairables, adn each variable will include one or more dimensions.
"""


class ContextMissingEmbeddingRepoError(Exception):
    """Context missing embedding repo error.

    This error indicates that the FetchDatasetsFromRagTool was invoked but the
    context passed to it did not contain an embedding repo.

    This is a programmer error and results in the exception flowing all the way
    back to the application level: nothing will be returned to the LLM, the
    agent run will exit, and the error will propagate all the way up the call
    stack to wherever the request originated from.

    Discussion: The propagation up the stack is probably correct for our use
    case. However, a case could be made to simply return this error in the tool
    call. A motivated developer could do so by changing the response class to
    return a value that indicates this error. An alternative would be to
    upgrade to Pydantic AI v2 and return this result as a `ToolFailed` error (which
    ultimately has the same effect).
    """

    def __init__(self) -> None:
        """Initializer."""
        super().__init__(
            "FetchDatasetsFromRagTool was invoked but no embedding store is in context."
        )


class _FetchDatasetsFromRagToolQuery(ToolInput):
    """Query arguments from FetchDatasetsFromRagTool.

    Filter datasets top n that match the provided query from the user.
    """

    model_config = ConfigDict(
        # Title in the JSON schema, w/o leading underscore to reduce confusion.
        title="FetchDatasetsFromRagToolQuery",
        extra="forbid",
        frozen=True,
    )

    user_query: str = Field(
        description=(
            "The string to match against dataset descriptions and labels. This string "
            "is generally expected to match the query from the user."
        )
    )


class _DatasetDistanceScore(BaseModel):
    model_config = ConfigDict(
        # Title in the JSON schema, w/o leading underscore to reduce confusion.
        title="DatasetDistanceScore",
        extra="forbid",
        frozen=True,
    )

    dataset: Dataset = Field(description="A dataset returned in the result.")
    distance: float = Field(
        description="The minimum distance metric returned comparing the "
        "embedding of the query with the embedding of the dataset's field."
    )


class _FetchDatasetsFromRagToolResult(BaseModel):
    model_config = ConfigDict(
        # Title in the JSON schema, w/o leading underscore to reduce confusion.
        title="FetchDatasetsFromRagToolResult",
        extra="forbid",
        frozen=True,
    )

    results: list[_DatasetDistanceScore] = Field(
        description=(
            "The datasets matching the provided filters, paired with the distance minimum distance "
            "score returned from the embedding store, sorted in ascending distance score order. "
            "The list of datasets will be no more than 10 items. If both the description and label "
            "in a dataset were in the top-10 items returned from the vector store, fewer than 10 "
            "datasets will be present."
        )
    )


class FetchDatasetsFromRagTool(
    AgentTool[_FetchDatasetsFromRagToolQuery, _FetchDatasetsFromRagToolResult]
):
    """A tool to fetch datasets through vector embedding of description and label."""

    name = "FetchDatasetsFromRagTool"
    description = TOOL_DESCRIPTION
    input_model = _FetchDatasetsFromRagToolQuery
    output_type = _FetchDatasetsFromRagToolResult

    @observe(as_type="tool", name=__name__)
    async def __call__(
        self,
        ctx: RunContext[AgentDeps],
        query: _FetchDatasetsFromRagToolQuery,
        /,
    ) -> _FetchDatasetsFromRagToolResult:
        """Fetch datasets whose description and label most closely match the user query."""
        embedding_repo = ctx.deps.chat_context.global_context.embedding_repo
        if embedding_repo is None:
            raise ContextMissingEmbeddingRepoError

        knowledge_graph_dao = ctx.deps.chat_context.global_context.knowledge_graph_dao

        document_search_result = embedding_repo.query(query.user_query)

        # List of pairs of (dataset_id, distance)
        datasets_and_distances = [
            (result.dataset_field_document.dataset_id, result.distance)
            for result in document_search_result.results
        ]

        results: dict[str, float] = {}
        for dataset_id, distance in datasets_and_distances:
            if dataset_id not in results or distance < results[dataset_id]:
                results[dataset_id] = distance

        return _FetchDatasetsFromRagToolResult(
            results=[
                _DatasetDistanceScore(
                    dataset=next(iter(knowledge_graph_dao.fetch_datasets(ids=[dataset]))),
                    distance=distance,
                )
                for dataset, distance in results.items()
            ]
        )
