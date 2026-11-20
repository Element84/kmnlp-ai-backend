from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

from dask.distributed import Client
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from natural_language_geocoding.place_lookup import PlaceLookup
from pydantic import BaseModel
from pydantic_ai import RunContext

from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from llm_agent.repository.embedding_index_repository import EmbeddingIndexRepository

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from llm_agent.tools.data_analysis_tool.types import GeometryRef, GeoTIFFRef, ImageRef


class DataPersistenceError(Exception):
    """Raised when data cannot be persisted."""


class DataRefNotResolvedError(Exception):
    """Raised when a data reference cannot be resolved."""

    def __init__(self, data_ref_or_key: "DataRef | str") -> None:
        """Constructor."""
        super().__init__(f"Data reference could not be resolved: {data_ref_or_key}")
        self.data_ref_or_key = data_ref_or_key


# Set `repr=False` so that this object does not serialize in Langfuse input arguments.
@dataclass(repr=False)
class LLMAgentContext:
    """Library-wide context for effectively-global objects.

    This context is meant to be used for dependency injection of objects that
    are meant to be reused throughout the life of the application.
    """

    place_lookup: PlaceLookup
    knowledge_graph_dao: KnowledgeGraphDAO
    embedding_repo: EmbeddingIndexRepository | None
    # IMPORTANT: dask_client must be initialized with asynchronous=True
    # Example: Client(scheduler_address, asynchronous=True) noqa: ERA001
    # This is validated at runtime in compute_dask_array().
    dask_client: Client
    bedrock_client: BedrockRuntimeClient


class DataRef(BaseModel, frozen=True):
    """Reference to persisted data."""

    data_ref_type: str
    key: str
    name: str
    description: str | None = None


class ChatContext(ABC):
    """Context for a single conversation."""

    @property
    @abstractmethod
    def global_context(self) -> LLMAgentContext:
        """Global context for the entire application."""

    @abstractmethod
    def persist_data(
        self,
        data_or_path: Any,  # noqa: ANN401
        *,
        data_type: Literal["geometry", "image", "geotiff", "animation"],
    ) -> str:
        """Persist data and return a reference to it."""

    @overload
    @abstractmethod
    def retrieve_data(self, data_ref: "GeometryRef") -> "BaseGeometry": ...

    @overload
    @abstractmethod
    def retrieve_data(self, data_ref: "GeoTIFFRef") -> str: ...

    @overload
    @abstractmethod
    def retrieve_data(self, data_ref: "ImageRef") -> bytes: ...


class ToolProgressUpdate[UpdateT: BaseModel](BaseModel):
    """Progress update for a tool call."""

    tool_call_id: str
    update: UpdateT


# A callback used to surface intermediate progress updates from within a tool/agent run.
type ProgressCallback = Callable[[ToolProgressUpdate[Any]], None]


class ToolCallContext:
    """Context for LLM tool calls."""

    tool_call_id: str
    chat_context: ChatContext
    progress_callback: Callable[[ToolProgressUpdate[Any]], None] | None

    def __init__(
        self,
        tool_call_id: str,
        chat_context: ChatContext,
        progress_callback: Callable[[ToolProgressUpdate[Any]], None] | None = None,
    ) -> None:
        """Constructor."""
        self.tool_call_id = tool_call_id
        self.chat_context = chat_context
        self.progress_callback = progress_callback

    def update_progress(self, update: BaseModel) -> None:
        """Call the progress callback if it is set."""
        if self.progress_callback is None:
            return

        self.progress_callback(
            ToolProgressUpdate(
                tool_call_id=self.tool_call_id,
                update=update,
            )
        )

    def make_progress_callback_for_nested_tool(
        self,
    ) -> Callable[[ToolProgressUpdate[ToolProgressUpdate[Any]]], None]:
        """Make a progress callback to pass to nested tools.

        A "nested tool" is a tool called by an agent called by the current tool. The goal is to
        wrap progress updates Matryoshka-style, such that the outer-most tool call ID is that of
        the orchestration agent's tool.
        """

        def _progress_callback(update: ToolProgressUpdate[ToolProgressUpdate[Any]]) -> None:
            self.update_progress(update)

        return _progress_callback


@dataclass
class AgentDeps:
    """Base dependencies injected into every Pydantic AI agent in this library."""

    chat_context: ChatContext
    progress_callback: ProgressCallback | None = None

    def update_progress(self, ctx: RunContext["AgentDeps"], update: BaseModel) -> None:
        """Call the progress callback if it is set, tagging it with the tool-call id."""
        if self.progress_callback is None:
            return

        # `tool_call_id` is populated whenever progress is reported from within a tool call.
        tool_call_id = ctx.tool_call_id
        if tool_call_id is None:
            return

        self.progress_callback(
            ToolProgressUpdate(
                tool_call_id=tool_call_id,
                update=update,
            )
        )

    def make_progress_callback_for_nested_tool(
        self, ctx: RunContext["AgentDeps"]
    ) -> Callable[[ToolProgressUpdate[Any]], None]:
        """Make a progress callback to pass to nested agents/tools.

        A "nested tool" is a tool called by an agent called by the current tool. Progress
        updates are wrapped Matryoshka-style, such that the outer-most tool call ID is that
        of the current (outer) tool's call.
        """

        def _progress_callback(update: ToolProgressUpdate[Any]) -> None:
            self.update_progress(ctx, update)

        return _progress_callback
