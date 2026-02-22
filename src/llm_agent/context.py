from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dask.distributed import Client
from mypy_boto3_s3.client import S3Client
from natural_language_geocoding.place_lookup import PlaceLookup
from pydantic import BaseModel, ConfigDict, Field

from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO


# Set `repr=False` so that this object does not serialize in Langfuse input arguments.
@dataclass(repr=False)
class LLMAgentContext:
    """Library-wide context for effectively-global objects.

    This context is meant to be used for dependency injection of objects that
    are meant to be reused throughout the life of the application.
    """

    place_lookup: PlaceLookup
    s3_client: S3Client
    knowledge_graph_dao: KnowledgeGraphDAO
    dask_client: Client


class ToolProgressUpdate[UpdateT: BaseModel](BaseModel):
    """Progress update for a tool call."""

    tool_call_id: str
    update: UpdateT


class ToolCallContext(BaseModel):
    """Context for LLM tool calls."""

    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    tool_call_id: str
    global_context: LLMAgentContext = Field(exclude=True)
    progress_callback: Callable[[ToolProgressUpdate[Any]], None] | None = None

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
