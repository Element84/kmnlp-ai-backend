import logging

from pydantic import ConfigDict, Field
from pydantic_ai import BinaryContent, RunContext

from llm_agent.context import AgentDeps
from llm_agent.tools.agent_tool import AgentTool, ToolInput
from llm_agent.tools.data_analysis_tool.types import ImageRef

log = logging.getLogger(__name__)

_TOOL_DESCRIPTION = (
    "Resolves a data reference to its actual content. "
    "Currently only supports resolving image references to images."
)


class ResolveDataRefToolInput(ToolInput):
    """Input for ResolveDataRefTool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_ref: ImageRef = Field(
        description="Reference to the data to resolve. Currently, only supports image references."
    )


class ResolveDataRefTool(AgentTool[ResolveDataRefToolInput, BinaryContent]):
    """Resolves data references to their actual content.

    Currently only supports resolving ``ImageRef``s to PNG image content.
    """

    name = "ResolveDataRef"
    description = _TOOL_DESCRIPTION
    input_model = ResolveDataRefToolInput
    output_type = BinaryContent

    async def __call__(
        self, ctx: RunContext[AgentDeps], tool_input: ResolveDataRefToolInput, /
    ) -> BinaryContent:
        """Resolve an image reference to its PNG image content."""
        image_bytes = ctx.deps.chat_context.retrieve_data(tool_input.data_ref)
        # Pyright misreads the Pydantic-dataclass __init__ as (*args, **kwargs).
        return BinaryContent(data=image_bytes, media_type="image/png")  # pyright: ignore[reportCallIssue]
