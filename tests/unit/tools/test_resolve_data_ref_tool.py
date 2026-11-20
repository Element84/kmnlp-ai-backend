from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

from pydantic_ai import BinaryContent, RunContext

from llm_agent.context import AgentDeps
from llm_agent.tools.data_analysis_tool.types import ImageRef
from llm_agent.tools.resolve_data_ref_tool import (
    ResolveDataRefTool,
    ResolveDataRefToolInput,
)
from tests.utils.mocks.context import MockChatContext

resolve_data_ref_tool = ResolveDataRefTool()


class TestResolveDataRefTool:
    def test_name_set_correctly(self):
        assert ResolveDataRefTool().name == "ResolveDataRef"

    def test_no_error_on_repr(self):
        with does_not_raise():
            _ = repr(resolve_data_ref_tool)

    async def test_resolve_image_ref_execution(
        self,
        chat_context: MockChatContext,
        make_run_context: Callable[..., RunContext[AgentDeps]],
    ):
        image_bytes = b"\x89PNG\r\n\x1a\n_fake_png_bytes_"
        key = chat_context.persist_data(image_bytes, data_type="image")
        image_ref = ImageRef(key=key, name="an_image", description="a test image")

        tool = ResolveDataRefTool()
        ctx = make_run_context(chat_context)
        result = await tool(ctx, ResolveDataRefToolInput(data_ref=image_ref))

        # Pyright misreads the Pydantic-dataclass __init__ as (*args, **kwargs).
        expected = BinaryContent(data=image_bytes, media_type="image/png")  # pyright: ignore[reportCallIssue]
        assert result == expected
