from collections.abc import Callable
from typing import cast

import pytest
from anyio import to_thread
from e84_geoai_common.llm.core import LLMToolUseContent

from llm_agent.context import LLMAgentContext, ToolCallContext
from llm_agent.tools.response_shaping_tools.text_response_tool import (
    TextResponse,
    text_response_tool,
)
from llm_agent.utils.utils import EnrichedLLMToolResultContent


@pytest.fixture
def context(global_context: Callable[..., LLMAgentContext]):
    return ToolCallContext(tool_call_id="", global_context=global_context())


class TestTextResponseTool:
    async def test_io(self, context: ToolCallContext):
        tool_use_request = LLMToolUseContent(
            id="abc123", name=text_response_tool.tool_spec.name, input={"text": "lorem ipsum"}
        )
        result = await to_thread.run_sync(text_response_tool.execute, context, tool_use_request)

        assert isinstance(result, EnrichedLLMToolResultContent)
        result = cast("EnrichedLLMToolResultContent[TextResponse]", result)
        assert result.model_dump(mode="json") == {
            "id": "abc123",
            "status": "success",
            "content": [{"data": {}}],
        }
        assert isinstance(result.data, TextResponse)
        assert result.data.model_dump(mode="json") == tool_use_request.input
