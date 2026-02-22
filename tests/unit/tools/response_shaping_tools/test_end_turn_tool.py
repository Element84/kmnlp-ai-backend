from collections.abc import Callable
from typing import cast

import pytest
from anyio import to_thread
from e84_geoai_common.llm.core import LLMToolUseContent

from llm_agent.context import LLMAgentContext, ToolCallContext
from llm_agent.tools.response_shaping_tools.end_turn_tool import EmptyInput, end_turn_tool
from llm_agent.utils.utils import EnrichedLLMToolResultContent


@pytest.fixture
def context(global_context: Callable[..., LLMAgentContext]):
    return ToolCallContext(tool_call_id="", global_context=global_context())


class TestEndTurnTool:
    async def test_io(self, context: ToolCallContext):
        tool_use_request = LLMToolUseContent(
            id="abc123", name=end_turn_tool.tool_spec.name, input={}
        )
        result = await to_thread.run_sync(end_turn_tool.execute, context, tool_use_request)

        assert isinstance(result, EnrichedLLMToolResultContent)
        result = cast("EnrichedLLMToolResultContent[EmptyInput]", result)
        assert result.model_dump(mode="json") == {
            "id": "abc123",
            "status": "success",
            "content": [{"data": {}}],
        }
        assert isinstance(result.data, EmptyInput)
        assert result.data.model_dump(mode="json") == tool_use_request.input
