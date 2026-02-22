from collections.abc import Callable
from typing import cast

import pytest
from anyio import to_thread
from e84_geoai_common.llm.core import LLMToolUseContent

from llm_agent.context import LLMAgentContext, ToolCallContext
from llm_agent.tools.response_shaping_tools.agent_plan_tool import (
    AgentPlan,
    AgentPlanOrUpdate,
    AgentPlanUpdate,
    agent_plan_tool,
)
from llm_agent.utils.utils import EnrichedLLMToolResultContent


@pytest.fixture
def context(global_context: Callable[..., LLMAgentContext]):
    return ToolCallContext(tool_call_id="", global_context=global_context())


class TestAgentPlanTool:
    async def test_plan(self, context: ToolCallContext):
        tool_use_request = LLMToolUseContent(
            id="abc123",
            name=agent_plan_tool.tool_spec.name,
            input={
                "plan_or_update": {
                    "content_type": "plan",
                    "id": "plan-1",
                    "title": "fancy analysis",
                    "goal": "do stuff",
                    "steps": [
                        {
                            "id": "step-1",
                            "title": "first step",
                            "status": "in_progress",
                            "outcome_summary": None,
                        }
                    ],
                }
            },
        )
        result = await to_thread.run_sync(agent_plan_tool.execute, context, tool_use_request)

        assert isinstance(result, EnrichedLLMToolResultContent)
        result = cast("EnrichedLLMToolResultContent[AgentPlanOrUpdate]", result)
        assert result.model_dump(mode="json") == {
            "id": "abc123",
            "status": "success",
            "content": [{"data": {}}],
        }
        assert isinstance(result.data, AgentPlanOrUpdate)
        assert isinstance(result.data.plan_or_update, AgentPlan)
        assert result.data.model_dump(mode="json") == tool_use_request.input

    async def test_plan_update(self, context: ToolCallContext):
        tool_use_request = LLMToolUseContent(
            id="abc123",
            name=agent_plan_tool.tool_spec.name,
            input={
                "plan_or_update": {
                    "content_type": "plan_update",
                    "id": "plan-1",
                    "step_updates": [
                        {
                            "id": "step-1",
                            "status": "success",
                            "outcome_summary": "done",
                        }
                    ],
                }
            },
        )
        result = await to_thread.run_sync(agent_plan_tool.execute, context, tool_use_request)

        assert isinstance(result, EnrichedLLMToolResultContent)
        result = cast("EnrichedLLMToolResultContent[AgentPlanOrUpdate]", result)
        assert result.model_dump(mode="json") == {
            "id": "abc123",
            "status": "success",
            "content": [{"data": {}}],
        }
        assert isinstance(result.data, AgentPlanOrUpdate)
        assert isinstance(result.data.plan_or_update, AgentPlanUpdate)
        assert result.data.model_dump(mode="json") == tool_use_request.input
