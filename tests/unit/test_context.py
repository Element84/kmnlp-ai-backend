from collections.abc import Callable
from typing import Any

from langfuse._utils.serializer import EventSerializer

from llm_agent.context import LLMAgentContext, ToolCallContext, ToolProgressUpdate


class TestToolCallContext:
    def test_serialization_without_progress_callback(
        self, global_context: Callable[..., LLMAgentContext]
    ):
        context = ToolCallContext(
            tool_call_id="the-id",
            global_context=global_context(),
            progress_callback=None,
        )
        serialized = EventSerializer().default(context)

        assert serialized == {"tool_call_id": "the-id", "progress_callback": None}

    def test_serialization_with_progress_callback(
        self, global_context: Callable[..., LLMAgentContext]
    ):
        def _callback(_: ToolProgressUpdate[Any]) -> None:
            return None

        context = ToolCallContext(
            tool_call_id="the-id",
            global_context=global_context(),
            progress_callback=_callback,
        )
        serialized = EventSerializer().default(context)

        assert serialized["tool_call_id"] == "the-id"
        assert serialized["progress_callback"] is not None
