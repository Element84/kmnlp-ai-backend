from typing import Any

from langfuse._utils.serializer import EventSerializer

from llm_agent.context import ToolCallContext, ToolProgressUpdate
from tests.utils.mocks.context import MockChatContext


class TestLangfuseContextSerialization:
    def test_serialization_without_progress_callback(self, chat_context: MockChatContext):
        context = ToolCallContext(
            tool_call_id="the-id",
            chat_context=chat_context,
            progress_callback=None,
        )
        serialized = EventSerializer().default(context)

        assert serialized == {
            "tool_call_id": "the-id",
            "chat_context": {},
            "progress_callback": None,
        }

    def test_serialization_with_progress_callback(self, chat_context: MockChatContext):
        def _callback(_: ToolProgressUpdate[Any]) -> None:
            return None

        context = ToolCallContext(
            tool_call_id="the-id",
            chat_context=chat_context,
            progress_callback=_callback,
        )
        serialized = EventSerializer().default(context)

        assert serialized == {
            "tool_call_id": "the-id",
            "chat_context": {},
            "progress_callback": {},
        }
