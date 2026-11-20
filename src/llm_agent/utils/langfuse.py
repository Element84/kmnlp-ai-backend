"""Helpers for Langfuse tracing."""

from collections.abc import Iterable
from typing import Any, cast

from pydantic_ai import AgentRunResultEvent
from pydantic_core import to_json


def stream_events_to_final_output(events: Iterable[Any]) -> str:
    """Reduce an agent's streamed events to the run's final output.

    Intended as the ``transform_to_string`` argument of ``@observe`` on a streaming
    method. Langfuse otherwise records the entire event list -- every text and
    thinking delta -- as the span's output.
    """
    for event in events:
        if isinstance(event, AgentRunResultEvent):
            output: Any = cast("AgentRunResultEvent[Any]", event).result.output
            return output if isinstance(output, str) else to_json(output).decode()
    return ""
