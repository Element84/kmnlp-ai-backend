"""LLM Agents and associated utils."""

import atexit

from e84_geoai_common.llm.models.claude import CLAUDE_4_6_SONNET
from langfuse import get_client

PYTHON_AGENT_DEFAULT_LLM_MODEL_ID = CLAUDE_4_6_SONNET
ALGORITHM_AGENT_DEFAULT_LLM_MODEL_ID = CLAUDE_4_6_SONNET
DATASET_PICKING_AGENT_DEFAULT_LLM_MODEL_ID = CLAUDE_4_6_SONNET
ORCHESTRATION_AGENT_DEFAULT_LLM_MODEL_ID = CLAUDE_4_6_SONNET


def _flush_langfuse_events() -> None:
    """Flush Langfuse Events.

    The Langfuse client batches events. If the program exits, the last batch of
    events may not have been set yet. This function sends any remaining events
    to Langfuse.
    """
    langfuse_client = get_client()
    langfuse_client.flush()


# Attempt to flush events when the process exits
atexit.register(_flush_langfuse_events)
