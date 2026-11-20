import os
from typing import Any

import boto3
import botocore.config
import numpy as np
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from pydantic import BaseModel

from llm_agent.context import ChatContext, DataRef, LLMAgentContext


class MockChatContext(ChatContext):
    """A mock ChatContext for testing purposes."""

    def __init__(self, global_context: LLMAgentContext) -> None:  # noqa: D107
        self._global_context = global_context

    @property
    def global_context(self) -> LLMAgentContext:
        """Return the global context."""
        return self._global_context

    def persist_data(self, data_or_path: object, *, data_type: str) -> str:
        """Mock persist_data method that returns a dummy data reference."""
        return f"mock_data_ref_for_{data_type}_{hash(data_or_path)}"

    def retrieve_data(self, data_ref: DataRef) -> Any:  # noqa: ANN401
        """Mock retrieve_data method that returns a dummy data object."""
        return f"mock_data_for_{data_ref.data_ref_type}_{data_ref.key}"


class SummaryStats(BaseModel):
    """Summary statistics."""

    mean: float
    std: float
    min: float
    max: float
    values: list[float]

    @staticmethod
    def from_values(values: list[float]) -> "SummaryStats":
        """Populate from a list of values."""
        values_arr = np.array(values)
        return SummaryStats(
            mean=float(values_arr.mean()),
            std=float(values_arr.std()),
            min=float(values_arr.min()),
            max=float(values_arr.max()),
            values=values,
        )


def build_eval_client() -> BedrockRuntimeClient:
    """Build client to use for evaluations.

    In CI, it is proxied through LiteLLM. The `AWS_BEARER_TOKEN_BEDROCK`
    variable is set to an API Key from LiteLLM, which requires us to go through
    the proxy for the requests to authenticate.

    NOTE: (kndl) Should a future developer decide to instead remove that
    environment variable, there's no obvious downside to the evaluations (other
    than that their costs won't be limited by LiteLLM).
    """
    return boto3.client(  # type: ignore[reportUnknownMemberType]
        "bedrock-runtime",
        config=botocore.config.Config(read_timeout=300, connect_timeout=1),
        endpoint_url=os.getenv("LITE_LLM_PROXY_URL"),
    )
