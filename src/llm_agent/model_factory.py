"""Helpers for constructing Pydantic AI models backed by Amazon Bedrock."""

from typing import TYPE_CHECKING

from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from pydantic_ai.settings import ModelSettings

CLAUDE_4_6_SONNET = "us.anthropic.claude-sonnet-4-6"


def make_bedrock_model(
    bedrock_client: "BedrockRuntimeClient",
    model_id: str,
    model_settings: "ModelSettings | None" = None,
) -> BedrockConverseModel:
    """Build a Pydantic AI Bedrock model from an existing boto3 Bedrock client.

    Args:
        bedrock_client: An initialized boto3 ``bedrock-runtime`` client.
        model_id: The Bedrock model id to use. Defaults to Claude 4 Sonnet.
        model_settings: Optional Pydantic AI model settings to override the defaults.

    Returns:
        A ``BedrockConverseModel`` ready to pass to an ``Agent``.
    """
    return BedrockConverseModel(
        model_id, provider=BedrockProvider(bedrock_client=bedrock_client), settings=model_settings
    )
