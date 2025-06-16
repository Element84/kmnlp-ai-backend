import datetime
from typing import Any

from e84_geoai_common.llm.models import (
    CLAUDE_BEDROCK_MODEL_IDS,
    BedrockClaudeLLM,
)
from pydantic import ValidationError

DEFAULT_MODEL_NAME = "Claude 3.7 Sonnet"
DEFAULT_LLM = BedrockClaudeLLM(
    model_id=CLAUDE_BEDROCK_MODEL_IDS[DEFAULT_MODEL_NAME]
)


def get_current_date(fmt: str = "%A, %B %d, %Y") -> str:
    """Get current date formatted by default as 'Monday, March 17, 2025'."""
    return datetime.datetime.now(tz=datetime.UTC).strftime(fmt)


class EnrichedValidationError(Exception):
    """Pydantic ValidationError bundled with the input that caused it.

    Attributes:
        error (ValidationError): The original Pydantic ValidationError.
        data (Union[str, dict[str, Any]]): The input data that failed
            validation.
    """

    def __init__(
        self, error: ValidationError, data: str | dict[str, Any]
    ) -> None:
        """Constructor.

        Args:
            error (ValidationError): The original Pydantic ValidationError.
            data (Union[str, dict[str, Any]]): The input data that failed
                validation.
        """
        self.error = error
        self.data = data
