import logging

from pydantic import BaseModel

from llm_agent.context import ToolCallContext
from llm_agent.tools.llm_tool_typed import ErrorResponse, LLMToolTyped, ToolOutput

log = logging.getLogger(__name__)


class ResponseValidationError(Exception):
    """Exception raised when response validation fails."""


class EmptyOutput(BaseModel):
    """Empty output."""


class ResponseShapingTool[InputT: BaseModel](LLMToolTyped[InputT, InputT, EmptyOutput]):
    """Base class for tools used to enforce structured outputs."""

    def __init__(self, *, name: str, description: str, response_model: type[InputT]) -> None:
        """Constructor.

        Args:
            name (str): Name of the tool.
            description (str): Description of the tool.
            response_model (type[InputT]): Pydantic model describing the expected output.
        """
        _description = f"{description}\nThis tool returns an empty output on success."

        super().__init__(
            name=name,
            description=_description,
            input_model=response_model,
            llm_output_model=EmptyOutput,
            user_output_model=response_model,
        )

    def _execute_typed(
        self,
        context: ToolCallContext,  # noqa: ARG002
        tool_input: InputT,
    ) -> ToolOutput[InputT, EmptyOutput] | ErrorResponse:
        """Typed execution function."""
        try:
            tool_input = self._validate_tool_input(tool_input)
        except ResponseValidationError as e:
            log.exception("Response failed validation: %s", tool_input)
            return ErrorResponse(message=f"Validation error: {e!r}")
        except Exception:
            log.exception("Error during response validation: %s", tool_input)
            return ErrorResponse(message="Unexpected runtime error during validation")

        return ToolOutput(user=tool_input, llm=EmptyOutput())

    def _validate_tool_input(self, tool_input: InputT) -> InputT:
        """Perform extra validation on the tool input.

        Subclasses should override this method to implement custom validation logic.
        """
        return tool_input
