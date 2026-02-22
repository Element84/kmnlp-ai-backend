import json
import logging
from abc import ABC, abstractmethod
from typing import TypedDict

from e84_geoai_common.llm.core import ExecutableLLMTool, JSONContent, LLMTool, LLMToolUseContent
from pydantic import BaseModel, Field, RootModel, ValidationError

from llm_agent.context import ToolCallContext
from llm_agent.utils.utils import EnrichedLLMToolResultContent

log = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    """A response object indicating that the tool call failed."""

    message: str = Field(description="The error message.")


class ToolOutput[UserOutputT: BaseModel, LLMOutputT: BaseModel](TypedDict):
    """Tool outputs for the user and the LLM."""

    user: UserOutputT
    llm: LLMOutputT


class LLMToolTyped[InputT: BaseModel, UserOutputT: BaseModel, LLMOutputT: BaseModel](
    ExecutableLLMTool[ToolCallContext], ABC
):
    """More strongly typed executable LLM tool."""

    input_model: type[InputT]
    llm_output_model: type[LLMOutputT]
    user_output_model: type[UserOutputT]
    error_model: type[ErrorResponse]

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_model: type[InputT],
        llm_output_model: type[LLMOutputT],
        user_output_model: type[UserOutputT],
    ) -> None:
        """Initializer.

        Args:
            name: The name of the tool.
            description: A description of the tool.
            input_model: Pydantic model for the tool input.
            llm_output_model: Pydantic model for the tool output to the LLM.
            user_output_model: Pydantic model for the tool output to the user.
        """
        # Dynamically define a class so that the model schema has a title of
        # "MyToolResponse" instead of "RootModel[Union[Foo, Bar]]".
        output_model_name = f"{name}Response"

        class OutputModel(RootModel[llm_output_model | ErrorResponse], title=output_model_name):
            pass

        super().__init__(
            tool_spec=LLMTool(
                name=name,
                description=description,
                input_model=input_model,
                output_model=OutputModel,
            ),
        )

        self.input_model = input_model
        self.llm_output_model = llm_output_model
        self.user_output_model = user_output_model
        self.error_model = ErrorResponse

    def _unwrap_input(self, tool_use_request: LLMToolUseContent) -> InputT | ErrorResponse:
        """Extract and validate the tool input from an LLMToolUseContent."""
        inp = tool_use_request.input
        try:
            return self.input_model.model_validate(inp)
        except (json.JSONDecodeError, ValidationError) as e:
            match e:
                case json.JSONDecodeError():
                    err_msg = f"JSONDecodeError: {e!s}"
                case ValidationError():
                    errors = e.errors(include_url=False)
                    err_msg = (
                        f"Pydantic ValidationError: tool input does not conform to schema: {errors}"
                    )
            return ErrorResponse(message=err_msg)

    def _wrap_output(
        self,
        tool_use_request: LLMToolUseContent,
        output: ToolOutput[UserOutputT, LLMOutputT] | ErrorResponse,
    ) -> EnrichedLLMToolResultContent[UserOutputT] | EnrichedLLMToolResultContent[ErrorResponse]:
        """Wrap the tool output or error in an EnrichedLLMToolResultContent."""
        if isinstance(output, ErrorResponse):
            error = output
            return EnrichedLLMToolResultContent[ErrorResponse](
                id=tool_use_request.id,
                status="error",
                content=[JSONContent(data=error.model_dump(mode="json"))],
                data=error,
            )
        return EnrichedLLMToolResultContent[self.user_output_model](
            id=tool_use_request.id,
            status="success",
            content=[JSONContent(data=output["llm"].model_dump(mode="json"))],
            data=output["user"],
        )

    def _execute(
        self,
        context: ToolCallContext,
        tool_use_request: LLMToolUseContent,
    ) -> EnrichedLLMToolResultContent[UserOutputT] | EnrichedLLMToolResultContent[ErrorResponse]:
        """Unwrap tool input, execute the tool, and wrap the output."""
        tool_input_or_error = self._unwrap_input(tool_use_request)
        if isinstance(tool_input_or_error, ErrorResponse):
            return self._wrap_output(tool_use_request, tool_input_or_error)

        try:
            output = self._execute_typed(context, tool_input_or_error)
        except Exception as e:
            log.exception("Unhandled exception in tool call: %s", tool_use_request)
            output = ErrorResponse(
                message=(f"An unhandled exception was encountered: `{type(e).__name__}`")
            )

        return self._wrap_output(tool_use_request, output)

    @abstractmethod
    def _execute_typed(
        self, context: ToolCallContext, tool_input: InputT
    ) -> ToolOutput[UserOutputT, LLMOutputT] | ErrorResponse:
        """Typed execution function to be implemented by subclasses."""
