import logging
from collections.abc import Callable

from e84_geoai_common.llm.core import (
    LLMTool,
    LLMToolResultContent,
    LLMToolUseContent,
    TextContent,
)
from pydantic import BaseModel

log = logging.getLogger(__name__)


class BaseTool(LLMTool):
    """Base LLM Tool that manages generically useful cases.

    This base class is intended to provide a facility to:
    * Make it easy to construct a tool type that provides an execution function
    with enforced types.
    * Wrap unhandled exceptions in an `LLMToolResultContent` so that the
    execution of a conversation can continue.
    """

    _execution_function: Callable[[LLMToolUseContent], LLMToolResultContent]

    def __init__(
        self,
        description: str,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        execution_func: Callable[[LLMToolUseContent], LLMToolResultContent],
    ) -> None:
        """Initializer."""
        super().__init__(
            name=str(type(self).__name__),
            description=description,
            input_model=input_model,
            output_model=output_model,
            execution_func=self._exec_and_wrap_unhandled_exception,
        )
        self._execution_function = execution_func

    def _exec_and_wrap_unhandled_exception(
        self, tool_use_request: LLMToolUseContent
    ) -> LLMToolResultContent:
        try:
            return self._execution_function(tool_use_request)
        except Exception as e:
            log.exception(
                "Unhandled exception in tool with name: %s", self.name
            )
            return LLMToolResultContent(
                id=tool_use_request.id,
                content=[
                    TextContent(
                        text=(
                            "An unhandled exception was encountered. That "
                            f"exception was: {e!r}"
                        )
                    )
                ],
                status="error",
            )
