import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

import json_repair
from langfuse import observe  # type: ignore[reportUnknownVariableType]
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import RunContext, Tool
from pydantic_ai.exceptions import ModelRetry

from llm_agent.context import AgentDeps

log = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    """A response object indicating that the tool call failed.

    Tools return this (rather than raising) for both anticipated, domain-level
    failures and for unexpected exceptions caught by the agent-invocation safety
    net (see :func:`build_safe_tool`). Returning it as a tool result lets the
    calling agent observe the failure and recover, instead of aborting the run.
    """

    message: str = Field(description="The error message.")


class ToolInput(BaseModel):
    """Base class for tool input models with lenient JSON-string parsing.

    Models sometimes emit a tool call whose arguments are a JSON *string* that
    decodes to an object (occasionally with minor corruption, such as a trailing
    ``}``) instead of a JSON object. This before-validator repairs such strings
    into dicts before field validation so the tool call can still succeed.
    """

    @model_validator(mode="before")
    @classmethod
    def _parse_json_strings(cls, data: Any) -> Any:  # noqa: ANN401
        """Repair a JSON-string payload into a dict before validation."""
        if isinstance(data, str):
            # ``json_repair.loads`` handles cases like an extra ``}``. It always
            # returns something rather than raising; if the data is completely
            # unreasonable, let Pydantic raise the validation error.
            log.warning("Tool input was a string. Attempting to parse as JSON. data: %s", data)
            return json_repair.loads(data)
        return data


class AgentTool[InputT: BaseModel, OutputT](ABC):
    """Base class for Pydantic AI tools used by this project's agents.

    Subclasses must:
    * define ``name``, ``description``, ``input_model`` (the concrete Pydantic
      model for the tool's arguments), and ``output_type`` (the ``__call__``
      return annotation) class attributes, and
    * implement ``async __call__(self, ctx, tool_input)``.

    Pydantic AI builds the tool's args schema from ``input_model``, flattening
    its fields into the tool's parameters. ``output_type`` is used to build the
    tool's return schema (see :func:`build_safe_tool`).

    ``__call__`` is the *pure* tool logic and is allowed to raise. Register an
    instance on an ``Agent`` via :meth:`as_pydantic_tool`, which wraps it in a
    catch-all safety net that converts unhandled exceptions into an
    :class:`ErrorResponse` result (see :func:`build_safe_tool`).
    """

    name: str
    description: str
    input_model: type[InputT]
    output_type: Any

    @abstractmethod
    async def __call__(self, ctx: RunContext[AgentDeps], tool_input: InputT, /) -> OutputT:
        """Execute the tool with the given run context and parsed input model."""
        ...

    def as_pydantic_tool(self) -> Tool[AgentDeps]:
        """Convert this tool into a Pydantic AI ``Tool`` ready for registration."""
        return build_safe_tool(
            self.__call__,
            name=self.name,
            description=self.description,
            input_annotation=self.input_model,
            output_annotation=self.output_type,
        )


def build_safe_tool(
    fn: Callable[[RunContext[AgentDeps], Any], Awaitable[Any]],
    *,
    name: str,
    description: str,
    input_annotation: Any,  # noqa: ANN401
    output_annotation: Any,  # noqa: ANN401
) -> Tool[AgentDeps]:
    """Build a Pydantic AI ``Tool`` that wraps ``fn`` with a catch-all safety net.

    Unhandled exceptions raised by ``fn`` are logged and returned to the calling
    agent as an :class:`ErrorResponse` (matching the legacy tool behavior), so a
    tool crash surfaces as a recoverable tool result rather than aborting the
    agent run. ``ModelRetry`` is re-raised so Pydantic AI's retry flow still
    works.

    ``input_annotation`` and ``output_annotation`` are applied to the wrapper's
    ``tool_input`` parameter and return type so Pydantic AI can build the args
    schema and (when ``include_return_schema`` is enabled) the return schema. The
    wrapper erases ``fn``'s own annotations, and a bare ``TypeVar`` cannot be
    schematized, so the concrete annotations must be supplied here.

    ``fn`` is wrapped in a Langfuse ``tool`` observation named after the tool, so every
    tool registered this way is traced without needing its own ``@observe``.
    """
    observed_fn = observe(as_type="tool", name=name)(fn)

    async def _safe(ctx: RunContext[AgentDeps], tool_input: Any) -> Any:  # noqa: ANN401
        try:
            return await observed_fn(ctx, tool_input)
        except ModelRetry:
            raise
        except Exception as exc:
            log.exception("Unhandled exception in tool %r", name)
            return ErrorResponse(
                message=f"An unhandled exception was encountered: `{type(exc).__name__}`"
            )

    _safe.__annotations__ = {
        "ctx": RunContext[AgentDeps],
        "tool_input": input_annotation,
        "return": output_annotation,
    }
    return Tool(
        _safe,
        takes_ctx=True,
        name=name,
        description=description,
        include_return_schema=True,
    )
