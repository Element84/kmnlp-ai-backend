import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any, cast

from anyio import to_thread
from e84_geoai_common.llm.core.llm import (
    LLMAssistantMessage,
    LLMInferenceConfig,
    LLMMessage,
    LLMToolChoice,
    LLMToolResultContent,
    LLMToolUseContent,
    LLMUserMessage,
)
from langfuse import observe  # pyright: ignore[reportUnknownVariableType]
from rich.pretty import pretty_repr

from llm_agent.agents.tools_config import ToolsConfig
from llm_agent.context import LLMAgentContext, ToolCallContext, ToolProgressUpdate
from llm_agent.tools.llm_tool_typed import ErrorResponse
from llm_agent.tools.response_shaping_tools.end_turn_tool import end_turn_tool
from llm_agent.utils.prompt_as_service import PromptService
from llm_agent.utils.utils import (
    EnrichedLLMToolResultContent,
)

log = logging.getLogger(__name__)

MAX_ALLOWED_TOOL_CALLS_PER_TURN = 16


class MaxAllowedToolCallsExceededError(Exception):
    """Max allowed number of tool calls exceeded."""


class MalformedLLMResponseError(Exception):
    """Base class for malformed LLM response errors."""

    def __init__(self, llm_msg: LLMMessage) -> None:  # noqa: D107
        super().__init__(f"{llm_msg!r}")


class NonToolUseStopReasonError(MalformedLLMResponseError):
    """Stop reason is something other than tool_use."""


class LLMResponseWithoutToolUseError(MalformedLLMResponseError):
    """LLM response has no tool calls."""


class NonToolUseContentError(MalformedLLMResponseError):
    """LLM response has content other than tool calls."""


class ToolOnlyAgent:
    """Agent that orchestrates a complex workflow using tools.

    This agent is stateless and does not store conversation history.
    The conversation must be passed to the run method on each call.
    """

    inference_cfg: LLMInferenceConfig
    tools_config: ToolsConfig

    prompt_service: PromptService

    def __init__(
        self,
        prompt_service: PromptService,
        *,
        tools_config: ToolsConfig | None = None,
        inference_cfg: LLMInferenceConfig,
    ) -> None:
        """Constructor.

        Args:
            prompt_service (PromptService): Service to call to invoke the model

            tools_config (ToolsArg): Tools config for the model invocation to
                use. None to get default.

            inference_cfg (LLMInferenceConfig | None): Inference config.
                Defaults to None.
        """
        if inference_cfg.tool_choice:
            if inference_cfg.tool_choice.mode != "force_tool_use":
                msg = "LLMInferenceConfig.tool_choice mode must be 'force_tool_use'"
                raise ValueError(msg)
        else:
            inference_cfg.tool_choice = LLMToolChoice(mode="force_tool_use", tool_name=None)

        if inference_cfg.json_mode:
            msg = "ToolOnlyAgent is not compatible with JSON mode"
            raise ValueError(msg)
        if inference_cfg.tools:
            msg = "LLMInferenceConfig.tools must not be set"
            raise ValueError(msg)

        self.tools_config = tools_config or ToolsConfig(
            tools=[],
            end_turn_tool=end_turn_tool,
            max_allowed_tool_calls_per_turn=MAX_ALLOWED_TOOL_CALLS_PER_TURN,
        )

        self.prompt_service = prompt_service

        if not inference_cfg.tools:
            inference_cfg.tools = [
                t.tool_spec for t in self.tools_config.tool_name_to_tool.values()
            ]

        self.inference_cfg = inference_cfg

    @observe(as_type="agent")
    async def run(
        self,
        context: LLMAgentContext,
        conversation: list[LLMMessage],
        progress_callback: Callable[[ToolProgressUpdate[Any]], None] | None = None,
    ) -> AsyncGenerator[LLMToolUseContent | LLMToolResultContent]:
        """Prompt the agent with a conversation and yield tool calls and results.

        This implements an agentic loop where the LLM is repeatedly prompted
        with the current conversation until it decides to end the turn
        using the end-turn tool, or the maximum allowed number of tool calls
        per turn is reached.

        The conversation is modified in place by appending the LLM responses
        and tool results.

        Args:
            context: The tool context for executing tools.
            conversation: The conversation history.
            progress_callback: Optional callback to pass to tool executions. The callback should
                accept two arguments: the tool call ID and a progress update object.
        """
        # Use the subclass as the current name
        for _ in range(self.tools_config.max_allowed_tool_calls_per_turn):
            response = await self._generate_llm_response_to_conversation(conversation)

            try:
                tool_calls = self._extract_tool_calls(response)
            except MalformedLLMResponseError:
                # The Bedrock API has been observed to sometimes return malformed responses.
                # If that happens, retry once.
                log.exception("Malformed LLM response")
                response = await self._generate_llm_response_to_conversation(conversation)
                tool_calls = self._extract_tool_calls(response)

            conversation.append(response)

            for tool_call in tool_calls:
                yield tool_call

            tool_results: list[LLMToolResultContent] = []
            end_turn_tool_called = False
            tool_error = False
            for tool_call in tool_calls:
                tool = self.tools_config.tool_name_to_tool[tool_call.name]

                if tool == self.tools_config.end_turn_tool:
                    end_turn_tool_called = True

                tool_call_context = ToolCallContext(
                    tool_call_id=tool_call.id,
                    global_context=context,
                    progress_callback=progress_callback,
                )
                tool_result = await to_thread.run_sync(tool.execute, tool_call_context, tool_call)

                if isinstance(tool_result, EnrichedLLMToolResultContent) and isinstance(
                    tool_result.data,  # pyright: ignore[reportUnknownMemberType]
                    ErrorResponse,
                ):
                    tool_error = True

                tool_results.append(tool_result)  # pyright: ignore[reportUnknownArgumentType]
                log.info("Tool result: %s", pretty_repr(tool_result))
                yield tool_result

            next_msg_to_llm = LLMUserMessage(content=tool_results)
            conversation.append(next_msg_to_llm)

            if end_turn_tool_called and not tool_error:
                break
        else:
            raise MaxAllowedToolCallsExceededError

    async def _generate_llm_response_to_conversation(
        self, conversation: list[LLMMessage]
    ) -> LLMAssistantMessage:
        """Prompt LLM with current conversation and return the response."""
        log.debug(pretty_repr(conversation))
        response = await self.prompt_service((conversation, self.inference_cfg))
        return response

    def _extract_tool_calls(self, response: LLMAssistantMessage) -> list[LLMToolUseContent]:
        if response.metadata and response.metadata.stop_reason != "tool_use":
            raise NonToolUseStopReasonError(response)

        tool_calls = [c for c in response.content if isinstance(c, LLMToolUseContent)]
        if len(tool_calls) == 0:
            raise LLMResponseWithoutToolUseError(response)

        if len(tool_calls) != len(response.content):
            raise NonToolUseContentError(response)

        return cast("list[LLMToolUseContent]", response.content)
