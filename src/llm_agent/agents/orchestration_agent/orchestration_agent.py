import logging
from collections.abc import AsyncGenerator, Sequence
from string import Template
from typing import Any

from langfuse import observe  # type: ignore[reportUnknownVariableType]
from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.models.bedrock import BedrockModelSettings
from pydantic_ai.usage import UsageLimits
from rich.pretty import pretty_repr

from llm_agent.agents.orchestration_agent.interface import UserMessage
from llm_agent.context import AgentDeps, ChatContext, ProgressCallback
from llm_agent.tools.agent_tool import AgentTool
from llm_agent.tools.response_shaping_tool import ResponseShapingTool
from llm_agent.utils.langfuse import stream_events_to_final_output
from llm_agent.utils.utils import get_current_date

log = logging.getLogger(__name__)

DEFAULT_MAX_ALLOWED_TOOL_CALLS_PER_TURN = 20

DEFAULT_MAX_OUTPUT_TOKENS = 10_000

_SYSTEM_PROMPT_TEMPLATE = Template("""\
Task:
You are a helpful assistant, specializing in assisting users with complex \
data-related tasks by breaking them down into simpler sub-tasks and completing them \
using the tools at your disposal.

Input schema:
You will receive input in the following JSON schema:
```json
${input_schema}
```

Persona:
- You help users interpret the results from your tools.
- You maintain a professional tone.
- Your responses are clear and to-the-point without being curt.

Constraints:
- You interleave brief explanatory text with tool calls, and parallelize tool calls where \
reasonable. When you have finished addressing the user's request, reply with a plain text \
message; replying with text and no tool calls ends your turn.
- You DO NOT write any code.
- Your tools are capable of geocoding natural language queries to locations. Do not attempt to \
reproduce geospatial coordinates from memory unless explicitly instructed to do so.
- Your tools might not be able to fulfill all user queries; you inform the user \
when this is the case.
- Your tools might run into errors. Consult with the user before retrying unless you are absolutely
sure about how to fix it.
- You describe your actions at a higher level of abstraction e.g. "I'm doing XYZ" \
instead of "I'm using such-and-such tool", unless asked otherwise.
- For multi-step tasks, communicate the sequence of steps you plan to take up front, and \
report progress as you complete them.

Target audience:
- The users are experts.

Context:
- The current date is: ${current_date}.
- The users will interact with you through a chat UI.
- The users will be able to see tool calls and outputs in the UI.
- The user's message may include an attached_data section listing references to data \
they have provided. You can pass these references to your tools or resolve image references \
to view them.
""")


class OrchestrationAgent:
    """Agent that orchestrates a complex workflow using tools.

    Wraps a Pydantic AI ``Agent`` whose output is free text: the model interleaves
    text with tool calls, and a text-only reply (no tool calls) ends the turn.
    """

    def __init__(
        self,
        model: Model,
        *,
        system_prompt: str | None = None,
        tools: Sequence[AgentTool[Any, Any]] | None = None,
        response_shaping_tools: Sequence[ResponseShapingTool[Any]] | None = None,
        max_allowed_tool_calls_per_turn: int = DEFAULT_MAX_ALLOWED_TOOL_CALLS_PER_TURN,
    ) -> None:
        """Constructor.

        Args:
            model: The Pydantic AI model to drive the agent. Build one with
                ``llm_agent.model_factory.make_bedrock_model``.
            system_prompt: Optional system prompt to override the default.
            tools: Tools the agent may use. The consumer is responsible for any
                wrapping (e.g. ``TrackingTool``) before passing them in.
            response_shaping_tools: Tools used to elicit structured outputs (e.g.
                an agent plan). Their input schemas are the structured models the
                agent can emit.
            max_allowed_tool_calls_per_turn: Maximum allowed tool calls per turn.
        """
        registered_tools = [
            *(tool.as_pydantic_tool() for tool in tools or []),
            *(tool.as_pydantic_tool() for tool in response_shaping_tools or []),
        ]
        self._usage_limits = UsageLimits(tool_calls_limit=max_allowed_tool_calls_per_turn)

        model_settings = BedrockModelSettings(
            bedrock_cache_instructions=True,
            bedrock_cache_tool_definitions=True,
        )
        if model.settings:
            if model.settings.get("temperature") is None:
                model.settings["temperature"] = 0.0
            if model.settings.get("max_tokens") is None:
                model.settings["max_tokens"] = DEFAULT_MAX_OUTPUT_TOKENS

        self._agent = Agent(
            model,
            deps_type=AgentDeps,
            output_type=str,
            instructions=system_prompt or _build_system_prompt(),
            model_settings=model_settings,
            tools=registered_tools,
        )

    @observe(as_type="agent", name=__name__)
    async def run(
        self,
        chat_context: ChatContext,
        user_message: UserMessage,
        *,
        message_history: list[ModelMessage] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> str:
        """Run the agent and return the final text response."""
        deps = AgentDeps(chat_context=chat_context, progress_callback=progress_callback)
        result = await self._agent.run(
            user_message.model_dump_json(),
            deps=deps,
            message_history=message_history,
            usage_limits=self._usage_limits,
        )
        log.info("OrchestrationAgent response: %s", pretty_repr(result.output))
        return result.output

    @observe(as_type="agent", name=__name__, transform_to_string=stream_events_to_final_output)
    async def run_stream(
        self,
        chat_context: ChatContext,
        user_message: UserMessage,
        *,
        message_history: list[ModelMessage] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AsyncGenerator[AgentStreamEvent | AgentRunResultEvent[str]]:
        """Run the agent, streaming text deltas plus tool call/result events in realtime."""
        deps = AgentDeps(chat_context=chat_context, progress_callback=progress_callback)
        async with self._agent.run_stream_events(
            user_message.model_dump_json(),
            deps=deps,
            message_history=message_history,
            usage_limits=self._usage_limits,
        ) as stream:
            async for event in stream:
                log.debug("OrchestrationAgent stream event: %s", pretty_repr(event))
                yield event


def _build_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.substitute(
        input_schema=UserMessage.model_json_schema(), current_date=get_current_date()
    )
