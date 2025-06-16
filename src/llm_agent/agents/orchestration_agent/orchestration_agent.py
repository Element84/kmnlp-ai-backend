import logging
from collections.abc import AsyncGenerator, Iterable
from string import Template

from anyio import to_thread
from e84_geoai_common.llm.core import (
    LLM,
    Agent,
    LLMInferenceConfig,
    LLMMessage,
    LLMTool,
    LLMToolChoice,
    LLMToolResultContent,
    LLMToolUseContent,
)
from rich.pretty import pretty_repr

from llm_agent.utils import get_current_date

log = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS = 10_000
MAX_TOOL_USE_TURNS = 10

ORCHESTRATION_AGENT_SYSTEM_PROMPT_TEMPLATE = Template("""\
Task:
You are a helpful assistant, specializing in assisting users with complex \
data-related tasks by breaking them down into simpler sub-tasks and completing them \
using the tools at your disposal.

Persona:
- You help users interpret the results from your tools.
- You maintain a professional tone.
- Your responses are clear and to-the-point without being curt.

Constraints:
- You DO NOT write any code, but generating JSON is fine.
- Your tools might not be able to fulfill all user queries; you inform the user \
when this is the case.
- You describe your actions at a higher level of abstraction e.g. "I'm doing XYZ" \
instead of "I'm using such-and-such tool", unless asked otherwise.

Target audience:
- The users are experts.

Context:
- The current date is: ${current_date}.
- The users will interact with you through a chat UI.
- The users will be able to see tool calls and outputs in the UI.
""")  # noqa: E501


class MaxToolUseTurnsExceededError(Exception):
    """Max allowed number of tool-use turns exceeded."""


# NOTE: We have decided to defer unifying the implementations of running
# queries and handling tool call between the `OrchestrationAgent` and the
# `DatasetPickingAgent`. However, the logic is highly similar and should be
# consolidated in something like a `ToolUsingAgent` the next time we write
# similar functionality.
class OrchestrationAgent(Agent):
    """Agent that orchestrates a complex workflow using tools."""

    def __init__(
        self,
        llm: "LLM",
        *,
        tools: list[LLMTool],
        inference_cfg: LLMInferenceConfig | None = None,
    ) -> None:
        """Constructor.

        Args:
            llm (LLM): An LLM instance.
            tools (list[LLMTool]): Tools that the agent may use.
            inference_cfg (LLMInferenceConfig | None): Inference config.
                Defaults to None.
        """
        self.llm = llm
        self.tools = tools
        self.tool_name_to_tool = {t.name: t for t in tools}
        self.system_prompt = (
            ORCHESTRATION_AGENT_SYSTEM_PROMPT_TEMPLATE.substitute(
                current_date=get_current_date(),
            )
        )
        log.debug("System prompt:\n%s", pretty_repr(self.system_prompt))
        log.debug("Tools:\n")
        for tool in self.tools:
            if tool.input_model is None:
                continue
            log.debug(
                "\n%s", pretty_repr(tool.input_model.model_json_schema())
            )
        if inference_cfg is not None:
            if inference_cfg.system_prompt is not None:
                msg = (
                    "system_prompt should not be set in the provided "
                    "LLMInferenceConfig"
                )
                raise ValueError(msg)
            inference_cfg = inference_cfg.model_copy(
                update={
                    "system_prompt": self.system_prompt,
                    "json_mode": False,
                }
            )
        else:
            inference_cfg = LLMInferenceConfig(
                system_prompt=self.system_prompt,
                tools=self.tools,
                tool_choice=LLMToolChoice(mode="optional", tool_name=None),
                temperature=0,
                json_mode=False,
                max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            )
        self.inference_cfg = inference_cfg

    async def run(
        self, msg: LLMMessage
    ) -> AsyncGenerator[LLMMessage | LLMToolResultContent]:
        """Prompt the agent with a message.

        If the LLM's response does not contain any tool-use requests, the
        returned iterator will only yield a single message.

        If the response does contain tool-use requests, the returned iterator
        will additionally yield tool-use results, and the LLM's response to
        those results, as they become available. There may be several rounds of
        these if the LLM issues new tool-use requests in response to results of
        previous tool-use requests.
        """
        conversation = [msg]
        for _ in range(MAX_TOOL_USE_TURNS):
            response = await self._prompt_for_single_response(conversation)
            yield response

            tool_use_requests = [
                c for c in response.content if isinstance(c, LLMToolUseContent)
            ]
            if len(tool_use_requests) == 0:
                break

            log.info("Using tools: %s", pretty_repr(tool_use_requests))

            tool_results: list[LLMToolResultContent] = []
            async for tool_result in self._use_tools(tool_use_requests):
                tool_results.append(tool_result)
                yield tool_result
            log.info("Tool results: %s", pretty_repr(tool_results))

            conversation.append(response)
            conversation.append(LLMMessage(role="user", content=tool_results))
        else:
            raise MaxToolUseTurnsExceededError

    async def _prompt_for_single_response(
        self, messages: list[LLMMessage]
    ) -> LLMMessage:
        """Prompt agent for a single response.

        Adds both the input message and the response to self.messages.
        """

        # Useful because `to_thread.run_sync` takes args of (func, *args),
        # which breaks kwargs and can be hard to reason about.
        #
        # Not `async` because `to_thread.run_sync` runs a synchronous function
        # in a worker thread.
        def _llm_prompt_shim(messages: list[LLMMessage]) -> LLMMessage:
            return self.llm.prompt(
                messages=messages, inference_cfg=self.inference_cfg
            )

        response = await to_thread.run_sync(_llm_prompt_shim, messages)
        return response

    async def _use_tools(
        self, tool_use_requests: Iterable[LLMToolUseContent]
    ) -> AsyncGenerator[LLMToolResultContent]:
        """Execute tools and yield results as they become available."""
        for tool_use_request in tool_use_requests:
            tool = self.tool_name_to_tool[tool_use_request.name]
            tool_result = await to_thread.run_sync(
                tool.execute, tool_use_request
            )
            yield tool_result

    @property
    def prompt_template(self) -> str:
        """The prompt template used by the agent."""
        return ""
