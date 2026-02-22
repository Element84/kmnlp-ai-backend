import logging
from collections.abc import AsyncGenerator, Callable
from string import Template
from typing import Any, override

from e84_geoai_common.llm.core import (
    LLMInferenceConfig,
    LLMMessage,
    LLMToolChoice,
    LLMToolResultContent,
    LLMToolUseContent,
)
from langfuse import observe  # pyright: ignore[reportUnknownVariableType]
from rich.pretty import pretty_repr

from llm_agent.agents.tool_only_agent import (
    ToolOnlyAgent,
    ToolsConfig,
)
from llm_agent.context import LLMAgentContext, ToolProgressUpdate
from llm_agent.utils.prompt_as_service import PromptService
from llm_agent.utils.utils import get_current_date

log = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS = 10_000

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
- You communicate using tool calls only and try to parallelize tool calls where reasonable.
- You DO NOT write any code.
- Your tools are capable of geocoding natural language queries to locations. Do not attempt to \
reproduce geospatial coordinates from memory unless explicitly instructed to do so.
- Your tools might not be able to fulfill all user queries; you inform the user \
when this is the case.
- Your tools might run into errors. Consult with the user before retrying unless you are absolutely
sure about how to fix it.
- You describe your actions at a higher level of abstraction e.g. "I'm doing XYZ" \
instead of "I'm using such-and-such tool", unless asked otherwise.

Target audience:
- The users are experts.

Context:
- The current date is: ${current_date}.
- The users will interact with you through a chat UI.
- The users will be able to see tool calls and outputs in the UI.
""")


# NOTE: We have decided to defer unifying the implementations of running
# queries and handling tool call between the `OrchestrationAgent` and the
# `DatasetPickingAgent`. However, the logic is highly similar and should be
# consolidated in something like a `ToolUsingAgent` the next time we write
# similar functionality.
class OrchestrationAgent(ToolOnlyAgent):
    """Agent that orchestrates a complex workflow using tools."""

    def __init__(
        self,
        prompt_service: PromptService,
        *,
        tools_config: ToolsConfig | None = None,
        inference_cfg: LLMInferenceConfig | None = None,
    ) -> None:
        """Constructor.

        Args:
            prompt_service (PromptService): Service to call to invoke the model

            tools_config (ToolsConfig): ToolsConfig for the model invocation to
                use. None to get default.

            inference_cfg (LLMInferenceConfig | None): Inference config.
        """
        self.system_prompt = ORCHESTRATION_AGENT_SYSTEM_PROMPT_TEMPLATE.substitute(
            current_date=get_current_date(),
        )
        if inference_cfg is not None:
            if inference_cfg.system_prompt is not None:
                msg = "LLMInferenceConfig.system_prompt must not be set"
                raise ValueError(msg)
        else:
            inference_cfg = LLMInferenceConfig(
                system_prompt=self.system_prompt,
                tool_choice=LLMToolChoice(mode="force_tool_use", tool_name=None),
                temperature=0,
                json_mode=False,
                max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            )

        super().__init__(
            prompt_service=prompt_service,
            tools_config=tools_config,
            inference_cfg=inference_cfg,
        )

        log.info("System prompt:\n%s", pretty_repr(self.system_prompt))
        log.info("Tools:\n\n%s", _tool_output_from_tools_config(self.tools_config))

    @observe(as_type="agent", name="OrchestrationAgent.run")
    @override
    def run(
        self,
        context: LLMAgentContext,
        conversation: list[LLMMessage],
        progress_callback: Callable[[ToolProgressUpdate[Any]], None] | None = None,
    ) -> AsyncGenerator[LLMToolUseContent | LLMToolResultContent]:
        return super().run(context, conversation, progress_callback)


def _tool_output_from_tools_config(tools_config: ToolsConfig) -> str:
    messages = [
        pretty_repr(tool.tool_spec.input_model.model_json_schema())
        for tool in tools_config.tool_name_to_tool.values()
        if tool.tool_spec.input_model is not None
    ]

    return "\n".join(messages)
