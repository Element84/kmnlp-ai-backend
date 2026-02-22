import logging
import operator
from abc import ABC
from collections.abc import AsyncGenerator, Callable
from functools import reduce
from string import Template
from types import UnionType
from typing import Any, cast

from e84_geoai_common.llm.core.llm import (
    LLM,
    CachePointContent,
    ExecutableLLMTool,
    LLMInferenceConfig,
    LLMUserMessage,
    TextContent,
)
from langfuse import observe  # pyright: ignore[reportUnknownVariableType]
from pydantic import BaseModel
from rich.pretty import pretty_repr

from llm_agent.agents.tool_only_agent import (
    MAX_ALLOWED_TOOL_CALLS_PER_TURN,
    ToolOnlyAgent,
    ToolsConfig,
)
from llm_agent.context import LLMAgentContext, ToolCallContext, ToolProgressUpdate
from llm_agent.tools.llm_tool_typed import ErrorResponse
from llm_agent.tools.response_shaping_tools.end_turn_tool import end_turn_tool
from llm_agent.tools.response_shaping_tools.response_shaping_tool import ResponseShapingTool
from llm_agent.utils.prompt_as_service import PromptServiceAdapter
from llm_agent.utils.utils import EnrichedLLMToolResultContent

log = logging.getLogger(__name__)

MAX_ALLOWED_CACHE_POINTS = 2

DEFAULT_PROMPT_TEMPLATE = Template("""
```json
${input_json}
```
""")


class TypedToolOnlyAgent[InputT: BaseModel, OutputT: BaseModel | UnionType](ToolOnlyAgent, ABC):
    """Tool-only agent that takes typed inputs and produces typed outputs."""

    prompt_template: Template
    input_model: type[InputT]
    output_type: type[OutputT]

    def __init__(  # noqa: PLR0913
        self,
        llm: LLM,
        input_model: type[InputT],
        output_models: list[type[BaseModel]],
        *,
        response_shaping_tools: list[ResponseShapingTool[Any]],
        other_tools: list[ExecutableLLMTool[ToolCallContext]] | None = None,
        end_turn_tool: ExecutableLLMTool[ToolCallContext] = end_turn_tool,
        prompt_template: Template = DEFAULT_PROMPT_TEMPLATE,
        inference_cfg: LLMInferenceConfig,
        max_allowed_tool_calls_per_turn: int = MAX_ALLOWED_TOOL_CALLS_PER_TURN,
    ) -> None:
        """Constructor.

        Args:
            llm: The LLM to use.
            input_model: The Pydantic model class for input.
            output_models: List of Pydantic model classes that the agent can output.
                Other types of outputs will be ignored.
            response_shaping_tools: Response shaping tools. Output types of each tool must be
                included in output_models.
            other_tools: Other LLM tools that the agent may use.
            end_turn_tool (ExecutableLLMTool[LLMAgentContext]): Tool for ending a turn.
                Defaults to end_turn_tool.
            prompt_template: Template for the prompt.
            inference_cfg: Inference configuration for the LLM.
            max_allowed_tool_calls_per_turn: Maximum allowed tool calls per turn.
        """
        all_tools = response_shaping_tools + (other_tools or [])
        super().__init__(
            PromptServiceAdapter(llm.prompt),
            tools_config=ToolsConfig(
                tools=all_tools,
                end_turn_tool=end_turn_tool,
                max_allowed_tool_calls_per_turn=max_allowed_tool_calls_per_turn,
            ),
            inference_cfg=inference_cfg,
        )
        self.prompt_template = prompt_template
        self.input_model = input_model
        self.output_models = set(output_models)
        self.output_type = reduce(operator.or_, self.output_models)
        self.response_shaping_tools = response_shaping_tools
        self.cache_point_counter = 0

        tool_output_models = {tool.user_output_model for tool in self.response_shaping_tools}
        if not self.output_models.issuperset(tool_output_models):
            raise TypeError(
                f"Agent output models {self.output_models} do not contain all response shaping "
                f"tool output models {tool_output_models}.",
            )

    @observe(as_type="agent", name=f"{__name__}.run_typed")
    async def run_typed(
        self,
        context: LLMAgentContext,
        inp: InputT,
        progress_callback: Callable[[ToolProgressUpdate[Any]], None] | None = None,
    ) -> AsyncGenerator[OutputT | ErrorResponse]:
        """Run the agent with typed input and yield typed outputs."""
        response_stream = self.run(
            context,
            conversation=[self.input_to_message(inp)],
            progress_callback=progress_callback,
        )
        async for response in response_stream:
            if not isinstance(response, EnrichedLLMToolResultContent):
                continue
            response = cast(
                "EnrichedLLMToolResultContent[ErrorResponse] | EnrichedLLMToolResultContent[Any]",
                response,
            )
            if isinstance(response.data, ErrorResponse):
                log.error("Error response from agent: %s", pretty_repr(response.data))
                yield response.data
            elif isinstance(response.data, self.output_type):
                log.info("Typed output from agent: %s", pretty_repr(response.data))
                yield response.data

    def input_to_message(self, inp: InputT) -> LLMUserMessage:
        """Convert typed input to LLMUserMessage."""
        prompt_str = self.prompt_template.substitute(input_json=inp.model_dump_json())

        # Naively insert a cache point on every user message until limit is reached.
        if self.cache_point_counter < MAX_ALLOWED_CACHE_POINTS:
            self.cache_point_counter += 1
            content = [TextContent(text=prompt_str), CachePointContent()]
        else:
            content = [TextContent(text=prompt_str)]

        return LLMUserMessage(content=content)
