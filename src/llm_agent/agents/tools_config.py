from dataclasses import dataclass

from e84_geoai_common.llm.core import ExecutableLLMTool

from llm_agent.context import ToolCallContext


@dataclass()
class ToolsConfig:
    """Bundling of tools available to be called by LLM.

    tools (list[ExecutableLLMTool[ToolCallContext]] | None): represent tools
    for the agent to use, excluding the end-turn tool.

    end_turn_tool (ExecutableLLMTool[ToolCallContext]): the tool to end the
        LLM's turn.

    max_allowed_tool_calls_per_turn (int): Max allowed tool calls per turn.
    """

    tools: list[ExecutableLLMTool[ToolCallContext]]
    end_turn_tool: ExecutableLLMTool[ToolCallContext]
    max_allowed_tool_calls_per_turn: int

    @property
    def tool_name_to_tool(self) -> dict[str, ExecutableLLMTool[ToolCallContext]]:
        """Dictionary of tool names to tool instances."""
        return {t.tool_spec.name: t for t in [*self.tools, self.end_turn_tool]}
