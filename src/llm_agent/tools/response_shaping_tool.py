from pydantic import BaseModel
from pydantic_ai import RunContext

from llm_agent.context import AgentDeps
from llm_agent.tools.agent_tool import AgentTool


class ResponseShapingToolAck(BaseModel):
    """Acknowledgement returned by a :class:`ResponseShapingTool`.

    The structured response payload itself is surfaced to the caller through the
    tool-call event stream; the tool only returns this lightweight ack so the
    model can continue its turn.
    """

    success: bool = True


class ResponseShapingTool[ResponseT: BaseModel](AgentTool[ResponseT, ResponseShapingToolAck]):
    """Tool used to elicit a structured response from an agent.

    The tool's input schema is the provided ``response_model``; calling it does
    nothing other than acknowledge receipt. The structured payload is surfaced to
    the caller via the tool-call event stream.
    """

    def __init__(self, *, name: str, description: str, response_model: type[ResponseT]) -> None:
        """Constructor.

        Args:
            name: Name of the tool.
            description: Description of the tool.
            response_model: Pydantic model describing the expected structured response.
        """
        self.name = name
        self.description = f"{description}\nThis tool returns a simple acknowledgement on success."
        self.input_model = response_model
        self.output_type = ResponseShapingToolAck

    async def __call__(
        self,
        ctx: RunContext[AgentDeps],  # noqa: ARG002
        tool_input: ResponseT,  # noqa: ARG002
        /,
    ) -> ResponseShapingToolAck:
        """Acknowledge the structured response; the payload rides on the event stream."""
        return ResponseShapingToolAck()
