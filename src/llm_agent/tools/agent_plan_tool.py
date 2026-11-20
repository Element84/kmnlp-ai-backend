from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_agent.tools.response_shaping_tool import ResponseShapingTool


class AgentStep(BaseModel, **ConfigDict(extra="forbid")):
    """A step."""

    id: str
    title: str
    status: Literal["pending", "in_progress", "success", "failure", "cancelled"]
    outcome_summary: str | None = None


class AgentPlan(BaseModel, extra="forbid"):
    """A plan."""

    content_type: Literal["plan"]
    id: str
    title: str
    goal: str
    steps: list[AgentStep] = Field(min_length=1)


class AgentStepUpdate(BaseModel):
    """Updates to existing agent step."""

    id: str
    status: Literal["pending", "in_progress", "success", "failure", "cancelled"] | None = None
    outcome_summary: str | None = None


class AgentPlanUpdate(BaseModel, extra="forbid"):
    """Updates to an existing agent plan."""

    content_type: Literal["plan_update"]
    id: str
    step_updates: list[AgentStepUpdate] = Field(min_length=1)


# This is not a RootModel because the Bedrock API requires a tool's input schema to have a "type"
# field which is missing from RootModel's JSON schema.
class AgentPlanOrUpdate(BaseModel, extra="forbid"):
    """Plan or an update to an existing plan."""

    plan_or_update: AgentPlan | AgentPlanUpdate


agent_plan_tool = ResponseShapingTool(
    name="AgentPlanTool",
    description="Tool to use to communicate and update the sequence of steps that you plan to "
    "take to solve a problem.",
    response_model=AgentPlanOrUpdate,
)
