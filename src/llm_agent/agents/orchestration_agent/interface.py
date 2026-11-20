from pydantic import BaseModel

from llm_agent.context import DataRef


class UserMessage(BaseModel):
    """A message from the user."""

    user_message: str
    attached_data: list[DataRef] | None = None
