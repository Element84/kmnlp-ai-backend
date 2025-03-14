from abc import ABC
from typing import Generic, Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    FunctionSet,
)


class InputVariable(BaseModel):
    """An input variable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    type_as_str: Literal["DataArray"] = Field(
        description="The type of the variable, specified as a string.",
    )
    description: str


class OutputVariable(BaseModel):
    """An output variable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    type_as_str: Literal["str", "int", "float", "list", "plot"] = Field(
        description="The approximate type of the variable, "
        "specified as a string.",
    )
    description: str


class ProblemSpec(BaseModel, ABC):
    """Base problem specification class."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SolutionSpec(BaseModel, ABC):
    """Base solution specification class."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AlgorithmProblemSpec(ProblemSpec):
    """Specification of an algorithmic problem."""

    statement: str = Field(description="The problem statement.")
    inputs_provided: list[InputVariable] = Field(
        description="List of variables that can be assumed to already exist "
        "and can be safely referenced in the algorithm.",
    )
    outputs_requested: list[OutputVariable] = Field(
        description="Variables that must be included as outputs of "
        "the algorithm.",
    )


class AlgorithmSolutionSpec(SolutionSpec, Generic[FunctionSet]):
    """Specification of an algorithmic solution."""

    thoughts: str = Field(
        default="",
        description="Understanding of the problem and plan of attack. "
        "Identify which of the available operations might be relevant to the "
        "solution.",
    )
    algorithm: Algorithm[FunctionSet] = Field(
        description="The algorithm. The statements and return variables of "
        "the algorithm can be set to empty lists in case the problem is "
        "malformed or not solvable.",
    )
    explanation: str = Field(
        default="",
        description="Additional explanation of the algorithm or "
        "an explanation for why the problem is not solvable using the "
        "available operations.",
    )
