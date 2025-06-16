from abc import ABC
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

FunctionSet = TypeVar("FunctionSet", bound="Function")


class Expression(ABC, BaseModel):
    """Base class for an expression that can be evaluated."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Function(Expression):
    """Base function class."""

    func_name: Literal["Function"] = Field(
        description="Used to figure out which model to deserialize JSON to.",
    )


class FunctionCall(Expression, Generic[FunctionSet]):
    """A function call."""

    func: FunctionSet = Field(
        # For more info: https://docs.pydantic.dev/2.9/concepts/unions/#discriminated-unions
        discriminator="func_name",
        description="Fields for instantiating the corresponding function "
        "model.",
    )


class VariableReference(Expression):
    """Reference to an already-defined variable."""

    name: str = Field(description="Name of the variable.")


class VariableAssignment(Expression, Generic[FunctionSet]):
    """Define a new variable."""

    name: str = Field(description="Name of the variable.")
    value: (
        str
        | int
        | float
        | list[str]
        | list[int]
        | list[float]
        | list[VariableReference]
        | FunctionCall[FunctionSet]
    ) = Field(
        description="Value to be assigned to the variable. "
        "Can be a literal, a list of literals, or a call to one of the "
        "available functions.",
    )
    description: str = Field(
        default="", description="Optional brief description of the variable."
    )


class Algorithm(BaseModel, Generic[FunctionSet]):
    """Representation of an algorithm.

    IMPORTANT: The output of each function call must be assigned to a variable.
    Function calls CANNOT be composed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    statements: list[VariableAssignment[FunctionSet]] = Field(
        description="List of VariableAssignment expressions, each "
        "constituting a single step in the algorithm.",
    )
    output_variables: list[VariableReference] = Field(
        description="List of references to variables that constitute the "
        "outputs of the algorithm.",
    )
