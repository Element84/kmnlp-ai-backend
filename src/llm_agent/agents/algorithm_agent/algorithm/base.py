import operator
from abc import ABC
from collections.abc import Iterable
from functools import reduce
from typing import Annotated, Literal

from pydantic import BaseModel, Field, create_model


class Expression(BaseModel, ABC, frozen=True, extra="forbid"):
    """Base class for an expression that can be evaluated."""

    type: str


class Function(Expression, ABC, frozen=True, extra="forbid"):
    """Base function class."""

    type: Literal["Function"] = "Function"
    func_name: str = Field(
        description="Used to figure out which model to deserialize JSON to.",
    )


class BaseFunctionCall(Expression, frozen=True, extra="forbid"):
    """Base function call."""

    type: Literal["FunctionCall"] = "FunctionCall"
    func: Function


class FunctionCallWithoutFunctionsError(Exception):
    """Building a function call model without functions is not allowed."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__("Function list cannot be empty.")


def make_function_call_model(
    allowed_funcs: Iterable[type[Function]],
) -> type[BaseFunctionCall]:
    """Dynamically build a FunctionCall model that only accepts given funcs."""
    allowed_funcs = list(allowed_funcs)
    if len(allowed_funcs) == 0:
        raise FunctionCallWithoutFunctionsError
    allowed_funcs_union_type = reduce(operator.or_, allowed_funcs)
    func_field = Annotated[
        allowed_funcs_union_type,
        Field(
            description="Fields for instantiating the corresponding function model.",
            # For more info: https://docs.pydantic.dev/2.9/concepts/unions/#discriminated-unions
            discriminator="func_name",
        ),
    ]
    return create_model(
        "FunctionCall",
        __base__=BaseFunctionCall,
        __doc__="A function call.",
        # model fields:
        func=func_field,
    )


class VariableReference(Expression, frozen=True):
    """Reference to an already-defined variable."""

    type: Literal["VariableReference"] = "VariableReference"
    name: str = Field(description="Name of the variable.")


class VariableAssignment[FunctionCall: BaseFunctionCall](Expression, frozen=True, extra="forbid"):
    """Define a new variable."""

    type: Literal["VariableAssignment"] = "VariableAssignment"
    name: str = Field(description="Name of the variable.")
    value: (
        str
        | int
        | float
        | list[str]
        | list[int]
        | list[float]
        | list[VariableReference]
        | FunctionCall
    ) = Field(
        description="Value to be assigned to the variable. "
        "Can be a literal, a list of literals, or a call to one of the "
        "available functions.",
    )
    description: str = Field(default="", description="Optional brief description of the variable.")


class Algorithm[FunctionCall: BaseFunctionCall](BaseModel, frozen=True, extra="forbid"):
    """Representation of an algorithm.

    IMPORTANT: The output of each function call must be assigned to a variable.
    Function calls CANNOT be composed.
    """

    statements: list[VariableAssignment[FunctionCall]] = Field(
        description="List of VariableAssignment expressions, each "
        "constituting a single step in the algorithm.",
    )
    output_variables: list[VariableReference] = Field(
        description="List of references to variables that constitute the outputs of the algorithm.",
    )
