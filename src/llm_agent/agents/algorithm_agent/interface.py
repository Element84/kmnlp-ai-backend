from abc import ABC
from collections.abc import Collection
from datetime import datetime, timedelta
from os.path import basename
from typing import Generic, Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    FunctionSet,
)
from llm_agent.model.dataset import Dataset
from llm_agent.model.variable import Variable

# InputVariable should be a union of supported input variable types
type InputVariable = SpatioTemporalDataArrayVariable


class UnsupportedVariableDimsError(Exception):
    """Variable has an unsupported set of dimensions."""

    def __init__(self, dims: Collection[str]) -> None:  # noqa: D107
        super().__init__(
            'Only variables with dims == {"latitude", "longitude", "time"} '
            f"are currently supported. Got dims: {dims}."
        )


class SpatioTemporalDataArrayVariable(BaseModel):
    """A spatiotemporal DataArray variable with (time, lat, lon) dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Name of the variable.")
    variable_type: Literal["DataArray"] = Field(
        description="The type of the variable, specified as a string.",
    )
    description: str | None = Field(description="Description of the variable.")
    units: str | None = Field(
        description="The units of the values in the data."
    )
    spatial_coverage: Literal["global"] = Field(
        description="The spatial area for which the variable has values."
    )
    temporal_coverage: tuple[datetime, datetime] = Field(
        description="The start and end times that the variable covers"
    )
    temporal_resolution: timedelta | None = Field(
        description=(
            "The resolution of the time field for the variable, or the time "
            "between each data point"
        )
    )


def input_variable_from_model(
    variable: Variable, dataset: Dataset
) -> InputVariable:
    """Transform from model.Variable to SpatioTemporalDataArrayVariable.

    Args:
        variable (Variable): The model variable to transform
        dataset (Dataset): The dataset to which the variable belongs.
            Used to provide dataset-level metadata to the `InputVariable`.

    Returns:
        InputVariable: An `InputVariable`.

    Raises:
        UnsupportedVariableDimsError: If variable does not have
            dims == {"latitude", "longitude", "time"}.
    """

    def _name_from_maybe_iri(iri: str | None) -> str | None:
        return basename(iri) if iri is not None else None

    # The type: ignore is for a pyright false positive about frozenset != set.
    # The comparison in fact works as expected.
    if variable.dimensions != {"latitude", "longitude", "time"}:  # type: ignore[reportUnnecessaryComparison]
        raise UnsupportedVariableDimsError(variable.dimensions)

    return SpatioTemporalDataArrayVariable(
        name=variable.field_name,
        variable_type="DataArray",
        description=variable.description,
        units=_name_from_maybe_iri(variable.units),
        spatial_coverage=dataset.spatial_coverage,
        temporal_coverage=dataset.temporal_coverage,
        temporal_resolution=dataset.temporal_resolution,
    )


class OutputVariable(BaseModel):
    """A variable to be included in the output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Name of the variable in snake case.")
    variable_type: Literal[
        "scalar_float",
        "datetime_string",
        "latitude_deg_float",
        "longitude_deg_float",
        "list[scalar_float]",
        "list[datetime_string]",
        "list[latitude_deg_float]",
        "list[longitude_deg_float]",
        "raster_plot",
        "bar_plot",
        "line_plot",
    ] = Field(
        description="The type of the variable. The list[] types are not "
        "currently supported and should not be used. Request each item "
        "individually."
    )
    description: str = Field(description="Description of the variable.")


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
        description="Understanding of the problem, understanding of the "
        "inputs, understanding of the outputs, and reasoning for the "
        "solution. Answer the following: Which of the given inputs are "
        "relevant to the problem? How do the temporal and spatial coverages "
        "of the inputs relate to each other and to the problem? How do the "
        "temporal and spatial resolutions of the inputs relate to each other "
        "and to the problem? How should the temporal resolutions be handled "
        "in reduction and group-by operations, if applicable. Identify "
        "potentially tricky aspects of the problem or the data, if any. "
        "It is important to determine at this stage whether all the "
        "requested outputs are constructable STRICTLY using only the "
        "available functions. You must report an error if any of them are "
        "not. The cost of an incorrect solution is very high, so err on the "
        "side of declining a solution in case the problem is unclear."
    )
    pseudocode: str = Field(
        description="Pseudocode in Python-like syntax for the algorithm, "
        "making use of the available functions. Ensure that types and "
        "dimensions of function inputs align with function definitions. Use "
        "single quotes for strings inside the pseudocode to avoid collisions "
        "with the rest of the JSON output. IMPORTANT: just like with the "
        "algorithm schema, the output of each function call must be assigned "
        "to a variable. Function calls MUST NOT be composed."
    )
    algorithm: Algorithm[FunctionSet] = Field(
        description="The algorithm. The statements and return variables of "
        "the algorithm can be set to empty lists in case of error.",
    )
    explanation: str = Field(
        description="Additional explanation for the solution. Can be an empty "
        "string in case of error.",
    )
    error_message: str | None = Field(
        default=None,
        description="In case the problem is not solvable e.g. if it is "
        "unclear, malformed, unrelated to the data, too complex to be solved "
        "using the available functions, or for any other reason, this field "
        "should explain why. Otherwise, this should be null.",
    )
