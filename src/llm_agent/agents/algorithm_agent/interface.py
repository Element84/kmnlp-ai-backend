from abc import ABC
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Annotated, Literal

from geojson_pydantic.geometries import Geometry
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, field_validator

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    BaseFunctionCall,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets import (
    FunctionSet,
    GeneralDataArrayFunctionSet,
    GeneralPlottingFunctionSet,
    SpatioTemporalDataArrayFunctionSet,
    TemporalDataArrayFunctionSet,
)
from llm_agent.model.dataset import Dataset
from llm_agent.model.dimension import Dimension
from llm_agent.model.variable import Variable

# InputVariable should be a union of supported input variable types
type InputVariable = (
    SpatioTemporalDataArrayVariable | TemporalDataArrayVariable | GeneralDataArrayVariable
)

DimensionSet = Annotated[
    frozenset[str],
    PlainSerializer(lambda dims: sorted(dims), return_type=list[str], when_used="json"),
]


class GeneralDataArrayVariable(BaseModel):
    """A DataArray variable with arbitrary dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Name of the variable.")
    variable_type: Literal["DataArray"] = Field(
        default="DataArray",
        description="The type of the variable, specified as a string.",
    )
    description: str | None = Field(description="Description of the variable.")
    dimensions: list[Dimension] = Field(description="Dimensions of the DataArray.")
    units: str | None = Field(description="The units of the values in the data.")


class TemporalDataArrayVariable(BaseModel):
    """A DataArray variable with a time dim plus other arbitrary dims."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Name of the variable.")
    variable_type: Literal["DataArray"] = Field(
        default="DataArray",
        description="The type of the variable, specified as a string.",
    )
    description: str | None = Field(description="Description of the variable.")
    dimensions: list[Dimension] = Field(description="Dimensions of the DataArray.")
    time_dim: str = Field(description="The time dimension of the DataArray.")
    units: str | None = Field(description="The units of the values in the data.")
    temporal_coverage: tuple[datetime, datetime] = Field(
        description="The start and end times that the variable covers"
    )
    temporal_resolution: timedelta | None = Field(
        description=(
            "The resolution of the time field for the variable, or the time between each data point"
        )
    )


class SpatioTemporalDataArrayVariable(BaseModel):
    """A spatiotemporal DataArray variable with (time, lat, lon) dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Name of the variable.")
    variable_type: Literal["DataArray"] = Field(
        default="DataArray",
        description="The type of the variable, specified as a string.",
    )
    description: str | None = Field(description="Description of the variable.")
    dimensions: list[Dimension] = Field(description="Dimensions of the DataArray.")
    time_dim: str = Field(description="The time dimension of the DataArray.")
    units: str | None = Field(description="The units of the values in the data.")
    spatial_coverage: Literal["global"] = Field(
        description="The spatial area for which the variable has values."
    )
    temporal_coverage: tuple[datetime, datetime] = Field(
        description="The start and end times that the variable covers"
    )
    temporal_resolution: timedelta | None = Field(
        description=(
            "The resolution of the time field for the variable, or the time between each data point"
        )
    )

    @field_validator("dimensions")
    @classmethod
    def _ensure_lat_lon_dims(cls, dims: list[Dimension]) -> list[Dimension]:
        dim_std_names = {dim.cf_standard_name for dim in dims}
        if not {"latitude", "longitude"}.issubset(dim_std_names):
            msg = (
                "SpatioTemporalDataArrayVariable must have both latitude and longitude dimensions."
            )
            raise ValueError(msg)
        return dims


def input_variable_from_model(variable: Variable, dataset: Dataset) -> InputVariable:
    """Transform from model.Variable to SpatioTemporalDataArrayVariable.

    Args:
        variable (Variable): The model variable to transform
        dataset (Dataset): The dataset to which the variable belongs.
            Used to provide dataset-level metadata to the `InputVariable`.

    Returns:
        InputVariable: An `InputVariable`.
    """
    if variable.is_spatiotemporal():
        return SpatioTemporalDataArrayVariable(
            name=variable.field_name,
            description=variable.description,
            dimensions=variable.dimensions,
            time_dim=next(dim.field_name for dim in variable.dimensions if dim.is_temporal()),
            units=variable.units,
            spatial_coverage=dataset.spatial_coverage,
            temporal_coverage=dataset.temporal_coverage,
            temporal_resolution=dataset.temporal_resolution,
        )
    if variable.is_temporal():
        return TemporalDataArrayVariable(
            name=variable.field_name,
            description=variable.description,
            dimensions=variable.dimensions,
            time_dim=next(dim.field_name for dim in variable.dimensions if dim.is_temporal()),
            units=variable.units,
            temporal_coverage=dataset.temporal_coverage,
            temporal_resolution=dataset.temporal_resolution,
        )
    return GeneralDataArrayVariable(
        name=variable.field_name,
        description=variable.description,
        dimensions=variable.dimensions,
        units=variable.units,
    )


def input_vars_to_function_set(
    input_vars: Sequence[InputVariable],
) -> FunctionSet:
    """Build function set for given input variables."""
    func_sets = [_input_var_to_function_set(v) for v in input_vars]
    if len(func_sets) == 0:
        return FunctionSet()
    return FunctionSet.merge([*func_sets, GeneralPlottingFunctionSet()])


def _input_var_to_function_set(input_var: InputVariable) -> FunctionSet:
    """Map input variable to appropriate function set."""
    dim_field_names = frozenset(dim.field_name for dim in input_var.dimensions)
    match input_var:
        case SpatioTemporalDataArrayVariable():
            return SpatioTemporalDataArrayFunctionSet(allowed_dims=dim_field_names)
        case TemporalDataArrayVariable():
            return TemporalDataArrayFunctionSet(allowed_dims=dim_field_names)
        case GeneralDataArrayVariable():
            return GeneralDataArrayFunctionSet(allowed_dims=dim_field_names)


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
    inputs_provided: Sequence[InputVariable] = Field(
        description="List of variables that can be assumed to already exist "
        "and can be safely referenced in the algorithm.",
        min_length=1,
    )
    user_provided_geometry: Geometry | None = Field(
        description="The user-provided geometry the algorithm should use for analysis.",
    )
    outputs_requested: list[OutputVariable] = Field(
        description="Variables that must be included as outputs of the algorithm.",
        min_length=1,
    )


class AlgorithmSolutionSpec[FunctionCall: BaseFunctionCall](SolutionSpec):
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
    algorithm: Algorithm[FunctionCall] = Field(
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
