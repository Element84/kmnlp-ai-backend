import logging
from abc import ABC
from collections.abc import Sequence
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, field_validator

from llm_agent.agents.algorithm_agent.algorithm.base import Algorithm, BaseFunctionCall
from llm_agent.agents.algorithm_agent.algorithm.function_sets import (
    ContourPlottingFunctionSet,
    EnsemblePlottingFunctionSet,
    FunctionSet,
    GeneralDataArrayFunctionSet,
    GeneralPlottingFunctionSet,
    GeospatialPlottingFunctionSet,
    SpatioTemporalDataArrayFunctionSet,
    SSTPlottingFunctionSet,
    TemporalDataArrayFunctionSet,
    WindPlottingFunctionSet,
)
from llm_agent.model.dataset import Dataset
from llm_agent.model.dimension import Dimension
from llm_agent.model.variable import Variable

log = logging.getLogger(__name__)

type DataArrayVariable = (
    SpatioTemporalDataArrayVariable | TemporalDataArrayVariable | GeneralDataArrayVariable
)

# InputVariable should be a union of supported input variable types
type InputVariable = DataArrayVariable | GeometryVariable

DimensionSet = Annotated[
    frozenset[str],
    PlainSerializer(lambda dims: sorted(dims), return_type=list[str], when_used="json"),
]


class GeometryVariable(BaseModel):
    """A variable representing a geometry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Name of the variable.")
    variable_type: Literal["geometry"] = Field(
        default="geometry",
        description="The type of the variable, specified as a string.",
    )
    description: str | None = Field(description="Description of the variable.")
    geometry_type: Literal["polygon"] = Field(
        description="The type of geometry represented by this variable."
    )


class GeneralDataArrayVariable(BaseModel):
    """A DataArray variable with arbitrary dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Name of the variable.")
    dataset_name: str = Field(description="Name of the dataset the variable belongs to.")
    variable_type: Literal["DataArray"] = Field(
        default="DataArray",
        description="The type of the variable, specified as a string.",
    )
    standard_name: str | None = Field(
        default=None, description="The CF standard name of the variable, if applicable. "
    )
    description: str | None = Field(description="Description of the variable.")
    dimensions: list[Dimension] = Field(description="Dimensions of the DataArray.")
    units: str | None = Field(description="The units of the values in the data.")


class TemporalDataArrayVariable(GeneralDataArrayVariable):
    """A DataArray variable with a time dim plus other arbitrary dims."""

    time_dim: str = Field(description="The time dimension of the DataArray.")
    temporal_coverage: tuple[datetime, datetime] = Field(
        description="The start and end times that the variable covers"
    )
    temporal_resolution: timedelta | None = Field(
        description=(
            "The resolution of the time field for the variable, or the time between each data point"
        )
    )


class SpatioTemporalDataArrayVariable(GeneralDataArrayVariable):
    """A spatiotemporal DataArray variable.

    If spatial_coverage is "global", then the lat and lon dims are guaranteed to be in the range
    [-90, 90] and [-180, 180], respectively.
    """

    time_dim: str = Field(description="The time dimension of the DataArray.")
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


def input_variable_from_model(variable: Variable, dataset: Dataset) -> DataArrayVariable:
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
            dataset_name=variable.dataset_name,
            description=variable.description,
            standard_name=variable.standard_name,
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
            dataset_name=variable.dataset_name,
            description=variable.description,
            standard_name=variable.standard_name,
            dimensions=variable.dimensions,
            time_dim=next(dim.field_name for dim in variable.dimensions if dim.is_temporal()),
            units=variable.units,
            temporal_coverage=dataset.temporal_coverage,
            temporal_resolution=dataset.temporal_resolution,
        )
    return GeneralDataArrayVariable(
        name=variable.field_name,
        dataset_name=variable.dataset_name,
        description=variable.description,
        standard_name=variable.standard_name,
        dimensions=variable.dimensions,
        units=variable.units,
    )


def input_vars_to_function_set(input_vars: Sequence[InputVariable]) -> FunctionSet:
    """Build function set for given input variables."""
    func_sets = [_input_var_to_function_set(v) for v in input_vars]
    if len(func_sets) == 0:
        return FunctionSet()
    return FunctionSet.merge(func_sets)


def _input_var_to_function_set(input_var: InputVariable) -> FunctionSet:
    """Map input variable to appropriate function set."""
    if isinstance(input_var, GeometryVariable):
        return FunctionSet()  # TODO: add geometry functions when needed

    dim_field_names = frozenset(dim.field_name for dim in input_var.dimensions)
    match input_var:
        case SpatioTemporalDataArrayVariable():
            fs: list[FunctionSet] = [
                SpatioTemporalDataArrayFunctionSet(allowed_dims=dim_field_names),
                GeospatialPlottingFunctionSet(),
                ContourPlottingFunctionSet(),
            ]
            if input_var.standard_name is not None:
                log.debug("Input variable standard name: %s", input_var.standard_name)
                # These are not exact string matches because standard_name is a URI like
                # http://mmisw.org/ont/cf/parameter/sea_surface_temperature.
                if "sea_surface_temperature" in input_var.standard_name:
                    fs.append(SSTPlottingFunctionSet())
                if (
                    "eastward_wind" in input_var.standard_name
                    or "northward_wind" in input_var.standard_name
                ):
                    fs.append(WindPlottingFunctionSet())
            if "ensemble_member" in dim_field_names:
                fs.append(EnsemblePlottingFunctionSet())
            return FunctionSet.merge(fs)

        case TemporalDataArrayVariable():
            return FunctionSet.merge(
                [
                    TemporalDataArrayFunctionSet(allowed_dims=dim_field_names),
                    GeneralPlottingFunctionSet(),
                ]
            )

        case GeneralDataArrayVariable():
            return FunctionSet.merge(
                [
                    GeneralDataArrayFunctionSet(allowed_dims=dim_field_names),
                    GeneralPlottingFunctionSet(),
                ]
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
        "raster_plot",
        "bar_plot",
        "histogram",
        "scatter_plot",
        "line_plot",
        "plot_grid",
        "geotiff",
        "animation",
    ] = Field(
        description="The type of the variable. "
        "Use 'geotiff' only if the user query implies that the user wants to see the data "
        "rendered on an interactive map in the UI (e.g. 'show me on the map ...') and only if "
        "a single raster output is requested; otherwise, use "
        "'raster_plot', which will render as static plot images in the chat UI. If the user "
        "specifies styling specifics, that is a hint to use 'raster_plot'."
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
    outputs_requested: list[OutputVariable] = Field(
        description="Variables that must be included as outputs of the algorithm.",
        min_length=1,
    )


class AlgorithmSolutionSpec[FunctionCall: BaseFunctionCall](SolutionSpec):
    """Specification of an algorithmic solution."""

    analysis: str = Field(
        description=dedent("""\
            First step: Understanding of the problem, understanding of the
            inputs, understanding of the requested outputs, and reasoning for the
            solution.

            Answer the following:
            - Which of the given inputs are relevant to the problem?
            - What is the optimal order in which to apply filters and operations to the data so as
            to minimize the amount of data that needs to be processed?
            - Identify potentially tricky aspects of the problem or the data, if any.
            - If the data is spatiotemporal, how do the temporal and spatial coverages of the
            inputs relate to each other and to the problem?
            - If the data is spatiotemporal, how do the temporal and spatial resolutions of the
            inputs relate to each other and to the problem?
            - How should the temporal resolutions be handled in reduction and group-by operations,
            if applicable?

            It is important to determine at this stage whether all the
            requested outputs are constructable STRICTLY using only the
            available functions. You must report an error if any of them are
            not. The cost of an incorrect solution is very high, so err on the
            side of declining a solution in case the problem is unclear.\
            """)
    )
    pseudocode: str = Field(
        description="Second step: Pseudocode in Python-like syntax for the algorithm, "
        "making use of the available functions. Ensure that types and "
        "dimensions of function inputs align with function definitions. Use "
        "single quotes for strings inside the pseudocode to avoid collisions "
        "with the rest of the JSON output. IMPORTANT: just like with the "
        "algorithm schema, the output of each function call must be assigned "
        "to a variable. Function calls MUST NOT be composed."
    )
    algorithm: Algorithm[FunctionCall] = Field(
        description="Third step: The algorithm. The statements and return variables of "
        "the algorithm can be set to empty lists in case of error.",
    )
    explanation: str = Field(
        description="Fourth step: Any additional explanation for the solution, if needed. Can be "
        "an empty string in case of error.",
    )
    error_message: str | None = Field(
        default=None,
        description="In case the problem is not solvable e.g. if it is "
        "unclear, malformed, unrelated to the data, too complex to be solved "
        "using the available functions, or for any other reason, this field "
        "should explain why. Otherwise, this should be null.",
    )
