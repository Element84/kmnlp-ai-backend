from collections.abc import Iterable
from typing import Annotated, Any, Literal, Self

from geojson_pydantic.geometries import Geometry
from pydantic import BaseModel, Field, create_model, model_validator

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Function,
    VariableReference,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets import (
    FunctionSet,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.utils import (
    concat_camel_case,
)

_SliceBoundsT = tuple[str, str] | tuple[float, float]


####################
# General functions
####################
class BinaryOperation(Function, frozen=True):
    """Apply a binary operation to numeric values or DataArrays."""

    func_name: Literal["BinaryOperation"]
    left: int | float | VariableReference = Field(description="The left value in the operation.")
    right: int | float | VariableReference = Field(description="The right value in the operation.")
    op: Literal["add", "sub", "mul", "div"]


class ZeroDimDataArrayToScalar(Function, frozen=True):
    """Convert a 0-dim DataArray to a scalar (int or float)."""

    func_name: Literal["ZeroDimDataArrayToScalar"]
    data_array: VariableReference = Field(
        description="A 0-dim DataArray. This will usually be the output of a reduction operation."
    )


class DataArrayToList(Function, frozen=True):
    """Convert a 1D DataArray to a list of scalars."""

    func_name: Literal["DataArrayToList"]
    data_array: VariableReference = Field(description="A 1D DataArray.")


class SliceDataArray(Function, frozen=True):
    """Slice a DataArray along given dims."""

    # func_name is a str here instead of a Literal so that when we narrow the
    # type to a Literal in the subclass in with_constrained_dims(), the type
    # checker doesn't complain about an incompatible variable override.
    func_name: str = "SliceDataArray"
    data_array: VariableReference = Field(description="The DataArray to slice.")
    dim_slices: dict[str, _SliceBoundsT] = Field(
        default_factory=dict[str, _SliceBoundsT],
        description="Mapping from dim names to (start, end) tuples representing slice ranges. "
        "Start and end are inclusive. "
        "If the dim is temporal, start and end must be ISO 8601 datetimes with an optional "
        "timezone-naive time component like 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS', which must "
        "make sense WRT the temporal resolution of the data (if known). "
        "If the dim is a durational time dimension (e.g. forecast lead time), start and end "
        "should be 0-indexed ISO 8601 durations like ('P0D', 'P2D') for the first 3 days, etc. "
        "For non-temporal dims, the slice values should be floats. "
        "To filter to a single value, set the start and end of the slice to be the same.",
    )
    dim_index_slices: dict[str, tuple[int, int | None]] = Field(
        default_factory=dict[str, tuple[int, int | None]],
        description="Mapping from dim names to (start_idx, end_idx) tuples representing slice "
        "ranges in terms of integer 0-indexed indices. Negative indices not allowed. "
        "Start_idx is inclusive and end_idx is exclusive. "
        "end_idx can also be None to indicate slicing until the end of the dimension. "
        "This is an alternative to dim_slices that allows for slicing by integer index "
        "rather than by coordinate value. Each dimension must only appear in either dim_slices or "
        "dim_index_slices, not both.",
    )

    @classmethod
    def with_constrained_dims(cls, allowed_dims: Iterable[str]) -> "type[SliceDataArray]":
        """Create class variant with constrained dims names."""
        original_field = cls.model_fields["dim_slices"]
        name = f"SliceDataArray_{concat_camel_case(sorted(allowed_dims))}"
        allowed_dims_type = Literal[*allowed_dims]
        return create_model(
            name,
            __base__=SliceDataArray,
            __doc__="Slice a DataArray along given dims. "
            "Dim names must strictly match allowed dims.",
            # field defs:
            func_name=Literal[name],
            dim_slices=Annotated[
                # unlike parent class, the dict keys are restricted to allowed
                # dim names
                dict[allowed_dims_type, _SliceBoundsT],
                Field(
                    default_factory=dict[allowed_dims_type, _SliceBoundsT],
                    description=original_field.description,
                ),
            ],
            dim_index_slices=Annotated[
                dict[allowed_dims_type, tuple[int, int | None]],
                Field(
                    default_factory=dict[allowed_dims_type, tuple[int, int | None]],
                    description=SliceDataArray.model_fields["dim_index_slices"].description,
                ),
            ],
        )

    @model_validator(mode="after")
    def _dim_name_exclusivity(self) -> Self:
        """Validate each dim name appears in either dim_slices or dim_index_slices, not both."""
        dim_slices_keys = set(self.dim_slices.keys())
        dim_index_slices_keys = set(self.dim_index_slices.keys())
        overlap = dim_slices_keys.intersection(dim_index_slices_keys)
        if overlap:
            raise ValueError(
                f"Dim names {overlap} appear in both dim_slices and dim_index_slices. "
                "Each dim name must only appear in one."
            )
        return self


class ReduceDataArray(Function, frozen=True):
    """Reduce a DataArray along some dims using an op."""

    # func_name is a str here instead of a Literal so that when we narrow the
    # type to a Literal in the subclass in with_constrained_dims(), the type
    # checker doesn't complain about an incompatible variable override.
    func_name: str = "ReduceDataArray"
    data_array: VariableReference = Field(description="The DataArray to reduce.")
    op: Literal["sum", "mean", "std", "min", "max"]
    dim: str | tuple[str, ...] | None = Field(
        default=None, description="Dimension(s) to reduce. If None, reduces all dims."
    )

    @classmethod
    def with_constrained_dims(cls, allowed_dims: Iterable[str]) -> "type[ReduceDataArray]":
        """Create class variant with constrained dims names."""
        original_field = cls.model_fields["dim"]
        allowed_dims_type = Literal[*allowed_dims]
        name = f"ReduceDataArray_{concat_camel_case(sorted(allowed_dims))}"
        return create_model(
            name,
            __base__=ReduceDataArray,
            __doc__="Reduce a DataArray along some dims using an op. "
            "Dim names must strictly match allowed dims.",
            # field defs:
            func_name=Literal[name],
            dim=Annotated[
                allowed_dims_type | tuple[allowed_dims_type, ...] | None,
                Field(
                    default=original_field.default,
                    description=original_field.description,
                ),
            ],
        )


class PlotDataArraysAsRaster(Function, frozen=True):
    """Plot 2D DataArrays as rasters and return the figure.

    If multiple DataArrays are provided, they will be plotted side-by-side as
    sub-plots in the same figure.
    """

    func_name: Literal["PlotDataArraysAsRaster"]
    data_arrays: list[VariableReference] = Field(description="List of 2D DataArrays to plot.")
    subplot_grid_shape: tuple[int, int] = Field(
        description="The shape of the subplot grid as a (nrows, ncols) tuple.",
    )
    title: str = Field("", description="Plot title.")


#####################
# Temporal functions
#####################
class GroupByTimeAndReduceDataArray(Function, frozen=True):
    """Group DataArray by some time unit and apply a reduction op.

    This collapses all non-time dimensions and returns a 1-dimensional time
    series.
    """

    func_name: Literal["GroupByTimeAndReduceDataArray"]
    data_array: VariableReference = Field(
        description="The DataArray to group by some time unit and reduce.",
    )
    op: Literal["sum", "mean", "std", "min", "max"]
    time_unit: Literal[
        "date",
        "week",
        "month",
        "quarter",
        "year",
        "dayofweek",
        "dayofmonth",
        "dayofyear",
    ] = Field(description="Time unit to group by.")


class PlotDataArraysAsTimeSeries(Function, frozen=True):
    """Plot 1D DataArrays as time series and return the figure.

    If multiple DataArrays are provided, they will be plotted as separate
    lines on the same figure.
    """

    func_name: Literal["PlotDataArraysAsTimeSeries"]
    data_arrays: list[VariableReference] = Field(description="DataArrays to plot.")
    labels: list[str] = Field(description="Legend labels for the data_arrays.")
    force_overlaid: bool = Field(
        default=False,
        description="If the time series are for non-overlapping intervals "
        "but you still want to overlay them for comparison. "
        "Defaults to False.",
    )
    title: str = Field("", description="Plot title.")


############################
# Spatio-temporal functions
############################


class FilterDataArrayByPolygon(Function, frozen=True):
    """Spatially filter a (time, lat, lon) DataArray with a polygon.

    The array is cropped to the bounds of the polygon and any points falling
    outside the polygon are masked out.
    """

    func_name: Literal["FilterDataArrayByPolygon"]
    data_array: VariableReference = Field(description="The (time, lat, lon) DataArray to filter.")
    polygon: VariableReference = Field(description="A Polygon describing some geographical area.")


class PointCoords(BaseModel):
    """Coordinates of a single point as scalar values.

    Also includes the value at that point.
    """

    coords: dict[str, str | float] = Field(
        description="Mapping from dim names to their values. "
        "E.g. {'time': '2023-01-01', 'lat': 30.0, 'lon': -90.0}"
    )
    value: float = Field(description="Value at the point represented by the coords.")


class SelectPointCoordsKey(Function, frozen=True):
    """Select an individual value from a PointCoords object, as a scalar."""

    func_name: Literal["SelectPointCoordsKey"]
    point_coords: VariableReference
    key: str = Field(
        description="The key in the coords dict to select. E.g. 'time', 'lat', or 'lon'."
    )

    @classmethod
    def model_json_schema(  # type:ignore[reportIncompatibleMethodOverride]
        cls, *args: list[Any], **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Include definition of PointCoords in JSON schema."""
        schema = super().model_json_schema(*args, **kwargs)  # type: ignore[reportUnknownArgumentType]
        schema["$defs"]["PointCoords"] = PointCoords.model_json_schema()
        return schema


class DataArrayArgmaxCoords(Function, frozen=True):
    """Return location of max value in DataArray as a PointCoords object."""

    func_name: Literal["DataArrayArgmaxCoords"]
    data_array: VariableReference


class DataArrayArgminCoords(Function, frozen=True):
    """Return location of min value in DataArray as a PontCoords object."""

    func_name: Literal["DataArrayArgminCoords"]
    data_array: VariableReference


class GeocodeNaturalLanguageDescription(Function, frozen=True):
    """Geocode natural language description of an area into a polygon."""

    func_name: Literal["GeocodeNaturalLanguageDescription"]
    text: str = Field(
        description="Natural language description of a location. "
        'Example inputs: "Caspian Sea", "Florida, north of Miami", '
        '"within 1 mile of the coast of Boston" etc.',
    )


class AssignUserProvidedGeometry(Function, frozen=True):
    """Assign a user-provided geometry to a variable to reference in future steps."""

    func_name: Literal["AssignUserProvidedGeometry"]

    geometry: Geometry = Field(
        description="The user-provided geometry the algorithm should use for analysis."
    )


################
# Function sets
################


class GeneralDataArrayFunctionSet(FunctionSet):
    """Functions for DataArrays with arbitrary dims."""

    def __init__(self, allowed_dims: Iterable[str]) -> None:
        """Constructor.

        Args:
            allowed_dims (Iterable[str]): Known dim names to constrain
                functions (that take dim names) to. This enables the dim name
                arguments to be validated via pydantic at runtime.
        """
        super().__init__(
            [
                BinaryOperation,
                ZeroDimDataArrayToScalar,
                SliceDataArray.with_constrained_dims(allowed_dims),
                ReduceDataArray.with_constrained_dims(allowed_dims),
                DataArrayArgmaxCoords,
                DataArrayArgminCoords,
                SelectPointCoordsKey,
                PlotDataArraysAsRaster,
            ]
        )


class TemporalDataArrayFunctionSet(FunctionSet):
    """Functions for DataArrays with a time dim plus other arbitrary dims."""

    def __init__(self, allowed_dims: Iterable[str]) -> None:
        """Constructor.

        Args:
            allowed_dims (Iterable[str]): Known dim names to constrain
                functions (that take dim names) to. This enables the dim name
                arguments to be validated via pydantic at runtime.
        """
        super().__init__(
            [
                *GeneralDataArrayFunctionSet(allowed_dims).functions,
                GroupByTimeAndReduceDataArray,
                PlotDataArraysAsTimeSeries,
            ]
        )


class SpatioTemporalDataArrayFunctionSet(FunctionSet):
    """Functions for DataArrays with (time, lat, lon) dims."""

    def __init__(self, allowed_dims: Iterable[str] | None = None) -> None:
        """Constructor.

        Args:
            allowed_dims (Iterable[str]): Known dim names to constrain
                functions (that take dim names) to. This enables the dim name
                arguments to be validated via pydantic at runtime. If None,
                will be set to ("time", "lat", "lon"). Defaults to None.
        """
        allowed_dims = allowed_dims or ["time", "lat", "lon"]
        super().__init__(
            [
                *TemporalDataArrayFunctionSet(allowed_dims).functions,
                GeocodeNaturalLanguageDescription,
                AssignUserProvidedGeometry,
                FilterDataArrayByPolygon,
            ]
        )
