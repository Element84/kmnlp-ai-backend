from typing import Any, Literal

from pydantic import BaseModel, Field

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Function,
    VariableReference,
)


class BinaryOperation(Function):
    """Apply a binary operation to numeric values or DataArrays."""

    func_name: Literal["BinaryOperation"]
    left: int | float | VariableReference = Field(
        description="The left value in the operation."
    )
    right: int | float | VariableReference = Field(
        description="The right value in the operation."
    )
    op: Literal["add", "sub", "mul", "div"]


class FilterDataArray(Function):
    """Filter a (time, lat, lon) DataArray by slicing along given dims.

    To filter to a single value, set the start and end of the range to be the
    same.
    """

    func_name: Literal["FilterDataArray"]
    data_array: VariableReference = Field(
        description="The (time, lat, lon) DataArray to slice."
    )
    time_range: tuple[str, str] | None = Field(
        default=None,
        description="Slice to apply to the time dim. "
        "Start and stop must be date strings of the form 'YYYY-MM-DD'. "
        "Defaults to None (no slicing).",
    )
    lat_range: tuple[float, float] | None = Field(
        default=None,
        description="Slice to apply to the time dim. "
        "Start and stop must be floats in the range [-90, 90]. "
        "Defaults to None (no slicing).",
    )
    lon_range: tuple[float, float] | None = Field(
        default=None,
        description="Slice to apply to the time dim. "
        "Start and stop must be floats in the range [-180, 180]. "
        "Defaults to None (no slicing).",
    )


class FilterDataArrayByPolygon(Function):
    """Spatially filter a (time, lat, lon) DataArray with a polygon.

    The array is cropped to the bounds of the polygon and any points falling
    outside the polygon are masked out.
    """

    func_name: Literal["FilterDataArrayByPolygon"]
    data_array: VariableReference = Field(
        description="The (time, lat, lon) DataArray to filter."
    )
    polygon: VariableReference = Field(
        description="A Polygon describing some geographical area."
    )


class ReduceDataArray(Function):
    """Reduce a (time, lat, lon) DataArray along some dims using an op."""

    func_name: Literal["ReduceDataArray"]
    data_array: VariableReference = Field(
        description="The (time, lat, lon) DataArray to reduce."
    )
    op: Literal["sum", "mean", "std", "min", "max"]
    dim: str | tuple[Literal["time", "lat", "lon"], ...] | None = None


class GroupByTimeAndReduceDataArray(Function):
    """Group DataArray by some time unit and apply a reduction op.

    This collapses the spatial dimensions and returns a 1-dimensional time
    series.
    """

    func_name: Literal["GroupByTimeAndReduceDataArray"]
    data_array: VariableReference = Field(
        description="The (time, lat, lon) DataArray to groupby some time unit "
        "and reduce.",
    )
    op: Literal["sum", "mean", "std", "min", "max"]
    time_unit: Literal[
        "time_dim",
        "year",
        "quarter",
        "month",
        "week",
        "dayofmonth",
        "dayofyear",
        "dayofweek",
    ] = Field(
        description="If 'time_dim', the DataArray will be grouped using the "
        "existing time dimension. For others, the output 1D DataArray's "
        "single dim will not be called 'time' and will not have the same "
        "coords as the input DataArray.",
    )


class PointCoords(BaseModel):
    """Coordinates of a single spatio-temporal point."""

    time: str
    lat: float
    lon: float


class SelectPointCoordsDim(Function):
    """Select an individual value from a PointCoords object."""

    func_name: Literal["SelectPointCoordsDim"]
    point_coords: VariableReference
    dim: Literal["time", "lat", "lon"]

    @classmethod
    def model_json_schema(  # type:ignore[reportIncompatibleMethodOverride]
        cls, *args: list[Any], **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Include definition of PointCoords in JSON schema."""
        schema = super().model_json_schema(*args, **kwargs)  # type: ignore[reportUnknownArgumentType]
        schema["$defs"]["PointCoords"] = PointCoords.model_json_schema()
        return schema


class DataArrayArgmaxCoords(Function):
    """Return location of max value in DataArray as a PontCoords object."""

    func_name: Literal["DataArrayArgmaxCoords"]
    data_array: VariableReference = Field(
        description="A (time, lat, lon) DataArray."
    )


class DataArrayArgminCoords(Function):
    """Return location of min value in DataArray as a PontCoords object."""

    func_name: Literal["DataArrayArgminCoords"]
    data_array: VariableReference


class DataArrayToScalar(Function):
    """Convert a 0-dim DataArray to a scalar (int or float)."""

    func_name: Literal["DataArrayToScalar"]
    data_array: VariableReference = Field(
        description="A (time, lat, lon) DataArray."
    )


class DataArrayToList(Function):
    """Convert a 1D DataArray to a list of scalars."""

    func_name: Literal["DataArrayToList"]
    data_array: VariableReference = Field(description="A 1D DataArray.")


class GeocodeNaturalLanguageDescription(Function):
    """Geocode natural language description of an area into a polygon."""

    func_name: Literal["GeocodeNaturalLanguageDescription"]
    text: str = Field(
        description="Natural language description of a location. "
        'Example inputs: "Caspian Sea", "Florida, north of Miami", '
        '"within 1 mile of the coast of Boston" etc.',
    )


class PlotDataArraysAsRaster(Function):
    """Plot (time, lat, lon) DataArrays as rasters and return the figure.

    If multiple DataArrays are provided, they will be plotted side-by-side as
    sub-plots in the same figure.
    """

    func_name: Literal["PlotDataArraysAsRaster"]
    data_arrays: list[VariableReference] = Field(
        description="DataArray to plot. If the size of the time dimension is "
        "greater than 1, only the first time step will be plotted.",
    )
    subplot_grid_shape: tuple[int, int] = Field(
        description="The shape of the subplot grid as a (nrows, ncols) tuple.",
    )
    title: str = Field("", description="Plot title.")


class PlotDataArraysAsTimeSeries(Function):
    """Plot 1D DataArrays as time series and return the figure.

    If multiple DataArrays are provided, they will be plotted as separate
    lines on the same plot.
    """

    func_name: Literal["PlotDataArraysAsTimeSeries"]
    data_arrays: list[VariableReference] = Field(
        description="DataArrays to plot. If the size of the time dimension is "
        "greater than 1, only the first time step will be plotted.",
    )
    title: str = Field("", description="Plot title.")


DataArrayFunctionSet = (
    FilterDataArray
    | BinaryOperation
    | DataArrayArgmaxCoords
    | DataArrayArgminCoords
    | DataArrayToScalar
    | FilterDataArrayByPolygon
    | GeocodeNaturalLanguageDescription
    | GroupByTimeAndReduceDataArray
    | ReduceDataArray
    | SelectPointCoordsDim
    | PlotDataArraysAsTimeSeries
    | PlotDataArraysAsRaster
)
