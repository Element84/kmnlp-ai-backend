from typing import Literal

from pydantic import BaseModel, Field

from llm_agent.agents.algorithm_agent.algorithm.base import Function, VariableReference
from llm_agent.agents.algorithm_agent.algorithm.function_sets.base import FunctionSet


class MapProjectionSpec(BaseModel):
    """Spec for map projection parameters."""

    projection: Literal[
        "PlateCarree",
        "NorthPolarStereo",
        "SouthPolarStereo",
        "Robinson",
        "AlbersEqualArea",
        "EqualEarth",
        "InterruptedGoodeHomolosine",
    ] = Field(default="PlateCarree", description="Projection to use for spatial plotting.")
    central_longitude: float = Field(
        default=0.0,
        description="Central longitude for the map projection.",
    )


class GeospatialSpec(BaseModel):
    """Spec for geospatial plotting parameters."""

    map_projection: MapProjectionSpec = Field(default_factory=MapProjectionSpec)
    add_coastlines: bool = True
    add_borders: bool = True
    add_rivers: bool = True
    add_lakes: bool = True
    add_gridlines: bool = False


class RasterStyleSpec(BaseModel):
    """Spec for styling raster plots."""

    cmap: str = Field(default="plasma", description="Color map to use for the plot.")
    cmap_vmin: float | None = None
    cmap_vmax: float | None = None
    add_colorbar: bool = True
    colorbar_label: str = Field(default="", description="Label for the colorbar.")


class AnnotationSpec(BaseModel):
    """Spec for plot annotations like title and axis labels."""

    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None


class PlotSpec(BaseModel, frozen=True):
    """Base spec for plotting functions."""

    spec_name: str


class PlotGridSpec[PlotSpecT: PlotSpec](BaseModel):
    """Spec for plotting a grid of subplots."""

    spec_name: Literal["PlotGridSpec"] = "PlotGridSpec"
    ncols: int = Field(default=1, description="Number of columns in the subplot grid.")
    subplot_specs: list[PlotSpecT] = Field(
        description="List of plot specs for each subplot in the grid.", min_length=2
    )
    super_title: str | None = Field(default=None, description="Title for the entire grid of plots.")


class GeneralRasterPlotSpec(PlotSpec, frozen=True):
    """Spec for plotting general 2D raster data."""

    spec_name: Literal["GeneralRasterPlotSpec"] = "GeneralRasterPlotSpec"
    data_array: VariableReference = Field(description="The data array to plot.")
    style: RasterStyleSpec = Field(default_factory=RasterStyleSpec)
    annotation: AnnotationSpec = Field(default_factory=AnnotationSpec)


class GeospatialRasterPlotSpec(PlotSpec, frozen=True):
    """Spec for plotting geospatial raster data."""

    spec_name: Literal["GeospatialRasterPlotSpec"] = "GeospatialRasterPlotSpec"
    data_array: VariableReference = Field(description="The data array to plot.")
    geospatial: GeospatialSpec = Field(default_factory=GeospatialSpec)
    style: RasterStyleSpec = Field(default_factory=RasterStyleSpec)
    annotation: AnnotationSpec = Field(default_factory=AnnotationSpec)


class AnimationSpec[PlotSpecT: PlotSpec](BaseModel):
    """Spec for creating animations."""

    spec_name: Literal["AnimationSpec"] = "AnimationSpec"
    plot_spec: PlotSpecT = Field(
        description="Plot spec for each frame of the animation with a 3D data array."
    )
    frame_dim: str = Field(description="Name of the dimension to animate over.")
    frame_interval_seconds: float = 0.5


class LineStyleSpec(BaseModel):
    """Spec for styling line plots."""

    color: str | None = Field(
        default=None, description="Color of the line. Uses default palette if not set."
    )
    linestyle: Literal["-", "--", "-."] = Field(
        default="-", description="Style of the line (e.g., solid, dashed, etc.)."
    )
    linewidth: float = Field(default=1, description="Width of the line.")


class LinePlotSpec(PlotSpec, frozen=True):
    """Spec for plotting line data."""

    spec_name: Literal["LinePlotSpec"] = "LinePlotSpec"
    data_array: VariableReference = Field(description="A 1D data array to plot.")
    data_array_error: VariableReference | None = Field(
        default=None,
        description="Optional 1D data array of the same shape as data_array containing error "
        "values to plot as shaded regions around the line.",
    )
    error_display_type: Literal["shaded", "errorbars"] = Field(
        default="shaded",
        description="How to display the error values: as shaded regions or error bars.",
    )
    add_linear_trendline: bool = Field(
        default=False,
        description="Whether to add a linear trendline to the plot. "
        "This will show up as a dotted, less prominent line of the same color as the main line.",
    )
    linear_trendline_time_unit: Literal["day", "year", "decade"] = Field(
        default="year",
        description="If x is temporal, this specifies the time unit for the linear trendline slope."
        "For example, degC per year for temperature trends. Defaults to 'year'. "
        "Ignored if x is not temporal.",
    )
    style: LineStyleSpec = Field(default_factory=LineStyleSpec)
    annotation: AnnotationSpec = Field(default_factory=AnnotationSpec)


class MultiLinePlotSpec(PlotSpec, frozen=True):
    """Spec for plotting multiple lines on the same plot."""

    spec_name: Literal["MultiLinePlotSpec"] = "MultiLinePlotSpec"
    specs: list[LinePlotSpec] = Field(
        description="List of LinePlotSpecs for each line to plot. Their titles will be used "
        "as legend keys. By default, each line will be styled with a unique color from the "
        "default color palette if colors have not been explicitly specified."
    )
    annotation: AnnotationSpec = Field(
        default_factory=AnnotationSpec,
        description="Axis labels here will override those in the individual LinePlotSpecs.",
    )


class BarPlotSpec(PlotSpec, frozen=True):
    """Spec for plotting a bar chart."""

    spec_name: Literal["BarPlotSpec"] = "BarPlotSpec"
    heights: list[VariableReference] = Field(description="Values of the bars.")
    labels: list[str] = Field(description="Labels of the bars.")
    annotation: AnnotationSpec = Field(default_factory=AnnotationSpec)


class HistogramPlotSpec(PlotSpec, frozen=True):
    """Spec for plotting a histogram of a DataArray's values.

    The DataArray is flattened across all of its dimensions and the distribution of its values is
    shown as a histogram. NaNs (e.g. masked-out points) are ignored. Use this to inspect the
    distribution of a quantity, e.g. the spread of daily precipitation intensities.
    """

    spec_name: Literal["HistogramPlotSpec"] = "HistogramPlotSpec"
    data_array: VariableReference = Field(
        description="The DataArray whose value distribution to plot, flattened across all dims."
    )
    bins: int = Field(default=30, ge=1, description="Number of histogram bins.")
    density: bool = Field(
        default=False,
        description="If True, show a normalized probability density instead of raw counts.",
    )
    annotation: AnnotationSpec = Field(default_factory=AnnotationSpec)


class ScatterPlotSpec(PlotSpec, frozen=True):
    """Spec for a scatter plot of one 1D DataArray against another.

    Plots paired values of two 1D DataArrays as points to reveal their relationship (e.g. sea
    surface temperature vs sea-ice concentration). Both arrays must be 1D and the same length;
    pairs where either value is NaN are dropped.
    """

    spec_name: Literal["ScatterPlotSpec"] = "ScatterPlotSpec"
    x_data_array: VariableReference = Field(description="A 1D DataArray for the x axis.")
    y_data_array: VariableReference = Field(
        description="A 1D DataArray for the y axis, the same length as x_data_array."
    )
    add_regression_line: bool = Field(
        default=False,
        description="If True, overlay a linear least-squares regression line to summarize the "
        "relationship.",
    )
    annotation: AnnotationSpec = Field(default_factory=AnnotationSpec)


class MakePlot[PlotSpecT: PlotSpec](Function, frozen=True):
    """Function to create a plot based on a given specification."""

    func_name: Literal["MakePlot"] = "MakePlot"
    plot_spec: PlotSpecT | PlotGridSpec[PlotSpecT]


class MakeAnimation[PlotSpecT: PlotSpec](Function, frozen=True):
    """Function to create an animation based on a given specification."""

    func_name: Literal["MakeAnimation"] = "MakeAnimation"
    animation_spec: AnimationSpec[PlotSpecT]


class GeneralPlottingFunctionSet(FunctionSet):
    """General plotting functions."""

    def __init__(self) -> None:
        """Constructor."""
        spec_union = (
            LinePlotSpec
            | MultiLinePlotSpec
            | GeneralRasterPlotSpec
            | BarPlotSpec
            | HistogramPlotSpec
            | ScatterPlotSpec
        )
        super().__init__(
            [
                MakePlot[spec_union],
                MakeAnimation[spec_union],
            ],
            func_name_to_discriminators={
                "MakePlot": ("spec_name",),
                "MakeAnimation": ("spec_name",),
            },
        )


class GeospatialPlottingFunctionSet(FunctionSet):
    """Geospatial plotting functions."""

    def __init__(self) -> None:
        """Constructor."""
        spec_union = (
            LinePlotSpec
            | MultiLinePlotSpec
            | GeneralRasterPlotSpec
            | GeospatialRasterPlotSpec
            | BarPlotSpec
            | HistogramPlotSpec
            | ScatterPlotSpec
        )
        super().__init__(
            [
                MakePlot[spec_union],
                MakeAnimation[spec_union],
            ],
            func_name_to_discriminators={
                "MakePlot": ("spec_name",),
                "MakeAnimation": ("spec_name",),
            },
        )
