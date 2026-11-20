from typing import Literal

from pydantic import Field

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.function_sets.base import FunctionSet
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    MakePlot,
    MapProjectionSpec,
    PlotSpec,
)


class ContourMapPlotSpec(PlotSpec, frozen=True):
    """Specialized spec for plotting a 2D geospatial field as a contour map.

    Renders a 2D (lat, lon) DataArray as contours on a map. Use for smooth scalar fields where
    iso-lines are informative (e.g. mean sea level pressure, geopotential height, temperature).
    Filled contours (the default) give a banded color map; unfilled contours draw labeled iso-lines
    that can be overlaid to read exact values.
    """

    spec_name: Literal["ContourMapPlotSpec"] = "ContourMapPlotSpec"
    data_array: VariableReference = Field(
        description="A 2D DataArray with 'lat' and 'lon' dimensions to contour."
    )
    filled: bool = Field(
        default=True,
        description="If True, draw filled contours (contourf). If False, draw unfilled contour "
        "iso-lines.",
    )
    levels: int = Field(
        default=10,
        ge=1,
        description="Number of contour levels to draw.",
    )
    label_contours: bool = Field(
        default=False,
        description="If True, add inline numeric labels on the contour lines. Most useful for "
        "unfilled contours.",
    )
    cmap: str = Field(default="viridis", description="Color map to use for the contours.")
    colorbar_min_and_max: tuple[float | None, float | None] = (None, None)
    map_projection: MapProjectionSpec = Field(default_factory=MapProjectionSpec)
    title: str | None = Field(default=None)


class ContourPlottingFunctionSet(FunctionSet):
    """Contour map plotting functions."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__(
            [MakePlot[ContourMapPlotSpec]],
            func_name_to_discriminators={"MakePlot": ("spec_name",)},
        )
