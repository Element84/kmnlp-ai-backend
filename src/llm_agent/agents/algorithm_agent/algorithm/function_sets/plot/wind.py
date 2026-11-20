from typing import Literal

from pydantic import Field

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.function_sets.base import FunctionSet
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    MakeAnimation,
    MakePlot,
    MapProjectionSpec,
    PlotSpec,
)


class WindVectorPlotSpec(PlotSpec, frozen=True):
    """Specialized spec for plotting a 2D wind field as vectors on a map.

    Draws wind direction and magnitude from the eastward (u) and northward (v) wind components.
    Should be used by default for visualizing wind fields, unless there is a good reason not to.
    Both components must be 2D with 'lat' and 'lon' dimensions and share the same grid.
    """

    spec_name: Literal["WindVectorPlotSpec"] = "WindVectorPlotSpec"
    u_data_array: VariableReference = Field(
        description="Eastward (u) wind component. Must be a 2D DataArray with 'lat' and 'lon' dims."
    )
    v_data_array: VariableReference = Field(
        description="Northward (v) wind component. Must share the same grid as the u component."
    )
    barbs: bool = Field(
        default=False,
        description="If True, draw meteorological wind barbs instead of arrows. Barbs are the "
        "conventional way to show wind speed and direction on weather maps.",
    )
    color_by_speed: bool = Field(
        default=True,
        description="If True, color the arrows by wind speed and add a colorbar. "
        "Only relevant when barbs is False.",
    )
    subsample: int = Field(
        default=1,
        ge=1,
        description="Plot every Nth grid point along each axis to reduce visual clutter on dense "
        "grids (e.g. 5). A value of 1 plots every grid point.",
    )
    map_projection: MapProjectionSpec = Field(default_factory=MapProjectionSpec)
    title: str | None = Field(default=None)


class WindPlottingFunctionSet(FunctionSet):
    """Wind vector plotting functions."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__(
            [
                MakePlot[WindVectorPlotSpec],
                MakeAnimation[WindVectorPlotSpec],
            ],
            func_name_to_discriminators={
                "MakePlot": ("spec_name",),
                "MakeAnimation": ("spec_name",),
            },
        )
