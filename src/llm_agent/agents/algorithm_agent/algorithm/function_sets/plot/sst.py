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


class SSTPlotSpec(PlotSpec, frozen=True):
    """Specialized spec for plotting sea surface temperature (SST) data.

    Uses best practices for visualizing SST and should be used by default for SST data, unless
    there is a good reason not to.
    """

    spec_name: Literal["SSTPlotSpec"] = "SSTPlotSpec"
    sst_data_array: VariableReference
    sea_ice_data_array: VariableReference | None = Field(
        default=None,
        description="Optional sea ice concentration or boolean ice mask data array to overlay on "
        "the SST plot.",
    )
    ice_threshold: float = Field(
        default=0.15,
        description="Sea ice concentration threshold (as a fraction) for plotting ice. "
        "Only relevant if sea_ice_data_array is provided. "
        "Ignored if sea_ice_data_array is a boolean ice mask.",
    )
    map_projection: MapProjectionSpec = Field(default_factory=MapProjectionSpec)
    colorbar_min_and_max: tuple[float | None, float | None] = (None, None)
    title: str | None = Field(default=None)


class SSTAnomalyPlotSpec(PlotSpec, frozen=True):
    """Specialized spec for plotting sea surface temperature anomaly (SSTA) data.

    Uses best practices for visualizing SSTA and should be used by default for SSTA data, unless
    there is a good reason not to.
    """

    spec_name: Literal["SSTAnomalyPlotSpec"] = "SSTAnomalyPlotSpec"
    ssta_data_array: VariableReference
    map_projection: MapProjectionSpec = Field(default_factory=MapProjectionSpec)
    colorbar_min_and_max: tuple[float | None, float | None] = (None, None)
    title: str | None = Field(default=None)


class SSTPlottingFunctionSet(FunctionSet):
    """SST plotting functions."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__(
            [
                MakePlot[SSTPlotSpec | SSTAnomalyPlotSpec],
                MakeAnimation[SSTPlotSpec | SSTAnomalyPlotSpec],
            ],
            func_name_to_discriminators={
                "MakePlot": ("spec_name",),
                "MakeAnimation": ("spec_name",),
            },
        )
