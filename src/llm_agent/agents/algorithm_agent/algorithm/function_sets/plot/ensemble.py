from typing import Literal

from pydantic import Field

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.function_sets.base import FunctionSet
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    AnnotationSpec,
    MakePlot,
    PlotSpec,
)


class EnsemblePlumePlotSpec(PlotSpec, frozen=True):
    """Specialized spec for plotting an ensemble forecast as a "plume" chart.

    Takes a 2D DataArray with an ensemble dimension and one other dimension (typically lead time or
    time). Draws each ensemble member as a thin, faint line, a bold ensemble-mean line, and a
    shaded band showing the spread across members. Should be used by default for visualizing
    ensemble forecast uncertainty (e.g. GEFS or ECMWF ensemble output at a point), unless there is
    a good reason not to.
    """

    spec_name: Literal["EnsemblePlumePlotSpec"] = "EnsemblePlumePlotSpec"
    data_array: VariableReference = Field(
        description="A 2D DataArray with an ensemble dimension and one other dimension (the x "
        "axis, e.g. lead_time). Typically the forecast at a single point across all members."
    )
    ensemble_dim: str = Field(
        default="ensemble_member",
        description="Name of the ensemble dimension to collapse across for the mean and spread.",
    )
    show_members: bool = Field(
        default=True,
        description="If True, draw each individual ensemble member as a thin, faint line.",
    )
    show_mean: bool = Field(
        default=True, description="If True, draw the ensemble mean as a bold line."
    )
    band_type: Literal["none", "minmax", "iqr", "p10_p90"] = Field(
        default="p10_p90",
        description="Shaded spread band across members: 'none' for no band, 'minmax' for the full "
        "range, 'iqr' for the 25th-75th percentile range, or 'p10_p90' for the 10th-90th "
        "percentile range.",
    )
    annotation: AnnotationSpec = Field(default_factory=AnnotationSpec)


class EnsemblePlottingFunctionSet(FunctionSet):
    """Ensemble forecast plotting functions."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__(
            [MakePlot[EnsemblePlumePlotSpec]],
            func_name_to_discriminators={"MakePlot": ("spec_name",)},
        )
