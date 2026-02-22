from typing import Literal

from pydantic import Field

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Function,
    VariableReference,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.base import (
    FunctionSet,
)


class PlotBarChart(Function, frozen=True):
    """Plot a bar chart."""

    func_name: Literal["PlotBarChart"]
    labels: list[str] = Field(description="Labels of the bars.")
    heights: list[VariableReference] = Field(description="Values of the bars.")
    title: str


class GeneralPlottingFunctionSet(FunctionSet):
    """General plotting functions."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__([PlotBarChart])
