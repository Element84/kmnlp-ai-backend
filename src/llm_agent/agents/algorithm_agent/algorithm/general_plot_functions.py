from typing import Literal

from pydantic import Field

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Function,
    VariableReference,
)


class PlotBarChart(Function):
    """Plot a bar chart."""

    func_name: Literal["PlotBarChart"]
    labels: list[str] = Field(description="Labels of the bars.")
    heights: list[VariableReference] = Field(description="Values of the bars.")
    title: str


GeneralPlottingFunctionSet = PlotBarChart
