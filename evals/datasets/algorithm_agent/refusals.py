"""Refusal / ambiguity eval cases for the algorithm agent.

Each case expects the agent to decline with an ``error_message`` rather than produce an algorithm:
the request is unanswerable from the available data, out of temporal range, too ambiguous, or
outside what the function set can express.
"""
# ruff: noqa: E501, D103, N806
# pyright: reportPrivateUsage=false

from collections.abc import Callable

from llm_agent.agents.algorithm_agent.algorithm.base import Algorithm, BaseFunctionCall
from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmProblemSpec,
    AlgorithmSolutionSpec,
    OutputVariable,
)
from llm_agent.evaluation.algorithm_agent import AlgorithmAgentEvalInput

from .common import _FunctionCall, _input_vars


def _refusal(
    *,
    case_id: str,
    statement: str,
    input_keys: list[str],
    output: OutputVariable,
    error_message: str,
) -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    """Build a refusal case: an empty algorithm carrying an ``error_message``."""
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id=case_id,
        problem=AlgorithmProblemSpec(
            statement=statement,
            inputs_provided=[_input_vars[k] for k in input_keys],
            outputs_requested=[output],
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](statements=[], output_variables=[]),
            explanation="",
            error_message=error_message,
        ),
    )


def refusal_variable_absent() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    return _refusal(
        case_id="refusal_variable_absent",
        statement="What was the sea-surface salinity in the Baltic Sea in 2023?",
        input_keys=["sst"],
        output=OutputVariable(
            name="salinity_value",
            variable_type="scalar_float",
            description="Mean sea-surface salinity value.",
        ),
        error_message="No sea-surface salinity variable is available in the provided data.",
    )


def refusal_time_out_of_range() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    return _refusal(
        case_id="refusal_time_out_of_range",
        statement="Map the sea surface temperature for the year 1850.",
        input_keys=["sst"],
        output=OutputVariable(
            name="sst_plot",
            variable_type="raster_plot",
            description="SST map for 1850.",
        ),
        error_message="The requested year (1850) is outside the temporal coverage of the dataset.",
    )


def ambiguous_recent_coast() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    return _refusal(
        case_id="ambiguous_recent_coast",
        statement="Show me the recent temperature near the coast.",
        input_keys=["temp2m"],
        output=OutputVariable(
            name="temperature_plot",
            variable_type="raster_plot",
            description="Recent coastal temperature.",
        ),
        error_message="The request is ambiguous: no specific region, time period, or definition of 'recent' is given.",
    )


def refusal_unsupported_op() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    return _refusal(
        case_id="refusal_unsupported_op",
        statement="Fit a neural network to predict next month's SST from the historical record.",
        input_keys=["sst"],
        output=OutputVariable(
            name="predicted_sst_plot",
            variable_type="raster_plot",
            description="Predicted next-month SST.",
        ),
        error_message="Forecasting via a trained model is not expressible with the available functions.",
    )


CASES: list[Callable[[], AlgorithmAgentEvalInput[BaseFunctionCall]]] = [
    refusal_variable_absent,
    refusal_time_out_of_range,
    ambiguous_recent_coast,
    refusal_unsupported_op,
]
