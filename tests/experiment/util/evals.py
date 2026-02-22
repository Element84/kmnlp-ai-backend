# ruff: noqa: ARG001, A002, ANN401
import logging
from typing import Any, cast

from e84_geoai_common.llm.core import LLMAssistantMessage, LLMToolUseContent
from langfuse import Evaluation
from langfuse.experiment import ExperimentItemResult
from pydantic import ValidationError

from llm_agent.tools.data_analysis_tool.data_analysis_tool import (
    DataAnalysisToolInput,
)

log = logging.getLogger(__name__)

IS_DATA_ANALYSIS_TOOL_CALL_EVAL_NAME: str = "is_data_analysis_tool_call"
PROBLEM_STATEMENT_DOES_NOT_HAVE_BOUNDING_BOX_EVAL_NAME: str = "no_bounding_box"

RUN_EVAL_FRACTION_EXPECTED_CALL_EVAL_NAME: str = "run_eval_fraction_expected_call"
RUN_EVAL_FRACTION_NO_BOUNDING_BOX_EVAL_NAME: str = "run_eval_fraction_no_bounding_box"


def eval_has_data_analysis_tool_call(
    *,
    input: Any,
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None,
    **kwargs: dict[str, Any],
) -> list[Evaluation]:
    """Evaluate run experiment item for whether it made a data_analysis_tool call.

    This function's signature must match Langfuse's `EvaluatorFunction`. Its
    args represent the result from running an ExperimentItem, and it returns an
    Evaluation whether the output was a valid data analysis tool call.

    Args:
        input (Any): The input from the experiment item in the dataset.

        output (Any): The output from running the experiment item. Expected to
            be an `LLMAssistantMessage` in this context.

        expected_output (Any): The expected output recorded in the experiment
            item in the dataset. Unused in this context.

        metadata (dict[str, Any] | None): The metadata from the experiment item
            in the datasets. Unused in this context.

        **kwargs (dict[str, Any]): Additional kwargs provided at runtime.
            Unused in this context.

    Returns:
        Evaluation with name `is_data_analysis_tool_call` and boolean value
        indicating whether the output is parsed as a data analysis tool call by
        the application.
    """
    log.debug("eval_has_data_analysis_tool_call for output: %s", output)
    message = _output_as_llm_assistant_message(output)
    if message is None:
        log.debug("is_data_analysis_tool_call output value: %s", False)  # noqa: FBT003
        return [Evaluation(name=IS_DATA_ANALYSIS_TOOL_CALL_EVAL_NAME, value=False)]

    try:
        if _data_analysis_tool_call(message):
            log.debug("is_data_analysis_tool_call output value: %s", True)  # noqa: FBT003
            return [Evaluation(name=IS_DATA_ANALYSIS_TOOL_CALL_EVAL_NAME, value=True)]
    except Exception:
        log.exception("Error parsing data analysis tool call. Message is: %s", message)
        log.debug("is_data_analysis_tool_call output value: %s", False)  # noqa: FBT003
        return [Evaluation(name=IS_DATA_ANALYSIS_TOOL_CALL_EVAL_NAME, value=False)]

    log.debug("is_data_analysis_tool_call output value: %s", False)  # noqa: FBT003
    return [Evaluation(name=IS_DATA_ANALYSIS_TOOL_CALL_EVAL_NAME, value=False)]


def eval_problem_statement_does_not_pass_bounding_box(
    *,
    input: Any,
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None,
    **kwargs: dict[str, Any],
) -> list[Evaluation]:
    """Evaluate output data analysis tool call's problem statement.

    This function's signature must match Langfuse's `EvaluatorFunction`. Its
    args represent the result from running an ExperimentItem, and it returns an
    Evaluation whether the output was a valid data analysis tool call.

    The evaluator will retun no evaluation if the output from the experiment
    item is not a data analysis tool call. If it is a data analysis tool call,
    it will return whether the problem statement matches particular heuristics:
    namely the lack of presence of some keywords we don't expected to see:
    "box", "latitude," and "longitude."

    Args:
        input (Any): The input from the experiment item in the dataset.

        output (Any): The output from running the experiment item. Expected to
            be an `LLMAssistantMessage` in this context.

        expected_output (Any): The expected output recorded in the experiment
            item in the dataset. Unused in this context.

        metadata (dict[str, Any] | None): The metadata from the experiment item
            in the datasets. Unused in this context.

        **kwargs (dict[str, Any]): Additional kwargs provided at runtime.
            Unused in this context.

    Returns:
        A list of type `Evaluation`:
            * Empty list if the output being evaluated does not have a data
                analysis tool call.
            * A list with a single item in it indicating whether the tool call
                matched the no bounding box heuristics.
    """
    log.debug("eval_problem_statement_does_not_pass_bounding_box output: %s", output)
    message = _output_as_llm_assistant_message(output)
    if message is None:
        log.debug("eval_problem_statement_does_not_pass_bounding_box value: %s", None)
        return []

    tool_call = _data_analysis_tool_call(message)

    if tool_call is None:
        log.debug("eval_problem_statement_does_not_pass_bounding_box value: %s", None)
        return []

    lower_case_problem_statement = tool_call.problem_statement.lower()
    has_bounding_box = (
        ("box" not in lower_case_problem_statement)
        and ("latitude" not in lower_case_problem_statement)
        and ("longitude" not in lower_case_problem_statement)
    )
    log.debug("eval_problem_statement_does_not_pass_bounding_box value: %s", has_bounding_box)
    return [
        Evaluation(
            name=PROBLEM_STATEMENT_DOES_NOT_HAVE_BOUNDING_BOX_EVAL_NAME, value=has_bounding_box
        )
    ]


def run_eval_fraction_expected_call(
    *,
    item_results: list[ExperimentItemResult],
    **kwargs: dict[str, Any],
) -> list[Evaluation]:
    """Evaluate the full dataset run for data analysis tool calls.

    This function's signature must match Langfuse's `RunEvaluatorFunction`. Its
    args represent all of the evaluations run on the individual experiment
    items in the dataset. It returns a list of evaluations for summarizing
    those individual evaluations.

    This run evaluator will return 2 run evaluations: one representing how
    often the provided evaluations indicate that an experiment item's output
    contained a data analysis too call, and the other representing how often
    that tool call's arguments passed the heuristics indicating that it did not
    provide a bounding box to the algorithm agent.

    Args:
        item_results (list[ExperimentItemResult]): All of the experiment item
            results from the dataset run, representing the input and expected
            output, actual output, and evaulations.

        **kwargs (dict[str, Any]): Additional kwargs provided at runtime.
            Unused in this context.

    Returns:
        A list of type `Evaluation` with 2 items:
            * One item representing the fraction of experiment items whose
                output is parsed as a valid data analysis tool call.
            * One item representing the fraction of experiment items whose data
                analysis tool call arguments match the provided heuristics,
                discarding those that are not parseable as a data analysis tool
                call.
    """
    log.debug(
        "Run eval for item results: %s",
        [
            f"{evaluation.name}, {evaluation.value}"
            for result in item_results
            for evaluation in result.evaluations
        ],
    )
    data_analysis_tool_call_evals = [
        evaluation
        for item_result in item_results
        for evaluation in item_result.evaluations
        if evaluation.name == IS_DATA_ANALYSIS_TOOL_CALL_EVAL_NAME
    ]

    tool_no_bounding_box_evals = [
        evaluation
        for item_result in item_results
        for evaluation in item_result.evaluations
        if evaluation.name == PROBLEM_STATEMENT_DOES_NOT_HAVE_BOUNDING_BOX_EVAL_NAME
    ]

    mean_score_tool_call = _mean_from_eval_list(data_analysis_tool_call_evals)
    mean_no_bounding_box = _mean_from_eval_list(tool_no_bounding_box_evals)

    return [
        Evaluation(name=RUN_EVAL_FRACTION_EXPECTED_CALL_EVAL_NAME, value=mean_score_tool_call),
        Evaluation(name=RUN_EVAL_FRACTION_NO_BOUNDING_BOX_EVAL_NAME, value=mean_no_bounding_box),
    ]


def _data_analysis_tool_call(llm_response: LLMAssistantMessage) -> DataAnalysisToolInput | None:
    if isinstance(llm_response.content, list):
        data_analysis_tool_use_contents = [
            c.input["tool_input"]
            for c in llm_response.content
            if isinstance(c, LLMToolUseContent)
            if c.name == "DataAnalysisToolWithTracking"
        ]

        if len(data_analysis_tool_use_contents) == 0:
            log.warning(
                "Unable to find parseable data analysis tool call in response: %s", llm_response
            )
            return None

        try:
            parsed_content = DataAnalysisToolInput.model_validate(
                data_analysis_tool_use_contents[0]
            )
        except ValidationError:
            log.exception(
                "Error parsing DataAnalysisToolInput from response. Contents: %s",
                data_analysis_tool_use_contents[0],
            )
            raise
        else:
            return parsed_content

    # Content was a str or something else we didn't expect
    log.warning("LLM Response did not have have a data analysis tool call: %s", llm_response)
    return None


def _output_as_llm_assistant_message(output: Any) -> LLMAssistantMessage | None:
    if isinstance(output, LLMAssistantMessage):
        return output

    log.warning("Received ouput not of type `LLMAssistantMessage`: %s", output)
    return None


def _mean_from_eval_list(evals: list[Evaluation]) -> float:
    return sum([cast("float", e.value) for e in evals]) / len(evals)
