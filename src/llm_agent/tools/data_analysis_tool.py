import base64
import io
import json
import logging
import os
from typing import Any

import fsspec  # type: ignore[reportMissingTypeStubs]
import xarray as xr
from anyio import from_thread, to_thread
from dask.distributed import Client
from e84_geoai_common.llm.core import (
    Base64ImageContent,
    JSONContent,
    LLMToolResultContent,
    LLMToolUseContent,
)
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rich.pretty import pretty_repr

from llm_agent.agents.algorithm_agent.algorithm import (
    DataArrayFunctionSet,
    GeneralPlottingFunctionSet,
)
from llm_agent.agents.algorithm_agent.algorithm.exec import execute_algorithm
from llm_agent.agents.algorithm_agent.algorithm_agent import AlgorithmAgent
from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmProblemSpec,
    AlgorithmSolutionSpec,
    InputVariable,
    OutputVariable,
    UnsupportedVariableDimsError,
    input_variable_from_model,
)
from llm_agent.agents.scientific_python_agent.utils.agent_code_utils import (
    compute,
)
from llm_agent.dao import knowledge_graph_dao
from llm_agent.tools.base_tool import BaseTool
from llm_agent.utils import DEFAULT_LLM, EnrichedValidationError

log = logging.getLogger(__name__)


SolutionSpecT = AlgorithmSolutionSpec[
    DataArrayFunctionSet | GeneralPlottingFunctionSet
]


class InvalidVariableIRIError(Exception):
    """Indicates a Variable model could not be built given a provided IRI."""

    def __init__(self, variable_iri: str) -> None:
        """Initializer."""
        self.variable_iri = variable_iri


class DataAnalysisToolInput(BaseModel):
    """Inputs to the DataAnalysisTool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    problem_statement: str = Field(
        description="A clear and unambiguous statement of the problem.",
    )
    variable_ids: list[str] = Field(
        description="Variable IDs that should be used in the data analysis.",
    )
    outputs_requested: list[OutputVariable] = Field(
        description="Variables that must be included as outputs of "
        "the data analysis.",
    )


class DataAnalysisToolOutput(BaseModel):
    """DataAnalysisTool outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_outputs: dict[str, str | float]
    image_outputs: list[str] | None = Field(
        description="Variable names for which images are attached in this "
        "response, in the same order as the names here."
    )
    solution_pseudocode: str | None = Field(
        description="Pseudocode for the solution in Python-like syntax."
    )
    error_message: str | None = Field(
        default=None,
        description="Error message in case the tool encounters a problem; "
        "null otherwise.",
    )


def _variable_id_to_algo_input_variable(var_id: str) -> InputVariable:
    """Make a variable compatible with AlgorithmAgent's interface.

    Args:
        var_id (str): The IRI in the knowledge graph of the variable to look up

    Returns:
        InputVariable: The full `InuptVariable` that can be used during
            execution

    Raises:
        InvalidVariableIRI: Raised if the knowledge graph does not have the
            provided IRI, or if it cannot build a valid model type from the
            item with that IRI.
    """
    looked_up_var = knowledge_graph_dao.get_variable_by_iri(var_id)
    dataset = knowledge_graph_dao.get_dataset_by_variable_iri(var_id)

    if looked_up_var is None or dataset is None:
        raise InvalidVariableIRIError(variable_iri=var_id)

    out = input_variable_from_model(looked_up_var, dataset)
    return out


async def _get_variable_data(var_id: str) -> xr.DataArray:
    """Load variable data.

    Args:
        var_id (str): The IRI in the knowledge graph of the variable to
            hydrate data for

    Returns:
        xr.DataArray: A data array with the data for the provided variable IRI

    Raises:
        InvalidVariableIRI: Raised if the knowledge graph does not have the
            provided IRI, or if it cannot build a valid model type from the
            item with that IRI.

    """

    def _get_mapper_shim(source: str) -> fsspec.FSMap:
        return fsspec.get_mapper(  # type: ignore[reportUnknownMemberType]
            "reference://", fo=source, remote_protocol="s3"
        )

    def _open_zarr_shim(mapper: fsspec.FSMap) -> xr.Dataset:
        return xr.open_zarr(mapper, consolidated=False)  # type: ignore[reportUnknownMemberType]

    input_variable = _variable_id_to_algo_input_variable(var_id)
    dataset = knowledge_graph_dao.get_dataset_by_variable_iri(var_id)

    if dataset is None:
        raise InvalidVariableIRIError(variable_iri=var_id)

    mapper = await to_thread.run_sync(_get_mapper_shim, dataset.source)

    log.debug("Opening Zarr with reference file for root: %s", mapper.root)  # type: ignore[reportUnknownMemberType]

    ds = await to_thread.run_sync(_open_zarr_shim, mapper)
    da = ds[input_variable.name]
    da = da.unify_chunks()
    return da


def _fig_to_image_content(fig: Figure) -> Base64ImageContent:
    """Convert matplotlib figure to base 64 encoded string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2)  # type: ignore[reportUnknownMemberType]
    plt.close(fig)
    image_bytes = buf.getvalue()
    image_content = Base64ImageContent(
        media_type="image/png",
        data=base64.b64encode(image_bytes).decode("utf-8"),
    )
    return image_content


def _ensure_valid_output_type(val: Any) -> str | int | float:  # noqa: ANN401
    """Ensure output value is a scalar."""
    if isinstance(val, str | int | float):
        return val
    if isinstance(val, xr.DataArray):
        return compute(val).item()
    raise TypeError


# NOTE: Visible for testing
def algo_outputs_to_tool_result_contents(
    algo_outputs: dict[str, Any],
    pseudocode: str | None,
    error_message: str | None = None,
) -> list[JSONContent | Base64ImageContent]:
    """Prepare algorithm execution outputs to be fed back to the LLM."""
    data_outputs = {
        k: _ensure_valid_output_type(v)
        for k, v in algo_outputs.items()
        if not isinstance(v, Figure)
    }
    plot_outputs = {
        k: v for k, v in algo_outputs.items() if isinstance(v, Figure)
    }
    tool_output = DataAnalysisToolOutput(
        data_outputs=data_outputs,
        image_outputs=(list(plot_outputs.keys()) if plot_outputs else None),
        solution_pseudocode=pseudocode,
        error_message=error_message,
    )
    json_content = JSONContent(data=tool_output.model_dump())
    image_contents = [
        _fig_to_image_content(fig) for fig in plot_outputs.values()
    ]
    return [json_content, *image_contents]


def _prepare_inputs[P: AlgorithmProblemSpec, S: SolutionSpecT](
    agent: AlgorithmAgent[P, S],
    *,
    tool_use_request_id: str,
    tool_input: DataAnalysisToolInput,
) -> P | LLMToolResultContent:
    # prepare inputs to the AlgorithmAgent
    try:
        algo_input_vars = [
            _variable_id_to_algo_input_variable(v_id)
            for v_id in tool_input.variable_ids
        ]
    except InvalidVariableIRIError as e:
        log.exception(
            "Invalid variable id building input variable: %s", e.variable_iri
        )
        return LLMToolResultContent(
            id=tool_use_request_id,
            content=algo_outputs_to_tool_result_contents(
                algo_outputs={},
                pseudocode="",
                error_message="Error encountered building input variable.",
            ),
            status="error",
        )
    except UnsupportedVariableDimsError as e:
        log.exception("Unsupported dims in input variable.")
        return LLMToolResultContent(
            id=tool_use_request_id,
            content=algo_outputs_to_tool_result_contents(
                algo_outputs={},
                pseudocode=None,
                error_message=str(e),
            ),
            status="error",
        )

    if any("list" in o.variable_type for o in tool_input.outputs_requested):
        return LLMToolResultContent(
            id=tool_use_request_id,
            content=algo_outputs_to_tool_result_contents(
                algo_outputs={},
                pseudocode="",
                error_message="list[] types are not yet supported for output "
                "variables. Request each item individually.",
            ),
            status="error",
        )
    return agent.problem_spec(
        statement=tool_input.problem_statement,
        outputs_requested=tool_input.outputs_requested,
        inputs_provided=algo_input_vars,
    )


async def _construct_algorithm[P: AlgorithmProblemSpec, S: SolutionSpecT](
    agent: AlgorithmAgent[P, S], problem: P, *, tool_use_request_id: str
) -> S | LLMToolResultContent:
    try:
        solution = await agent.run(problem)
    except (
        json.JSONDecodeError,
        ValidationError,
        EnrichedValidationError,
    ):
        log.exception("Invalid JSON or  error validating solution")
        return LLMToolResultContent(
            id=tool_use_request_id,
            content=algo_outputs_to_tool_result_contents(
                algo_outputs={},
                pseudocode="",
                error_message="Failed to generate a valid solution. "
                "Consider asking for fewer or simpler outputs.",
            ),
            status="error",
        )
    if solution.error_message is not None:
        return LLMToolResultContent(
            id=tool_use_request_id,
            content=algo_outputs_to_tool_result_contents(
                algo_outputs={},
                pseudocode="",
                error_message=solution.error_message,
            ),
            status="error",
        )
    return solution


async def _execute_algorithm(
    problem: AlgorithmProblemSpec,
    solution: SolutionSpecT,
    *,
    tool_input: DataAnalysisToolInput,
    tool_use_request_id: str,
) -> dict[str, Any] | LLMToolResultContent:
    dask_scheduler_address = os.getenv("DASK_ADDRESS")
    _ = await Client(dask_scheduler_address, asynchronous=True)

    try:
        input_vars = {
            v.name: await _get_variable_data(v_id)
            for v_id, v in zip(
                tool_input.variable_ids,
                problem.inputs_provided,
                strict=True,
            )
        }
    except InvalidVariableIRIError:
        log.exception("Invalid variable id building data.")
        return LLMToolResultContent(
            id=tool_use_request_id,
            content=algo_outputs_to_tool_result_contents(
                algo_outputs={},
                pseudocode=solution.pseudocode,
                error_message="Error encountered building data.",
            ),
            status="error",
        )
    except UnsupportedVariableDimsError as e:
        log.exception("Unsupported dims in input variable.")
        return LLMToolResultContent(
            id=tool_use_request_id,
            content=algo_outputs_to_tool_result_contents(
                algo_outputs={},
                pseudocode=None,
                error_message=str(e),
            ),
            status="error",
        )

    try:
        algo_outputs = await execute_algorithm(
            solution.algorithm, input_vars=input_vars
        )
    except Exception:
        log.exception("Error executing algorithm.")
        return LLMToolResultContent(
            id=tool_use_request_id,
            content=algo_outputs_to_tool_result_contents(
                algo_outputs={},
                pseudocode=solution.pseudocode,
                error_message="Error encountered when executing solution.",
            ),
            status="error",
        )

    return algo_outputs


async def _a_execute_data_analysis_tool(
    tool_use_request: LLMToolUseContent,
) -> LLMToolResultContent:
    """Execute DataAnalysisTool tool."""
    agent = AlgorithmAgent(
        DEFAULT_LLM,
        problem_spec=AlgorithmProblemSpec,
        solution_spec=SolutionSpecT,
    )
    tool_input = DataAnalysisToolInput.model_validate(tool_use_request.input)

    # prepare AlgorithmAgent inputs
    problem_spec_or_error = _prepare_inputs(
        agent, tool_use_request_id=tool_use_request.id, tool_input=tool_input
    )
    if isinstance(problem_spec_or_error, LLMToolResultContent):
        return problem_spec_or_error
    problem = problem_spec_or_error
    log.debug("Problem:\n%s", pretty_repr(problem))

    # run AlgorithmAgent
    solution_or_error = await _construct_algorithm(
        agent, problem, tool_use_request_id=tool_use_request.id
    )
    if isinstance(solution_or_error, LLMToolResultContent):
        return solution_or_error
    solution = solution_or_error
    log.debug("Solution:\n%s", pretty_repr(solution))

    # execute algorithm
    algo_outputs_or_error = await _execute_algorithm(
        problem,
        solution,
        tool_input=tool_input,
        tool_use_request_id=tool_use_request.id,
    )
    if isinstance(algo_outputs_or_error, LLMToolResultContent):
        return algo_outputs_or_error
    algo_outputs = algo_outputs_or_error
    log.debug("Algorithm execution outputs:\n%s", pretty_repr(algo_outputs))

    out = LLMToolResultContent(
        id=tool_use_request.id,
        content=algo_outputs_to_tool_result_contents(
            algo_outputs, pseudocode=solution.pseudocode
        ),
        status="success",
    )
    return out


class DataAnalysisTool(BaseTool):
    """Tool to generate algorithm solution and run it."""

    def __init__(self) -> None:
        """Initializer."""
        super().__init__(
            description=(
                "A tool that can perform complex data analysis to compute "
                "quantitative results from data. It can also create data "
                "visualizations. This tool simply constructs a solution, "
                "executes it, and returns the results. It has no ability to "
                "inspect the outputs it produces, nor does it remember "
                "context from previous calls."
            ),
            input_model=DataAnalysisToolInput,
            output_model=DataAnalysisToolOutput,
            execution_func=self._execute_data_analysis_tool,
        )

    def _execute_data_analysis_tool(
        self,
        tool_use_request: LLMToolUseContent,
    ) -> LLMToolResultContent:
        """Execute DataAnalysisTool tool."""
        return from_thread.run(_a_execute_data_analysis_tool, tool_use_request)


data_analysis_tool = DataAnalysisTool()
