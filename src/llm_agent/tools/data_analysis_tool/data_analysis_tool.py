import base64
import io
import logging
from typing import Any, Self

import xarray as xr
from anyio import from_thread, to_thread
from e84_geoai_common.llm.core import Base64ImageContent, JSONContent, LLMToolUseContent
from e84_geoai_common.llm.models.claude import BedrockClaudeLLM
from e84_geoai_common.util import timed_function
from geojson_pydantic.geometries import Geometry
from langfuse import observe  # type: ignore[reportUnknownVariableType]
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.pretty import pretty_repr

from llm_agent import ALGORITHM_AGENT_DEFAULT_LLM_MODEL_ID
from llm_agent.agents.algorithm_agent.algorithm.base import (
    BaseFunctionCall,
    make_function_call_model,
)
from llm_agent.agents.algorithm_agent.algorithm.exec import (
    AlgorithmExecutionError,
    execute_algorithm,
)
from llm_agent.agents.algorithm_agent.algorithm_agent import AlgorithmAgent
from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmProblemSpec,
    AlgorithmSolutionSpec,
    InputVariable,
    OutputVariable,
    input_variable_from_model,
    input_vars_to_function_set,
)
from llm_agent.agents.scientific_python_agent.utils.agent_code_utils import (
    DaskComputeError,
    DataSizeTooLargeError,
    compute,
)
from llm_agent.agents.tool_only_agent import MaxAllowedToolCallsExceededError
from llm_agent.context import ToolCallContext
from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from llm_agent.model.dataset import DatasetSource
from llm_agent.tools.data_analysis_tool.utils import normalized_data_array
from llm_agent.tools.llm_tool_input import LLMToolInput
from llm_agent.tools.llm_tool_typed import ErrorResponse, LLMToolTyped, ToolOutput
from llm_agent.transformers.geotiff_transformers import GeoTIFFRef
from llm_agent.utils.utils import EnrichedLLMToolResultContent, PrivateAttr

log = logging.getLogger(__name__)


TOOL_DESCRIPTION = """\
A tool that can perform some basic data analysis to compute results from data. \
It works by constructing an algorithm (via an LLM) by composing various pre-defined functions and \
then executing it. These functions are currently limited to operations such as \
geocoding, data filtering, aggregation, computing basic statistics, and visualization. If it is \
unable to solve a problem using these functions, it returns an error message stating so.

This tool simply constructs a solution, executes it, and returns the results. \
It has no ability to inspect the outputs it produces, nor does it remember context from \
previous calls.

It works best when the problem is well-defined and tightly scoped and there are few requested \
outputs. If the user's question is too broad or vague, it should not be passed directly to this \
tool. Instead, consider asking the user for clarification.

Calls to this tool are expensive, so avoid requesting extra outputs that the user has not \
explicitly asked for.\
"""


class InvalidVariableIDError(Exception):
    """Indicates a Variable model could not be built given a provided ID."""

    def __init__(self, variable_id: str) -> None:
        """Initializer."""
        self.variable_id = variable_id


class DataAnalysisToolInput(LLMToolInput):
    """Inputs to the DataAnalysisTool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    problem_statement: str = Field(
        description=(
            "A clear and unambiguous statement of the problem. This can include place names, "
            "which will be geocoded into precise polygons for the analysis."
        ),
    )
    user_provided_geometry: Geometry | None = Field(
        description=(
            "A geometry provided by the user that indicates the area to which the analysis should "
            "be limited."
        ),
        default=None,
    )
    variable_ids: list[str] = Field(
        description="Variable IDs that should be used in the data analysis.",
        min_length=1,
    )
    outputs_requested: list[OutputVariable] = Field(
        description="Variables that must be included as outputs of the data analysis.",
        min_length=1,
    )


class DataAnalysisToolOutput(BaseModel):
    """DataAnalysisTool outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_outputs: dict[str, str | int | float | GeoTIFFRef] | None = Field(
        description="Mapping of variable names to their computed values. Any GeoTIFFRefs "
        "will be rendered as raster layers on a map in the UI."
    )
    image_outputs: list[str] | None = Field(
        description="Variable names for which images are attached in this "
        "response, in the same order as the names here."
    )
    solution_pseudocode: str | None = Field(
        description="Pseudocode for the solution in Python-like syntax."
    )
    _image_data: list[Base64ImageContent] = PrivateAttr(default_factory=list[Base64ImageContent])

    def __init__(
        self,
        *,
        data_outputs: dict[str, str | int | float | GeoTIFFRef] | None,
        image_outputs: list[str] | None = None,
        solution_pseudocode: str | None = None,
        image_data: list[Base64ImageContent] | None = None,
    ) -> None:
        """Constructor."""
        super().__init__(
            data_outputs=data_outputs,
            image_outputs=image_outputs,
            solution_pseudocode=solution_pseudocode,
        )
        self._image_data = image_data or []
        self._validate_image_info_consistent_with_data()

    @property
    def image_data(self) -> list[Base64ImageContent]:
        """Access image data."""
        return self._image_data

    @model_validator(mode="after")
    def _validate_nonempty(self) -> Self:
        if not (self.data_outputs or self.image_outputs):
            msg = "Must provide at least one of data_outputs and image_outputs."
            raise ValueError(msg)
        return self

    def _validate_image_info_consistent_with_data(self) -> Self:
        num_image_outputs = len(self.image_outputs) if self.image_outputs else 0
        num_images = len(self._image_data) if self._image_data else 0
        if num_image_outputs != num_images:
            msg = f"len(image_outputs) != len(images): {num_image_outputs} != {num_images}"
            raise ValueError(msg)
        return self


class DataAnalysisToolProblemFormulatedUpdate(BaseModel):
    """Intermediate result indicating a problem has been formulated."""

    problem: AlgorithmProblemSpec


class DataAnalysisToolSolutionAttemptUpdate(BaseModel):
    """Update indicating a solution attempt is in progress."""

    attempt_num: int


class DataAnalysisToolSolutionGeneratedUpdate[S: AlgorithmSolutionSpec[BaseFunctionCall]](
    BaseModel
):
    """Intermediate result indicating a solution has been generated."""

    solution: S


def _variable_id_to_algo_input_variable(
    knowledge_graph_dao: KnowledgeGraphDAO, var_id: str
) -> InputVariable:
    """Make a variable compatible with AlgorithmAgent's interface.

    Args:
        knowledge_graph_dao (KnowledgeGraphDAO): The knowledge graph DAO to use for lookups
        var_id (str): The ID in the knowledge graph of the variable to look up

    Returns:
        InputVariable: The full `InputVariable` that can be used during
            execution

    Raises:
        InvalidVariableIDError: Raised if the knowledge graph does not have the
            provided ID, or if it cannot build a valid model type from the
            item with that ID.
    """
    looked_up_var = knowledge_graph_dao.get_variable_by_id(var_id)
    dataset = knowledge_graph_dao.get_dataset_by_variable_id(var_id)

    if looked_up_var is None or dataset is None:
        raise InvalidVariableIDError(variable_id=var_id)

    out = input_variable_from_model(looked_up_var, dataset)
    return out


@observe(as_type="chain")
async def _get_variable_data(knowledge_graph_dao: KnowledgeGraphDAO, var_id: str) -> xr.DataArray:
    """Load variable data.

    Args:
        knowledge_graph_dao (KnowledgeGraphDAO): The knowledge graph DAO to use for lookups
        var_id (str): The ID in the knowledge graph of the variable to
            hydrate data for

    Returns:
        xr.DataArray: A data array with the data for the provided variable ID

    Raises:
        InvalidVariableIDError: Raised if the knowledge graph does not have the
            provided ID, or if it cannot build a valid model type from the
            item with that ID.

    """

    def _open_dataset_shim(source: DatasetSource) -> xr.Dataset:
        match source.format:
            case "zarr":
                return xr.open_zarr(source.uri, chunks="auto")  # pyright: ignore[reportUnknownMemberType]
            case "kerchunk-parquet":
                return xr.open_dataset(  # pyright: ignore[reportUnknownMemberType]
                    source.uri,
                    engine="kerchunk",
                    storage_options={
                        "remote_protocol": "s3",
                        "target_protocol": "s3",
                    },
                    chunks="auto",
                )

    input_variable = _variable_id_to_algo_input_variable(knowledge_graph_dao, var_id)
    dataset = knowledge_graph_dao.get_dataset_by_variable_id(var_id)

    if dataset is None:
        raise InvalidVariableIDError(variable_id=var_id)

    uri = dataset.source
    log.info("Opening Zarr with reference file: %s", uri)

    ds = await to_thread.run_sync(_open_dataset_shim, uri)
    da = ds[input_variable.name]
    da = normalized_data_array(da)

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


def _ensure_valid_output_type(val: Any) -> str | int | float | GeoTIFFRef:  # noqa: ANN401
    """Ensure output value is a scalar."""
    if isinstance(val, str | int | float | GeoTIFFRef):
        return val
    if isinstance(val, xr.DataArray):
        return compute(val).item()
    raise TypeError


def _algo_outputs_to_tool_output(
    *, algo_outputs: dict[str, Any] | None = None, pseudocode: str | None = None
) -> DataAnalysisToolOutput:
    """Prepare algorithm execution outputs to be fed back to the LLM."""
    algo_outputs = algo_outputs or {}
    data_outputs = {
        k: _ensure_valid_output_type(v)
        for k, v in algo_outputs.items()
        if not isinstance(v, Figure)
    }
    plot_outputs = {k: v for k, v in algo_outputs.items() if isinstance(v, Figure)}
    image_contents = [_fig_to_image_content(fig) for fig in plot_outputs.values()]

    return DataAnalysisToolOutput(
        data_outputs=data_outputs,
        image_outputs=(list(plot_outputs.keys()) if plot_outputs else None),
        solution_pseudocode=pseudocode,
        image_data=image_contents,
    )


def _prepare_problem[P: AlgorithmProblemSpec](
    knowledge_graph_dao: KnowledgeGraphDAO, problem_spec: type[P], tool_input: DataAnalysisToolInput
) -> P | ErrorResponse:
    # prepare inputs to the AlgorithmAgent
    try:
        algo_input_vars = [
            _variable_id_to_algo_input_variable(knowledge_graph_dao, v_id)
            for v_id in tool_input.variable_ids
        ]
    except InvalidVariableIDError as e:
        log.exception("Invalid variable ID: %s", e.variable_id)
        return ErrorResponse(
            message=f"Could not find data for variable: {e.variable_id}",
        )

    if any("list" in o.variable_type for o in tool_input.outputs_requested):
        return ErrorResponse(
            message="list[] types are not yet supported for output "
            "variables. Request each item individually.",
        )
    return problem_spec(
        statement=tool_input.problem_statement,
        outputs_requested=tool_input.outputs_requested,
        inputs_provided=algo_input_vars,
        user_provided_geometry=tool_input.user_provided_geometry,
    )


async def _construct_solution[F: BaseFunctionCall](
    context: ToolCallContext, agent: AlgorithmAgent[F], problem: AlgorithmProblemSpec
) -> AlgorithmSolutionSpec[F] | ErrorResponse:
    try:
        attempts = 0
        context.update_progress(DataAnalysisToolSolutionAttemptUpdate(attempt_num=attempts + 1))
        async for output in agent.run_typed(context.global_context, inp=problem):
            if isinstance(output, ErrorResponse):
                attempts += 1
                context.update_progress(
                    DataAnalysisToolSolutionAttemptUpdate(attempt_num=attempts + 1)
                )
            else:
                solution = output
                break
        else:
            return ErrorResponse(
                message="Failed to generate a valid solution. "
                "Consider asking for fewer or simpler outputs.",
            )
    except MaxAllowedToolCallsExceededError:
        return ErrorResponse(
            message="Failed to generate a valid solution. "
            "Consider asking for fewer or simpler outputs.",
        )

    if solution.error_message is not None:
        return ErrorResponse(message=solution.error_message)

    return solution


async def _execute_algorithm(
    context: ToolCallContext,
    input_vars: dict[str, xr.DataArray],
    solution: AlgorithmSolutionSpec[BaseFunctionCall],
) -> dict[str, Any] | ErrorResponse:
    try:
        algo_outputs = await execute_algorithm(context, solution.algorithm, input_vars=input_vars)
    except (DataSizeTooLargeError, DaskComputeError) as e:
        log.exception("Error executing algorithm.")
        return ErrorResponse(message=f"Error encountered while executing solution: {e!r}")
    except AlgorithmExecutionError as e:
        log.exception("Algorithm execution error.")
        return ErrorResponse(
            message=f"Error encountered while executing solution: {e!r}. "
            "This might or might not be fixable. If retrying, consider including this error info "
            "in the problem statement of the next call to help the LLM avoid this error in "
            "the future.",
        )

    return algo_outputs


async def _extract_arguments_from_input(
    context: ToolCallContext, problem: AlgorithmProblemSpec, variable_ids: list[str]
) -> dict[str, xr.DataArray] | ErrorResponse:
    try:
        input_vars = {
            v.name: await _get_variable_data(context.global_context.knowledge_graph_dao, v_id)
            for v_id, v in zip(variable_ids, problem.inputs_provided, strict=True)
        }
    except InvalidVariableIDError as e:
        log.exception("Invalid variable ID.")
        return ErrorResponse(message=f"Could not find data for variable: {e.variable_id}")

    return input_vars


async def _a_execute_data_analysis_tool(
    context: ToolCallContext, inp: DataAnalysisToolInput
) -> DataAnalysisToolOutput | ErrorResponse:
    """Execute DataAnalysisTool tool."""
    # prepare AlgorithmAgent inputs
    problem_spec_or_error = _prepare_problem(
        context.global_context.knowledge_graph_dao,
        problem_spec=AlgorithmProblemSpec,
        tool_input=inp,
    )
    if isinstance(problem_spec_or_error, ErrorResponse):
        return problem_spec_or_error
    problem = problem_spec_or_error
    log.info("Problem:\n%s", pretty_repr(problem))

    context.update_progress(DataAnalysisToolProblemFormulatedUpdate(problem=problem))

    # configure agent
    function_set = input_vars_to_function_set(problem.inputs_provided)
    function_call_model = make_function_call_model(function_set.functions)

    agent = AlgorithmAgent(
        llm=BedrockClaudeLLM(model_id=ALGORITHM_AGENT_DEFAULT_LLM_MODEL_ID),
        problem=problem,
        function_call_type=function_call_model,
    )

    # run AlgorithmAgent
    solution_or_error = await _construct_solution(context, agent, problem)
    if isinstance(solution_or_error, ErrorResponse):
        return solution_or_error
    solution = solution_or_error
    log.info("Solution:\n%s", pretty_repr(solution))

    context.update_progress(DataAnalysisToolSolutionGeneratedUpdate(solution=solution))

    # extract arguments
    input_vars = await _extract_arguments_from_input(context, problem, inp.variable_ids)
    if isinstance(input_vars, ErrorResponse):
        return input_vars

    # execute algorithm
    algo_outputs_or_error = await _execute_algorithm(context, input_vars, solution)
    if isinstance(algo_outputs_or_error, ErrorResponse):
        return algo_outputs_or_error
    algo_outputs = algo_outputs_or_error
    log.info("Algorithm execution outputs:\n%s", pretty_repr(algo_outputs))

    return _algo_outputs_to_tool_output(algo_outputs=algo_outputs, pseudocode=solution.pseudocode)


class DataAnalysisTool(
    LLMToolTyped[DataAnalysisToolInput, DataAnalysisToolOutput, DataAnalysisToolOutput]
):
    """Tool to generate algorithm solution and run it."""

    def __init__(self) -> None:
        """Initializer."""
        super().__init__(
            name="DataAnalysisTool",
            description=TOOL_DESCRIPTION,
            input_model=DataAnalysisToolInput,
            user_output_model=DataAnalysisToolOutput,
            llm_output_model=DataAnalysisToolOutput,
        )

    @observe(as_type="tool", name=__name__)
    @timed_function()
    def _execute_typed(
        self, context: ToolCallContext, tool_input: DataAnalysisToolInput
    ) -> ToolOutput[DataAnalysisToolOutput, DataAnalysisToolOutput] | ErrorResponse:
        """Execute DataAnalysisTool tool."""
        out = from_thread.run(_a_execute_data_analysis_tool, context, tool_input)
        if isinstance(out, ErrorResponse):
            return out
        return ToolOutput(user=out, llm=out)

    def _wrap_output(
        self,
        tool_use_request: LLMToolUseContent,
        output: ToolOutput[DataAnalysisToolOutput, DataAnalysisToolOutput] | ErrorResponse,
    ) -> (
        EnrichedLLMToolResultContent[DataAnalysisToolOutput]
        | EnrichedLLMToolResultContent[ErrorResponse]
    ):
        if isinstance(output, ErrorResponse):
            error = output
            return EnrichedLLMToolResultContent[ErrorResponse](
                id=tool_use_request.id,
                status="error",
                content=[JSONContent(data=error.model_dump(mode="json"))],
                data=error,
            )
        out = output["user"]
        return EnrichedLLMToolResultContent[DataAnalysisToolOutput](
            id=tool_use_request.id,
            status="success",
            content=[JSONContent(data=out.model_dump(mode="json")), *out.image_data],
            data=out,
        )
