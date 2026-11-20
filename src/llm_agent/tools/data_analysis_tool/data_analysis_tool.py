import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import xarray as xr
from anyio import to_thread
from icechunk import Credentials, Repository, S3Credentials, s3_storage
from langfuse import observe  # type: ignore[reportUnknownVariableType]
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import AgentRunResultEvent, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.models import Model
from rich.pretty import pretty_repr

from llm_agent.agents.algorithm_agent.algorithm.base import (
    BaseFunctionCall,
    make_function_call_model,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.base import (
    AlgorithmExecutionError,
    ExecutionConfig,
    execute_algorithm,
)
from llm_agent.agents.algorithm_agent.algorithm_agent import (
    AlgorithmAgent,
    AlgorithmAgentStreamEvent,
)
from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmProblemSpec,
    AlgorithmSolutionSpec,
    DataArrayVariable,
    GeometryVariable,
    InputVariable,
    OutputVariable,
    input_variable_from_model,
    input_vars_to_function_set,
)
from llm_agent.context import AgentDeps, ToolCallContext
from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from llm_agent.model.dataset import DatasetSource
from llm_agent.tools.agent_tool import AgentTool, ErrorResponse, ToolInput
from llm_agent.tools.data_analysis_tool.types import AnimationRef, GeometryRef, GeoTIFFRef, ImageRef
from llm_agent.tools.data_analysis_tool.utils import algorithm_to_pseudocode, normalized_data_array
from llm_agent.utils.dask import DaskComputeError, DataSizeTooLargeError

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

log = logging.getLogger(__name__)


TOOL_DESCRIPTION = """\
A tool that can perform some basic data analysis to compute results from data. \
It works by constructing an algorithm (via an LLM) by composing various pre-defined functions and \
then executing it. If it is unable to solve a problem using these functions, it returns an error \
message stating so.

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


class DataAnalysisToolInput(ToolInput):
    """Inputs to the DataAnalysisTool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    problem_statement: str = Field(
        description=("A clear and unambiguous statement of the problem."),
    )
    input_data: list[str | GeometryRef] = Field(
        description="List of input data items, which can be either (1) variable IDs (strings) from "
        "the results of the data search tool or (2) geometry references returned by the geocoding "
        "tool or provided by the user. References passed here should have their original keys but "
        "the name and description fields can be rewritten for clarity; the names should be brief "
        "and in snake-case.",
        min_length=1,
    )
    outputs_requested: list[OutputVariable] = Field(
        description="Variables that must be included as outputs of the data analysis.",
        min_length=1,
    )


class DataAnalysisToolOutput(BaseModel):
    """DataAnalysisTool outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outputs: Mapping[str, str | int | float | ImageRef | GeoTIFFRef | AnimationRef] | None = Field(
        description="Computed values of the requested outputs."
    )
    solution_pseudocode: str | None = Field(
        description="Pseudocode for the solution in Python-like syntax."
    )


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
    pseudocode: str = Field(
        description="Pseudocode generated deterministically from the algorithm as opposed to "
        "the pseudocode field inside the solution which is written by the LLM."
    )


@observe(as_type="chain")
def _variable_id_to_algo_input_variable(
    knowledge_graph_dao: KnowledgeGraphDAO, var_id: str
) -> DataArrayVariable:
    """Make a variable compatible with AlgorithmAgent's interface.

    Args:
        knowledge_graph_dao (KnowledgeGraphDAO): The knowledge graph DAO to use for lookups
        var_id (str): The ID in the knowledge graph of the variable to look up

    Returns:
        DataArrayVariable: The full `DataArrayVariable` that can be used during execution.

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


def open_dataset_from_source(source: DatasetSource) -> xr.Dataset:
    """Open an xarray Dataset from a DatasetSource.

    Args:
        source: The dataset source containing format and URI information

    Returns:
        xr.Dataset: The opened dataset

    Raises:
        ValueError: If the S3 URI format is invalid for icechunk datasets
    """
    match source.format:
        case "zarr":
            return xr.open_zarr(source.uri, chunks="auto")  # pyright: ignore[reportUnknownMemberType]
        case "icechunk-zarr":
            # Parse S3 URI: s3://bucket/prefix
            s3_uri = source.uri
            if not s3_uri.startswith("s3://"):
                raise ValueError(f"Invalid S3 URI for icechunk: {s3_uri}")
            s3_path = s3_uri[5:]  # Remove "s3://"
            bucket, _, prefix = s3_path.partition("/")

            # Create storage, open repository, and get readonly session
            storage = s3_storage(bucket=bucket, prefix=prefix)
            config = Repository.fetch_config(storage)
            if config is None:
                raise ValueError("Cannot fetch Icechunk repo.")
            vccs = config.virtual_chunk_containers

            repo = Repository.open(
                storage=storage,
                authorize_virtual_chunk_access={
                    vcc.url_prefix: Credentials.S3(S3Credentials.Anonymous())
                    for vcc in vccs.values()
                }
                if vccs
                else None,
            )
            session = repo.readonly_session(branch="main")

            return xr.open_dataset(  # pyright: ignore[reportUnknownMemberType]
                session.store,  # pyright: ignore[reportArgumentType]
                engine="zarr",
                chunks="auto",
            )
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
    input_variable = _variable_id_to_algo_input_variable(knowledge_graph_dao, var_id)
    dataset = knowledge_graph_dao.get_dataset_by_variable_id(var_id)

    if dataset is None:
        raise InvalidVariableIDError(variable_id=var_id)

    source = dataset.source
    log.info("Opening dataset from source: %s (format: %s)", source.uri, source.format)

    ds = await to_thread.run_sync(open_dataset_from_source, source)
    da = ds[input_variable.name]
    da = normalized_data_array(da)

    return da


@observe(as_type="chain")
async def _ensure_valid_output_type(
    val: Any,  # noqa: ANN401
) -> str | int | float | ImageRef | GeoTIFFRef | AnimationRef:
    """Ensure output value is a scalar."""
    if not isinstance(val, str | int | float | ImageRef | GeoTIFFRef | AnimationRef):
        raise TypeError
    return val


@observe(as_type="chain")
async def _algo_outputs_to_tool_output(
    *, algo_outputs: dict[str, Any] | None = None, pseudocode: str | None = None
) -> DataAnalysisToolOutput:
    """Prepare algorithm execution outputs to be fed back to the LLM."""
    algo_outputs = algo_outputs or {}
    outputs = {k: await _ensure_valid_output_type(v) for k, v in algo_outputs.items()}
    return DataAnalysisToolOutput(outputs=outputs, solution_pseudocode=pseudocode)


@observe(as_type="chain")
def _prepare_problem[P: AlgorithmProblemSpec](
    knowledge_graph_dao: KnowledgeGraphDAO, problem_spec: type[P], tool_input: DataAnalysisToolInput
) -> tuple[P, dict[str, InputVariable]] | ErrorResponse:
    """Prepare inputs to the AlgorithmAgent."""
    algo_input_vars: dict[str, InputVariable] = {}
    try:
        for inp in tool_input.input_data:
            match inp:
                case str():
                    algo_input_vars[inp] = _variable_id_to_algo_input_variable(
                        knowledge_graph_dao, inp
                    )
                case GeometryRef():
                    algo_input_vars[inp.name] = GeometryVariable(
                        name=inp.name,
                        variable_type="geometry",
                        geometry_type="polygon",
                        description=inp.description,
                    )
    except InvalidVariableIDError as e:
        log.exception("Invalid variable ID: %s", e.variable_id)
        return ErrorResponse(
            message=f"Could not find data for variable: {e.variable_id}",
        )
    problem_spec_instance = problem_spec(
        statement=tool_input.problem_statement,
        outputs_requested=tool_input.outputs_requested,
        inputs_provided=list(algo_input_vars.values()),
    )
    return problem_spec_instance, algo_input_vars


@observe(as_type="chain")
async def _construct_solution[F: BaseFunctionCall](
    ctx: RunContext[AgentDeps], agent: AlgorithmAgent[F]
) -> AlgorithmSolutionSpec[F] | ErrorResponse:
    ctx.deps.update_progress(ctx, DataAnalysisToolSolutionAttemptUpdate(attempt_num=1))
    try:
        solution = await _run_algorithm_agent(agent, ctx)
    except (UnexpectedModelBehavior, UsageLimitExceeded):
        log.exception("AlgorithmAgent failed to produce a valid solution.")
        return ErrorResponse(
            message="Failed to generate a valid solution. "
            "Consider asking for fewer or simpler outputs.",
        )

    if solution.error_message:
        return ErrorResponse(message=solution.error_message)

    return solution


class AlgorithmAgentThinkingDeltaUpdate(BaseModel):
    """Content delta for the Algorithm agent's thinking."""

    content_delta: str


class AlgorithmAgentThinkingCompleteUpdate(BaseModel):
    """Algorithm agent's full thinking content."""

    content: str


@observe(as_type="chain")
async def _run_algorithm_agent[F: BaseFunctionCall](
    agent: AlgorithmAgent[F], ctx: RunContext[AgentDeps]
) -> AlgorithmSolutionSpec[F]:
    agent_stream = agent.run_stream(
        ctx.deps.chat_context,
        progress_callback=ctx.deps.make_progress_callback_for_nested_tool(ctx),
    )
    solution: AlgorithmSolutionSpec[F] | None = None
    # Iterate to exhaustion rather than returning early: Langfuse's `@observe` on an async
    # generator only ends (and therefore exports) its span when the generator is exhausted.
    async for event in agent_stream:
        solution = _handle_algorithm_agent_event(ctx, event) or solution

    if solution is None:
        raise UnexpectedModelBehavior("Algorithm agent run terminated without producing solution.")

    return solution


def _handle_algorithm_agent_event[F: BaseFunctionCall](
    ctx: RunContext[AgentDeps], event: AlgorithmAgentStreamEvent[F]
) -> AlgorithmSolutionSpec[F] | None:
    """Report progress for a stream event, returning the solution if the run finished."""
    match event:
        case PartDeltaEvent(delta=ThinkingPartDelta(content_delta=str(content_delta))):
            log.debug("\n[Thinking Part Delta] %s", content_delta)
            ctx.deps.update_progress(
                ctx, AlgorithmAgentThinkingDeltaUpdate(content_delta=content_delta)
            )
        case PartStartEvent(part=ThinkingPart(content=content)):
            log.debug("\n[Thinking Part Delta] %s", content)
            ctx.deps.update_progress(ctx, AlgorithmAgentThinkingDeltaUpdate(content_delta=content))
        case PartEndEvent(part=ThinkingPart(content=content)):
            log.info("\n[Thinking] %s", content)
            ctx.deps.update_progress(ctx, AlgorithmAgentThinkingCompleteUpdate(content=content))
        case AgentRunResultEvent(result=final_result):
            log.info("\n[Run Finished] The agent run has completely finished.")
            return final_result.output
        case _:
            pass
    return None


@observe(as_type="chain")
async def _execute_algorithm(
    tool_call_context: ToolCallContext,
    input_vars: "dict[str, xr.DataArray | BaseGeometry]",
    solution: AlgorithmSolutionSpec[BaseFunctionCall],
    exec_config: ExecutionConfig,
) -> dict[str, Any] | ErrorResponse:
    try:
        algo_outputs = await execute_algorithm(
            tool_call_context, solution.algorithm, input_vars=input_vars, exec_config=exec_config
        )
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


@observe(as_type="chain")
async def _algo_input_vars_to_exec_input_vars(
    ctx: RunContext[AgentDeps],
    algo_input_vars: dict[str, InputVariable],
    tool_input: DataAnalysisToolInput,
) -> "dict[str, xr.DataArray | BaseGeometry] | ErrorResponse":
    try:
        input_vars: dict[str, xr.DataArray | BaseGeometry] = {}
        geom_refs = {
            d.name.lower().replace(" ", "_").replace("-", "_"): d
            for d in tool_input.input_data
            if isinstance(d, GeometryRef)
        }
        for k, v in algo_input_vars.items():
            match v:
                case GeometryVariable():
                    geom_ref = geom_refs[v.name]
                    input_vars[v.name] = ctx.deps.chat_context.retrieve_data(geom_ref)
                case _:
                    input_vars[v.name] = await _get_variable_data(
                        ctx.deps.chat_context.global_context.knowledge_graph_dao, var_id=k
                    )

    except InvalidVariableIDError as e:
        log.exception("Invalid variable ID.")
        return ErrorResponse(message=f"Could not find data for variable: {e.variable_id}")

    return input_vars


@observe(as_type="chain")
async def _a_execute_data_analysis_tool(
    ctx: RunContext[AgentDeps],
    inp: DataAnalysisToolInput,
    model: Model,
    exec_config: ExecutionConfig,
) -> DataAnalysisToolOutput | ErrorResponse:
    """Execute DataAnalysisTool tool."""
    knowledge_graph_dao = ctx.deps.chat_context.global_context.knowledge_graph_dao

    # prepare AlgorithmAgent inputs
    problem_spec_or_error = _prepare_problem(
        knowledge_graph_dao,
        problem_spec=AlgorithmProblemSpec,
        tool_input=inp,
    )
    if isinstance(problem_spec_or_error, ErrorResponse):
        return problem_spec_or_error
    problem, algo_input_vars = problem_spec_or_error
    log.info("Problem:\n%s", pretty_repr(problem))

    ctx.deps.update_progress(ctx, DataAnalysisToolProblemFormulatedUpdate(problem=problem))

    # configure agent
    function_set = input_vars_to_function_set(problem.inputs_provided)
    function_call_model = make_function_call_model(function_set.functions)

    agent = AlgorithmAgent(
        model,
        problem=problem,
        function_call_type=function_call_model,
    )

    # run AlgorithmAgent
    solution_or_error = await _construct_solution(ctx, agent)
    if isinstance(solution_or_error, ErrorResponse):
        return solution_or_error
    solution = solution_or_error
    log.info("Solution:\n%s", pretty_repr(solution))

    ctx.deps.update_progress(
        ctx,
        DataAnalysisToolSolutionGeneratedUpdate(
            solution=solution, pseudocode=algorithm_to_pseudocode(solution.algorithm)
        ),
    )

    # extract arguments
    exec_input_vars = await _algo_input_vars_to_exec_input_vars(ctx, algo_input_vars, inp)
    if isinstance(exec_input_vars, ErrorResponse):
        return exec_input_vars

    # execute algorithm
    tool_call_context = ToolCallContext(
        tool_call_id=ctx.tool_call_id or "",
        chat_context=ctx.deps.chat_context,
        progress_callback=ctx.deps.progress_callback,
    )
    algo_outputs_or_error = await _execute_algorithm(
        tool_call_context, exec_input_vars, solution, exec_config=exec_config
    )
    if isinstance(algo_outputs_or_error, ErrorResponse):
        return algo_outputs_or_error
    algo_outputs = algo_outputs_or_error
    log.info("Algorithm execution outputs:\n%s", pretty_repr(algo_outputs))

    return await _algo_outputs_to_tool_output(
        algo_outputs=algo_outputs, pseudocode=algorithm_to_pseudocode(solution.algorithm)
    )


class DataAnalysisTool(AgentTool[DataAnalysisToolInput, DataAnalysisToolOutput | ErrorResponse]):
    """Tool to generate an algorithm solution and run it."""

    name = "DataAnalysisTool"
    description = TOOL_DESCRIPTION
    input_model = DataAnalysisToolInput
    output_type = DataAnalysisToolOutput | ErrorResponse

    def __init__(self, model: Model, exec_config: ExecutionConfig | None = None) -> None:
        """Initializer.

        Args:
            model: The Pydantic AI model used to drive the nested ``AlgorithmAgent``.
            exec_config: Optional execution configuration for running the generated algorithm.
        """
        self._model = model
        self._exec_config = exec_config or ExecutionConfig()

    async def __call__(
        self, ctx: RunContext[AgentDeps], tool_input: DataAnalysisToolInput, /
    ) -> DataAnalysisToolOutput | ErrorResponse:
        """Generate an algorithm solution for the problem, execute it, and return the results."""
        return await _a_execute_data_analysis_tool(ctx, tool_input, self._model, self._exec_config)
