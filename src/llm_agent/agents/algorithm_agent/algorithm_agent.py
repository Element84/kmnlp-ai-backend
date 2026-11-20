import logging
from collections.abc import AsyncGenerator
from string import Template

from langfuse import observe  # type: ignore[reportUnknownVariableType]
from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    AgentStreamEvent,
    ModelRetry,
    PromptedOutput,
    RunContext,
)
from pydantic_ai.models import Model
from pydantic_ai.models.bedrock import BedrockModelSettings
from pydantic_ai.usage import RunUsage, UsageLimits
from rich.pretty import pretty_repr

from llm_agent.agents.algorithm_agent.algorithm.base import BaseFunctionCall
from llm_agent.agents.algorithm_agent.algorithm.type_check import (
    AlgorithmValidationError,
    InferredDataArrayVariable,
    InferredResolutionlessTemporalDataArrayVariable,
    InferredTemporalDataArrayVariable,
    InferredVariable,
    type_check_algorithm,
)
from llm_agent.agents.algorithm_agent.algorithm.utils import TimeCoords
from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmProblemSpec,
    AlgorithmSolutionSpec,
    GeneralDataArrayVariable,
    GeometryVariable,
    SpatioTemporalDataArrayVariable,
    TemporalDataArrayVariable,
)
from llm_agent.context import AgentDeps, ChatContext, ProgressCallback
from llm_agent.utils.langfuse import stream_events_to_final_output
from llm_agent.utils.utils import get_current_date

log = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS = 10_000

DEFAULT_NUM_RETRIES = 3

type AlgorithmAgentStreamEvent[FunctionCallT: BaseFunctionCall] = (
    AgentStreamEvent | AgentRunResultEvent[AlgorithmSolutionSpec[FunctionCallT]]
)

ALGORITHM_AGENT_SYSTEM_PROMPT_TEMPLATE = Template("""\
You are an LLM specializing in designing algorithms to solve problems and \
describing them using the given specifications and constraints.

The problems to solve will be specified using this schema:
<problem_schema>
```json
${problem_schema}
```
</problem_schema>
""")

# Uses `{schema}` (`str.format` syntax, per `PromptedOutput`), not `${...}` (`string.Template`).
PROMPTED_OUTPUT_TEMPLATE = """
Respond with a JSON object for the algorithm solution that's compatible with this schema:
<solution_schema>
```json
{schema}
```
</solution_schema>

Don't include any text or Markdown fencing before or after the JSON object.
"""

CURRENT_DATE_TEMPLATE = Template("""
---
The current date is: ${current_date}.
""")

ALGORITHM_AGENT_PROMPT_TEMPLATE = Template("""\
Design an algorithm to fulfill the request given below.

<problem_spec>
```json
${input_json}
```
</problem_spec>
""")


def _is_thinking_enabled(model: Model) -> bool:
    """Whether extended thinking is turned on in the model's settings."""
    settings = model.settings
    if not settings:
        return False
    return bool(settings.get("thinking")) or bool(
        settings.get("bedrock_additional_model_requests_fields", {}).get("thinking")
    )


class AlgorithmAgent[FunctionCallT: BaseFunctionCall]:
    """Agent that designs algorithms to answer questions using data.

    Wraps a Pydantic AI agent whose single structured output is an
    ``AlgorithmSolutionSpec``. ``type_check_algorithm`` runs as an output
    validator: a failed check raises ``ModelRetry`` so the model can correct its
    solution, up to ``retries`` times. The agent is built per problem because
    the solution schema is parameterized by the problem's ``function_call_type``.
    """

    def __init__(
        self,
        model: Model,
        *,
        problem: AlgorithmProblemSpec,
        function_call_type: type[FunctionCallT],
        num_retries: int = DEFAULT_NUM_RETRIES,
    ) -> None:
        """Constructor.

        Args:
            model: The Pydantic AI model to drive the agent. Build one with
                ``llm_agent.model_factory.make_bedrock_model``.
            problem: Input problem that the agent will solve.
            function_call_type: Function call type. Expected to be a union of the
                available functions for this problem.
            num_retries: Number of retries for the output validator.
        """
        self._problem = problem
        self._solution_type = AlgorithmSolutionSpec[function_call_type]
        self._usage_limits = UsageLimits(tool_calls_limit=num_retries)

        model_settings = BedrockModelSettings(
            bedrock_cache_instructions=True,
            bedrock_cache_tool_definitions=True,
        )
        if model.settings:
            if model.settings.get("temperature") is None:
                model.settings["temperature"] = 0.0
            if model.settings.get("max_tokens") is None:
                model.settings["max_tokens"] = DEFAULT_MAX_OUTPUT_TOKENS

        log.info("Building AlgorithmAgent with model settings: %s", model.settings)
        log.debug(
            "Solution schema: %s",
            pretty_repr(self._solution_type.model_json_schema(), indent_size=2),
        )

        # When thinking is enabled, Bedrock cannot force the structured-output tool call and
        # falls back to native structured output, whose validator rejects the JSON-Schema
        # `oneOf` that Pydantic emits for discriminated unions (e.g. the function-call and
        # plot-spec unions). Prompted output keeps the schema in the prompt and parses/validates
        # the result client-side, sidestepping that limitation.
        if _is_thinking_enabled(model):
            log.info("Thinking enabled; using prompted structured output for the solution.")
            output_type = PromptedOutput(self._solution_type, template=PROMPTED_OUTPUT_TEMPLATE)
        else:
            output_type = self._solution_type

        self._agent = Agent(
            model,
            deps_type=AgentDeps,
            output_type=output_type,
            retries=num_retries,
            instructions=ALGORITHM_AGENT_SYSTEM_PROMPT_TEMPLATE.substitute(
                problem_schema=pretty_repr(AlgorithmProblemSpec.model_json_schema(), indent_size=2),
            ),
            model_settings=model_settings,
        )

        # Registered as a dynamic instruction so it's sorted after the PromptedOutput schema block.
        self._agent.instructions(self._current_date_instructions)

        self._agent.output_validator(self._validate_solution)

    @observe(as_type="agent", name=__name__)
    async def run(
        self, chat_context: ChatContext, *, progress_callback: ProgressCallback | None = None
    ) -> tuple[AlgorithmSolutionSpec[FunctionCallT], RunUsage]:
        """Run the agent and return the validated algorithm solution."""
        deps = AgentDeps(chat_context=chat_context, progress_callback=progress_callback)
        result = await self._agent.run(
            self._format_problem(), deps=deps, usage_limits=self._usage_limits
        )
        log.info("AlgorithmAgent solution: %s", pretty_repr(result.output))
        return result.output, result.usage

    @observe(as_type="agent", name=__name__, transform_to_string=stream_events_to_final_output)
    async def run_stream(
        self, chat_context: ChatContext, *, progress_callback: ProgressCallback | None = None
    ) -> AsyncGenerator[AlgorithmAgentStreamEvent[FunctionCallT]]:
        """Run the agent, streaming each event (solution attempts, retries, final output)."""
        deps = AgentDeps(chat_context=chat_context, progress_callback=progress_callback)
        async with self._agent.run_stream_events(
            self._format_problem(), deps=deps, usage_limits=self._usage_limits
        ) as stream:
            async for event in stream:
                yield event

    def _format_problem(self) -> str:
        return ALGORITHM_AGENT_PROMPT_TEMPLATE.substitute(
            input_json=self._problem.model_dump_json()
        )

    @staticmethod
    def _current_date_instructions() -> str:
        return CURRENT_DATE_TEMPLATE.substitute(current_date=get_current_date())

    @observe(as_type="guardrail", name="AlgorithmAgent._validate_solution")
    def _validate_solution(
        self,
        ctx: RunContext[AgentDeps],  # noqa: ARG002
        solution: AlgorithmSolutionSpec[FunctionCallT],
    ) -> AlgorithmSolutionSpec[FunctionCallT]:
        """Type-check the solution; raise ``ModelRetry`` so the model can fix mistakes."""
        if solution.error_message:
            # No need to type check if the LLM reported an error.
            return solution
        try:
            self._type_check_solution(solution)
        except AlgorithmValidationError as e:
            log.info("Algorithm solution failed type check, requesting retry: %s", e)
            raise ModelRetry(str(e)) from e
        return solution

    @observe(as_type="guardrail", name="AlgorithmAgent._type_check_solution")
    def _type_check_solution(self, solution: AlgorithmSolutionSpec[FunctionCallT]) -> None:
        expected_outputs: dict[str, InferredVariable] = {}
        for v in self._problem.outputs_requested:
            match v.variable_type:
                case "scalar_float" | "latitude_deg_float" | "longitude_deg_float":
                    expected_outputs[v.name] = InferredVariable(type="float", name=v.name)
                case "datetime_string":
                    expected_outputs[v.name] = InferredVariable(type="str", name=v.name)
                case (
                    "raster_plot"
                    | "line_plot"
                    | "bar_plot"
                    | "histogram"
                    | "scatter_plot"
                    | "plot_grid"
                    | "geotiff"
                    | "animation"
                ):
                    expected_outputs[v.name] = InferredVariable(type=v.variable_type, name=v.name)
                case _:
                    msg = f"Unsupported output variable type: {v.variable_type}"
                    raise AlgorithmValidationError(msg)

        inferred_input_vars, valid_time_dims = self._get_inferred_input_vars(self._problem)

        type_check_algorithm(
            solution.algorithm,
            input_vars=inferred_input_vars,
            expected_outputs=expected_outputs,
            valid_time_dims=valid_time_dims,
        )

    def _get_inferred_input_vars(
        self, problem: AlgorithmProblemSpec
    ) -> tuple[dict[str, InferredVariable], set[str]]:
        inferred_vars: dict[str, InferredVariable] = {}
        valid_time_dims = set[str]()
        for input_var in problem.inputs_provided:
            match input_var:
                case GeometryVariable():
                    inferred_vars[input_var.name] = InferredVariable(
                        type="geometry", name=input_var.name
                    )
                case TemporalDataArrayVariable() | SpatioTemporalDataArrayVariable():
                    dims = frozenset([d.field_name for d in input_var.dimensions])
                    valid_time_dims.add(input_var.time_dim)
                    if input_var.temporal_resolution:
                        time_coords = TimeCoords(
                            *input_var.temporal_coverage, freq=input_var.temporal_resolution
                        )
                        inferred_vars[input_var.name] = InferredTemporalDataArrayVariable(
                            name=input_var.name,
                            dims=dims,
                            time_dim=input_var.time_dim,
                            time_coords=time_coords,
                            shape={
                                **dict.fromkeys(dims, "unknown"),
                                input_var.time_dim: len(time_coords),
                            },
                        )
                    else:
                        # Time dim is known but resolution isn't, so exact time sizes are unknown.
                        inferred_vars[input_var.name] = (
                            InferredResolutionlessTemporalDataArrayVariable(
                                name=input_var.name,
                                dims=dims,
                                time_dim=input_var.time_dim,
                                shape=dict.fromkeys(dims, "unknown"),
                            )
                        )
                case GeneralDataArrayVariable():
                    dims = frozenset([d.field_name for d in input_var.dimensions])
                    inferred_vars[input_var.name] = InferredDataArrayVariable(
                        name=input_var.name,
                        dims=dims,
                        shape=dict.fromkeys(dims, "unknown"),
                    )
        return inferred_vars, valid_time_dims
