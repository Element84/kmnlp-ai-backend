import logging
from collections.abc import AsyncGenerator, Callable
from string import Template
from typing import Any, override

from e84_geoai_common.llm.core import LLM, LLMInferenceConfig
from langfuse import observe  # pyright: ignore[reportUnknownVariableType]
from rich.pretty import pretty_repr

from llm_agent.agents.algorithm_agent.algorithm.base import BaseFunctionCall
from llm_agent.agents.algorithm_agent.algorithm.type_check import (
    InferredDataArrayVariable,
    InferredTemporalDataArrayVariable,
    InferredVariable,
    type_check_algorithm,
)
from llm_agent.agents.algorithm_agent.algorithm.utils import TimeCoords
from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmProblemSpec,
    AlgorithmSolutionSpec,
    GeneralDataArrayVariable,
    SpatioTemporalDataArrayVariable,
    TemporalDataArrayVariable,
)
from llm_agent.agents.typed_tool_only_agent import TypedToolOnlyAgent
from llm_agent.context import LLMAgentContext, ToolProgressUpdate
from llm_agent.tools.llm_tool_typed import ErrorResponse
from llm_agent.tools.response_shaping_tools.response_shaping_tool import (
    ResponseShapingTool,
    ResponseValidationError,
)
from llm_agent.utils.utils import get_current_date

log = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS = 10_000

MAX_ALLOWED_TOOL_CALLS_PER_TURN = 8

ALGORITHM_AGENT_SYSTEM_PROMPT_TEMPLATE = Template("""\
You are an LLM specializing in designing algorithms to solve problems and \
describing them using the given specifications and constraints.

The problems to solve will be specified using this schema:
<problem_schema>
```json
${problem_schema}
```
</problem_schema>

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


class AlgorithmAgent[FunctionCallT: BaseFunctionCall](
    TypedToolOnlyAgent[AlgorithmProblemSpec, AlgorithmSolutionSpec[FunctionCallT]]
):
    """Agent that designs algorithms to answer questions using data."""

    problem: AlgorithmProblemSpec
    system_prompt: str
    solution_spec: type[AlgorithmSolutionSpec[FunctionCallT]]

    def __init__(
        self,
        llm: "LLM",
        *,
        problem: AlgorithmProblemSpec,
        function_call_type: type[FunctionCallT],
        inference_cfg: LLMInferenceConfig | None = None,
        max_allowed_tool_calls_per_turn: int = MAX_ALLOWED_TOOL_CALLS_PER_TURN,
    ) -> None:
        """Constructor.

        Args:
            llm (LLM): An LLM instance.
            problem (AlgorithmProblemSpec): Input problem that the agent will solve.
            function_call_type (type[FunctionCallT]): Function call type. Expected to be
                a union of available functions.
            inference_cfg (LLMInferenceConfig | None): Inference config.
                Defaults to None.
            max_allowed_tool_calls_per_turn (int): Max allowed tool calls per turn.
        """
        self.problem = problem
        self.system_prompt = ALGORITHM_AGENT_SYSTEM_PROMPT_TEMPLATE.substitute(
            problem_schema=pretty_repr(AlgorithmProblemSpec.model_json_schema(), indent_size=2),
            solution_schema=pretty_repr(
                AlgorithmSolutionSpec[function_call_type].model_json_schema(), indent_size=2
            ),
            current_date=get_current_date(),
        )
        self.solution_spec = AlgorithmSolutionSpec[function_call_type]
        if inference_cfg is not None:
            if inference_cfg.system_prompt is not None:
                msg = "LLMInferenceConfig.system_prompt must not be set"
                raise ValueError(msg)
        else:
            inference_cfg = LLMInferenceConfig(
                system_prompt=self.system_prompt,
                temperature=0,
                max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            )

        solution_tool = AlgorithmAgentSolutionTool[function_call_type](
            problem=problem,
            solution_model=AlgorithmSolutionSpec[function_call_type],
        )

        super().__init__(
            llm=llm,
            input_model=AlgorithmProblemSpec,
            output_models=[AlgorithmSolutionSpec[function_call_type]],
            response_shaping_tools=[solution_tool],
            inference_cfg=inference_cfg,
            prompt_template=ALGORITHM_AGENT_PROMPT_TEMPLATE,
            max_allowed_tool_calls_per_turn=max_allowed_tool_calls_per_turn,
        )

    @observe(as_type="agent", name="AlgorithmAgent.run_typed")
    @override
    def run_typed(
        self,
        context: LLMAgentContext,
        inp: AlgorithmProblemSpec,
        progress_callback: Callable[[ToolProgressUpdate[Any]], None] | None = None,
    ) -> AsyncGenerator[AlgorithmSolutionSpec[FunctionCallT] | ErrorResponse]:
        return super().run_typed(context, inp, progress_callback)


class AlgorithmAgentSolutionTool[FunctionCallT: BaseFunctionCall](
    ResponseShapingTool[AlgorithmSolutionSpec[FunctionCallT]]
):
    """Tool for outputting algorithm agent solutions."""

    def __init__(
        self,
        problem: AlgorithmProblemSpec,
        solution_model: type[AlgorithmSolutionSpec[FunctionCallT]],
    ) -> None:
        """Constructor.

        Args:
            problem (ProblemSpecType): Input problem that the agent will solve.
            solution_model (type[SolutionSpecT]): Pydantic model describing
                how solutions will be specified by the agent.
        """
        super().__init__(
            name="AlgorithmSolutionTool",
            description="Output and validate an algorithm solution. The algorithm must strictly "
            "conform to the given detailed schema. If it does not, an error will be returned. "
            "In case of repeated validation failures, you may return an error response instead of "
            "attempting further corrections.",
            response_model=solution_model,
        )
        self.problem = problem

    def _validate_tool_input(
        self, tool_input: AlgorithmSolutionSpec[FunctionCallT]
    ) -> AlgorithmSolutionSpec[FunctionCallT]:
        """Type check algorithm."""
        if tool_input.error_message:
            # No need to type check if the LLM reported an error
            return tool_input

        expected_outputs: dict[str, InferredVariable] = {}
        for v in self.problem.outputs_requested:
            match v.variable_type:
                case "scalar_float" | "latitude_deg_float" | "longitude_deg_float":
                    expected_outputs[v.name] = InferredVariable(type="float", name=v.name)
                case "datetime_string":
                    expected_outputs[v.name] = InferredVariable(type="str", name=v.name)
                case "raster_plot":
                    expected_outputs[v.name] = InferredVariable(type="RasterPlot", name=v.name)
                case "line_plot":
                    expected_outputs[v.name] = InferredVariable(type="LinePlot", name=v.name)
                case "bar_plot":
                    expected_outputs[v.name] = InferredVariable(type="BarChart", name=v.name)
                case _:
                    msg = f"Unsupported output variable type: {v.variable_type}"
                    raise ResponseValidationError(msg)

        inferred_input_vars, valid_time_dims = self._get_inferred_input_vars(self.problem)

        type_check_algorithm(
            tool_input.algorithm,
            input_vars=inferred_input_vars,
            expected_outputs=expected_outputs,
            valid_time_dims=valid_time_dims,
        )

        return tool_input

    def _get_inferred_input_vars(
        self, problem: AlgorithmProblemSpec
    ) -> tuple[dict[str, InferredVariable], set[str]]:
        inferred_vars: dict[str, InferredVariable] = {}
        valid_time_dims = set[str]()
        for input_var in problem.inputs_provided:
            dims = frozenset([d.field_name for d in input_var.dimensions])
            match input_var:
                case GeneralDataArrayVariable():
                    inferred_vars[input_var.name] = InferredDataArrayVariable(
                        name=input_var.name,
                        dims=dims,
                        shape=dict.fromkeys(dims, "unknown"),
                    )
                case TemporalDataArrayVariable() | SpatioTemporalDataArrayVariable():
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
                        inferred_vars[input_var.name] = InferredDataArrayVariable(
                            name=input_var.name,
                            dims=dims,
                            shape=dict.fromkeys(dims, "unknown"),
                        )
        return inferred_vars, valid_time_dims
