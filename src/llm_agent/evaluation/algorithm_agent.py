# Ignore warnings about missing public method docstrings since their purpose is
# obvious.
# ruff: noqa: D102

import json
import logging
from collections import Counter
from typing import Any, Self
from unittest.mock import Mock

import anyio
from e84_geoai_common.llm.core import LLMInferenceConfig
from e84_geoai_common.llm.models.claude import (
    CLAUDE_BEDROCK_MODEL_IDS,
    BedrockClaudeLLM,
)
from pydantic import BaseModel, model_validator
from rich.pretty import pretty_repr
from tqdm import tqdm

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    BaseFunctionCall,
    make_function_call_model,
)
from llm_agent.agents.algorithm_agent.algorithm_agent import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    AlgorithmAgent,
)
from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmProblemSpec,
    AlgorithmSolutionSpec,
    input_vars_to_function_set,
)
from llm_agent.agents.tool_only_agent import MaxAllowedToolCallsExceededError
from llm_agent.context import LLMAgentContext
from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from llm_agent.evaluation.base_eval import (
    BaseEvaluator,
    EvalDataset,
    EvalInput,
    EvalReport,
    ExperimentConfig,
    SingleEvalResult,
)
from llm_agent.evaluation.utils import SummaryStats
from llm_agent.tools.llm_tool_typed import ErrorResponse
from llm_agent.utils.utils import EnrichedLLMToolResultContent, EnrichedValidationError

log = logging.getLogger(__name__)

context = LLMAgentContext(
    place_lookup=Mock(),
    s3_client=Mock(),
    knowledge_graph_dao=KnowledgeGraphDAO(),
    dask_client=Mock(),
)


class AlgorithmAgentExperimentConfig(BaseModel):
    """Name and parameters of the experiment."""

    name: str
    dataset_id: str
    dataset_version: str
    model_name: str
    temperature: float

    @staticmethod
    def from_eval_exp_config(
        experiment_config: ExperimentConfig,
    ) -> "AlgorithmAgentExperimentConfig":
        return AlgorithmAgentExperimentConfig.model_validate(
            {
                "name": experiment_config.name,
                "dataset_id": experiment_config.dataset_id,
                "dataset_version": experiment_config.dataset_version,
                **experiment_config.params,
            }
        )

    def to_eval_exp_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name=self.name,
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
            params={
                "model_name": self.model_name,
                "temperature": self.temperature,
            },
        )


class AlgorithmAgentEvalInput[FunctionCallT: BaseFunctionCall](BaseModel):
    """A single evaluation test input and associated reference output."""

    id: str
    problem: AlgorithmProblemSpec
    reference_solution: AlgorithmSolutionSpec[FunctionCallT]
    problem_attrs: "AlgorithmAgentProblemAttrs | None" = None
    solution_attrs: "AlgorithmAgentSolutionAttrs | None" = None

    @model_validator(mode="after")
    def _init_problem_attrs(self) -> Self:
        if self.problem_attrs is None:
            self.problem_attrs = AlgorithmAgentProblemAttrs(
                num_inputs=len(self.problem.inputs_provided),
                num_requested_outputs=len(self.problem.outputs_requested),
                query_length=len(self.problem.statement),
            )
        return self

    @model_validator(mode="after")
    def _init_solution_attrs(self) -> Self:
        if self.solution_attrs is None:
            self.solution_attrs = AlgorithmAgentSolutionAttrs(
                num_statements=len(self.reference_solution.algorithm.statements),
                num_outputs=len(self.reference_solution.algorithm.output_variables),
                ops=_get_algorithm_ops(self.reference_solution.algorithm),
                is_refusal=(self.reference_solution.error_message is not None),
            )
        return self

    @staticmethod
    def from_eval_input(
        eval_input: EvalInput,
    ) -> "tuple[AlgorithmAgentEvalInput[BaseFunctionCall], type[BaseFunctionCall]]":
        problem = AlgorithmProblemSpec.model_validate(eval_input.input)
        function_set = input_vars_to_function_set(problem.inputs_provided)
        function_call_type = make_function_call_model(function_set.functions)
        alg_eval_input = AlgorithmAgentEvalInput[function_call_type](
            id=eval_input.id,
            problem=AlgorithmProblemSpec.model_validate(eval_input.input),
            reference_solution=AlgorithmSolutionSpec[function_call_type].model_validate(
                eval_input.reference_output
            ),
        )
        return alg_eval_input, function_call_type

    def to_eval_input(self) -> EvalInput:
        attrs: dict[str, Any] = {}
        if self.problem_attrs is not None:
            attrs.update(**self.problem_attrs.model_dump())
        if self.solution_attrs is not None:
            attrs.update(**self.solution_attrs.model_dump())

        return EvalInput(
            id=self.id,
            input=self.problem.model_dump(),
            reference_output=self.reference_solution.model_dump(),
            attrs=attrs,
        )


class AlgorithmAgentProblemAttrs(BaseModel):
    """Attributes describing an algorithm agent problem."""

    num_inputs: int
    num_requested_outputs: int
    query_length: int


class AlgorithmAgentSolutionAttrs(BaseModel):
    """Attributes describing an algorithm agent solution."""

    num_statements: int
    num_outputs: int
    ops: list[str]
    is_refusal: bool


class AlgorithmAgentSingleEvalMetrics(BaseModel):
    """Metrics for a single eval test case."""

    format_compliant: bool = False
    schema_compliant: bool | None = None
    is_refusal: bool | None = None
    generated_solution_attrs: AlgorithmAgentSolutionAttrs | None = None
    statement_count_diff: int | None = None
    output_variables_count_diff: int | None = None
    matching_ops: list[str] | None = None
    missing_ops: list[str] | None = None
    extra_ops: list[str] | None = None


class ParsingFailure(BaseModel):
    """Agent output that failed to parse."""

    failed_output: dict[str, Any] | str
    error_message: str


class AlgorithmAgentSingleEvalResult[FunctionCallT: BaseFunctionCall](BaseModel):
    """Results for a single eval test case."""

    input: AlgorithmAgentEvalInput[FunctionCallT]
    output: AlgorithmSolutionSpec[FunctionCallT] | ParsingFailure | None
    metrics: AlgorithmAgentSingleEvalMetrics

    @staticmethod
    def from_single_eval_result(
        eval_result: SingleEvalResult,
    ) -> "tuple[AlgorithmAgentSingleEvalResult[BaseFunctionCall], type[BaseFunctionCall]]":
        eval_input, function_call_type = AlgorithmAgentEvalInput.from_eval_input(eval_result.input)
        out = AlgorithmAgentSingleEvalResult[function_call_type].model_validate(
            {
                "input": eval_input,
                "output": eval_result.output,
                "metrics": eval_result.metrics,
            }
        )
        return out, function_call_type

    def to_single_eval_result(self) -> SingleEvalResult:
        output = self.output.model_dump() if self.output is not None else None
        return SingleEvalResult(
            input=self.input.to_eval_input(),
            output=output,
            metrics=self.metrics.model_dump(),
        )


class AlgorithmAgentAggregatedMetrics(BaseModel):
    """Aggregated metrics."""

    num_tests: int
    pct_format_compliant: float
    pct_schema_compliant: float
    pct_correct_refusal: float
    pct_incorrect_refusal: float
    pct_statement_count_mismatch: float
    pct_output_variables_count_mismatch: float
    pct_ops_mismatch: float


class AlgorithmAgentEvalReport(BaseModel):
    """Eval report."""

    experiment_config: AlgorithmAgentExperimentConfig
    # We use Any here because each item in the list might have a different
    # FunctionCall model. We CANNOT use BaseFunctionCall because that causes
    # pydantic to implicitly cast items to the BaseFunctionCall class and the
    # contained function instance to the base Function class i.e. they lose all
    # their fields except func_name and they don't show up in the report JSON.
    results: list[AlgorithmAgentSingleEvalResult[Any]]
    aggregated_metrics: AlgorithmAgentAggregatedMetrics

    @staticmethod
    def from_eval_report(
        eval_report: EvalReport,
    ) -> "AlgorithmAgentEvalReport":
        if eval_report.results is None:
            raise ValueError

        return AlgorithmAgentEvalReport(
            experiment_config=AlgorithmAgentExperimentConfig.from_eval_exp_config(
                eval_report.experiment_config
            ),
            results=[
                AlgorithmAgentSingleEvalResult.from_single_eval_result(r)[0]
                for r in eval_report.results
            ],
            aggregated_metrics=AlgorithmAgentAggregatedMetrics.model_validate(
                eval_report.aggregated_metrics
            ),
        )

    def to_eval_report(self) -> EvalReport:
        return EvalReport(
            experiment_config=self.experiment_config.to_eval_exp_config(),
            results=[r.to_single_eval_result() for r in self.results],
            aggregated_metrics=self.aggregated_metrics.model_dump(),
        )


class AlgorithmAgentMultiRunMetrics(BaseModel):
    """Aggregated metrics, averaged over multiple experiment runs."""

    num_runs: int
    pct_format_compliant: SummaryStats
    pct_schema_compliant: SummaryStats
    pct_correct_refusal: SummaryStats
    pct_incorrect_refusal: SummaryStats
    pct_statement_count_mismatch: SummaryStats
    pct_output_variables_count_mismatch: SummaryStats
    pct_ops_mismatch: SummaryStats


class AlgorithmAgentMultiRunEvalReport(BaseModel):
    """Eval report aggregated over multiple experiment runs."""

    experiment_config: AlgorithmAgentExperimentConfig
    aggregated_metrics: AlgorithmAgentMultiRunMetrics

    @staticmethod
    def from_eval_report(
        eval_report: EvalReport,
    ) -> "AlgorithmAgentMultiRunEvalReport":
        return AlgorithmAgentMultiRunEvalReport(
            experiment_config=AlgorithmAgentExperimentConfig.from_eval_exp_config(
                eval_report.experiment_config
            ),
            aggregated_metrics=AlgorithmAgentMultiRunMetrics.model_validate(
                eval_report.aggregated_metrics
            ),
        )

    def to_eval_report(self) -> EvalReport:
        return EvalReport(
            experiment_config=self.experiment_config.to_eval_exp_config(),
            results=None,
            aggregated_metrics=self.aggregated_metrics.model_dump(),
        )


class AlgorithmAgentEvaluator(BaseEvaluator):
    """Algorithm agent evaluator."""

    async def _eval_dataset(
        self,
        experiment_config: ExperimentConfig,
        dataset: EvalDataset,
    ) -> EvalReport:
        algo_dataset = [AlgorithmAgentEvalInput.from_eval_input(item) for item in dataset.items]
        algo_exp_config = AlgorithmAgentExperimentConfig.from_eval_exp_config(experiment_config)
        algo_report = await self._eval_dataset_typed(algo_exp_config, algo_dataset)
        eval_report = algo_report.to_eval_report()
        return eval_report

    async def _eval_dataset_typed(
        self,
        experiment_config: AlgorithmAgentExperimentConfig,
        dataset: list[
            tuple[
                AlgorithmAgentEvalInput[BaseFunctionCall],
                type[BaseFunctionCall],
            ]
        ],
    ) -> AlgorithmAgentEvalReport:
        results: list[AlgorithmAgentSingleEvalResult[BaseFunctionCall]] = []
        with tqdm(dataset, desc="Evaluating") as bar:
            for inp, function_call_type in bar:
                # Workaround for Bedrock throttling errors
                await anyio.sleep(5)
                result = await self._eval_single_case(
                    experiment_config=experiment_config,
                    inp=inp,
                    function_call_type=function_call_type,
                )
                results.append(result)
        return AlgorithmAgentEvalReport(
            experiment_config=experiment_config,
            results=results,
            aggregated_metrics=_aggregate_metrics(results),
        )

    async def _eval_single_case[FunctionCallT: BaseFunctionCall](
        self,
        *,
        experiment_config: AlgorithmAgentExperimentConfig,
        inp: AlgorithmAgentEvalInput[FunctionCallT],
        function_call_type: type[FunctionCallT],
    ) -> AlgorithmAgentSingleEvalResult[FunctionCallT]:
        llm = BedrockClaudeLLM(model_id=CLAUDE_BEDROCK_MODEL_IDS[experiment_config.model_name])
        inference_cfg = LLMInferenceConfig(
            temperature=experiment_config.temperature,
            max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        )
        agent = AlgorithmAgent(
            llm,
            problem=inp.problem,
            function_call_type=function_call_type,
            inference_cfg=inference_cfg,
        )
        ref_soln_attrs = _get_solution_attrs(inp.reference_solution)
        try:
            # FUTURE: do a more thorough refactoring of the evaluation now that
            # agents returns a stream of outputs.
            outputs: list[
                EnrichedLLMToolResultContent[ErrorResponse]
                | EnrichedLLMToolResultContent[AlgorithmSolutionSpec[FunctionCallT]]
            ] = [
                output
                async for output in agent.run(
                    context,
                    conversation=[agent.input_to_message(inp.problem)],
                )
                if isinstance(output, EnrichedLLMToolResultContent)
            ]
            errors = [o.data for o in outputs if isinstance(o.data, ErrorResponse)]
            valid_solution_attempts = [
                o.data for o in outputs if isinstance(o.data, AlgorithmSolutionSpec)
            ]
            if len(valid_solution_attempts) == 0:
                return AlgorithmAgentSingleEvalResult(
                    input=inp,
                    output=ParsingFailure(
                        failed_output="",
                        error_message=f"{errors!r}",
                    ),
                    metrics=AlgorithmAgentSingleEvalMetrics(
                        format_compliant=True,
                        schema_compliant=False,
                    ),
                )
            solution = valid_solution_attempts[-1]

        except json.JSONDecodeError as e:
            return AlgorithmAgentSingleEvalResult(
                input=inp,
                output=ParsingFailure(
                    failed_output=e.doc,
                    error_message=f"JSONDecodeError: {e!s}",
                ),
                metrics=AlgorithmAgentSingleEvalMetrics(
                    format_compliant=False,
                ),
            )
        except EnrichedValidationError as e:
            # Pydantic ValidationError contains multiple sub-errors when
            # validation fails. Each sub-error has an "input" field showing
            # the invalid data. Since the same invalid input can trigger
            # multiple validation rules, we extract and deduplicate the error
            # inputs for clearer reporting.
            error_inputs = [json.dumps(e["input"]) for e in e.error.errors()]
            unique_error_inputs = list(set(error_inputs))
            parsing_error = f"ValidationErrors:\n{pretty_repr(unique_error_inputs)}"
            return AlgorithmAgentSingleEvalResult(
                input=inp,
                output=ParsingFailure(
                    failed_output=e.data,
                    error_message=parsing_error,
                ),
                metrics=AlgorithmAgentSingleEvalMetrics(
                    format_compliant=True,
                    schema_compliant=False,
                ),
            )
        except MaxAllowedToolCallsExceededError:
            return AlgorithmAgentSingleEvalResult(
                input=inp,
                output=ParsingFailure(
                    failed_output="",
                    error_message="Max allowed tool calls exceeded",
                ),
                metrics=AlgorithmAgentSingleEvalMetrics(
                    format_compliant=True,
                    schema_compliant=False,
                ),
            )

        if solution.error_message is not None:
            return AlgorithmAgentSingleEvalResult[FunctionCallT](
                input=inp,
                output=solution,
                metrics=AlgorithmAgentSingleEvalMetrics(
                    format_compliant=True,
                    schema_compliant=True,
                    is_refusal=True,
                ),
            )

        gen_soln_attrs = _get_solution_attrs(solution)
        matching_ops, missing_ops, extra_ops = _compare_ops(gen_soln_attrs.ops, ref_soln_attrs.ops)

        return AlgorithmAgentSingleEvalResult[FunctionCallT](
            input=inp,
            output=solution,
            metrics=AlgorithmAgentSingleEvalMetrics(
                format_compliant=True,
                schema_compliant=True,
                is_refusal=False,
                generated_solution_attrs=gen_soln_attrs,
                statement_count_diff=(
                    ref_soln_attrs.num_statements - gen_soln_attrs.num_statements
                ),
                output_variables_count_diff=(
                    ref_soln_attrs.num_outputs - gen_soln_attrs.num_outputs
                ),
                matching_ops=matching_ops,
                missing_ops=missing_ops,
                extra_ops=extra_ops,
            ),
        )

    async def _aggregate_eval_reports(self, reports: list[EvalReport]) -> EvalReport:
        reports_custom = [AlgorithmAgentEvalReport.from_eval_report(r) for r in reports]
        exp_config = reports_custom[0].experiment_config
        run_metrics = [r.aggregated_metrics for r in reports_custom]
        keys = AlgorithmAgentAggregatedMetrics.model_fields.keys() - "num_tests"
        summary_stats = {
            k: SummaryStats.from_values([getattr(m, k) for m in run_metrics]) for k in keys
        }
        aggregated_metrics = AlgorithmAgentMultiRunMetrics.model_validate(
            {"num_runs": len(reports), **summary_stats}
        )
        return AlgorithmAgentMultiRunEvalReport(
            experiment_config=exp_config,
            aggregated_metrics=aggregated_metrics,
        ).to_eval_report()


def _get_solution_attrs[FunctionCallT: BaseFunctionCall](
    solution: AlgorithmSolutionSpec[FunctionCallT],
) -> AlgorithmAgentSolutionAttrs:
    ops = _get_algorithm_ops(solution.algorithm)
    return AlgorithmAgentSolutionAttrs(
        num_statements=len(solution.algorithm.statements),
        num_outputs=len(solution.algorithm.output_variables),
        ops=ops,
        is_refusal=(solution.error_message is not None),
    )


def _compare_ops(ops: list[str], ref_ops: list[str]) -> tuple[list[str], list[str], list[str]]:
    ops_counts = Counter(ops)
    ref_ops_counts = Counter(ref_ops)
    ops_union = set(ref_ops_counts.keys()).union(set(ops_counts.keys()))
    diff = {op: ops_counts[op] - ref_ops_counts[op] for op in ops_union}
    matching_ops = [op for op, count in diff.items() if count == 0]
    missing_ops = [op for op, count in diff.items() if count < 0]
    extra_ops = [op for op, count in diff.items() if count > 0]
    return matching_ops, missing_ops, extra_ops


def _pct_true(items: list[bool]) -> float:
    if (n := len(items)) == 0:
        return 0
    return round(100 * sum(items) / n, 2)


def _aggregate_metrics[FunctionCallT: BaseFunctionCall](
    results: list[AlgorithmAgentSingleEvalResult[FunctionCallT]],
) -> AlgorithmAgentAggregatedMetrics:
    format_compliant = [r.metrics.format_compliant for r in results]
    schema_compliant = [r.metrics.schema_compliant is True for r in results]
    correct_refusals = [
        r.metrics.is_refusal is True
        for r in results
        if (r.input.solution_attrs is not None and r.input.solution_attrs.is_refusal)
    ]
    incorrect_refusal = [
        r.metrics.is_refusal is True
        for r in results
        if (r.input.solution_attrs is None or not r.input.solution_attrs.is_refusal)
    ]
    statement_count_mismatch = [
        r.metrics.statement_count_diff > 0
        for r in results
        if r.metrics.statement_count_diff is not None
    ]
    output_variables_count_mismatch = [
        r.metrics.output_variables_count_diff > 0
        for r in results
        if r.metrics.output_variables_count_diff is not None
    ]
    ops_mismatch = [
        (len(r.metrics.missing_ops) + len(r.metrics.extra_ops)) > 0
        for r in results
        if (r.metrics.missing_ops is not None and r.metrics.extra_ops is not None)
    ]
    return AlgorithmAgentAggregatedMetrics(
        num_tests=len(results),
        pct_format_compliant=_pct_true(format_compliant),
        pct_schema_compliant=_pct_true(schema_compliant),
        pct_correct_refusal=_pct_true(correct_refusals),
        pct_incorrect_refusal=_pct_true(incorrect_refusal),
        pct_statement_count_mismatch=_pct_true(statement_count_mismatch),
        pct_output_variables_count_mismatch=_pct_true(output_variables_count_mismatch),
        pct_ops_mismatch=_pct_true(ops_mismatch),
    )


def _get_algorithm_ops[FunctionCallT: BaseFunctionCall](
    algorithm: Algorithm[FunctionCallT],
) -> list[str]:
    return [
        type(s.value.func).__name__
        for s in algorithm.statements
        if isinstance(s.value, BaseFunctionCall)
    ]
