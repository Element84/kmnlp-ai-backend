# Ignore warnings about missing public method docstrings since their purpose is
# obvious.
# ruff: noqa: D102
import logging
from collections import Counter
from typing import Any, Literal, Self, cast
from unittest.mock import Mock

import anyio
from chromadb import EphemeralClient
from e84_geoai_common.llm.models.claude import CLAUDE_4_5_HAIKU
from pydantic import BaseModel, computed_field, field_validator, model_validator
from pydantic_ai import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models.bedrock import BedrockModelSettings
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage
from rich.pretty import pretty_repr
from tqdm import tqdm

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    BaseFunctionCall,
    make_function_call_model,
)
from llm_agent.agents.algorithm_agent.algorithm_agent import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_NUM_RETRIES,
    AlgorithmAgent,
)
from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmProblemSpec,
    AlgorithmSolutionSpec,
    input_vars_to_function_set,
)
from llm_agent.context import LLMAgentContext
from llm_agent.dao.chromadb_embedding_dao import ChromaDBEmbeddingDAO
from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from llm_agent.evaluation.base_eval import (
    BaseEvaluator,
    EvalDataset,
    EvalInput,
    EvalReport,
    ExperimentConfig,
    SingleEvalResult,
)
from llm_agent.evaluation.utils import MockChatContext, SummaryStats, build_eval_client
from llm_agent.model_factory import make_bedrock_model
from llm_agent.repository.embedding_index_repository import EmbeddingIndexRepository

log = logging.getLogger(__name__)


context = MockChatContext(
    LLMAgentContext(
        place_lookup=Mock(),
        knowledge_graph_dao=KnowledgeGraphDAO(),
        embedding_repo=EmbeddingIndexRepository(
            ChromaDBEmbeddingDAO(EphemeralClient(), "datasets"),
            KnowledgeGraphDAO().fetch_datasets(),
        ),
        dask_client=Mock(),
        bedrock_client=Mock(),
    )
)


class AlgorithmAgentExperimentConfig(BaseModel):
    """Name and parameters of the experiment."""

    name: str
    dataset_id: str
    dataset_version: str
    model_name: str
    temperature: float
    thinking: bool | Literal["minimal", "low", "medium", "high", "xhigh"] = False
    num_retries: int = DEFAULT_NUM_RETRIES

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
                "thinking": self.thinking,
                "num_retries": self.num_retries,
            },
        )


class AlgorithmAgentEvalInput[FunctionCallT: BaseFunctionCall](BaseModel):
    """A single evaluation test input and associated reference output."""

    id: str
    problem: AlgorithmProblemSpec
    reference_solution: (
        AlgorithmSolutionSpec[FunctionCallT] | list[AlgorithmSolutionSpec[FunctionCallT]]
    )
    problem_attrs: "AlgorithmAgentProblemAttrs | None" = None
    solution_attrs: "AlgorithmAgentSolutionAttrs | None" = None
    # Ops the agent may emit beyond the reference without penalty (valid optimizations,
    # e.g. a loose bbox slice before a polygon filter). Each listed occurrence forgives one
    # extra occurrence of that op.
    allowed_extra_ops: list[str] = []

    @field_validator("reference_solution")
    @classmethod
    def _reject_empty_reference_list(
        cls,
        v: "AlgorithmSolutionSpec[FunctionCallT] | list[AlgorithmSolutionSpec[FunctionCallT]]",
    ) -> "AlgorithmSolutionSpec[FunctionCallT] | list[AlgorithmSolutionSpec[FunctionCallT]]":
        if isinstance(v, list) and not v:
            msg = "reference_solution must not be an empty list."
            raise ValueError(msg)
        return v

    @property
    def reference_solutions(self) -> list[AlgorithmSolutionSpec[FunctionCallT]]:
        """Reference solution(s) normalized to a list."""
        ref = self.reference_solution
        return ref if isinstance(ref, list) else [ref]

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
            primary = self.reference_solutions[0]
            self.solution_attrs = AlgorithmAgentSolutionAttrs(
                num_statements=len(primary.algorithm.statements),
                num_outputs=len(primary.algorithm.output_variables),
                ops=_get_algorithm_ops(primary.algorithm),
                is_refusal=any(bool(s.error_message) for s in self.reference_solutions),
            )
        return self

    @staticmethod
    def from_eval_input(
        eval_input: EvalInput,
    ) -> "tuple[AlgorithmAgentEvalInput[BaseFunctionCall], type[BaseFunctionCall]]":
        problem = AlgorithmProblemSpec.model_validate(eval_input.input)
        function_set = input_vars_to_function_set(problem.inputs_provided)
        function_call_type = make_function_call_model(function_set.functions)
        solution_type = AlgorithmSolutionSpec[function_call_type]
        reference_output = eval_input.reference_output
        if isinstance(reference_output, dict) and isinstance(
            reference_output.get("solutions"), list
        ):
            reference_solution = [
                solution_type.model_validate(s) for s in reference_output["solutions"]
            ]
        else:
            reference_solution = solution_type.model_validate(reference_output)
        alg_eval_input = AlgorithmAgentEvalInput[function_call_type](
            id=eval_input.id,
            problem=AlgorithmProblemSpec.model_validate(eval_input.input),
            reference_solution=reference_solution,
            allowed_extra_ops=(eval_input.attrs or {}).get("allowed_extra_ops", []),
        )
        return alg_eval_input, function_call_type

    def to_eval_input(self) -> EvalInput:
        attrs: dict[str, Any] = {}
        if self.problem_attrs is not None:
            attrs.update(**self.problem_attrs.model_dump())
        if self.solution_attrs is not None:
            attrs.update(**self.solution_attrs.model_dump())
        attrs["allowed_extra_ops"] = self.allowed_extra_ops

        # A single solution keeps the legacy flat shape; a list is wrapped under "solutions".
        ref = self.reference_solution
        if isinstance(ref, list):
            reference_output: dict[str, Any] = {"solutions": [s.model_dump() for s in ref]}
        else:
            reference_output = ref.model_dump()

        return EvalInput(
            id=self.id,
            input=self.problem.model_dump(),
            reference_output=reference_output,
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
    usage: RunUsage | None = None

    @computed_field
    @property
    def correct(self) -> bool:
        if self.is_refusal is None:
            return False
        if self.is_refusal:
            return self.format_compliant and bool(self.schema_compliant)
        # statement_count_diff is intentionally excluded: since every statement is an op, it is
        # redundant with the ops check except when it disagrees with allowed_extra_ops forgiveness.
        return (
            self.format_compliant
            and bool(self.schema_compliant)
            and self.output_variables_count_diff == 0
            and not bool(self.missing_ops)
            and not bool(self.extra_ops)
        )


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
    pct_correct: float = -1
    pct_format_compliant: float
    pct_schema_compliant: float
    pct_correct_refusal: float
    pct_incorrect_refusal: float
    pct_statement_count_mismatch: float
    pct_output_variables_count_mismatch: float
    pct_ops_mismatch: float
    # Usage totals summed across all cases in the run (defaults keep older reports loadable).
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_requests: int = 0
    total_tool_calls: int = 0


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
    pct_correct: SummaryStats
    pct_format_compliant: SummaryStats
    pct_schema_compliant: SummaryStats
    pct_correct_refusal: SummaryStats
    pct_incorrect_refusal: SummaryStats
    pct_statement_count_mismatch: SummaryStats
    pct_output_variables_count_mismatch: SummaryStats
    pct_ops_mismatch: SummaryStats
    total_input_tokens: SummaryStats
    total_output_tokens: SummaryStats
    total_cache_read_tokens: SummaryStats
    total_cache_write_tokens: SummaryStats
    total_requests: SummaryStats
    total_tool_calls: SummaryStats


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
        log.info(
            "AlgorithmAgentExperimentConfig: %s",
            pretty_repr(algo_exp_config, indent_size=2),
        )
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

    async def _make_agent[FunctionCallT: BaseFunctionCall](
        self,
        *,
        experiment_config: AlgorithmAgentExperimentConfig,
        inp: AlgorithmAgentEvalInput[FunctionCallT],
        function_call_type: type[FunctionCallT],
    ) -> AlgorithmAgent[FunctionCallT]:
        if (
            experiment_config.model_name == CLAUDE_4_5_HAIKU
            and experiment_config.thinking == "high"
        ):
            # Thinking needs to be configured differently for the old Haiku 4.5 model.
            log.info("Using special thinking config for Haiku 4.5")
            model_settings = BedrockModelSettings(
                temperature=experiment_config.temperature,
                bedrock_additional_model_requests_fields={
                    "thinking": {
                        "type": "enabled",
                        # The API requires max_tokens to be greater than the thinking budget
                        "budget_tokens": DEFAULT_MAX_OUTPUT_TOKENS - 1,
                    }
                },
            )
        else:
            model_settings = ModelSettings(
                temperature=experiment_config.temperature,
                thinking=experiment_config.thinking,
            )
        model = make_bedrock_model(
            build_eval_client(),
            model_id=experiment_config.model_name,
            model_settings=model_settings,
        )
        return AlgorithmAgent(
            model,
            problem=inp.problem,
            function_call_type=function_call_type,
            num_retries=experiment_config.num_retries,
        )

    async def _eval_single_case[FunctionCallT: BaseFunctionCall](
        self,
        *,
        experiment_config: AlgorithmAgentExperimentConfig,
        inp: AlgorithmAgentEvalInput[FunctionCallT],
        function_call_type: type[FunctionCallT],
    ) -> AlgorithmAgentSingleEvalResult[FunctionCallT]:
        agent = await self._make_agent(
            experiment_config=experiment_config,
            inp=inp,
            function_call_type=function_call_type,
        )
        try:
            solution, usage = await agent.run(context)
            log.info("Usage: %s", usage)
        except UsageLimitExceeded:
            return AlgorithmAgentSingleEvalResult(
                # Raised when the agent cannot produce a valid, type-checked solution
                # within its retry budget (the output validator raises ModelRetry on
                # schema/type-check failures).
                input=inp,
                output=ParsingFailure(
                    failed_output="",
                    error_message="Usage limit exceeded",
                ),
                metrics=AlgorithmAgentSingleEvalMetrics(
                    format_compliant=True,
                    schema_compliant=False,
                ),
            )
        except UnexpectedModelBehavior as e:
            # Usually raised when token budget exceeded.
            return AlgorithmAgentSingleEvalResult(
                input=inp,
                output=ParsingFailure(
                    failed_output="",
                    error_message=f"{type(e).__name__}: {e!s}",
                ),
                metrics=AlgorithmAgentSingleEvalMetrics(
                    format_compliant=True,
                    schema_compliant=False,
                ),
            )

        return await self._eval_solution(
            inp=inp,
            solution=solution,
            usage=usage,
        )

    async def _eval_solution[FunctionCallT: BaseFunctionCall](
        self,
        *,
        inp: AlgorithmAgentEvalInput[FunctionCallT],
        solution: AlgorithmSolutionSpec[FunctionCallT],
        usage: RunUsage | None,
    ) -> AlgorithmAgentSingleEvalResult[FunctionCallT]:
        if solution.error_message:
            return AlgorithmAgentSingleEvalResult[FunctionCallT](
                input=inp,
                output=solution,
                metrics=AlgorithmAgentSingleEvalMetrics(
                    format_compliant=True,
                    schema_compliant=True,
                    is_refusal=True,
                    usage=usage,
                ),
            )

        gen_soln_attrs = _get_solution_attrs(solution)
        ref_soln_attrs, matching_ops, missing_ops, extra_ops = _select_best_reference(
            gen_soln_attrs.ops,
            [_get_solution_attrs(s) for s in inp.reference_solutions],
            inp.allowed_extra_ops,
        )

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
                usage=usage,
            ),
        )

    async def _reevaluate_report(self, report: EvalReport, dataset: EvalDataset) -> EvalReport:
        """Re-score a single run report's stored solutions against the current dataset."""
        current_inputs = {
            item.id: AlgorithmAgentEvalInput.from_eval_input(item)[0] for item in dataset.items
        }
        typed_report = AlgorithmAgentEvalReport.from_eval_report(report)
        rescored = [
            await self._rescore_result(result, current_inputs.get(result.input.id))
            for result in typed_report.results
        ]
        return AlgorithmAgentEvalReport(
            experiment_config=typed_report.experiment_config,
            results=rescored,
            aggregated_metrics=_aggregate_metrics(rescored),
        ).to_eval_report()

    async def _rescore_result(
        self,
        result: AlgorithmAgentSingleEvalResult[Any],
        current_input: "AlgorithmAgentEvalInput[BaseFunctionCall] | None",
    ) -> AlgorithmAgentSingleEvalResult[Any]:
        # Only generated solutions carry ops to re-score; parse/schema failures are left as-is.
        if not isinstance(result.output, AlgorithmSolutionSpec):
            return result
        if current_input is None:
            log.warning(
                "Case %s is not in the current dataset; re-scoring against its stored reference.",
                result.input.id,
            )
        inp = cast("AlgorithmAgentEvalInput[BaseFunctionCall]", current_input or result.input)
        solution = cast("AlgorithmSolutionSpec[BaseFunctionCall]", result.output)
        return await self._eval_solution(inp=inp, solution=solution, usage=result.metrics.usage)

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
        is_refusal=bool(solution.error_message),
    )


def _compare_ops(
    ops: list[str], ref_ops: list[str], allowed_extra_ops: list[str] | None = None
) -> tuple[list[str], list[str], list[str]]:
    ops_counts = Counter(ops)
    ref_ops_counts = Counter(ref_ops)
    ops_union = set(ref_ops_counts.keys()).union(set(ops_counts.keys()))
    diff = {op: ops_counts[op] - ref_ops_counts[op] for op in ops_union}
    # Forgive up to `allowance` extra occurrences of ops the case marks as acceptable.
    for op, allowance in Counter(allowed_extra_ops or []).items():
        if diff.get(op, 0) > 0:
            diff[op] = max(diff[op] - allowance, 0)
    matching_ops = [op for op, count in diff.items() if count == 0]
    missing_ops = [op for op, count in diff.items() if count < 0]
    extra_ops = [op for op, count in diff.items() if count > 0]
    return matching_ops, missing_ops, extra_ops


def _select_best_reference(
    gen_ops: list[str],
    ref_attrs_options: list[AlgorithmAgentSolutionAttrs],
    allowed_extra_ops: list[str] | None = None,
) -> tuple[AlgorithmAgentSolutionAttrs, list[str], list[str], list[str]]:
    """Score `gen_ops` against each reference and return the closest-matching one.

    "Closest" = fewest missing+extra ops, tie-broken by most matching ops. This lets a case list
    several equally valid solutions and be graded against whichever one the agent produced.
    """
    scored = [
        (ref_attrs, *_compare_ops(gen_ops, ref_attrs.ops, allowed_extra_ops))
        for ref_attrs in ref_attrs_options
    ]
    return min(scored, key=lambda s: (len(s[2]) + len(s[3]), -len(s[1])))


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
    correct = [r.metrics.correct for r in results]
    usages = [r.metrics.usage for r in results if r.metrics.usage is not None]
    return AlgorithmAgentAggregatedMetrics(
        num_tests=len(results),
        pct_format_compliant=_pct_true(format_compliant),
        pct_schema_compliant=_pct_true(schema_compliant),
        pct_correct_refusal=_pct_true(correct_refusals),
        pct_incorrect_refusal=_pct_true(incorrect_refusal),
        pct_statement_count_mismatch=_pct_true(statement_count_mismatch),
        pct_output_variables_count_mismatch=_pct_true(output_variables_count_mismatch),
        pct_ops_mismatch=_pct_true(ops_mismatch),
        pct_correct=_pct_true(correct),
        total_input_tokens=sum(u.input_tokens for u in usages),
        total_output_tokens=sum(u.output_tokens for u in usages),
        total_cache_read_tokens=sum(u.cache_read_tokens for u in usages),
        total_cache_write_tokens=sum(u.cache_write_tokens for u in usages),
        total_requests=sum(u.requests for u in usages),
        total_tool_calls=sum(u.tool_calls for u in usages),
    )


def _get_algorithm_ops[FunctionCallT: BaseFunctionCall](
    algorithm: Algorithm[FunctionCallT],
) -> list[str]:
    return [
        s.value.func.func_name
        for s in algorithm.statements
        if isinstance(s.value, BaseFunctionCall)
    ]
