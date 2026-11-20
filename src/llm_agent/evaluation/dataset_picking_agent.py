# Ignore warnings about missing public method docstrings since their purpose is
# obvious.
# ruff: noqa: D102
import logging
import statistics
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import Mock

from chromadb import EphemeralClient
from natural_language_geocoding.tests.canned_place_lookup import CannedPlaceLookup
from pydantic import BaseModel
from pydantic_ai import AgentRunResultEvent, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import FunctionToolCallEvent
from tqdm import tqdm

from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import (
    DatasetPickingAgent,
    DatasetSearchQuery,
    DatasetSearchResponse,
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
from llm_agent.tools.agent_tool import AgentTool
from llm_agent.tools.execute_sparql_query_tool import ExecuteSparqlQueryTool
from llm_agent.tools.fetch_datasets_tool import FetchDatasetsTool

log = logging.getLogger(__name__)

# Name Pydantic AI assigns to the structured-output tool; excluded from the
# recorded "real" tool calls (e.g. dataset fetches).
_OUTPUT_TOOL_NAME = "final_result"


@dataclass
class _IntermediateMetric:
    """Metric values for a single evaluation result."""

    format_compliant: bool
    schema_compliant: bool
    correct_refusal: bool | None
    incorrect_refusal: bool | None
    correct_non_refusal: bool | None
    incorrect_non_refusal: bool | None
    precision_datasets: float | None
    recall_datasets: float | None


class DatasetPickingResponseAttrs(BaseModel):
    """Attributes extracted from response."""

    is_refusal: bool
    tool_call_count: int
    dataset_ids: list[str]


class _DatasetPickingExperimentConfig(BaseModel):
    """Name and parameters of experiment."""

    name: str
    dataset_id: str
    dataset_version: str
    model_name: str
    temperature: float
    available_tool_names: list[str]

    @staticmethod
    def from_eval_exp_config(
        experiment_config: ExperimentConfig,
    ) -> "_DatasetPickingExperimentConfig":
        """Factory to build from base ExperimentConfig."""
        return _DatasetPickingExperimentConfig.model_validate(
            {
                "name": experiment_config.name,
                "dataset_id": experiment_config.dataset_id,
                "dataset_version": experiment_config.dataset_version,
                **experiment_config.params,
            }
        )

    def to_eval_exp_config(self) -> ExperimentConfig:
        """Convert back to base ExperimentConfig."""
        return ExperimentConfig(
            name=self.name,
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
            params={
                "model_name": self.model_name,
                "temperature": self.temperature,
                "available_tool_names": self.available_tool_names,
            },
        )


class DatasetPickingEvalInput(BaseModel):
    """A single evaluation test input and reference output."""

    id: str
    natural_language_query: str
    expected_response_attrs: DatasetPickingResponseAttrs

    @staticmethod
    def from_eval_input(eval_input: EvalInput) -> "DatasetPickingEvalInput":
        """Factory from base `EvalInput`."""
        if type(eval_input.reference_output) is str:
            raise NotImplementedError

        return DatasetPickingEvalInput(
            id=eval_input.id,
            natural_language_query=cast(
                "str",
                (cast("dict[str, Any]", eval_input.input))["natural_language_query"],
            ),
            expected_response_attrs=DatasetPickingResponseAttrs.model_validate(
                eval_input.reference_output
            ),
        )

    def to_eval_input(self) -> EvalInput:
        """Convert back to base `EvalInput`."""
        return EvalInput(
            id=self.id,
            input={"natural_language_query": self.natural_language_query},
            reference_output=self.expected_response_attrs.model_dump(mode="json"),
            attrs=None,
        )


class DatasetPickingSingleEvalMetrics(BaseModel):
    """Metrics for a single eval."""

    format_compliant: bool
    schema_compliant: bool
    is_too_many_prompt_turns: bool = False

    is_refusal: bool | None = None
    contains_datasets: bool | None = None
    contains_expected_datasets: bool | None = None
    precision_datasets: float | None = None
    recall_datasets: float | None = None


class DatasetPickingAggregatedMetrics(BaseModel):
    """Aggregated metrics."""

    pct_format_compliant: float
    pct_schema_compliant: float
    pct_correct_refusal: float
    pct_incorrect_refusal: float
    pct_correct_non_refusal: float
    pct_incorrect_non_refusal: float
    avg_precision_datasets: float
    avg_recall_datasets: float


class DatasetPickingAgentFailure(BaseModel):
    """Failure details."""

    name: str
    message: str
    failed_output: dict[str, Any] | str


class DatasetPickingToolCall(BaseModel):
    """A tool call made by the agent during a run."""

    name: str
    args: dict[str, Any]


class DatasetPickingAgentFullResponse(BaseModel):
    """Full agent response from a single turn."""

    tool_calls: list[DatasetPickingToolCall]
    final_response: DatasetSearchResponse | None


class DatasetPickingAgentSingleEvalResult(BaseModel):
    """Results for a single eval test case."""

    input: DatasetPickingEvalInput
    output: DatasetPickingAgentFullResponse | DatasetPickingAgentFailure
    metrics: DatasetPickingSingleEvalMetrics

    @staticmethod
    def from_single_eval_result(
        eval_result: SingleEvalResult,
    ) -> "DatasetPickingAgentSingleEvalResult":
        eval_input = DatasetPickingEvalInput.from_eval_input(eval_result.input)
        return DatasetPickingAgentSingleEvalResult.model_validate(
            {
                "input": eval_input,
                "output": eval_result.output,
                "metrics": eval_result.metrics,
            }
        )

    def to_single_eval_result(self) -> SingleEvalResult:
        """Convert back to `SingleEvalResult`."""
        return SingleEvalResult(
            input=self.input.to_eval_input(),
            output=self.output.model_dump(),
            metrics=self.metrics.model_dump(),
        )


class DatasetPickingEvalReport(BaseModel):
    """Eval report."""

    experiment_config: _DatasetPickingExperimentConfig
    results: list[DatasetPickingAgentSingleEvalResult]
    aggregated_metrics: DatasetPickingAggregatedMetrics

    @staticmethod
    def from_eval_report(
        eval_report: EvalReport,
    ) -> "DatasetPickingEvalReport":
        if eval_report.results is None:
            raise ValueError

        return DatasetPickingEvalReport(
            experiment_config=_DatasetPickingExperimentConfig.from_eval_exp_config(
                eval_report.experiment_config
            ),
            results=[
                DatasetPickingAgentSingleEvalResult.from_single_eval_result(r)
                for r in eval_report.results
            ],
            aggregated_metrics=DatasetPickingAggregatedMetrics.model_validate(
                eval_report.aggregated_metrics
            ),
        )

    def to_eval_report(self) -> EvalReport:
        """Convert back to base `EvalReport`."""
        return EvalReport(
            experiment_config=self.experiment_config.to_eval_exp_config(),
            results=[r.to_single_eval_result() for r in self.results],
            aggregated_metrics=self.aggregated_metrics.model_dump(),
        )


class DatasetPickingMultiRunMetrics(BaseModel):
    """Aggregated metrics, averaged over multiple experiment runs."""

    num_runs: int
    pct_format_compliant: SummaryStats
    pct_schema_compliant: SummaryStats
    pct_correct_refusal: SummaryStats
    pct_incorrect_refusal: SummaryStats
    pct_correct_non_refusal: SummaryStats
    pct_incorrect_non_refusal: SummaryStats
    avg_precision_datasets: SummaryStats
    avg_recall_datasets: SummaryStats


class DatasetPickingMultiRunEvalReport(BaseModel):
    """Eval report aggregated over multiple experiment runs."""

    experiment_config: _DatasetPickingExperimentConfig
    aggregated_metrics: DatasetPickingMultiRunMetrics

    @staticmethod
    def from_eval_report(
        eval_report: EvalReport,
    ) -> "DatasetPickingMultiRunEvalReport":
        return DatasetPickingMultiRunEvalReport(
            experiment_config=_DatasetPickingExperimentConfig.from_eval_exp_config(
                eval_report.experiment_config
            ),
            aggregated_metrics=DatasetPickingMultiRunMetrics.model_validate(
                eval_report.aggregated_metrics
            ),
        )

    def to_eval_report(self) -> EvalReport:
        return EvalReport(
            experiment_config=self.experiment_config.to_eval_exp_config(),
            results=None,
            aggregated_metrics=self.aggregated_metrics.model_dump(),
        )


# The context doesn't matter for LLM evaluation so we use this one as with unit tests.
context = MockChatContext(
    LLMAgentContext(
        place_lookup=CannedPlaceLookup(),
        knowledge_graph_dao=KnowledgeGraphDAO(),
        embedding_repo=EmbeddingIndexRepository(
            ChromaDBEmbeddingDAO(EphemeralClient(), "datasets"),
            KnowledgeGraphDAO().fetch_datasets(),
        ),
        dask_client=Mock(),
        bedrock_client=Mock(),
    )
)


class DatasetPickingAgentEvaluator(BaseEvaluator):
    """Dataset Picking Agent Evaluator."""

    async def _eval_dataset(
        self, experiment_config: ExperimentConfig, dataset: EvalDataset
    ) -> EvalReport:
        """Run eval dataset with given experiment config."""
        dataset_picking_eval_dataset = [
            DatasetPickingEvalInput.from_eval_input(item) for item in dataset.items
        ]
        dataset_picking_experiment_config = _DatasetPickingExperimentConfig.from_eval_exp_config(
            experiment_config
        )

        dataset_picking_report = await self._eval_dataset_typed(
            dataset_picking_experiment_config, dataset_picking_eval_dataset
        )
        eval_report = dataset_picking_report.to_eval_report()
        return eval_report

    async def _eval_dataset_typed(
        self,
        experiment_config: _DatasetPickingExperimentConfig,
        dataset: list[DatasetPickingEvalInput],
    ) -> DatasetPickingEvalReport:
        model = make_bedrock_model(build_eval_client(), model_id=experiment_config.model_name)
        tools = _tools_for_tool_names(experiment_config.available_tool_names)
        agent = DatasetPickingAgent(model, tools=tools, temperature=experiment_config.temperature)
        with tqdm(dataset, desc="Evaluating") as bar:
            results = [await self._eval_single_case(agent, case) for case in bar]
        return DatasetPickingEvalReport(
            experiment_config=experiment_config,
            results=results,
            aggregated_metrics=_aggregate_metrics(results),
        )

    async def _eval_single_case(
        self, agent: DatasetPickingAgent, case: DatasetPickingEvalInput
    ) -> DatasetPickingAgentSingleEvalResult:
        log.debug("Starting case with id: %s", case.id)
        try:
            tool_calls: list[DatasetPickingToolCall] = []
            final_response: DatasetSearchResponse | None = None
            async for event in agent.run_stream(
                context, DatasetSearchQuery(query=case.natural_language_query)
            ):
                if (
                    isinstance(event, FunctionToolCallEvent)
                    and event.part.tool_name != _OUTPUT_TOOL_NAME
                ):
                    tool_calls.append(
                        DatasetPickingToolCall(
                            name=event.part.tool_name, args=event.part.args_as_dict()
                        )
                    )
                elif isinstance(event, AgentRunResultEvent):
                    final_response = event.result.output
            response = DatasetPickingAgentFullResponse(
                tool_calls=tool_calls, final_response=final_response
            )
        except UsageLimitExceeded as e:
            message = "Tool-call usage limit exceeded."
            log.exception(message)
            return DatasetPickingAgentSingleEvalResult(
                input=case,
                output=DatasetPickingAgentFailure(
                    name=type(e).__name__, message=message, failed_output=str(e)
                ),
                metrics=DatasetPickingSingleEvalMetrics(
                    format_compliant=True,
                    schema_compliant=True,
                    is_too_many_prompt_turns=True,
                ),
            )
        except UnexpectedModelBehavior as e:
            message = "Agent failed to produce a valid structured response."
            log.exception(message)
            return DatasetPickingAgentSingleEvalResult(
                input=case,
                output=DatasetPickingAgentFailure(
                    name=type(e).__name__, message=message, failed_output=str(e)
                ),
                metrics=DatasetPickingSingleEvalMetrics(
                    format_compliant=True, schema_compliant=False
                ),
            )

        response_attrs = _response_attrs_for(response)

        metrics = DatasetPickingSingleEvalMetrics(
            format_compliant=True,
            schema_compliant=True,
            is_refusal=response_attrs.is_refusal,
            contains_datasets=len(response_attrs.dataset_ids) > 0,
            contains_expected_datasets=(
                response_attrs.dataset_ids == case.expected_response_attrs.dataset_ids
            ),
            precision_datasets=_precision(
                response_attrs.dataset_ids,
                case.expected_response_attrs.dataset_ids,
            ),
            recall_datasets=_recall(
                response_attrs.dataset_ids,
                case.expected_response_attrs.dataset_ids,
            ),
        )

        return DatasetPickingAgentSingleEvalResult(input=case, output=response, metrics=metrics)

    async def _aggregate_eval_reports(self, reports: list[EvalReport]) -> EvalReport:
        reports_custom = [DatasetPickingEvalReport.from_eval_report(r) for r in reports]
        exp_config = reports_custom[0].experiment_config
        run_metrics = [r.aggregated_metrics for r in reports_custom]
        keys = DatasetPickingAggregatedMetrics.model_fields.keys()
        summary_stats = {
            k: SummaryStats.from_values([getattr(m, k) for m in run_metrics]) for k in keys
        }
        aggregated_metrics = DatasetPickingMultiRunMetrics.model_validate(
            {"num_runs": len(reports), **summary_stats}
        )
        return DatasetPickingMultiRunEvalReport(
            experiment_config=exp_config,
            aggregated_metrics=aggregated_metrics,
        ).to_eval_report()


def _aggregate_metrics(
    results: list[DatasetPickingAgentSingleEvalResult],
) -> DatasetPickingAggregatedMetrics:
    intermediate_metrics = [_intermediate_metric_from_result(result) for result in results]

    return DatasetPickingAggregatedMetrics(
        pct_format_compliant=statistics.mean([m.format_compliant for m in intermediate_metrics])
        * 100.0,
        pct_schema_compliant=statistics.mean([m.schema_compliant for m in intermediate_metrics])
        * 100.0,
        pct_correct_refusal=statistics.mean(
            [m.correct_refusal for m in intermediate_metrics if m.correct_refusal is not None]
            or [0]
        )
        * 100.0,
        pct_incorrect_refusal=statistics.mean(
            [m.incorrect_refusal for m in intermediate_metrics if m.incorrect_refusal is not None]
            or [0]
        )
        * 100.0,
        pct_correct_non_refusal=statistics.mean(
            [m.correct_non_refusal for m in intermediate_metrics if m.correct_non_refusal] or [0]
        )
        * 100.0,
        pct_incorrect_non_refusal=statistics.mean(
            [m.incorrect_non_refusal for m in intermediate_metrics if m.incorrect_non_refusal]
            or [0]
        )
        * 100.0,
        avg_precision_datasets=statistics.mean(
            [
                precision_datasets
                for m in intermediate_metrics
                if (precision_datasets := m.precision_datasets) is not None
            ]
            or [0]
        )
        * 100.0,
        avg_recall_datasets=statistics.mean(
            [
                recall_datasets
                for m in intermediate_metrics
                if (recall_datasets := m.recall_datasets) is not None
            ]
            or [0]
        )
        * 100.0,
    )


def _response_attrs_for(response: DatasetPickingAgentFullResponse) -> DatasetPickingResponseAttrs:
    if response.final_response is None:
        return DatasetPickingResponseAttrs(
            is_refusal=False,
            tool_call_count=len(response.tool_calls),
            dataset_ids=[],
        )

    return DatasetPickingResponseAttrs(
        is_refusal=response.final_response.is_refusal,
        tool_call_count=len(response.tool_calls),
        dataset_ids=response.final_response.result or [],
    )


def _intermediate_metric_from_result(
    result: DatasetPickingAgentSingleEvalResult,
) -> _IntermediateMetric:
    return _IntermediateMetric(
        format_compliant=result.metrics.format_compliant,
        schema_compliant=result.metrics.schema_compliant,
        correct_refusal=_is_correct_refusal(result),
        incorrect_refusal=_is_incorrect_refusal(result),
        correct_non_refusal=_is_correct_non_refusal(result),
        incorrect_non_refusal=_is_incorrect_non_refusal(result),
        precision_datasets=result.metrics.precision_datasets,
        recall_datasets=result.metrics.recall_datasets,
    )


def _is_correct_refusal(result: DatasetPickingAgentSingleEvalResult) -> bool | None:
    if result.metrics.is_refusal:
        return result.input.expected_response_attrs.is_refusal
    return None


def _is_incorrect_refusal(result: DatasetPickingAgentSingleEvalResult) -> bool | None:
    if result.metrics.is_refusal:
        return not result.input.expected_response_attrs.is_refusal
    return None


def _is_correct_non_refusal(result: DatasetPickingAgentSingleEvalResult) -> bool | None:
    if not result.metrics.is_refusal:
        return not result.input.expected_response_attrs.is_refusal
    return None


def _is_incorrect_non_refusal(result: DatasetPickingAgentSingleEvalResult) -> bool | None:
    if not result.metrics.is_refusal:
        return result.input.expected_response_attrs.is_refusal
    return None


def _precision[T](actual: list[T], expected: list[T]) -> float | None:
    true_positives = len([d for d in actual if d in expected])
    false_positives = len([d for d in actual if d not in expected])

    if true_positives + false_positives == 0:
        return None

    return true_positives / (true_positives + false_positives)


def _recall[T](actual: list[T], expected: list[T]) -> float | None:
    true_positives = len([d for d in actual if d in expected])
    false_negatives = len([d for d in expected if d not in actual])

    if true_positives + false_negatives == 0:
        return None

    return true_positives / (true_positives + false_negatives)


def _tools_for_tool_names(tool_names: list[str]) -> list[AgentTool[Any, Any]]:
    tool_dict: dict[str, AgentTool[Any, Any]] = {
        "fetch_datasets_tool": FetchDatasetsTool(),
        "execute_sparql_query_tool": ExecuteSparqlQueryTool(),
    }
    return [tool_dict[tool_name] for tool_name in tool_names]
