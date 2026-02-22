# Ignore warnings about missing public method docstrings since their purpose is
# obvious.
# ruff: noqa: D102
import logging
import statistics
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, cast
from unittest.mock import Mock

import boto3
from e84_geoai_common.llm.core import (
    ExecutableLLMTool,
    LLMInferenceConfig,
    LLMMessage,
    LLMToolUseContent,
)
from e84_geoai_common.llm.models import (
    CLAUDE_BEDROCK_MODEL_IDS,
    BedrockClaudeLLM,
)
from natural_language_geocoding.tests.canned_place_lookup import CannedPlaceLookup
from pydantic import BaseModel, TypeAdapter
from tqdm import tqdm

from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import (
    DatasetPickingAgent,
    DatasetPickingAgentMessage,
    DatasetResponseTextContent,
    MaxPromptTurnsExceededError,
    UnexpectedResponseTypeError,
)
from llm_agent.context import LLMAgentContext, ToolCallContext
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
from llm_agent.tools.execute_sparql_query_tool import ExecuteSparqlQueryTool
from llm_agent.tools.fetch_datasets_tool import FetchDatasetsTool
from llm_agent.utils.prompt_as_service import PromptServiceAdapter
from llm_agent.utils.utils import EnrichedValidationError

log = logging.getLogger(__name__)


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
    explanatory_content_count: int
    dataset_ids: list[str]
    has_datasets_in_multiple_responses: bool


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


class DatasetPickingAgentSingleEvalResult(BaseModel):
    """Results for a single eval test case."""

    input: DatasetPickingEvalInput
    output: list[DatasetPickingAgentMessage] | DatasetPickingAgentFailure
    metrics: DatasetPickingSingleEvalMetrics

    _type_adapter = TypeAdapter(list[DatasetPickingAgentMessage])

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
        output: dict[str, Any] | list[dict[str, Any]] | None = None
        match self.output:
            case list():
                output = self._type_adapter.dump_python(self.output)
            case DatasetPickingAgentFailure():
                output = self.output.model_dump()

        return SingleEvalResult(
            input=self.input.to_eval_input(),
            output=output,
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
s3_client = boto3.client("s3")  # type: ignore[reportUnknownMemberType]
context = LLMAgentContext(
    place_lookup=CannedPlaceLookup(),
    s3_client=s3_client,
    knowledge_graph_dao=KnowledgeGraphDAO(),
    dask_client=Mock(),
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
        llm = BedrockClaudeLLM(model_id=CLAUDE_BEDROCK_MODEL_IDS[experiment_config.model_name])
        inference_cfg = LLMInferenceConfig(temperature=experiment_config.temperature)
        tools = _tools_for_tool_names(experiment_config.available_tool_names)
        agent = DatasetPickingAgent(
            PromptServiceAdapter(llm.prompt), inference_cfg=inference_cfg, tools=tools
        )
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
            messages = [
                message async for message in agent.run(context, case.natural_language_query)
            ]
        except JSONDecodeError as e:
            message = "Unable to parse JSON: %s"
            log.exception(message, e.doc)
            return DatasetPickingAgentSingleEvalResult(
                input=case,
                output=DatasetPickingAgentFailure(
                    name=type(e).__name__, message=message, failed_output=e.doc
                ),
                metrics=DatasetPickingSingleEvalMetrics(
                    format_compliant=False, schema_compliant=False
                ),
            )
        except EnrichedValidationError as e:
            message = "JSON parseable but did not fit model schema."
            log.exception(message)
            return DatasetPickingAgentSingleEvalResult(
                input=case,
                output=DatasetPickingAgentFailure(
                    name=type(e).__name__,
                    message=message,
                    failed_output=e.data,
                ),
                metrics=DatasetPickingSingleEvalMetrics(
                    format_compliant=True, schema_compliant=False
                ),
            )
        except UnexpectedResponseTypeError as e:
            message = f"Got a response with an unexpected content type: {type(e.content)}"
            log.exception(message)
            return DatasetPickingAgentSingleEvalResult(
                input=case,
                output=DatasetPickingAgentFailure(
                    name=type(e).__name__,
                    message=message,
                    failed_output=(e.content).model_dump(mode="json"),
                ),
                metrics=DatasetPickingSingleEvalMetrics(
                    format_compliant=False, schema_compliant=False
                ),
            )
        except MaxPromptTurnsExceededError as e:
            message = "Max tool use turns exceeded."
            log.exception(message)

            # Turn the conversation list of messages into a Python dict.
            #
            # In the event that serialization fails, handle by writing an empty
            # conversation to the results file so that execution continues and
            # aggregate metrics are accurate.
            try:
                type_adapter = TypeAdapter(list[LLMMessage])
                messages_in_conversation: list[dict[str, Any]] = type_adapter.dump_python(
                    e.conversation
                )
                conversation = {"conversation": messages_in_conversation}
            except Exception:  # noqa: BLE001
                conversation: dict[str, Any] = {}

            return DatasetPickingAgentSingleEvalResult(
                input=case,
                output=DatasetPickingAgentFailure(
                    name=type(e).__name__,
                    message=message,
                    failed_output=conversation,
                ),
                metrics=DatasetPickingSingleEvalMetrics(
                    format_compliant=True,
                    schema_compliant=True,
                    is_too_many_prompt_turns=True,
                ),
            )

        response_attrs = _response_attrs_for(messages)

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

        return DatasetPickingAgentSingleEvalResult(input=case, output=messages, metrics=metrics)

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


def _datasets_ids_in_last_message(
    messages: list[DatasetPickingAgentMessage],
) -> list[str]:
    # Return empty list if messages is empty.
    # Most useful for when expected output is empty during testing.
    if len(messages) == 0:
        return []

    last_message = messages[-1]
    return [
        dataset_i
        for content in last_message.parsed_content
        if isinstance(content, DatasetResponseTextContent)
        if content.result
        for dataset_i in content.result
    ]


def _tool_call_count(messages: list[DatasetPickingAgentMessage]) -> int:
    return len(
        [
            True
            for message in messages
            for content in message.parsed_content
            if isinstance(content, LLMToolUseContent)
        ]
    )


def _has_datasets_in_multiple_messages(
    messages: list[DatasetPickingAgentMessage],
) -> bool:
    return (
        len(
            [
                True
                for message in messages
                for parsed_content in message
                if isinstance(parsed_content, DatasetResponseTextContent)
                if parsed_content.result
            ]
        )
        > 1
    )


def _response_attrs_for(
    messages: list[DatasetPickingAgentMessage],
) -> DatasetPickingResponseAttrs:
    dataset_ids_in_last_message = _datasets_ids_in_last_message(messages)

    return DatasetPickingResponseAttrs(
        is_refusal=_is_refusal(messages),
        tool_call_count=_tool_call_count(messages),
        explanatory_content_count=_explanatory_content_count(messages),
        dataset_ids=dataset_ids_in_last_message,
        has_datasets_in_multiple_responses=_has_datasets_in_multiple_messages(messages),
    )


def _is_refusal(messages: list[DatasetPickingAgentMessage]) -> bool:
    # Cannot be a refusal if there are no messages.
    #
    # This is mostly useful for running with empty expected output to see
    # results.
    if len(messages) == 0:
        return False

    last_message = messages[-1]
    last_content = last_message.parsed_content[-1]

    match last_content:
        case DatasetResponseTextContent():
            return last_content.is_refusal
        case LLMToolUseContent():
            return False


def _explanatory_content_count(
    messages: list[DatasetPickingAgentMessage],
) -> int:
    return len(
        [
            True
            for message in messages
            for content in message.parsed_content
            if isinstance(content, DatasetResponseTextContent)
            if content.result is None
        ]
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


def _is_correct_refusal(
    result: DatasetPickingAgentSingleEvalResult,
) -> bool | None:
    if result.metrics.is_refusal:
        return result.input.expected_response_attrs.is_refusal

    return None


def _is_incorrect_refusal(
    result: DatasetPickingAgentSingleEvalResult,
) -> bool | None:
    if result.metrics.is_refusal:
        return not result.input.expected_response_attrs.is_refusal

    return None


def _is_correct_non_refusal(
    result: DatasetPickingAgentSingleEvalResult,
) -> bool | None:
    if not result.metrics.is_refusal:
        return not result.input.expected_response_attrs.is_refusal

    return None


def _is_incorrect_non_refusal(
    result: DatasetPickingAgentSingleEvalResult,
) -> bool | None:
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


def _tools_for_tool_names(tool_names: list[str]) -> list[ExecutableLLMTool[ToolCallContext]]:
    tool_dict = {
        "fetch_datasets_tool": FetchDatasetsTool(),
        "execute_sparql_query_tool": ExecuteSparqlQueryTool(),
    }
    return [tool_dict[tool_name] for tool_name in tool_names]
