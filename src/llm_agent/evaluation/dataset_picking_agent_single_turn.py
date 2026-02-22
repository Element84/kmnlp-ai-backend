# Ignore warnings about missing public method docstrings since their purpose is
# obvious.
# ruff: noqa: D102
import logging
import statistics
from json import JSONDecodeError
from typing import Any, cast

from e84_geoai_common.llm.core import (
    LLMInferenceConfig,
    LLMToolUseContent,
    LLMUserMessage,
    TextContent,
)
from e84_geoai_common.llm.models import (
    CLAUDE_BEDROCK_MODEL_IDS,
    BedrockClaudeLLM,
)
from pydantic import BaseModel
from tqdm import tqdm

from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import (
    DatasetPickingAgent,
    DatasetPickingAgentMessage,
)
from llm_agent.evaluation.base_eval import (
    BaseEvaluator,
    EvalDataset,
    EvalInput,
    EvalReport,
    ExperimentConfig,
    SingleEvalResult,
)
from llm_agent.evaluation.dataset_picking_agent import (
    DatasetPickingAgentFailure,
)
from llm_agent.evaluation.utils import SummaryStats
from llm_agent.utils.prompt_as_service import PromptServiceAdapter
from llm_agent.utils.utils import EnrichedValidationError

log = logging.getLogger(__name__)


class _DatasetPickingSingleTurnExperimentConfig(BaseModel):
    """Name and parameters of experiment."""

    name: str
    dataset_id: str
    dataset_version: str
    model_name: str
    temperature: float

    @staticmethod
    def from_eval_exp_config(
        experiment_config: ExperimentConfig,
    ) -> "_DatasetPickingSingleTurnExperimentConfig":
        """Factory to build from base ExperimentConfig."""
        return _DatasetPickingSingleTurnExperimentConfig.model_validate(
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
            },
        )


class DatasetPickingSingleTurnResponseAttrs(BaseModel):
    """Attributes extracted from response."""

    is_tool_call: bool
    tool_call_args: dict[str, Any] | None


class DatasetPickingSingleTurnEvalInput(BaseModel):
    """A single evaluation test input and reference output.

    Eventually, it might make sense for all dataset picking agent evals to use
    this form, as it evaluates whether a single call to the LLM returns the
    expected result.

    At the very least, this eval type is useful to validate that the LLM makes
    expected tool calls with the expected arguments.
    """

    id: str
    natural_language_query: str
    expected_response_attrs: DatasetPickingSingleTurnResponseAttrs

    @staticmethod
    def from_eval_input(
        eval_input: EvalInput,
    ) -> "DatasetPickingSingleTurnEvalInput":
        """Factory from base `EvalInput`."""
        if type(eval_input.reference_output) is str:
            raise NotImplementedError

        return DatasetPickingSingleTurnEvalInput(
            id=eval_input.id,
            natural_language_query=cast("str", eval_input.input),
            expected_response_attrs=DatasetPickingSingleTurnResponseAttrs.model_validate(
                eval_input.reference_output
            ),
        )

    def to_eval_input(self) -> EvalInput:
        """Convert back to base `EvalInput`."""
        return EvalInput(
            id=self.id,
            input=self.natural_language_query,
            reference_output=self.expected_response_attrs.model_dump(mode="json"),
            attrs=None,
        )


class DatasetPickingSingleTurnEvalMetrics(BaseModel):
    """Metrics for a single eval."""

    is_correct_message_type: bool
    tool_call_has_expected_args: bool | None


class DatasetPickingSingleTurnAggregatedMetrics(BaseModel):
    """Aggregated metrics."""

    pct_correct_message_type: float
    pct_correct_tool_call_args: float


class DatasetPickingAgentSingleTurnSingleEvalResult(BaseModel):
    """Results for a single eval test case."""

    input: DatasetPickingSingleTurnEvalInput
    output: DatasetPickingAgentMessage | DatasetPickingAgentFailure
    metrics: DatasetPickingSingleTurnEvalMetrics

    @staticmethod
    def from_single_eval_result(
        eval_result: SingleEvalResult,
    ) -> "DatasetPickingAgentSingleTurnSingleEvalResult":
        eval_input = DatasetPickingSingleTurnEvalInput.from_eval_input(eval_result.input)
        return DatasetPickingAgentSingleTurnSingleEvalResult.model_validate(
            {
                "input": eval_input,
                "output": eval_result.output,
                "metrics": eval_result.metrics,
            }
        )

    def to_single_eval_result(self) -> SingleEvalResult:
        """Convert back to `SingleEvalResult`."""
        output: dict[str, Any] | None = self.output.model_dump()

        return SingleEvalResult(
            input=self.input.to_eval_input(),
            output=output,
            metrics=self.metrics.model_dump(),
        )


class DatasetPickingSingleTurnEvalReport(BaseModel):
    """Eval report."""

    experiment_config: _DatasetPickingSingleTurnExperimentConfig
    results: list[DatasetPickingAgentSingleTurnSingleEvalResult]
    aggregated_metrics: DatasetPickingSingleTurnAggregatedMetrics

    @staticmethod
    def from_eval_report(
        eval_report: EvalReport,
    ) -> "DatasetPickingSingleTurnEvalReport":
        if eval_report.results is None:
            raise ValueError

        return DatasetPickingSingleTurnEvalReport(
            experiment_config=_DatasetPickingSingleTurnExperimentConfig.from_eval_exp_config(
                eval_report.experiment_config
            ),
            results=[
                DatasetPickingAgentSingleTurnSingleEvalResult.from_single_eval_result(r)
                for r in eval_report.results
            ],
            aggregated_metrics=DatasetPickingSingleTurnAggregatedMetrics.model_validate(
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


class DatasetPickingSingleTurnMultiRunMetrics(BaseModel):
    """Aggregated metrics, averaged over multiple experiment runs."""

    num_runs: int
    pct_correct_message_type: SummaryStats
    pct_correct_tool_call_args: SummaryStats


class DatasetPickingSingleTurnMultiRunEvalReport(BaseModel):
    """Eval report aggregated over multiple experiment runs."""

    experiment_config: _DatasetPickingSingleTurnExperimentConfig
    aggregated_metrics: DatasetPickingSingleTurnMultiRunMetrics

    @staticmethod
    def from_eval_report(
        eval_report: EvalReport,
    ) -> "DatasetPickingSingleTurnMultiRunEvalReport":
        return DatasetPickingSingleTurnMultiRunEvalReport(
            experiment_config=_DatasetPickingSingleTurnExperimentConfig.from_eval_exp_config(
                eval_report.experiment_config
            ),
            aggregated_metrics=DatasetPickingSingleTurnMultiRunMetrics.model_validate(
                eval_report.aggregated_metrics
            ),
        )

    def to_eval_report(self) -> EvalReport:
        return EvalReport(
            experiment_config=self.experiment_config.to_eval_exp_config(),
            results=None,
            aggregated_metrics=self.aggregated_metrics.model_dump(),
        )


class DatasetPickingAgentSingleTurnEvaluator(BaseEvaluator):
    """Dataset Picking Agent Single Turn Evaluator."""

    async def _eval_dataset(
        self,
        experiment_config: ExperimentConfig,
        dataset: EvalDataset,
    ) -> EvalReport:
        """Run eval dataset with a given experiment config."""
        typed_dataset = [
            DatasetPickingSingleTurnEvalInput.from_eval_input(item) for item in dataset.items
        ]

        typed_experiment_config = _DatasetPickingSingleTurnExperimentConfig.from_eval_exp_config(
            experiment_config
        )

        typed_report = await self._eval_dataset_typed(typed_experiment_config, typed_dataset)
        eval_report = typed_report.to_eval_report()
        return eval_report

    async def _eval_dataset_typed(
        self,
        experiment_config: _DatasetPickingSingleTurnExperimentConfig,
        dataset: list[DatasetPickingSingleTurnEvalInput],
    ) -> DatasetPickingSingleTurnEvalReport:
        llm = BedrockClaudeLLM(model_id=CLAUDE_BEDROCK_MODEL_IDS[experiment_config.model_name])
        inference_cfg = LLMInferenceConfig(temperature=experiment_config.temperature)
        agent = DatasetPickingAgent(
            prompt_service=PromptServiceAdapter(llm.prompt), inference_cfg=inference_cfg
        )
        with tqdm(dataset, desc="Evaluating") as bar:
            results = [await self._eval_single_case(agent, case) for case in bar]
        return DatasetPickingSingleTurnEvalReport(
            experiment_config=experiment_config,
            results=results,
            aggregated_metrics=_aggregate_metrics(results),
        )

    async def _eval_single_case(
        self,
        agent: DatasetPickingAgent,
        case: DatasetPickingSingleTurnEvalInput,
    ) -> DatasetPickingAgentSingleTurnSingleEvalResult:
        try:
            message = await agent.run_single(
                [LLMUserMessage(content=[TextContent(text=case.natural_language_query)])]
            )
        except JSONDecodeError as e:
            message = "Unable to parse JSON: %s"
            log.exception(message, e.doc)
            return DatasetPickingAgentSingleTurnSingleEvalResult(
                input=case,
                output=DatasetPickingAgentFailure(
                    name=type(e).__name__, message=message, failed_output=e.doc
                ),
                metrics=DatasetPickingSingleTurnEvalMetrics(
                    is_correct_message_type=False,
                    tool_call_has_expected_args=None,
                ),
            )
        except EnrichedValidationError as e:
            message = "JSON parseable but did not fit model schema."
            log.exception(message)
            return DatasetPickingAgentSingleTurnSingleEvalResult(
                input=case,
                output=DatasetPickingAgentFailure(
                    name=type(e).__name__,
                    message=message,
                    failed_output=e.data,
                ),
                metrics=DatasetPickingSingleTurnEvalMetrics(
                    is_correct_message_type=False,
                    tool_call_has_expected_args=None,
                ),
            )

        response_attrs = _response_attrs_for_message(message)
        is_correct_message_type = (
            response_attrs.is_tool_call == case.expected_response_attrs.is_tool_call
        )

        metrics = DatasetPickingSingleTurnEvalMetrics(
            is_correct_message_type=is_correct_message_type,
            tool_call_has_expected_args=_tool_call_has_expected_attrs(
                response_attrs, case.expected_response_attrs
            ),
        )

        return DatasetPickingAgentSingleTurnSingleEvalResult(
            input=case,
            output=message,
            metrics=metrics,
        )

    async def _aggregate_eval_reports(self, reports: list[EvalReport]) -> EvalReport:
        reports_custom = [DatasetPickingSingleTurnEvalReport.from_eval_report(r) for r in reports]
        exp_config = reports_custom[0].experiment_config
        run_metrics = [r.aggregated_metrics for r in reports_custom]
        keys = DatasetPickingSingleTurnAggregatedMetrics.model_fields.keys()
        summary_stats = {
            k: SummaryStats.from_values([getattr(m, k) for m in run_metrics]) for k in keys
        }
        aggregated_metrics = DatasetPickingSingleTurnMultiRunMetrics.model_validate(
            {"num_runs": len(reports), **summary_stats}
        )
        return DatasetPickingSingleTurnMultiRunEvalReport(
            experiment_config=exp_config,
            aggregated_metrics=aggregated_metrics,
        ).to_eval_report()


def _tool_call_has_expected_attrs(
    response_attrs: DatasetPickingSingleTurnResponseAttrs,
    expected_response_attrs: DatasetPickingSingleTurnResponseAttrs,
) -> bool | None:
    if response_attrs.is_tool_call and expected_response_attrs.is_tool_call:
        return response_attrs.tool_call_args == expected_response_attrs.tool_call_args

    if response_attrs.is_tool_call != expected_response_attrs.is_tool_call:
        return False

    if not response_attrs.is_tool_call and not expected_response_attrs.is_tool_call:
        return None

    log.error("Evaluation state should not be reachable. Check logic.")
    raise NotImplementedError


def _response_attrs_for_message(
    message: DatasetPickingAgentMessage,
) -> DatasetPickingSingleTurnResponseAttrs:
    tool_call_contents = [
        content for content in message.parsed_content if isinstance(content, LLMToolUseContent)
    ]

    is_tool_call = len(tool_call_contents) > 0

    if not is_tool_call:
        return DatasetPickingSingleTurnResponseAttrs(is_tool_call=False, tool_call_args=None)

    if len(tool_call_contents) > 1:
        log.error(
            "Evaluation doesn't support case where LLM responds with multiple "
            "tool calls in a single message. If this occurs, please update "
            "the evaluation code!"
        )
        raise NotImplementedError

    return DatasetPickingSingleTurnResponseAttrs(
        is_tool_call=is_tool_call, tool_call_args=tool_call_contents[0].input
    )


def _aggregate_metrics(
    results: list[DatasetPickingAgentSingleTurnSingleEvalResult],
) -> DatasetPickingSingleTurnAggregatedMetrics:
    return DatasetPickingSingleTurnAggregatedMetrics(
        pct_correct_message_type=statistics.mean(
            [result.metrics.is_correct_message_type for result in results]
        )
        * 100.0,
        pct_correct_tool_call_args=statistics.mean(
            [
                result.metrics.tool_call_has_expected_args
                for result in results
                if result.metrics.tool_call_has_expected_args is not None
            ]
        )
        * 100.0,
    )
