import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from llm_agent.utils.utils import model_from_file, model_to_file

log = logging.getLogger(__name__)


class ExperimentConfig(BaseModel):
    """Name and parameters of the experiment."""

    name: str
    dataset_id: str
    dataset_version: str
    params: dict[str, Any]


class EvalInput(BaseModel):
    """A single evaluation test input and associated reference output."""

    id: str
    input: str | dict[str, Any]
    reference_output: str | dict[str, Any]
    attrs: dict[str, Any] | None


class EvalDataset(BaseModel):
    """An evaluation dataset."""

    id: str
    version: str
    items: list[EvalInput]


class SingleEvalResult(BaseModel):
    """Evaluation result for a single test case."""

    input: EvalInput
    output: str | dict[str, Any] | list[dict[str, Any]] | None
    metrics: dict[str, Any]


class EvalReport(BaseModel):
    """Evaluation report for an experiment."""

    experiment_config: ExperimentConfig
    results: list[SingleEvalResult] | None
    aggregated_metrics: dict[str, Any]


class BaseEvaluator(ABC):
    """Base eval class."""

    async def eval_dataset(
        self, experiment_config: ExperimentConfig, dataset: EvalDataset
    ) -> EvalReport:
        """Evaluate a dataset using the given experiment config."""
        if not (
            experiment_config.dataset_id == dataset.id
            and experiment_config.dataset_version == dataset.version
        ):
            msg = (
                "Mismatch between the dataset provided and the "
                "dataset configured in the experiment config."
            )
            raise ValueError(msg)

        return await self._eval_dataset(experiment_config, dataset)

    @abstractmethod
    async def _eval_dataset(
        self, experiment_config: ExperimentConfig, dataset: EvalDataset
    ) -> EvalReport:
        """Evaluate a dataset using the given experiment config."""

    def save_eval_report(self, eval_report: EvalReport, path: str) -> None:
        """Save eval report as a JSON file."""
        model_to_file(path, eval_report)

    async def aggregate_eval_reports(self, report_paths: list[str], output_path: str) -> None:
        """Aggregate eval reports from multiple experiment runs."""
        if not report_paths:
            raise ValueError

        reports = [self._load_eval_report(path) for path in report_paths]

        unique_exps = {
            (
                r.experiment_config.name,
                r.experiment_config.dataset_id,
                r.experiment_config.dataset_version,
            )
            for r in reports
        }
        if len(unique_exps) > 1:
            msg = (
                "All reports must have the same experiment name, dataset ID, "
                f"and dataset version. Found {unique_exps}."
            )
            raise ValueError(msg)

        aggregated_report = await self._aggregate_eval_reports(reports)
        self.save_eval_report(aggregated_report, output_path)

    @abstractmethod
    async def _aggregate_eval_reports(self, reports: list[EvalReport]) -> EvalReport:
        """Aggregate eval reports from multiple experiment runs."""

    def _load_eval_report(self, path: str) -> EvalReport:
        return model_from_file(path, EvalReport)
