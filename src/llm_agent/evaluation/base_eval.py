import logging
from abc import ABC, abstractmethod
from pathlib import Path
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

        log.info(
            "Evaluating dataset %s version %s with experiment config %s",
            dataset.id,
            dataset.version,
            experiment_config,
        )
        return await self._eval_dataset(experiment_config, dataset)

    @abstractmethod
    async def _eval_dataset(
        self, experiment_config: ExperimentConfig, dataset: EvalDataset
    ) -> EvalReport:
        """Evaluate a dataset using the given experiment config."""

    def save_eval_report(self, eval_report: EvalReport, path: str) -> None:
        """Save eval report as a JSON file."""
        log.info("Saving eval report to %s", path)
        model_to_file(path, eval_report)

    async def aggregate_eval_reports(self, report_paths: list[str], output_path: str) -> None:
        """Aggregate eval reports from multiple experiment runs."""
        if not report_paths:
            raise ValueError("No report paths provided.")

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

        log.info(
            "Aggregating %d eval reports for experiment '%s' on dataset %s",
            len(reports),
            reports[0].experiment_config.name,
            reports[0].experiment_config.dataset_id,
        )
        aggregated_report = await self._aggregate_eval_reports(reports)
        self.save_eval_report(aggregated_report, output_path)

    @abstractmethod
    async def _aggregate_eval_reports(self, reports: list[EvalReport]) -> EvalReport:
        """Aggregate eval reports from multiple experiment runs."""

    async def reevaluate_experiment_dir(self, experiment_dir: str, dataset: EvalDataset) -> None:
        """Recompute metrics for a completed experiment run using the current eval logic.

        Loads each per-run report (`0.json`, `1.json`, ...) in `experiment_dir`, re-scores the
        already-generated outputs against `dataset` (matched by case id), rewrites each report in
        place, then regenerates `aggregated.json`. The agent is never re-run, so this is a cheap
        way to apply evaluation-logic changes to previously generated solutions.
        """
        run_paths = self._find_run_report_paths(experiment_dir)
        if not run_paths:
            msg = f"No per-run report files (0.json, 1.json, ...) found in {experiment_dir}."
            raise ValueError(msg)

        for path in run_paths:
            report = self._load_eval_report(path)
            cfg = report.experiment_config
            if cfg.dataset_id != dataset.id or cfg.dataset_version != dataset.version:
                msg = (
                    f"Report {path} was produced for dataset '{cfg.dataset_id}' "
                    f"v{cfg.dataset_version}, but the current dataset is '{dataset.id}' "
                    f"v{dataset.version}. Re-evaluation requires a matching dataset."
                )
                raise ValueError(msg)
            rescored = await self._reevaluate_report(report, dataset)
            self.save_eval_report(rescored, path)
            log.info("Re-evaluated %s", path)

        aggregated_path = str(Path(experiment_dir) / "aggregated.json")
        await self.aggregate_eval_reports(run_paths, aggregated_path)

    async def _reevaluate_report(self, report: EvalReport, dataset: EvalDataset) -> EvalReport:
        """Recompute a single run report's metrics from its stored outputs.

        Overridden per evaluator. Must reuse the report's outputs and must not re-run the agent.
        """
        msg = f"{type(self).__name__} does not support re-evaluation."
        raise NotImplementedError(msg)

    @staticmethod
    def _find_run_report_paths(experiment_dir: str) -> list[str]:
        """Return per-run report paths (`0.json`, `1.json`, ...) in ascending order."""
        dir_path = Path(experiment_dir)
        if not dir_path.is_dir():
            msg = f"Not a directory: {experiment_dir}"
            raise NotADirectoryError(msg)
        run_files = [p for p in dir_path.glob("*.json") if p.stem.isdigit()]
        return [str(p) for p in sorted(run_files, key=lambda p: int(p.stem))]

    def _load_eval_report(self, path: str) -> EvalReport:
        return model_from_file(path, EvalReport)
