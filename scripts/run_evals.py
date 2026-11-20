#!/usr/bin/env python

import argparse
import importlib.util
import logging
import os
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import partial
from os.path import join
from pathlib import Path
from typing import Literal

import anyio
from e84_geoai_common.llm.models.claude import CLAUDE_4_5_HAIKU, CLAUDE_4_6_OPUS, CLAUDE_4_6_SONNET
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from llm_agent.evaluation.algorithm_agent import AlgorithmAgentEvaluator
from llm_agent.evaluation.base_eval import BaseEvaluator, EvalDataset, ExperimentConfig
from llm_agent.evaluation.dataset_picking_agent import DatasetPickingAgentEvaluator

log = logging.getLogger(__name__)

DEFAULT_LOCAL_OUTPUT_ROOT = str((Path(__file__).parent / "../evals/outputs").resolve())
DEFAULT_RUNS_PER_EXPERIMENT = 3


class TaskEvalConfig(BaseModel):
    """Config for evaluating an LLM task."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    task_id: Literal[
        "algorithm_agent",
        "dataset_picking_agent",
    ]
    evaluator_class: type[BaseEvaluator]
    dataset_path: Path | str
    experiments: dict[str, ExperimentConfig]


class ExperimentSingleRunConfig(BaseModel):
    """Config for a single experiment run."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    dataset: EvalDataset
    experiment_config: ExperimentConfig
    output_path: str


class ExperimentExecutionConfig(BaseModel):
    """Config for a single experiment run."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    runs: list[ExperimentSingleRunConfig]
    aggregated_output_path: str


task_evals = {
    "algorithm_agent": TaskEvalConfig(
        task_id="algorithm_agent",
        evaluator_class=AlgorithmAgentEvaluator,
        dataset_path="evals.datasets.algorithm_agent",
        experiments={
            "claude_4_5_haiku_no_retry": ExperimentConfig(
                name="claude_4_5_haiku_no_retry",
                dataset_id="default",
                dataset_version="6",
                params={
                    "model_name": CLAUDE_4_5_HAIKU,
                    "temperature": 0.0,
                    "thinking": False,
                    "num_retries": 0,
                },
            ),
            "claude_4_6_sonnet_no_retry": ExperimentConfig(
                name="claude_4_6_sonnet_no_retry",
                dataset_id="default",
                dataset_version="6",
                params={
                    "model_name": CLAUDE_4_6_SONNET,
                    "temperature": 0.0,
                    "thinking": False,
                    "num_retries": 0,
                },
            ),
            "claude_4_6_opus_no_retry": ExperimentConfig(
                name="claude_4_6_opus_no_retry",
                dataset_id="default",
                dataset_version="6",
                params={
                    "model_name": CLAUDE_4_6_OPUS,
                    "temperature": 0.0,
                    "thinking": False,
                    "num_retries": 0,
                },
            ),
            "claude_4_5_haiku": ExperimentConfig(
                name="claude_4_5_haiku",
                dataset_id="default",
                dataset_version="6",
                params={
                    "model_name": CLAUDE_4_5_HAIKU,
                    "temperature": 0.0,
                    "thinking": False,
                },
            ),
            "claude_4_6_sonnet": ExperimentConfig(
                name="claude_4_6_sonnet",
                dataset_id="default",
                dataset_version="6",
                params={
                    "model_name": CLAUDE_4_6_SONNET,
                    "temperature": 0.0,
                    "thinking": False,
                },
            ),
            "claude_4_6_opus": ExperimentConfig(
                name="claude_4_6_opus",
                dataset_id="default",
                dataset_version="6",
                params={
                    "model_name": CLAUDE_4_6_OPUS,
                    "temperature": 0.0,
                    "thinking": False,
                },
            ),
            "claude_4_5_haiku_thinking": ExperimentConfig(
                name="claude_4_5_haiku_thinking",
                dataset_id="default",
                dataset_version="6",
                params={
                    "model_name": CLAUDE_4_5_HAIKU,
                    "temperature": 1.0,
                    "thinking": "high",
                },
            ),
            "claude_4_6_sonnet_thinking": ExperimentConfig(
                name="claude_4_6_sonnet_thinking",
                dataset_id="default",
                dataset_version="6",
                params={
                    "model_name": CLAUDE_4_6_SONNET,
                    "temperature": 1.0,
                    "thinking": "high",
                },
            ),
            "claude_4_6_opus_thinking": ExperimentConfig(
                name="claude_4_6_opus_thinking",
                dataset_id="default",
                dataset_version="6",
                params={
                    "model_name": CLAUDE_4_6_OPUS,
                    "temperature": 1.0,
                    "thinking": "high",
                },
            ),
        },
    ),
    "dataset_picking_agent": TaskEvalConfig(
        task_id="dataset_picking_agent",
        evaluator_class=DatasetPickingAgentEvaluator,
        dataset_path=Path("evals/datasets/dataset_picking_agent.py"),
        experiments={
            "claude_4_6_opus_fetch_tool": ExperimentConfig(
                name="claude_4_6_opus_fetch_tool",
                dataset_id="default",
                dataset_version="2",
                params={
                    "model_name": CLAUDE_4_6_OPUS,
                    "temperature": 0.0,
                    "available_tool_names": ["fetch_datasets_tool"],
                },
            ),
            "claude_4_6_opus_sparql_tool": ExperimentConfig(
                name="claude_4_6_opus_sparql_tool",
                dataset_id="default",
                dataset_version="2",
                params={
                    "model_name": CLAUDE_4_6_OPUS,
                    "temperature": 0.0,
                    "available_tool_names": ["execute_sparql_query_tool"],
                },
            ),
            "claude_4_6_opus_fetch_and_sparql_tools": ExperimentConfig(
                name="claude_4_6_opus_fetch_and_sparql_tools",
                dataset_id="default",
                dataset_version="2",
                params={
                    "model_name": CLAUDE_4_6_OPUS,
                    "temperature": 0.0,
                    "available_tool_names": [
                        "fetch_datasets_tool",
                        "execute_sparql_query_tool",
                    ],
                },
            ),
        },
    ),
}


async def run(  # noqa: PLR0913
    task_ids: Iterable[str],
    exp_names: Iterable[str] | None,
    output_root: str,
    tag: str | None,
    *,
    runs: int,
    test_run: bool = False,
) -> None:
    """Async main."""
    task_ids = task_ids or list(task_evals.keys())
    for task_id in task_ids:
        await evaluate_task(
            task_id,
            exp_names=exp_names,
            output_root=output_root,
            tag=tag,
            runs=runs,
            test_run=test_run,
        )


def _resolve_reeval_task_id(experiment_dir: str) -> str:
    """Infer the task id from an experiment output directory path."""
    parts = Path(experiment_dir).resolve().parts
    task_id = next((tid for tid in task_evals if tid in parts), None)
    if task_id is None:
        msg = (
            f"Could not determine the task from '{experiment_dir}'. Expected a path like "
            ".../<task_id>/dataset/<id>/version/<v>/<exp>/<timestamp>/"
        )
        raise ValueError(msg)
    return task_id


async def reevaluate(experiment_dir: str) -> None:
    """Re-score an already-completed experiment run using the current eval logic.

    Loads the per-run report JSONs in `experiment_dir`, recomputes metrics from the stored
    outputs against the current dataset, and rewrites the reports (plus the aggregate). The
    agent is not re-run.
    """
    task_id = _resolve_reeval_task_id(experiment_dir)
    task_eval_cfg = task_evals[task_id]
    evaluator = task_eval_cfg.evaluator_class()
    dataset = _load_dataset(task_eval_cfg.dataset_path)
    log.info("Re-evaluating '%s' outputs in %s", task_id, experiment_dir)
    await evaluator.reevaluate_experiment_dir(experiment_dir, dataset)


async def evaluate_task(  # noqa: PLR0913
    task_id: str,
    *,
    exp_names: Iterable[str] | None,
    output_root: str,
    tag: str | None = None,
    runs: int = 1,
    test_run: bool = False,
) -> None:
    """Evaluate task."""
    task_eval_cfg = task_evals[task_id]
    evaluator = task_eval_cfg.evaluator_class()
    exp_names = exp_names or list(task_eval_cfg.experiments.keys())

    log.info("Evaluating task: %s", task_id)
    log.info("Experiments: %s", exp_names)
    log.info("Output dir: %s", output_root)

    with tqdm(exp_names, desc=f"Evaluating {task_id}") as bar:
        for exp_name in bar:
            exp_exec_cfg = _get_experiment_exec_config(
                task_eval_cfg,
                exp_name,
                output_root=output_root,
                tag=tag,
                runs=runs,
                test_run=test_run,
            )
            bar.set_postfix_str(exp_name)
            await _run_experiment(evaluator=evaluator, exp_exec_cfg=exp_exec_cfg)


async def _run_experiment(
    evaluator: BaseEvaluator, exp_exec_cfg: ExperimentExecutionConfig
) -> None:
    """Run experiment for a task."""
    with tqdm(exp_exec_cfg.runs, desc="Executing experiment runs") as bar:
        for cfg in bar:
            await _execute_experiment_run(cfg, evaluator)

    await evaluator.aggregate_eval_reports(
        report_paths=[cfg.output_path for cfg in exp_exec_cfg.runs],
        output_path=exp_exec_cfg.aggregated_output_path,
    )


async def _execute_experiment_run(cfg: ExperimentSingleRunConfig, evaluator: BaseEvaluator) -> None:
    """Execute a single experiment run."""
    eval_report = await evaluator.eval_dataset(cfg.experiment_config, cfg.dataset)
    evaluator.save_eval_report(eval_report, cfg.output_path)


def _load_dataset(filepath_or_module: Path | str) -> EvalDataset:
    """Load the variable named "dataset" from a Python file or module.

    Args:
        filepath_or_module: Path to the Python file or a fully specified module
            path like "abc.def.ghi".

    Returns:
        EvalDataset: The loaded dataset.
    """
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    match filepath_or_module:
        case Path():
            abs_path = filepath_or_module.resolve()
            module_name = filepath_or_module.stem
            if (spec := importlib.util.spec_from_file_location(module_name, abs_path)) is None:
                msg = f"Could not create spec for {abs_path}"
                raise ImportError(msg)
        case str():
            module_name = filepath_or_module
            if (spec := importlib.util.find_spec(module_name)) is None:
                msg = f"Could not find module {module_name}"
                raise ImportError(msg)

    if spec.loader is None:
        msg = "spec.loader is None"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    dataset = module.dataset
    if not isinstance(dataset, EvalDataset):
        msg = "dataset is not an EvalDataset."
        raise TypeError(msg)

    return dataset


def _get_experiment_exec_config(  # noqa: PLR0913
    task: TaskEvalConfig,
    exp_name: str,
    *,
    output_root: str,
    tag: str | None = None,
    runs: int = 1,
    test_run: bool = False,
) -> ExperimentExecutionConfig:
    # s3://<bucket>/
    #   <task_id>/
    #       dataset/
    #           <dataset_id>/
    #               version/
    #                   <version>/
    #                       <exp_name>/
    #                           <timestamp>[_<tag>]/
    #                               0.json
    #                               1.json
    #                               ...
    #                               N.json
    #                               aggregated.json
    dataset = _load_dataset(task.dataset_path)
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H%M")
    exp_output_dir = join(
        output_root,
        task.task_id,
        "dataset",
        dataset.id,
        "version",
        str(dataset.version),
        exp_name,
        f"{timestamp}_{tag}" if tag else timestamp,
    )

    if test_run:
        runs = 1
        dataset.items = dataset.items[:3]

    return ExperimentExecutionConfig(
        runs=[
            ExperimentSingleRunConfig(
                dataset=dataset,
                experiment_config=task.experiments[exp_name],
                output_path=join(exp_output_dir, f"{i}.json"),
            )
            for i in range(runs)
        ],
        aggregated_output_path=join(exp_output_dir, "aggregated.json"),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("llm_agent").setLevel(logging.INFO)
    logging.getLogger("e84_geoai_common").setLevel(logging.WARNING)
    logging.getLogger("langfuse").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Run evaluations.")
    parser.add_argument(
        "--task_ids",
        nargs="+",
        type=str,
        default=task_evals.keys(),
        help="The tasks to evaluate. If not provided, all configured task evals will be run.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=str,
        help="Names of experiments to run. If not provided, all experiments "
        "configured for the task are run.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Optional tag to append to the timestamp dir: <timestamp>_<tag>.",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="If set, evals are written to S3. Otherwise, to ../evals/outputs/.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS_PER_EXPERIMENT,
        help="Number of runs per experiment.",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="If set, only 1 run is executed and the dataset is truncated to 3 items.",
    )
    parser.add_argument(
        "--reevaluate-dir",
        type=str,
        help="Path to a completed experiment run directory (containing 0.json, 1.json, ..., "
        "aggregated.json). If set, re-scores the already-generated outputs using the current "
        "eval logic and rewrites the JSONs in place instead of running the agent. The task is "
        "inferred from the path.",
    )

    args = parser.parse_args()

    if args.reevaluate_dir:
        anyio.run(partial(reevaluate, experiment_dir=args.reevaluate_dir))
        raise SystemExit(0)

    task_ids: set[str] = set(args.task_ids)
    experiments: set[str] | None = set(args.experiments) if args.experiments else None
    local: bool = not bool(args.remote)
    tag: str | None = args.tag
    runs: int = args.runs
    test_run: bool = bool(args.test_run)

    output_root = os.getenv("EVALS_OUTPUT_ROOT")
    if not output_root:
        if local:
            output_root = DEFAULT_LOCAL_OUTPUT_ROOT
        else:
            msg = "EVALS_OUTPUT_ROOT must be set if running with --remote."
            raise ValueError(msg)

    if test_run:
        log.info("[TEST RUN] Using runs=1 and truncating datasets to 3 items.")

    log.info("Evaluating tasks: %s", task_ids)
    log.info("Output root dir: %s", output_root)

    anyio.run(
        partial(
            run,
            task_ids=task_ids,
            exp_names=experiments,
            output_root=output_root,
            tag=tag,
            runs=runs,
            test_run=test_run,
        )
    )
