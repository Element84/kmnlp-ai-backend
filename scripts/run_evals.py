#!/usr/bin/env python

import argparse
import importlib.util
import logging
import os
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from os.path import join
from pathlib import Path
from typing import Literal

import anyio
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from llm_agent.evaluation.algorithm_agent import AlgorithmAgentEvaluator
from llm_agent.evaluation.base_eval import (
    BaseEvaluator,
    EvalDataset,
    ExperimentConfig,
)
from llm_agent.evaluation.dataset_picking_agent import (
    DatasetPickingAgentEvaluator,
)
from llm_agent.evaluation.dataset_picking_agent_single_turn import (
    DatasetPickingAgentSingleTurnEvaluator,
)

log = logging.getLogger(__name__)

DEFAULT_LOCAL_OUTPUT_ROOT = str((Path(__file__).parent / "../evals/outputs").resolve())
DEFAULT_RUNS_PER_EXPERIMENT = 3


class TaskEvalConfig(BaseModel):
    """Config for evaluating an LLM task."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    task_id: Literal[
        "algorithm_agent",
        "dataset_picking_agent",
        "dataset_picking_agent_single_turn",
    ]
    evaluator_class: type[BaseEvaluator]
    dataset_path: Path
    experiments: dict[str, ExperimentConfig]
    runs_per_experiment: int = DEFAULT_RUNS_PER_EXPERIMENT


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
        dataset_path=Path("evals/datasets/algorithm_agent.py"),
        experiments={
            "claude_4_sonnet": ExperimentConfig(
                name="claude_4_sonnet",
                dataset_id="default",
                dataset_version="1",
                params={
                    "model_name": "Claude 4 Sonnet",
                    "temperature": 0.0,
                },
            ),
        },
    ),
    "dataset_picking_agent": TaskEvalConfig(
        task_id="dataset_picking_agent",
        evaluator_class=DatasetPickingAgentEvaluator,
        dataset_path=Path("evals/datasets/dataset_picking_agent.py"),
        experiments={
            "claude_4_sonnet_fetch_tool": ExperimentConfig(
                name="claude_4_sonnet_fetch_tool",
                dataset_id="default",
                dataset_version="1",
                params={
                    "model_name": "Claude 4 Sonnet",
                    "temperature": 0.0,
                    "available_tool_names": ["fetch_datasets_tool"],
                },
            ),
            "claude_4_sonnet_sparql_tool": ExperimentConfig(
                name="claude_4_sonnet_sparql_tool",
                dataset_id="default",
                dataset_version="1",
                params={
                    "model_name": "Claude 4 Sonnet",
                    "temperature": 0.0,
                    "available_tool_names": ["execute_sparql_query_tool"],
                },
            ),
            "claude_4_sonnet_fetch_and_sparql_tools": ExperimentConfig(
                name="claude_4_sonnet_fetch_and_sparql_tools",
                dataset_id="default",
                dataset_version="1",
                params={
                    "model_name": "Claude 4 Sonnet",
                    "temperature": 0.0,
                    "available_tool_names": [
                        "fetch_datasets_tool",
                        "execute_sparql_query_tool",
                    ],
                },
            ),
        },
    ),
    "dataset_picking_agent_single_turn": TaskEvalConfig(
        task_id="dataset_picking_agent_single_turn",
        evaluator_class=DatasetPickingAgentSingleTurnEvaluator,
        dataset_path=Path("evals/datasets/dataset_picking_agent_single_turn.py"),
        experiments={
            "claude_4_sonnet": ExperimentConfig(
                name="claude_4_sonnet",
                dataset_id="default",
                dataset_version="2",
                params={
                    "model_name": "Claude 4 Sonnet",
                    "temperature": 0.0,
                },
            ),
        },
    ),
}


async def run(
    task_ids: Iterable[str],
    exp_names: Iterable[str] | None,
    output_root: str,
    tag: str | None,
) -> None:
    """Async main."""
    task_ids = task_ids or list(task_evals.keys())
    for task_id in task_ids:
        await evaluate_task(task_id, exp_names=exp_names, output_root=output_root, tag=tag)


async def evaluate_task(
    task_id: str,
    *,
    exp_names: Iterable[str] | None,
    output_root: str,
    tag: str | None = None,
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
                task_eval_cfg, exp_name, output_root=output_root, tag=tag
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


def _load_dataset(path: Path) -> EvalDataset:
    """Load the variable named "dataset" from the given Python file."""
    abs_path = path.resolve()
    module_name = path.stem

    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None:
        msg = f"Could not create spec for {abs_path}"
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


def _get_experiment_exec_config(
    task: TaskEvalConfig,
    exp_name: str,
    *,
    output_root: str,
    tag: str | None = None,
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
    return ExperimentExecutionConfig(
        runs=[
            ExperimentSingleRunConfig(
                dataset=dataset,
                experiment_config=task.experiments[exp_name],
                output_path=join(exp_output_dir, f"{i}.json"),
            )
            for i in range(task.runs_per_experiment)
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

    args = parser.parse_args()
    task_ids: set[str] = set(args.task_ids)
    experiments: set[str] | None = set(args.experiments) if args.experiments else None
    local: bool = not bool(args.remote)
    tag: str | None = args.tag

    output_root = os.getenv("EVALS_OUTPUT_ROOT")
    if not output_root:
        if local:
            output_root = DEFAULT_LOCAL_OUTPUT_ROOT
        else:
            msg = "EVALS_OUTPUT_ROOT must be set if running with --remote."
            raise ValueError(msg)

    log.info("Evaluating tasks: %s", task_ids)
    log.info("Output root dir: %s", output_root)

    anyio.run(run, task_ids, experiments, output_root, tag)
