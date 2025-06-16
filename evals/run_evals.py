import importlib.util
import logging
import sys
from datetime import UTC, datetime
from os.path import join
from pathlib import Path
from typing import TypedDict

from anyio import run

from llm_agent.evaluation.algorithm_agent import AlgorithmAgentEvaluator
from llm_agent.evaluation.base_eval import (
    BaseEvaluator,
    EvalDataset,
    ExperimentConfig,
)
from llm_agent.evaluation.dataset_picking_agent import (
    DatasetPickingAgentEvaluator,
)


class ExperimentExecutionConfig(TypedDict):
    """Experiment execution config."""

    evaluator_class: type[BaseEvaluator]
    dataset_path: Path
    config: ExperimentConfig
    output_dir: str


experiments = [
    ExperimentExecutionConfig(
        evaluator_class=AlgorithmAgentEvaluator,
        dataset_path=Path("evals/datasets/algorithm_agent.py"),
        config=ExperimentConfig(
            name="algo_agent_claude_35_haiku",
            params={
                "model_name": "Claude 3.5 Haiku",
                "temperature": 0.0,
            },
        ),
        output_dir="evals/outputs/",
    ),
    ExperimentExecutionConfig(
        evaluator_class=AlgorithmAgentEvaluator,
        dataset_path=Path("evals/datasets/algorithm_agent.py"),
        config=ExperimentConfig(
            name="algo_agent_claude_35_sonnet",
            params={
                "model_name": "Claude 3.5 Sonnet",
                "temperature": 0.0,
            },
        ),
        output_dir="evals/outputs/",
    ),
    ExperimentExecutionConfig(
        evaluator_class=AlgorithmAgentEvaluator,
        dataset_path=Path("evals/datasets/algorithm_agent.py"),
        config=ExperimentConfig(
            name="algo_agent_claude_37_sonnet",
            params={
                "model_name": "Claude 3.7 Sonnet",
                "temperature": 0.0,
            },
        ),
        output_dir="evals/outputs/",
    ),
    ExperimentExecutionConfig(
        evaluator_class=DatasetPickingAgentEvaluator,
        dataset_path=Path("evals/datasets/dataset_picking_agent.py"),
        config=ExperimentConfig(
            name="dataset_picking_claude_35_sonnet",
            params={
                "model_name": "Claude 3.5 Sonnet",
                "temperature": 0.0,
            },
        ),
        output_dir="evals/outputs",
    ),
    ExperimentExecutionConfig(
        evaluator_class=DatasetPickingAgentEvaluator,
        dataset_path=Path("evals/datasets/dataset_picking_agent.py"),
        config=ExperimentConfig(
            name="dataset_picking_claude_37_sonnet",
            params={
                "model_name": "Claude 3.7 Sonnet",
                "temperature": 0.0,
            },
        ),
        output_dir="evals/outputs",
    ),
]


async def run_experiment(exp_exec_cfg: ExperimentExecutionConfig) -> None:
    """Run a single experiment."""
    evaluator = exp_exec_cfg["evaluator_class"]()
    dataset = load_dataset(exp_exec_cfg["dataset_path"])
    eval_report = await evaluator.eval_dataset(exp_exec_cfg["config"], dataset)

    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H%M")
    out_path = join(
        exp_exec_cfg["output_dir"],
        f"{timestamp}_{exp_exec_cfg['config'].name}.json",
    )
    evaluator.save_eval_report(eval_report, out_path)


def load_dataset(path: Path) -> EvalDataset:
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


async def main() -> None:
    """Async main."""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("llm_agent").setLevel(logging.INFO)

    for exp_exec_cfg in experiments:
        await run_experiment(exp_exec_cfg)


if __name__ == "__main__":
    run(main)
