# ruff: noqa: A002 ANN401 ARG001
import logging
import socket
import statistics
from typing import Any, Literal

import pytest
from e84_geoai_common.llm.models.claude import CLAUDE_4_5_HAIKU, CLAUDE_4_6_OPUS, CLAUDE_4_6_SONNET
from langfuse import Evaluation, get_client
from langfuse._client.datasets import DatasetItemClient  # type: ignore[reportMissingTypeStubs]
from langfuse.experiment import ExperimentItem, ExperimentItemResult, ExperimentResult, TaskFunction
from pydantic import BaseModel
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import (
    DatasetPickingAgent,
    DatasetSearchQuery,
)
from llm_agent.context import ChatContext
from llm_agent.evaluation.utils import MockChatContext, build_eval_client
from llm_agent.tools.agent_tool import AgentTool
from llm_agent.tools.execute_sparql_query_tool import ExecuteSparqlQueryTool
from llm_agent.tools.fetch_datasets_from_rag_tool import FetchDatasetsFromRagTool
from llm_agent.tools.fetch_datasets_tool import FetchDatasetsTool
from tests.experiment.stats import f1_score, precision, recall
from tests.experiment.utils import (
    MeanAndTCI,
    extract_item_result_scores_by_name,
    extract_run_evaluator_scores_by_name,
    get_git_branch,
    get_git_dirty,
    get_git_sha,
    mean_and_t_ci,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Initialize client
langfuse = get_client()
bedrock_runtime_client = build_eval_client()

PRECISION_EVALUATOR_NAME = "precision"
RECALL_EVALUATOR_NAME = "recall"
F1_SCORE_EVALUATOR_NAME = "f1_score"

REFUSAL_RECALL_EVALUATOR_NAME = "refusal_recall"
REFUSAL_PRECISION_EVALUATOR_NAME = "refusal_precision"

AVERAGE_PRECISION_EVALUATOR_NAME = "average_precision"
AVERAGE_RECALL_EVALUATOR_NAME = "average_recall"
AVERAGE_F1_SCORE_EVALUATOR_NAME = "average_f1_score"

TOOL_NAMES = ["FetchDatasetsTool", "ExecuteSparqlQueryTool", "FetchDatasetsFromRagTool"]
TOOL_ABBREVIATIONS = {
    "FetchDatasetsTool": "fdt",
    "ExecuteSparqlQueryTool": "sqt",
    "FetchDatasetsFromRagTool": "rt",
}

MODELS = [
    CLAUDE_4_5_HAIKU,
    CLAUDE_4_6_SONNET,
    CLAUDE_4_6_OPUS,
]

TEMPERATURES = [0.0, 0.1]

ToolName = Literal["FetchDatasetsTool", "ExecuteSparqlQueryTool", "FetchDatasetsFromRagTool"]


def all_tool_combinations() -> list[tuple[ToolName, ...]]:
    """All non-empty combinations of 1, 2, or 3 tools, order-independent."""
    return [
        ("FetchDatasetsTool",),
        ("FetchDatasetsTool", "FetchDatasetsFromRagTool"),
        ("FetchDatasetsTool", "ExecuteSparqlQueryTool"),
        ("FetchDatasetsTool", "FetchDatasetsFromRagTool", "ExecuteSparqlQueryTool"),
        ("FetchDatasetsFromRagTool",),
        ("FetchDatasetsFromRagTool", "ExecuteSparqlQueryTool"),
        ("ExecuteSparqlQueryTool",),
    ]


def make_config_name(model_key: str, temperature: float, tools: tuple[str, ...]) -> str:
    temp_label = str(temperature).replace(".", "")
    tools_label = "-".join(TOOL_ABBREVIATIONS[t] for t in tools)
    return f"{model_key}-t{temp_label}-{tools_label}"


class ExperimentConfig(BaseModel):
    name: str
    description: str
    model: str
    tool_names: list[ToolName]
    temperature: float
    min_recall: float = 0.80
    min_precision: float = 0.80


CONFIGS = [
    ExperimentConfig(
        name=make_config_name(model, temperature, tools),
        description=f"model: {model}, temp: {temperature:.2f}, tools: {tools}",
        model=model,
        tool_names=list(tools),
        temperature=temperature,
        min_recall=0.50,
        min_precision=0.50,
    )
    for model in MODELS
    for temperature in TEMPERATURES
    for tools in all_tool_combinations()
]


def _tool_name_to_tool(name: str) -> AgentTool[Any, Any]:
    match name:
        case "FetchDatasetsTool":
            return FetchDatasetsTool()
        case "ExecuteSparqlQueryTool":
            return ExecuteSparqlQueryTool()
        case "FetchDatasetsFromRagTool":
            return FetchDatasetsFromRagTool()
        case _:
            raise ValueError(f"Unexpected tool name: {name}")

    # This shouldn't be needed, but both pyright and ruff want it.
    raise ValueError(f"Unexpected tool name: {name}")


def make_task(config: ExperimentConfig, chat_context: ChatContext) -> TaskFunction:
    agent = DatasetPickingAgent(
        BedrockConverseModel(
            config.model, provider=BedrockProvider(bedrock_client=bedrock_runtime_client)
        ),
        tools=[_tool_name_to_tool(tool) for tool in config.tool_names],
        temperature=config.temperature,
    )

    async def task(*, item: ExperimentItem, **kwargs: dict[str, Any]) -> Any:
        assert isinstance(item, DatasetItemClient)
        log.debug("running experiment item input, id %s, %s", item.input, item.id)
        try:
            res = await agent.run(chat_context, DatasetSearchQuery(query=item.input))
        except Exception:
            log.exception("agent.run failed for item with input: %s", item.input)
            raise
        log.debug("agent run output: %s", res)
        return res

    return task


def item_recall_evaluator(
    *,
    input: Any,
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None,
    **kwargs: dict[str, Any],
) -> list[Evaluation]:
    try:
        score = recall(
            set(output.result) if output.result is not None else set(),
            set(expected_output["dataset_ids"]),
        )
    except Exception:
        log.exception("Exception calculating recall for input: %s", input)
        raise

    return [
        Evaluation(
            name=RECALL_EVALUATOR_NAME,
            value=score,
            comment=None,
            metadata=metadata,
        )
    ]


def item_precision_evaluator(
    *,
    input: Any,
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None,
    **kwargs: dict[str, Any],
) -> list[Evaluation]:
    try:
        score = precision(
            set(output.result) if output.result is not None else set(),
            set(expected_output["dataset_ids"]),
        )
    except Exception:
        log.exception("Exception calculating precision for input: %s", input)
        raise

    if score is None:
        return []

    return [
        Evaluation(
            name=PRECISION_EVALUATOR_NAME,
            value=score,
            comment=None,
            metadata=metadata,
        )
    ]


def item_f1_score_evaluator(
    *,
    input: Any,
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None,
    **kwargs: dict[str, Any],
):
    try:
        score = f1_score(
            set(output.result) if output.result is not None else set(),
            set(expected_output["dataset_ids"]),
        )
    except Exception:
        log.exception("Exception calculating precision for input: %s", input)
        raise

    return [
        Evaluation(
            name=F1_SCORE_EVALUATOR_NAME,
            value=score,
            comment=None,
            metadata=metadata,
        )
    ]


def item_refusal_recall_evaluator(
    *,
    input: Any,
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None,
    **kwargs: dict[str, Any],
) -> list[Evaluation]:
    try:
        is_output_refusal = output.is_refusal
        is_expected_refusal = expected_output["is_refusal"]

        score: bool = is_output_refusal == is_expected_refusal
    except Exception:
        log.exception("Exception calculating correct refusal for input: %s", input)
        raise

    return [
        Evaluation(
            name=REFUSAL_RECALL_EVALUATOR_NAME,
            value=float(score),
            comment=None,
            metadata=metadata,
        )
    ]


def item_refusal_precision_evaluator(
    *,
    input: Any,
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None,
    **kwargs: dict[str, Any],
) -> list[Evaluation]:
    try:
        is_output_refusal = output.is_refusal
        is_expected_refusal = expected_output["is_refusal"]

        if not is_output_refusal:
            return []

        score: bool = is_output_refusal == is_expected_refusal
    except Exception:
        log.exception("Exception calculating correct refusal for input: %s", input)
        raise

    return [
        Evaluation(
            name=REFUSAL_PRECISION_EVALUATOR_NAME,
            value=float(score),
            comment=None,
            metadata=metadata,
        )
    ]


def avg_recall_run_evaluator(
    *, item_results: list[ExperimentItemResult], **kwargs: dict[str, Any]
) -> Evaluation:
    scores: list[float] = extract_item_result_scores_by_name(item_results, RECALL_EVALUATOR_NAME)
    average_recall = statistics.mean(scores)
    return Evaluation(
        name=AVERAGE_RECALL_EVALUATOR_NAME, value=average_recall, comment=None, metadata=kwargs
    )


def avg_precision_run_evaluator(
    *, item_results: list[ExperimentItemResult], **kwargs: dict[str, Any]
) -> Evaluation:
    scores: list[float] = extract_item_result_scores_by_name(item_results, PRECISION_EVALUATOR_NAME)
    average_precision = statistics.mean(scores)
    return Evaluation(
        name=AVERAGE_PRECISION_EVALUATOR_NAME,
        value=average_precision,
        comment=None,
        metadata=kwargs,
    )


def avg_f1_score_run_evaluator(
    *, item_results: list[ExperimentItemResult], **kwargs: dict[str, Any]
) -> Evaluation:
    scores: list[float] = extract_item_result_scores_by_name(item_results, F1_SCORE_EVALUATOR_NAME)
    average_f1_score = statistics.mean(scores)
    return Evaluation(
        name=AVERAGE_F1_SCORE_EVALUATOR_NAME, value=average_f1_score, comment=None, metadata=kwargs
    )


def avg_refusal_recall_evaluator(
    *, item_results: list[ExperimentItemResult], **kwargs: dict[str, Any]
) -> Evaluation:
    scores: list[float] = extract_item_result_scores_by_name(
        item_results, REFUSAL_RECALL_EVALUATOR_NAME
    )
    average_recall = statistics.mean(scores)
    return Evaluation(
        name="average_refusal_recall", value=average_recall, comment=None, metadata=kwargs
    )


def avg_refusal_precision_evaluator(
    *, item_results: list[ExperimentItemResult], **kwargs: dict[str, Any]
) -> Evaluation:
    scores: list[float] = extract_item_result_scores_by_name(
        item_results, REFUSAL_PRECISION_EVALUATOR_NAME
    )
    average_recall = statistics.mean(scores)
    return Evaluation(
        name="average_refusal_precision", value=average_recall, comment=None, metadata=kwargs
    )


def _log_top_level_metrics(result: ExperimentResult) -> None:
    log.info(
        "item evaluations: %s",
        [
            f"{evaluation.name}: {evaluation.value}"
            for result in result.item_results
            for evaluation in result.evaluations
        ],
    )
    log.info("run evaluations: %s", result.run_evaluations)


def _add_mean_and_t_ci_scores_to_result(
    result: ExperimentResult, name: str, mean_and_t_ci: MeanAndTCI
) -> None:
    langfuse.create_score(
        name=f"{name}_ci_mean",
        value=mean_and_t_ci.mean,
        dataset_run_id=result.dataset_run_id,
    )
    langfuse.create_score(
        name=f"{name}_ci_low",
        value=mean_and_t_ci.ci_low,
        dataset_run_id=result.dataset_run_id,
    )
    langfuse.create_score(
        name=f"{name}_ci_high",
        value=mean_and_t_ci.ci_high,
        dataset_run_id=result.dataset_run_id,
    )


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
@pytest.mark.integration
async def test_dataset_picking(config: ExperimentConfig, chat_context_integration: MockChatContext):
    dataset = langfuse.get_dataset("dataset_picking_agent_evals_from_sdk")

    run_metadata = {
        "git_sha": get_git_sha(),
        "git_branch": get_git_branch(),
        "git_dirty": get_git_dirty(),
        "config": config.model_dump(mode="json"),
        "hostname": socket.gethostname(),
    }

    results: list[ExperimentResult] = []
    for n in range(5):
        result = dataset.run_experiment(
            name=config.name,
            description=config.description,
            task=make_task(config, chat_context_integration),
            evaluators=[
                item_recall_evaluator,
                item_precision_evaluator,
                item_f1_score_evaluator,
                item_refusal_recall_evaluator,
                item_refusal_precision_evaluator,
            ],
            run_evaluators=[
                avg_recall_run_evaluator,
                avg_precision_run_evaluator,
                avg_f1_score_run_evaluator,
                avg_refusal_recall_evaluator,
                avg_refusal_precision_evaluator,
            ],
            metadata={**run_metadata, "run_index": n},
            max_concurrency=50,
        )

        _log_top_level_metrics(result)
        results.append(result)

    precision_scores = extract_run_evaluator_scores_by_name(
        results, AVERAGE_PRECISION_EVALUATOR_NAME
    )
    recall_scores = extract_run_evaluator_scores_by_name(results, AVERAGE_RECALL_EVALUATOR_NAME)
    f1_scores = extract_run_evaluator_scores_by_name(results, AVERAGE_F1_SCORE_EVALUATOR_NAME)

    recall_stats = mean_and_t_ci(recall_scores)
    precision_stats = mean_and_t_ci(precision_scores)
    f1_stats = mean_and_t_ci(f1_scores)

    last_result_for_config = results[-1]

    _add_mean_and_t_ci_scores_to_result(last_result_for_config, "precision", precision_stats)
    _add_mean_and_t_ci_scores_to_result(last_result_for_config, "recall", recall_stats)
    _add_mean_and_t_ci_scores_to_result(last_result_for_config, "f1_score", f1_stats)
