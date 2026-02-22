# Ignore "line too long" errors.
# ruff: noqa: E501

from datetime import UTC, datetime

from llm_agent.evaluation.base_eval import EvalDataset
from llm_agent.evaluation.dataset_picking_agent_single_turn import (
    DatasetPickingSingleTurnEvalInput,
    DatasetPickingSingleTurnResponseAttrs,
)

_dataset: list[DatasetPickingSingleTurnEvalInput] = [
    DatasetPickingSingleTurnEvalInput(
        id="fetch-all-datasets",
        natural_language_query="What datasets do you know about?",
        expected_response_attrs=DatasetPickingSingleTurnResponseAttrs(
            is_tool_call=True, tool_call_args={}
        ),
    ),
    DatasetPickingSingleTurnEvalInput(
        id="fetch-single-day",
        natural_language_query="Mean Sea Surface Temperature for the Black Sea on June 1, 2024.",
        expected_response_attrs=DatasetPickingSingleTurnResponseAttrs(
            is_tool_call=True,
            tool_call_args={
                "temporal_coverage_filter": {
                    "start": datetime(2024, 6, 1, tzinfo=UTC),
                    "end": datetime(2024, 6, 1, tzinfo=UTC),
                }
            },
        ),
    ),
    DatasetPickingSingleTurnEvalInput(
        id="fetch-range-since",
        natural_language_query="Mean Sea Surface Temperature for the Atlantic Coast since 2010.",
        expected_response_attrs=DatasetPickingSingleTurnResponseAttrs(
            is_tool_call=True,
            tool_call_args={
                "temporal_coverage_filter": {
                    "start": datetime(2010, 1, 1, tzinfo=UTC),
                    "end": None,
                }
            },
        ),
    ),
    DatasetPickingSingleTurnEvalInput(
        id="fetch-range-before",
        natural_language_query="Mean Sea Surface Temperature for the Pacific Coast with data until at least January 1, 2024.",
        expected_response_attrs=DatasetPickingSingleTurnResponseAttrs(
            is_tool_call=True,
            tool_call_args={
                "temporal_coverage_filter": {
                    "start": datetime(2024, 1, 1, tzinfo=UTC),
                    "end": None,
                }
            },
        ),
    ),
]

dataset = EvalDataset(
    id="default",
    version="2",
    items=[d.to_eval_input() for d in _dataset],
)
