from collections.abc import Awaitable, Callable
from contextlib import nullcontext as does_not_raise
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from pydantic_ai import RunContext

from llm_agent.context import AgentDeps, ToolProgressUpdate
from llm_agent.model.dataset import Dataset, DatasetSource
from llm_agent.model.dimension import Dimension
from llm_agent.model.variable import Variable
from llm_agent.tools.agent_tool import ErrorResponse
from llm_agent.tools.fetch_datasets_tool import (
    FetchDatasetsTool,
    KnowledgeGraphQueryUpdate,
    QueryResultsUpdate,
    _FetchDatasetsToolQuery,  # pyright: ignore[reportPrivateUsage]
    _TemporalCoverageFilter,  # pyright: ignore[reportPrivateUsage]
)
from tests.utils.mocks.context import MockChatContext

fetch_datasets_tool = FetchDatasetsTool()


class TestFetchDatasetsTool:
    def test_name_set_correctly(self):
        tool = FetchDatasetsTool()
        assert tool.name == "FetchDatasetsTool"

    def test_no_error_on_repr(self):
        with does_not_raise():
            _ = repr(fetch_datasets_tool)

    async def test_fetch_datasets_tool_execution(
        self,
        make_chat_context: Callable[..., MockChatContext],
        make_run_context: Callable[..., RunContext[AgentDeps]],
    ):
        tool = FetchDatasetsTool()

        progress_updates: list[ToolProgressUpdate[Any]] = []
        ctx = make_run_context(
            make_chat_context(knowledge_graph_dao=self._mock_knowledge_graph_dao()),
            progress_callback=progress_updates.append,
        )
        result = await tool(ctx, _FetchDatasetsToolQuery(detailed_view=True))

        assert result.model_dump(mode="json") == {
            "datasets": [
                {
                    "id": "id1",
                    "source": {
                        "format": "zarr",
                        "uri": "http://source1",
                    },
                    "label": "label1",
                    "description": "description1",
                    "landing_page": "http://example.com/landing_page/1",
                    "temporal_coverage": [
                        "2020-01-01T00:00:00Z",
                        "2024-11-30T12:00:00Z",
                    ],
                    "temporal_resolution": "P1D",
                    "spatial_coverage": "global",
                    "variables": [
                        {
                            "id": "var1",
                            "name": "variable 1",
                            "dataset_name": "dataset_name",
                            "field_name": "var1",
                            "standard_name": None,
                            "description": "description of variable 1",
                            "units": "units1",
                            "dimensions": [
                                {
                                    "id": "dim1",
                                    "field_name": "dim1",
                                    "alternate_names": [],
                                    "cf_standard_name": None,
                                },
                                {
                                    "id": "dim2",
                                    "field_name": "dim2",
                                    "alternate_names": [],
                                    "cf_standard_name": None,
                                },
                            ],
                        }
                    ],
                },
                {
                    "id": "id2",
                    "source": {
                        "format": "zarr",
                        "uri": "http://source2",
                    },
                    "label": "label2",
                    "description": "description2",
                    "landing_page": "http://example.com/landing_page/2",
                    "temporal_coverage": [
                        "2020-01-01T00:00:00Z",
                        "2024-11-30T12:00:00Z",
                    ],
                    "temporal_resolution": "P1D",
                    "spatial_coverage": "global",
                    "variables": [
                        {
                            "id": "var2",
                            "name": "variable 2",
                            "dataset_name": "dataset_name",
                            "field_name": "var2",
                            "standard_name": None,
                            "description": "description of variable 2",
                            "units": "units2",
                            "dimensions": [
                                {
                                    "id": "dim1",
                                    "field_name": "dim1",
                                    "alternate_names": [],
                                    "cf_standard_name": None,
                                },
                                {
                                    "id": "dim2",
                                    "field_name": "dim2",
                                    "alternate_names": [],
                                    "cf_standard_name": None,
                                },
                            ],
                        }
                    ],
                },
            ]
        }

        assert {type(update.update) for update in progress_updates} == {
            KnowledgeGraphQueryUpdate,
            QueryResultsUpdate,
        }
        for update in progress_updates:
            match update.update:
                case KnowledgeGraphQueryUpdate():
                    assert update.update.id_filter is None
                    assert update.update.temporal_filter is None
                    assert update.update.regex_filter is None
                case QueryResultsUpdate():
                    assert update.update.num_datasets == 2
                    assert update.update.execution_time_sec >= 0
                    assert update.update.detailed_view is True
                case _:
                    raise AssertionError(f"Unexpected update type: {type(update.update)}")

    async def test_fetch_datasets_tool_summary_view(
        self,
        make_chat_context: Callable[..., MockChatContext],
        make_run_context: Callable[..., RunContext[AgentDeps]],
    ):
        tool = FetchDatasetsTool()
        ctx = make_run_context(
            make_chat_context(knowledge_graph_dao=self._mock_knowledge_graph_dao())
        )
        result = await tool(ctx, _FetchDatasetsToolQuery(detailed_view=False))

        assert result.model_dump(mode="json") == {
            "datasets": [
                {
                    "id": "id1",
                    "label": "label1",
                    "description": "description1",
                    "temporal_coverage": [
                        "2020-01-01T00:00:00Z",
                        "2024-11-30T12:00:00Z",
                    ],
                    "spatial_coverage": "global",
                    "variables": ["variable 1"],
                },
                {
                    "id": "id2",
                    "label": "label2",
                    "description": "description2",
                    "temporal_coverage": [
                        "2020-01-01T00:00:00Z",
                        "2024-11-30T12:00:00Z",
                    ],
                    "spatial_coverage": "global",
                    "variables": ["variable 2"],
                },
            ]
        }

    async def test_fetch_datasets_tool_execution_start_time_filter(
        self,
        make_chat_context: Callable[..., MockChatContext],
        make_run_context: Callable[..., RunContext[AgentDeps]],
    ):
        mock_dao = self._mock_knowledge_graph_dao()
        ctx = make_run_context(make_chat_context(knowledge_graph_dao=mock_dao))
        tool = FetchDatasetsTool()
        _ = await tool(
            ctx,
            _FetchDatasetsToolQuery(
                temporal_coverage_filter=_TemporalCoverageFilter(
                    start=datetime(2020, 1, 1, tzinfo=UTC), end=None
                )
            ),
        )

        mock_dao.fetch_datasets.assert_called_once_with(
            ids=None,
            start_time=datetime(2020, 1, 1, tzinfo=UTC),
            end_time=None,
            regex_pattern=None,
        )

    async def test_fetch_datasets_tool_execution_with_end_time_filter(
        self,
        make_chat_context: Callable[..., MockChatContext],
        make_run_context: Callable[..., RunContext[AgentDeps]],
    ):
        mock_dao = self._mock_knowledge_graph_dao()
        ctx = make_run_context(make_chat_context(knowledge_graph_dao=mock_dao))
        tool = FetchDatasetsTool()
        _ = await tool(
            ctx,
            _FetchDatasetsToolQuery(
                temporal_coverage_filter=_TemporalCoverageFilter(
                    start=None, end=datetime(2025, 1, 1, tzinfo=UTC)
                )
            ),
        )

        mock_dao.fetch_datasets.assert_called_once_with(
            ids=None,
            start_time=None,
            end_time=datetime(2025, 1, 1, tzinfo=UTC),
            regex_pattern=None,
        )

    async def test_fetch_datasets_tool_execution_with_full_temporal_coverage_filter(
        self,
        make_chat_context: Callable[..., MockChatContext],
        make_run_context: Callable[..., RunContext[AgentDeps]],
    ):
        mock_dao = self._mock_knowledge_graph_dao()
        ctx = make_run_context(make_chat_context(knowledge_graph_dao=mock_dao))
        tool = FetchDatasetsTool()
        _ = await tool(
            ctx,
            _FetchDatasetsToolQuery(
                temporal_coverage_filter=_TemporalCoverageFilter(
                    start=datetime(2020, 1, 1, tzinfo=UTC),
                    end=datetime(2025, 1, 1, tzinfo=UTC),
                )
            ),
        )

        mock_dao.fetch_datasets.assert_called_once_with(
            ids=None,
            start_time=datetime(2020, 1, 1, tzinfo=UTC),
            end_time=datetime(2025, 1, 1, tzinfo=UTC),
            regex_pattern=None,
        )

    async def test_unhandled_exception(
        self,
        make_chat_context: Callable[..., MockChatContext],
        make_run_context: Callable[..., RunContext[AgentDeps]],
        invoke_tool_safely: Callable[..., Awaitable[Any]],
    ):
        mock_knowledge_graph_dao = Mock()
        mock_knowledge_graph_dao.fetch_datasets.side_effect = ZeroDivisionError
        ctx = make_run_context(make_chat_context(knowledge_graph_dao=mock_knowledge_graph_dao))

        tool = FetchDatasetsTool()
        result = await invoke_tool_safely(tool, ctx, _FetchDatasetsToolQuery())

        assert isinstance(result, ErrorResponse)
        assert result.message == "An unhandled exception was encountered: `ZeroDivisionError`"

    def _mock_knowledge_graph_dao(self) -> Mock:
        # NOTE: The following line has the effect of actually parsing the
        # Knowledge Graph. This isn't strictly needed, but it makes setting up
        # the mock easier. It might be nice to rework it at some point.
        dao = Mock()
        all_datasets_stub = [
            Dataset(
                id="id1",
                source=DatasetSource(format="zarr", uri="http://source1"),
                label="label1",
                description="description1",
                landing_page="http://example.com/landing_page/1",
                temporal_coverage=(
                    datetime(year=2020, month=1, day=1, tzinfo=UTC),
                    datetime(year=2024, month=11, day=30, hour=12, tzinfo=UTC),
                ),
                temporal_resolution=timedelta(days=1),
                variables=[
                    Variable(
                        id="var1",
                        name="variable 1",
                        dataset_name="dataset_name",
                        field_name="var1",
                        description="description of variable 1",
                        units="units1",
                        dimensions=[
                            Dimension(id="dim1", field_name="dim1"),
                            Dimension(id="dim2", field_name="dim2"),
                        ],
                    )
                ],
            ),
            Dataset(
                id="id2",
                source=DatasetSource(format="zarr", uri="http://source2"),
                label="label2",
                description="description2",
                landing_page="http://example.com/landing_page/2",
                temporal_coverage=(
                    datetime(year=2020, month=1, day=1, tzinfo=UTC),
                    datetime(year=2024, month=11, day=30, hour=12, tzinfo=UTC),
                ),
                temporal_resolution=timedelta(days=1),
                variables=[
                    Variable(
                        id="var2",
                        name="variable 2",
                        dataset_name="dataset_name",
                        field_name="var2",
                        description="description of variable 2",
                        units="units2",
                        dimensions=[
                            Dimension(id="dim1", field_name="dim1"),
                            Dimension(id="dim2", field_name="dim2"),
                        ],
                    )
                ],
            ),
        ]
        dao.fetch_datasets.return_value = all_datasets_stub

        return dao


class TestTemporalCoverageFilter:
    def test_no_temporal_coverage(self):
        with does_not_raise():
            _ = _TemporalCoverageFilter.model_validate(
                {
                    "start": None,
                    "end": None,
                }
            )

    def test_only_start_time(self):
        with does_not_raise():
            _ = _TemporalCoverageFilter.model_validate(
                {
                    "start": datetime(2020, 1, 1, tzinfo=UTC),
                    "end": None,
                }
            )

    def test_only_end_time(self):
        with does_not_raise():
            _ = _TemporalCoverageFilter.model_validate(
                {
                    "start": None,
                    "end": datetime(2025, 1, 1, tzinfo=UTC),
                }
            )

    def test_both_start_and_end_time(self):
        with does_not_raise():
            _ = _TemporalCoverageFilter.model_validate(
                {
                    "start": datetime(2020, 1, 1, tzinfo=UTC),
                    "end": datetime(2025, 1, 1, tzinfo=UTC),
                }
            )

    def test_start_equals_end(self):
        with does_not_raise():
            _ = _TemporalCoverageFilter.model_validate(
                {
                    "start": datetime(2020, 1, 1, tzinfo=UTC),
                    "end": datetime(2020, 1, 1, tzinfo=UTC),
                }
            )

    def test_start_time_after_end_time(self):
        with pytest.raises(ValidationError):
            _ = _TemporalCoverageFilter.model_validate(
                {
                    "start": datetime(2025, 1, 1, tzinfo=UTC),
                    "end": datetime(2020, 1, 1, tzinfo=UTC),
                }
            )

    def test_naive_datetime_converted_to_utc(self):
        temp_filter = _TemporalCoverageFilter.model_validate(
            {
                "start": datetime(2020, 1, 1),  # noqa: DTZ001
                "end": datetime(2025, 1, 1),  # noqa: DTZ001
            }
        )
        assert temp_filter.start == datetime(2020, 1, 1, tzinfo=UTC)
        assert temp_filter.end == datetime(2025, 1, 1, tzinfo=UTC)

    def test_naive_datetime_converted_to_utc_from_string(self):
        temp_filter = _TemporalCoverageFilter.model_validate(
            {
                "start": "2020-01-01T00:00:00",
                "end": "2025-01-01T00:00:00",
            }
        )
        assert temp_filter.start == datetime(2020, 1, 1, tzinfo=UTC)
        assert temp_filter.end == datetime(2025, 1, 1, tzinfo=UTC)
