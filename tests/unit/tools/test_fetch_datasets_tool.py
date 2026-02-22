from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest
from e84_geoai_common.llm.core import (
    JSONContent,
    LLMToolResultContent,
    LLMToolUseContent,
)
from pydantic import ValidationError

from llm_agent.context import LLMAgentContext, ToolCallContext
from llm_agent.model.dataset import Dataset, DatasetSource
from llm_agent.model.dimension import Dimension
from llm_agent.model.variable import Variable
from llm_agent.tools.fetch_datasets_tool import (
    FetchDatasetsTool,
    _FetchDatasetsToolResult,  # pyright: ignore[reportPrivateUsage]
    _TemporalCoverageFilter,  # pyright: ignore[reportPrivateUsage]
)
from llm_agent.utils.utils import EnrichedLLMToolResultContent

if TYPE_CHECKING:
    from pydantic import RootModel

    from llm_agent.tools.llm_tool_typed import ErrorResponse

fetch_datasets_tool = FetchDatasetsTool()


class TestFetchDatasetsTool:
    def test_name_set_correctly(self):
        tool = FetchDatasetsTool()
        assert tool.tool_spec.name == "FetchDatasetsTool"

    def test_no_error_on_repr(self):
        with does_not_raise():
            _ = repr(fetch_datasets_tool)

    def test_output_model(self):
        assert fetch_datasets_tool.tool_spec.output_model is not None
        assert (
            fetch_datasets_tool.tool_spec.output_model.model_json_schema()["title"]
            == "FetchDatasetsToolResponse"
        )
        success_response = {
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
                    "comment": "comment1",
                    "temporal_coverage": [
                        "2020-01-01T00:00:00Z",
                        "2024-11-30T12:00:00Z",
                    ],
                    "temporal_resolution": "P1D",
                    "spatial_coverage": "global",
                    "variables": [],
                },
            ]
        }

        with does_not_raise():
            res = cast(
                "RootModel[_FetchDatasetsToolResult | ErrorResponse]",
                fetch_datasets_tool.tool_spec.output_model.model_validate(success_response),
            )
        assert isinstance(res.root, fetch_datasets_tool.user_output_model)

        error_response = {"message": "Error"}
        with does_not_raise():
            err = cast(
                "RootModel[_FetchDatasetsToolResult | ErrorResponse]",
                fetch_datasets_tool.tool_spec.output_model.model_validate(error_response),
            )
        assert isinstance(err.root, fetch_datasets_tool.error_model)  # pyright: ignore[reportPrivateUsage]

    def test_fetch_datasets_tool_execution(self, global_context: Callable[..., LLMAgentContext]):
        tool = FetchDatasetsTool()
        context = ToolCallContext(
            tool_call_id="",
            global_context=global_context(knowledge_graph_dao=self._mock_knowledge_graph_dao()),
        )
        result: LLMToolResultContent = tool.execute(
            context,
            LLMToolUseContent(
                id="some-id",
                name="some-name",
                input={
                    "detailed_view": True,
                },
            ),
        )

        assert isinstance(result, EnrichedLLMToolResultContent)
        assert result.id == "some-id"
        assert result.status == "success"
        assert len(result.content) == 1
        assert result.content[0].model_dump(mode="json") == {
            "data": {
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
                                "field_name": "var1",
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
                                "field_name": "var2",
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
        }

    def test_fetch_datasets_tool_summary_view(self, global_context: Callable[..., LLMAgentContext]):
        tool = FetchDatasetsTool()
        context = ToolCallContext(
            tool_call_id="",
            global_context=global_context(knowledge_graph_dao=self._mock_knowledge_graph_dao()),
        )
        result: LLMToolResultContent = tool.execute(
            context,
            LLMToolUseContent(
                id="some-id",
                name="some-name",
                input={
                    "detailed_view": False,
                },
            ),
        )

        assert isinstance(result, EnrichedLLMToolResultContent)
        assert result.id == "some-id"
        assert result.status == "success"
        assert len(result.content) == 1
        assert result.content[0].model_dump(mode="json") == {
            "data": {
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
        }

    def test_fetch_datasets_tool_execution_start_time_filter(
        self, global_context: Callable[..., LLMAgentContext]
    ):
        mock_dao = self._mock_knowledge_graph_dao()
        context = ToolCallContext(
            tool_call_id="",
            global_context=global_context(knowledge_graph_dao=mock_dao),
        )
        tool = FetchDatasetsTool()
        _ = tool.execute(
            context,
            LLMToolUseContent(
                id="some-id",
                name="some-name",
                input={
                    "temporal_coverage_filter": {
                        "start": datetime(2020, 1, 1, tzinfo=UTC),
                        "end": None,
                    }
                },
            ),
        )

        mock_dao.fetch_datasets.assert_called_once_with(
            ids=None,
            start_time=datetime(2020, 1, 1, tzinfo=UTC),
            end_time=None,
            regex_pattern=None,
        )

    def test_fetch_datasets_tool_execution_with_end_time_filter(
        self, global_context: Callable[..., LLMAgentContext]
    ):
        mock_dao = self._mock_knowledge_graph_dao()
        context = ToolCallContext(
            tool_call_id="",
            global_context=global_context(knowledge_graph_dao=mock_dao),
        )
        tool = FetchDatasetsTool()
        _ = tool.execute(
            context,
            LLMToolUseContent(
                id="some-id",
                name="some-name",
                input={
                    "temporal_coverage_filter": {
                        "start": None,
                        "end": datetime(2025, 1, 1, tzinfo=UTC),
                    }
                },
            ),
        )

        mock_dao.fetch_datasets.assert_called_once_with(
            ids=None,
            start_time=None,
            end_time=datetime(2025, 1, 1, tzinfo=UTC),
            regex_pattern=None,
        )

    def test_fetch_datasets_tool_execution_with_full_temporal_coverage_filter(
        self, global_context: Callable[..., LLMAgentContext]
    ):
        mock_dao = self._mock_knowledge_graph_dao()
        context = ToolCallContext(
            tool_call_id="",
            global_context=global_context(knowledge_graph_dao=mock_dao),
        )
        tool = FetchDatasetsTool()
        _ = tool.execute(
            context,
            LLMToolUseContent(
                id="some-id",
                name="some-name",
                input={
                    "temporal_coverage_filter": {
                        "start": datetime(2020, 1, 1, tzinfo=UTC),
                        "end": datetime(2025, 1, 1, tzinfo=UTC),
                    }
                },
            ),
        )

        mock_dao.fetch_datasets.assert_called_once_with(
            ids=None,
            start_time=datetime(2020, 1, 1, tzinfo=UTC),
            end_time=datetime(2025, 1, 1, tzinfo=UTC),
            regex_pattern=None,
        )

    def test_unhandled_exception(self, global_context: Callable[..., LLMAgentContext]):
        mock_knowledge_graph_dao = Mock()
        mock_knowledge_graph_dao.fetch_datasets.side_effect = ZeroDivisionError
        context = ToolCallContext(
            tool_call_id="",
            global_context=global_context(knowledge_graph_dao=mock_knowledge_graph_dao),
        )

        tool = FetchDatasetsTool()
        result = tool.execute(context, LLMToolUseContent(id="some-id", name="some-name", input={}))

        assert result.id == "some-id"
        assert result.status == "error"
        assert result.content == [
            JSONContent(
                data={"message": ("An unhandled exception was encountered: `ZeroDivisionError`")}
            )
        ]

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
