from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

from e84_geoai_common.llm.core import (
    JSONContent,
    LLMToolResultContent,
    LLMToolUseContent,
    TextContent,
)

from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from llm_agent.model.dataset import Dataset
from llm_agent.tools.fetch_datasets_tool import FetchDatasetsTool


class TestFetchDatasetsTool:
    def test_name_set_correctly(self):
        dataset_picking_agent = Mock()
        tool = FetchDatasetsTool(dataset_picking_agent)
        assert tool.name == "FetchDatasetsTool"

    def test_fetch_datsets_tool_pydantic_props(self):
        mock_dao = self._mock_knowledge_graph_dao()
        tool = FetchDatasetsTool(mock_dao)
        assert tool.name == "FetchDatasetsTool"

    def test_fetch_datasets_tool_execution(self):
        mock_dao = self._mock_knowledge_graph_dao()
        tool = FetchDatasetsTool(mock_dao)
        result: LLMToolResultContent = tool.execute(
            LLMToolUseContent(id="some-id", name="some-name", input={})
        )

        assert result.id == "some-id"
        assert result.status == "success"
        assert len(result.content) == 1
        assert result.content[0].model_dump(mode="json") == JSONContent(
            data={
                "datasets": [
                    {
                        "iri": "iri1",
                        "source": "source1",
                        "label": "label1",
                        "description": None,
                        "temporal_coverage": [
                            "2020-01-01T00:00:00Z",
                            "2024-11-30T12:00:00Z",
                        ],
                        "temporal_resolution": "P1D",
                        "spatial_coverage": "global",
                        "variables": [],
                    },
                    {
                        "iri": "iri2",
                        "source": "source2",
                        "label": "label2",
                        "description": None,
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
        ).model_dump(mode="json")

    def test_unhandled_exception(self):
        mock_knowledge_graph_dao = Mock()
        mock_knowledge_graph_dao.fetch_variables.side_effect = (
            ZeroDivisionError
        )

        tool = FetchDatasetsTool(mock_knowledge_graph_dao)
        result = tool.execute(
            LLMToolUseContent(id="some-id", name="some-name", input={})
        )

        assert result.id == "some-id"
        assert result.status == "error"
        assert result.content == [
            TextContent(
                text="An unhandled exception was encountered. That exception "
                "was: ZeroDivisionError()"
            )
        ]

    def _mock_knowledge_graph_dao(self) -> KnowledgeGraphDAO:
        # NOTE: The following line has the effect of actually parsing the
        # Knowledge Graph. This isn't strictly needed, but it makes setting up
        # the mock easier. It might be nice to rework it at some point.
        dao = Mock()
        all_datasets_stub = [
            Dataset(
                iri="iri1",
                source="source1",
                label="label1",
                temporal_coverage=(
                    datetime(year=2020, month=1, day=1, tzinfo=UTC),
                    datetime(year=2024, month=11, day=30, hour=12, tzinfo=UTC),
                ),
                temporal_resolution=timedelta(days=1),
                variables=[],
            ),
            Dataset(
                iri="iri2",
                source="source2",
                label="label2",
                temporal_coverage=(
                    datetime(year=2020, month=1, day=1, tzinfo=UTC),
                    datetime(year=2024, month=11, day=30, hour=12, tzinfo=UTC),
                ),
                temporal_resolution=timedelta(days=1),
                variables=[],
            ),
        ]
        dao.fetch_variables.return_value = all_datasets_stub

        return dao
