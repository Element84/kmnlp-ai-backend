from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import Mock

import pytest
from chromadb import EphemeralClient
from pydantic_ai import RunContext

from llm_agent.context import AgentDeps, ToolProgressUpdate
from llm_agent.dao.chromadb_embedding_dao import ChromaDBEmbeddingDAO
from llm_agent.model.dataset import Dataset, DatasetSource
from llm_agent.model.dimension import Dimension
from llm_agent.model.variable import Variable
from llm_agent.repository.embedding_index_repository import EmbeddingIndexRepository
from llm_agent.tools.fetch_datasets_from_rag_tool import (
    FetchDatasetsFromRagTool,
    _FetchDatasetsFromRagToolQuery,  # type: ignore[reportPrivateUsage]
)
from tests.utils.mocks.context import MockChatContext


# Mark this test as an integratio test because it downloads the model if it
# isn't cached locally. In CI, it's never cached. It would be possible to hack
# around this by putting the model on the image. This is left an an exercise to
# the developer.
@pytest.mark.integration
class TestFetchDatasetsFromRagTool:
    async def test_fetch_datasets_tool_execution(
        self,
        make_chat_context: Callable[..., MockChatContext],
        make_run_context: Callable[..., RunContext[AgentDeps]],
    ):
        tool = FetchDatasetsFromRagTool()

        progress_updates: list[ToolProgressUpdate[Any]] = []
        mock_kg_dao = self._mock_knowledge_graph_dao()
        ctx = make_run_context(
            make_chat_context(
                knowledge_graph_dao=mock_kg_dao,
                embedding_repo=EmbeddingIndexRepository(
                    ChromaDBEmbeddingDAO(EphemeralClient(), "datasets"),
                    mock_kg_dao.fetch_datasets(),
                ),
            ),
            progress_callback=progress_updates.append,
        )
        result = await tool(ctx, _FetchDatasetsFromRagToolQuery(user_query="any"))

        # Distances vary sublty based on execution environment.
        assert result.model_dump(mode="json") == {
            "results": [
                {
                    "dataset": {
                        "description": "description1",
                        "id": "id1",
                        "label": "label1",
                        "landing_page": "http://example.com/landing_page/1",
                        "source": {
                            "format": "zarr",
                            "uri": "http://source1",
                        },
                        "spatial_coverage": "global",
                        "temporal_coverage": [
                            "2020-01-01T00:00:00Z",
                            "2024-11-30T12:00:00Z",
                        ],
                        "temporal_resolution": "P1D",
                        "variables": [
                            {
                                "dataset_name": "dataset_name",
                                "description": "description of variable 1",
                                "dimensions": [
                                    {
                                        "alternate_names": [],
                                        "cf_standard_name": None,
                                        "field_name": "dim1",
                                        "id": "dim1",
                                    },
                                    {
                                        "alternate_names": [],
                                        "cf_standard_name": None,
                                        "field_name": "dim2",
                                        "id": "dim2",
                                    },
                                ],
                                "field_name": "var1",
                                "id": "var1",
                                "name": "variable 1",
                                "standard_name": None,
                                "units": "units1",
                            },
                        ],
                    },
                    "distance": pytest.approx(1.4681206941604614, 1e-4),  # type: ignore[reportUnknownMemberType]
                },
                {
                    "dataset": {
                        "description": "description1",
                        "id": "id1",
                        "label": "label1",
                        "landing_page": "http://example.com/landing_page/1",
                        "source": {
                            "format": "zarr",
                            "uri": "http://source1",
                        },
                        "spatial_coverage": "global",
                        "temporal_coverage": [
                            "2020-01-01T00:00:00Z",
                            "2024-11-30T12:00:00Z",
                        ],
                        "temporal_resolution": "P1D",
                        "variables": [
                            {
                                "dataset_name": "dataset_name",
                                "description": "description of variable 1",
                                "dimensions": [
                                    {
                                        "alternate_names": [],
                                        "cf_standard_name": None,
                                        "field_name": "dim1",
                                        "id": "dim1",
                                    },
                                    {
                                        "alternate_names": [],
                                        "cf_standard_name": None,
                                        "field_name": "dim2",
                                        "id": "dim2",
                                    },
                                ],
                                "field_name": "var1",
                                "id": "var1",
                                "name": "variable 1",
                                "standard_name": None,
                                "units": "units1",
                            },
                        ],
                    },
                    "distance": pytest.approx(1.576291799545288, 1e-4),  # type: ignore[reportUnknownMemberType]
                },
            ],
        }

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
