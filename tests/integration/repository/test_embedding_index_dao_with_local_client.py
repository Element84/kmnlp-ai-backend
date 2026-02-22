from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from chromadb import EphemeralClient

from llm_agent.dao.chromadb_embedding_dao import (
    ChromaDBEmbeddingDAO,
)

if TYPE_CHECKING:
    from chromadb.api import ClientAPI

    from llm_agent.model.dataset import Dataset

from llm_agent.repository.embedding_index_dao import (
    EmbeddingIndexRepository,
)


@pytest.mark.integration
class TestEmbeddingIndexRepositoryWithChromaDB:
    """Test that the embedding retrieval actually works.

    What makese this an integration test? At runtime, querying the ChromaDB
    will download the embedding model if it doesn't already have it locally.
    While the model could certainly be mocked, the goal of tis test is to
    assert that the relevant fields are actually retrieval from the Embedding
    DB.

    One could replace the download function with one that returned a constant
    object, but that doesn't feel like it's worth the effort.
    """

    def test_retrieve_single_dataset(self):
        client: ClientAPI = EphemeralClient()

        dao = ChromaDBEmbeddingDAO(client, "datasets")
        repository = EmbeddingIndexRepository(dao)

        datasets: list[Dataset] = [
            Mock(
                id="http://kmnlp.noaa.gov/1",
                label="label 1",
                description="description 1",
                comment="comment 1",
            )
        ]

        # First, query should return empty results
        assert len(repository.query("label").results) == 0

        # Second, add datasets
        repository.add_datasets(datasets)

        # Third, make sure expected values are retrieved
        results = repository.query("label")

        assert results.query == "label"
        assert len(results.results) == 2

        assert [r.dataset_field_document.dataset_id for r in results.results] == [
            "http://kmnlp.noaa.gov/1",
            "http://kmnlp.noaa.gov/1",
        ]

        assert [r.dataset_field_document.field_value for r in results.results] == [
            "label 1",
            "description 1",
        ]
