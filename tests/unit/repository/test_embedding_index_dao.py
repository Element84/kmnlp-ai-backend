from typing import TYPE_CHECKING
from unittest.mock import Mock

from llm_agent.dao.embedding_dao import (
    DocumentSearchResult,
    DocumentSearchResults,
    DocumentWrapper,
    EmbeddingDAO,
)
from llm_agent.dao.models.embeddings_index_models import (
    DatasetFieldDocument,
    DatasetQueryResult,
    DatasetQueryResults,
)

if TYPE_CHECKING:
    from llm_agent.model.dataset import Dataset

from llm_agent.repository.embedding_index_dao import (
    EmbeddingIndexRepository,
)


class TestEmbeddingIndexRepository:
    def test_add_datasets(self):
        dao: EmbeddingDAO = Mock()
        datasets: list[Dataset] = [
            Mock(
                spec=["id, label, description, comment"],
                id="http://kmnlp.noaa.gov/1",
                label="label 1",
                description="description 1",
                comment="comment 1",
            ),
            Mock(
                spec=["id, label, description, comment"],
                id="http://kmnlp.noaa.gov/2",
                label="label 2",
                description="description 2",
                comment="comment 2",
            ),
        ]
        repository = EmbeddingIndexRepository(dao)

        repository.add_datasets(datasets)

        dao.add_documents.assert_called_once_with(
            [
                DocumentWrapper(
                    id="http://kmnlp.noaa.gov/1-description",
                    document="description 1",
                    metadata={
                        "dataset_id": "http://kmnlp.noaa.gov/1",
                        "field_name": "description",
                    },
                ),
                DocumentWrapper(
                    id="http://kmnlp.noaa.gov/1-label",
                    document="label 1",
                    metadata={
                        "dataset_id": "http://kmnlp.noaa.gov/1",
                        "field_name": "label",
                    },
                ),
                DocumentWrapper(
                    id="http://kmnlp.noaa.gov/2-description",
                    document="description 2",
                    metadata={
                        "dataset_id": "http://kmnlp.noaa.gov/2",
                        "field_name": "description",
                    },
                ),
                DocumentWrapper(
                    id="http://kmnlp.noaa.gov/2-label",
                    document="label 2",
                    metadata={
                        "dataset_id": "http://kmnlp.noaa.gov/2",
                        "field_name": "label",
                    },
                ),
            ]
        )

    def test_queries_collection(self):
        dao: EmbeddingDAO = Mock()

        document_search_results: DocumentSearchResults = DocumentSearchResults(
            query="sea surface temperature",
            results=[
                DocumentSearchResult(
                    distance=0.3,
                    document_wrapper=DocumentWrapper(
                        id="http://kmnlp.noaa.gov/1-description",
                        document="description 1",
                        metadata={
                            "dataset_id": "http://kmnlp.noaa.gov/1",
                            "field_name": "description",
                        },
                    ),
                ),
                DocumentSearchResult(
                    distance=0.2,
                    document_wrapper=DocumentWrapper(
                        id="http://kmnlp.noaa.gov/1-label",
                        document="label 1",
                        metadata={
                            "dataset_id": "http://kmnlp.noaa.gov/1",
                            "field_name": "label",
                        },
                    ),
                ),
            ],
        )
        dao.query.return_value = document_search_results

        repository = EmbeddingIndexRepository(dao)
        result = repository.query("sea surface temperature")

        # Assert result
        assert result == DatasetQueryResults(
            query="sea surface temperature",
            results=[
                DatasetQueryResult(
                    distance=0.3,
                    dataset_field_document=DatasetFieldDocument(
                        dataset_id="http://kmnlp.noaa.gov/1",
                        field_value="description 1",
                        field_name="description",
                    ),
                ),
                DatasetQueryResult(
                    distance=0.2,
                    dataset_field_document=DatasetFieldDocument(
                        dataset_id="http://kmnlp.noaa.gov/1",
                        field_value="label 1",
                        field_name="label",
                    ),
                ),
            ],
        )

        # Assert database was queried with expected args
        dao.query.assert_called_once_with(
            "sea surface temperature",
        )
