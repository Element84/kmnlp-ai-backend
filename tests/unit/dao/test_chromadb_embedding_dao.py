from typing import TYPE_CHECKING
from unittest.mock import Mock

if TYPE_CHECKING:
    import chromadb
    from chromadb.api import ClientAPI

from llm_agent.dao.chromadb_embedding_dao import (
    ChromaDBEmbeddingDAO,
)
from llm_agent.dao.embedding_dao import (
    DocumentSearchResult,
    DocumentSearchResults,
    DocumentWrapper,
)


class TestChromaDBEmbeddingDAO:
    def test_initialization(self):
        client: ClientAPI = Mock()

        dao = ChromaDBEmbeddingDAO(client, "datasets")
        assert dao.collection_name == "datasets"
        client.create_collection.assert_called_once_with("datasets")

    def test_add_documents(self):
        collection: chromadb.Collection = Mock()
        collection.add.return_value = None

        client: ClientAPI = Mock()
        client.create_collection.return_value = collection

        dao = ChromaDBEmbeddingDAO(client, "datasets")

        dao.add_documents(
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
            ]
        )

        collection.add.assert_called_once_with(
            ids=[
                "http://kmnlp.noaa.gov/1-description",
                "http://kmnlp.noaa.gov/1-label",
            ],
            documents=["description 1", "label 1"],
            metadatas=[
                {
                    "dataset_id": "http://kmnlp.noaa.gov/1",
                    "field_name": "description",
                },
                {
                    "dataset_id": "http://kmnlp.noaa.gov/1",
                    "field_name": "label",
                },
            ],
        )

    def test_query(self):
        query_result: chromadb.QueryResult = {
            "ids": [
                [
                    "http://kmnlp.noaa.gov/1-label",
                ]
            ],
            "embeddings": None,
            "documents": [["label 1"]],
            "uris": None,
            "data": None,
            "metadatas": [
                [
                    {
                        "dataset_id": "http://kmnlp.noaa.gov/1",
                        "field_name": "label",
                    },
                ]
            ],
            "distances": [[0.98]],
            "included": [],
        }
        collection: chromadb.Collection = Mock()
        collection.query.return_value = query_result

        client: ClientAPI = Mock()
        client.create_collection.return_value = collection

        dao = ChromaDBEmbeddingDAO(client, "datasets")

        results = dao.query("yo")

        assert results == DocumentSearchResults(
            query="yo",
            results=[
                DocumentSearchResult(
                    distance=0.98,
                    document_wrapper=DocumentWrapper(
                        id="http://kmnlp.noaa.gov/1-label",
                        document="label 1",
                        metadata={
                            "dataset_id": "http://kmnlp.noaa.gov/1",
                            "field_name": "label",
                        },
                    ),
                )
            ],
        )

        collection.query.assert_called_once_with(query_texts=["yo"], n_results=9)
