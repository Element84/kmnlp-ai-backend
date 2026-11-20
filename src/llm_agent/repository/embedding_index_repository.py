from collections.abc import Iterable

from llm_agent.dao.embedding_dao import EmbeddingDAO
from llm_agent.dao.models.embeddings_index_models import (
    DatasetQueryResults,
)
from llm_agent.model.dataset import Dataset
from llm_agent.transformers.transformers import (
    dataset_to_document_wrappers,
    document_search_results_to_dataset_query_results,
)


class EmbeddingIndexRepository:
    """Repository for accessing embeddings."""

    _embedding_dao: EmbeddingDAO

    def __init__(self, embedding_dao: EmbeddingDAO, datasets: list[Dataset]) -> None:
        """Initializer.

        Args:
            embedding_dao (EmbeddingDAO): An implementor of the
                `EmbeddingDAO` protocol, which is responsible for storing
                documents, creating embeddings for them, and executing queries
                against them.

            datasets (list[Dataset]): The list of datasets with which to
                initialize the vector store.

        """
        self._embedding_dao = embedding_dao

        self.add_datasets(datasets)

    def add_datasets(self, datasets: Iterable[Dataset]) -> None:
        """Add datasets to the embedding manager's store.

        Args:
            datasets (list[Dataset]): List of the datasets to add to the
                embedding dao's store

        """
        self._embedding_dao.add_documents(
            [
                document_wrapper
                for d in datasets
                for document_wrapper in dataset_to_document_wrappers(d)
            ]
        )

    def query(self, query: str) -> DatasetQueryResults:
        """Query the datasets collection."""
        return document_search_results_to_dataset_query_results(self._embedding_dao.query(query))
