from dataclasses import dataclass
from typing import Protocol


@dataclass()
class DocumentWrapper:
    """A wrapper for a document to store in an embedding."""

    id: str
    document: str
    metadata: dict[str, str]


@dataclass()
class DocumentSearchResult:
    """A single document result returned from the embeddings store."""

    distance: float
    document_wrapper: DocumentWrapper


@dataclass()
class DocumentSearchResults:
    """Search result type for queries."""

    query: str
    results: list[DocumentSearchResult]


class EmbeddingDAO(Protocol):
    """Interface for managing stored embeddings."""

    def add_documents(self, documents: list[DocumentWrapper]) -> None:
        """Add wrapped documents to the store, creating embeddings for them.

        Args:
            documents (list[DocumentWrapper]): Wrappers for the documents to
                get embeddings for and store, along with ids and metadata to
                associate with the created records.

        Returns:
            None
        """
        ...

    def query(self, query: str) -> DocumentSearchResults:
        """Query store for matching documents with a natural language query.

        Args:
            query (str): The natural language query to with which to search the
                stored documents, using an embedding similarity metric.

        Returns:
            list[DocumentWrapper]: A list of the most closely-matching
            documents, along with their distance scores, ids, and metadata.
        """
        ...
