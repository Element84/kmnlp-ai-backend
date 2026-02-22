import chromadb
from chromadb.api import ClientAPI

from llm_agent.dao.embedding_dao import (
    DocumentSearchResults,
    DocumentWrapper,
    EmbeddingDAO,
)
from llm_agent.transformers.transformers import (
    chromadb_query_result_to_document_search_results,
)


class ChromaDBEmbeddingDAO(EmbeddingDAO):
    """Store and retrieve documents in ChromaDB."""

    collection_name: str

    _client: ClientAPI
    _collection: chromadb.Collection

    def __init__(self, client: ClientAPI, collection_name: str) -> None:
        """Initializer.

        Creates a collection in ChromaDB with the given name and tracks it.
        """
        self.collection_name = collection_name

        self._client = client
        self._collection = client.create_collection("datasets")

    def add_documents(self, documents: list[DocumentWrapper]) -> None:
        """Add the provided documents to the collection."""
        self._collection.add(
            ids=[document.id for document in documents],
            documents=[document.document for document in documents],
            metadatas=[document.metadata for document in documents],
        )

    def query(self, query: str) -> DocumentSearchResults:
        """Query the collection for matching documents."""
        query_texts = [query]
        result: chromadb.QueryResult = self._collection.query(
            query_texts=query_texts,
            n_results=9,
        )

        return chromadb_query_result_to_document_search_results(query, result)
