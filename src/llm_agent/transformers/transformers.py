import chromadb

from llm_agent.dao.embedding_dao import (
    DocumentSearchResult,
    DocumentSearchResults,
    DocumentWrapper,
)
from llm_agent.dao.models.embeddings_index_models import (
    DatasetFieldDocument,
    DatasetQueryResult,
    DatasetQueryResults,
    QueryResultFieldIsNoneError,
    QueryResultFieldUnexpectedLengthError,
    QueryResultIdFieldUnexpectedLengthError,
    QueryResultMismatchedInnerResultLengthsError,
)
from llm_agent.model.dataset import Dataset


def chromadb_query_result_to_document_search_results(
    query: str,
    query_result: chromadb.QueryResult,
) -> DocumentSearchResults:
    """`(str, chromadb.QueryResult` -> `DocumentSearchResults`."""
    query_result_ids = query_result["ids"]

    if len(query_result_ids) != 1:
        raise QueryResultIdFieldUnexpectedLengthError(len(query_result_ids), 1)

    query_result_documents = query_result["documents"]
    query_result_metadatas = query_result["metadatas"]
    query_result_distances = query_result["distances"]

    if (
        query_result_documents is None
        or query_result_metadatas is None
        or query_result_distances is None
    ):
        none_fields = [
            key for key in ["documents", "metadatas", "distances"] if query_result[key] is None
        ]
        raise QueryResultFieldIsNoneError(none_fields)

    if (
        len(query_result_documents) != 1
        or len(query_result_metadatas) != 1
        or len(query_result_distances) != 1
    ):
        raise QueryResultFieldUnexpectedLengthError(
            query_length=1,
            documents_length=len(query_result_documents),
            metadatas_length=len(query_result_metadatas),
            distances_length=len(query_result_distances),
        )

    try:
        return DocumentSearchResults(
            query=query,
            results=[
                DocumentSearchResult(
                    distance=query_result_distances[0][i],
                    document_wrapper=DocumentWrapper(
                        id=query_result_ids[0][i],
                        document=query_result_documents[0][i],
                        metadata={k: str(v) for k, v in query_result_metadatas[0][i].items()},
                    ),
                )
                for i in range(len(query_result_documents[0]))
            ],
        )

    except IndexError as ie:
        raise QueryResultMismatchedInnerResultLengthsError(query_result) from ie


def dataset_to_document_wrappers(
    dataset: Dataset,
) -> tuple[DocumentWrapper, DocumentWrapper]:
    """Transform from domain dataset to DocumentWrapper for fields."""
    return (
        DocumentWrapper(
            id=f"{dataset.id}-description",
            document=dataset.description,
            metadata={
                "dataset_id": dataset.id,
                "field_name": "description",
            },
        ),
        DocumentWrapper(
            id=f"{dataset.id}-label",
            document=dataset.label,
            metadata={
                "dataset_id": dataset.id,
                "field_name": "label",
            },
        ),
    )


def document_search_results_to_dataset_query_results(
    document_search_results: DocumentSearchResults,
) -> DatasetQueryResults:
    """`DocumentSearchResults` -> `DatasetQueryResults`."""
    return DatasetQueryResults(
        query=document_search_results.query,
        results=[
            document_search_result_to_dataset_query_result(r)
            for r in document_search_results.results
        ],
    )


def document_search_result_to_dataset_query_result(
    document_search_result: DocumentSearchResult,
) -> DatasetQueryResult:
    """`DocumentSearchResult` -> `DatasetQueryResult`."""
    return DatasetQueryResult(
        distance=document_search_result.distance,
        dataset_field_document=document_wrapper_to_dataset_field_document(
            document_search_result.document_wrapper
        ),
    )


def document_wrapper_to_dataset_field_document(
    document_wrapper: DocumentWrapper,
) -> DatasetFieldDocument:
    """`DocumentWrapper` -> `DatasetFieldDocument`."""
    return DatasetFieldDocument(
        dataset_id=document_wrapper.metadata["dataset_id"],
        field_value=document_wrapper.document,
        field_name=document_wrapper.metadata["field_name"],
    )
