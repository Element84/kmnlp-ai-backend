from typing import TYPE_CHECKING
from unittest.mock import Mock

import chromadb
import pytest

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
from llm_agent.transformers.transformers import (
    chromadb_query_result_to_document_search_results,
    dataset_to_document_wrappers,
    document_search_result_to_dataset_query_result,
    document_search_results_to_dataset_query_results,
    document_wrapper_to_dataset_field_document,
)

if TYPE_CHECKING:
    from llm_agent.model.dataset import Dataset


def test_raises_on_result_id_length_gt_1():
    query_result = _default_chromadb_query_result()
    query_result["ids"] = [["1", "2"], ["1"]]

    with pytest.raises(QueryResultIdFieldUnexpectedLengthError):
        chromadb_query_result_to_document_search_results("yo", query_result)


def test_raises_on_documents_is_none():
    query_result = _default_chromadb_query_result()
    query_result["documents"] = None

    with pytest.raises(QueryResultFieldIsNoneError):
        chromadb_query_result_to_document_search_results("yo", query_result)


def test_raises_on_metadatas_is_none():
    query_result = _default_chromadb_query_result()
    query_result["metadatas"] = None

    with pytest.raises(QueryResultFieldIsNoneError):
        chromadb_query_result_to_document_search_results("yo", query_result)


def test_raises_on_distances_is_none():
    query_result = _default_chromadb_query_result()
    query_result["distances"] = None

    with pytest.raises(QueryResultFieldIsNoneError):
        chromadb_query_result_to_document_search_results("yo", query_result)


def test_raises_on_mismatched_ids_and_documents_length():
    query_result = _default_chromadb_query_result()
    query_result["documents"] = [["1", "2"], []]

    with pytest.raises(QueryResultFieldUnexpectedLengthError):
        chromadb_query_result_to_document_search_results("yo", query_result)


def test_raises_on_mismatched_ids_and_metadatas_length():
    query_result = _default_chromadb_query_result()
    query_result["metadatas"] = [[{"some": "value"}], [{"other": "value"}]]

    with pytest.raises(QueryResultFieldUnexpectedLengthError):
        chromadb_query_result_to_document_search_results("yo", query_result)


def test_raises_on_mismatched_ids_and_distances_length():
    query_result = _default_chromadb_query_result()
    query_result["distances"] = [[0.1], [0.2]]

    with pytest.raises(QueryResultFieldUnexpectedLengthError):
        chromadb_query_result_to_document_search_results("yo", query_result)


def test_raises_on_mismatched_result_field_length():
    query_result = _default_chromadb_query_result()
    query_result["documents"] = [["hello", "there"]]
    query_result["metadatas"] = [[{"some": "value", "other": "value"}]]
    query_result["distances"] = [[0.1, 0.2]]

    with pytest.raises(QueryResultMismatchedInnerResultLengthsError):
        chromadb_query_result_to_document_search_results("yo", query_result)


def test_happy_path_for_default_query_result():
    query_result = _default_chromadb_query_result()

    results = chromadb_query_result_to_document_search_results("yo", query_result)

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


def test_transform_complex_query_result():
    query_result: chromadb.QueryResult = {
        "ids": [
            [
                "http://kmnlp.noaa.gov/1-label",
                "http://kmnlp.noaa.gov/1-description",
                "http://kmnlp.noaa.gov/2-description",
            ],
        ],
        "embeddings": None,
        "documents": [
            ["label 1", "description 1", "description 2"],
        ],
        "uris": None,
        "data": None,
        "metadatas": [
            [
                {
                    "dataset_id": "http://kmnlp.noaa.gov/1",
                    "field_name": "label",
                },
                {
                    "dataset_id": "http://kmnlp.noaa.gov/1",
                    "field_name": "description",
                },
                {
                    "dataset_id": "http://kmnlp.noaa.gov/2",
                    "field_name": "description",
                },
            ],
        ],
        "distances": [[0.98, 0.97, 0.96]],
        "included": [],
    }
    results = chromadb_query_result_to_document_search_results("yo", query_result)

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
            ),
            DocumentSearchResult(
                distance=0.97,
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
                distance=0.96,
                document_wrapper=DocumentWrapper(
                    id="http://kmnlp.noaa.gov/2-description",
                    document="description 2",
                    metadata={
                        "dataset_id": "http://kmnlp.noaa.gov/2",
                        "field_name": "description",
                    },
                ),
            ),
        ],
    )


def test_dataset_to_document_wrappers():
    dataset: Dataset = Mock(
        id="http://kmnlp.noaa.gov/1",
        label="label 1",
        description="description 1",
        comment="comment 1",
    )

    document_wrappers = dataset_to_document_wrappers(dataset)

    assert document_wrappers == (
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
    )


def test_document_search_results_to_dataset_query_results():
    document_search_results = DocumentSearchResults(
        query="yo",
        results=[
            DocumentSearchResult(
                distance=0.5,
                document_wrapper=DocumentWrapper(
                    id="http://kmnlp.noaa.gov/1-label",
                    document="label 1",
                    metadata={
                        "dataset_id": "http://kmnlp.noaa.gov/1",
                        "field_name": "label",
                    },
                ),
            ),
            DocumentSearchResult(
                distance=0.4,
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
                distance=0.3,
                document_wrapper=DocumentWrapper(
                    id="http://kmnlp.noaa.gov/1-comment",
                    document="comment 1",
                    metadata={
                        "dataset_id": "http://kmnlp.noaa.gov/1",
                        "field_name": "comment",
                    },
                ),
            ),
        ],
    )

    results = document_search_results_to_dataset_query_results(document_search_results)
    assert results == DatasetQueryResults(
        query="yo",
        results=[
            DatasetQueryResult(
                distance=0.5,
                dataset_field_document=DatasetFieldDocument(
                    dataset_id="http://kmnlp.noaa.gov/1",
                    field_value="label 1",
                    field_name="label",
                ),
            ),
            DatasetQueryResult(
                distance=0.4,
                dataset_field_document=DatasetFieldDocument(
                    dataset_id="http://kmnlp.noaa.gov/1",
                    field_value="description 1",
                    field_name="description",
                ),
            ),
            DatasetQueryResult(
                distance=0.3,
                dataset_field_document=DatasetFieldDocument(
                    dataset_id="http://kmnlp.noaa.gov/1",
                    field_value="comment 1",
                    field_name="comment",
                ),
            ),
        ],
    )


def test_document_search_result_to_dataset_query_result():
    document_search_result = DocumentSearchResult(
        distance=0.5,
        document_wrapper=DocumentWrapper(
            id="http://kmnlp.noaa.gov/1-label",
            document="label 1",
            metadata={
                "dataset_id": "http://kmnlp.noaa.gov/1",
                "field_name": "label",
            },
        ),
    )

    result = document_search_result_to_dataset_query_result(document_search_result)

    assert result == DatasetQueryResult(
        distance=0.5,
        dataset_field_document=DatasetFieldDocument(
            dataset_id="http://kmnlp.noaa.gov/1",
            field_value="label 1",
            field_name="label",
        ),
    )


def test_document_wrapper_to_dataset_field_document():
    document_wrapper = DocumentWrapper(
        id="http://kmnlp.noaa.gov/1-label",
        document="label 1",
        metadata={
            "dataset_id": "http://kmnlp.noaa.gov/1",
            "field_name": "label",
        },
    )

    dataset_field_document = document_wrapper_to_dataset_field_document(document_wrapper)

    assert dataset_field_document == DatasetFieldDocument(
        dataset_id="http://kmnlp.noaa.gov/1",
        field_value="label 1",
        field_name="label",
    )


def _default_chromadb_query_result() -> chromadb.QueryResult:
    return {
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
