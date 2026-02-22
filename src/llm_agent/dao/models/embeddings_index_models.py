###############
# Error Types #
###############


from dataclasses import dataclass

import chromadb


class QueryResultFieldIsNoneError(ValueError):
    """`chromadb.QueryResult` includes an unexpected `None` value."""

    field_names: list[str]

    def __init__(self, field_names: list[str]) -> None:
        """Initializer."""
        super().__init__(f"Found unexpected `None` in `QueryResult` field: {field_names}")
        self.field_names = field_names


class QueryResultIdFieldUnexpectedLengthError(ValueError):
    """`chromadb.QueryResult` `ids` length does not match query length."""

    def __init__(
        self,
        ids_length: int,
        query_length: int,
    ) -> None:
        """Initializer."""
        super().__init__(
            "Some `chromadb.QueryResult` value length does not match query "
            f"length. Lenghts are: ids_length: {ids_length} query_length: "
            f"{query_length}"
        )


class QueryResultFieldUnexpectedLengthError(ValueError):
    """`chromadb.QueryResult` value length does not match query length."""

    def __init__(
        self,
        query_length: int,
        documents_length: int,
        metadatas_length: int,
        distances_length: int,
    ) -> None:
        """Initializer."""
        super().__init__(
            "Some `chromadb.QueryResult` value length does not match query "
            f"length. Lenghts are: query_length: {query_length}, "
            f"documents_length: {documents_length}, metadatas_length: "
            f"{metadatas_length},   distances_length: {distances_length}"
        )


class QueryResultMismatchedInnerResultLengthsError(ValueError):
    """`chromadb.QueryResult` mismatched inner result lengths.

    If this happens, it means that one of the result's documents, metadatas,
    and distances are not of the same length.
    """

    def __init__(self, query_result: chromadb.QueryResult) -> None:
        """Initializer."""
        super().__init__(f"Query result: {query_result}")


##################
# Success Models #
##################


@dataclass()
class DatasetFieldDocument:
    """A document representing the value of a field on a dataset instance."""

    dataset_id: str
    field_value: str
    field_name: str

    @property
    def id(self) -> str:
        """Unique ID for the field document.

        To build this id, concatenate the dataset ID (which is unique to the
        dataset) with the name of the field whose value makes up the document
        in the vector store.
        """
        return f"{self.dataset_id}-{self.field_name}"


@dataclass()
class DatasetQueryResult:
    """An item returned by querying the vector DB."""

    distance: float
    dataset_field_document: DatasetFieldDocument


@dataclass()
class DatasetQueryResults:
    """Dataset results from querying the embedding store."""

    query: str
    results: list[DatasetQueryResult]
