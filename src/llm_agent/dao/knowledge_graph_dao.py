# ruff: noqa: N815

from collections.abc import Sequence
from datetime import datetime
from functools import cached_property
from importlib.resources import as_file
from importlib.resources.abc import Traversable
from itertools import groupby
from os.path import basename, join
from typing import TYPE_CHECKING, cast

from langfuse import observe  # type: ignore[reportUnknownVariableType]
from pydantic import BaseModel, ConfigDict
from rdflib import Graph, URIRef
from rdflib.plugins.sparql import prepareQuery
from rdflib.term import Literal as RDFLiteral

if TYPE_CHECKING:
    from rdflib.query import ResultRow

from llm_agent.dao.utils import KG_TTL_PATHS
from llm_agent.model.dataset import Dataset, DatasetSource
from llm_agent.model.dimension import CanonicalDimension, Dimension
from llm_agent.model.variable import Variable

KG_NAMESPACE = "http://kmnlp-demo.noaa.gov/"

FETCH_ALL_NODES_QUERY = """
SELECT ?s ?p ?o
WHERE {
    ?s ?p ?o
}
"""

FETCH_DATASETS_QUERY = prepareQuery(
    """
PREFIX : <http://kmnlp-demo.noaa.gov/>
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX qb: <http://purl.org/linked-data/cube#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <https://schema.org/>

SELECT
    ?dataset_id
    ?datasetLabel
    ?description
    ?landingPage
    ?sourceUri
    ?sourceType
    ?startDate
    ?endDate
    ?temporalResolution
    ?variable_id
    ?variableName
    ?variableFieldName
    ?variableStandardName
    ?variableDescription
    ?variableUnits
    (GROUP_CONCAT(?dim_id; SEPARATOR=";") AS ?dimIds)
    (GROUP_CONCAT(?dimFieldName; SEPARATOR=";") AS ?dimFieldNames)
WHERE {
    ?dataset a dcat:Dataset ;
        rdfs:label ?datasetLabel ;
        dcat:distribution ?distribution ;
        dcat:startDate ?startDate  ;
        dcat:endDate ?endDate  ;
        dcterms:description ?description  ;
        qb:structure ?dataStructDef .

    ?distribution dcat:accessURL ?sourceUri .
    ?distribution dcat:mediaType ?sourceType .

    ?dataset schema:description ?description  .

    OPTIONAL { ?dataset schema:url ?landingPage } .
    OPTIONAL { ?dataset dcat:temporalResolution ?temporalResolution } .

    ?dataStructDef qb:component ?dataStructDefComp .
    ?dataStructDefComp qb:measure ?variable .

    ?variable rdfs:label ?variableLabel ;
        schema:alternateName ?variableFieldName .

    ?dataStructDef qb:component ?dataStructDefCompDim .
    ?dataStructDefCompDim qb:dimension ?dim ;
        :field_name ?dimFieldName .

    OPTIONAL { ?variable schema:name ?variableName } .
    OPTIONAL { ?variable :cf_standard_name ?variableStandardName } .
    OPTIONAL { ?variable schema:description ?variableDescription } .
    OPTIONAL { ?variable schema:unitText ?variableUnits } .

    FILTER (
        (!BOUND(?filterTimeRangeStart) || ?endDate >= ?filterTimeRangeStart) &&
        (!BOUND(?filterTimeRangeEnd) || ?startDate <= ?filterTimeRangeEnd)
    ) .

    FILTER (
        !BOUND(?pattern) ||
        regex(
            CONCAT(
                ?datasetLabel, " ",
                ?description, " ",
                ?variableName, " ",
                ?variableLabel, " ",
                ?variableFieldName, " ",
                IF(BOUND(?variableStandardName), STR(?variableStandardName), ""), " ",
                IF(BOUND(?variableDescription), ?variableDescription, "")),
            ?pattern,
            "is"
        )
    ) .

    BIND (STRAFTER(STR(?dataset), "__KG_NAMESPACE__") AS ?dataset_id) .
    BIND (STRAFTER(STR(?variable), "__KG_NAMESPACE__") AS ?variable_id) .
    BIND (STRAFTER(STR(?dim), "__KG_NAMESPACE__") AS ?dim_id) .
}
GROUP BY ?dataset_id ?variable_id
ORDER BY ?dataset_id ?variable_id
""".replace("__KG_NAMESPACE__", KG_NAMESPACE)
)


FETCH_VARIABLES_QUERY = prepareQuery(
    """
PREFIX : <http://kmnlp-demo.noaa.gov/>
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX qb: <http://purl.org/linked-data/cube#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <https://schema.org/>

SELECT
    ?dataset_id
    ?datasetLabel
    ?variable_id
    ?variableName
    ?variableFieldName
    ?variableStandardName
    ?variableDescription
    ?variableUnits
    (GROUP_CONCAT(?dim_id; SEPARATOR=";") AS ?dimIds)
    (GROUP_CONCAT(?dimFieldName; SEPARATOR=";") AS ?dimFieldNames)
WHERE {
    ?dataset a dcat:Dataset ;
        rdfs:label ?datasetLabel ;
        schema:variableMeasured ?variable ;
        dcterms:description ?datasetDescription ;
        qb:structure ?dataStructDef .

    ?dataStructDef qb:component ?dataStructDefComp .
    ?dataStructDefComp qb:measure ?variable .

    ?dataStructDef qb:component ?dataStructDefCompDim .
    ?dataStructDefCompDim qb:dimension ?dim ;
        :field_name ?dimFieldName .

    ?variable rdfs:label ?variableLabel ;
        schema:alternateName ?variableFieldName .

    OPTIONAL { ?variable schema:name ?variableName } .
    OPTIONAL { ?variable :cf_standard_name ?variableStandardName } .
    OPTIONAL { ?variable schema:description ?variableDescription } .
    OPTIONAL { ?variable schema:unitText ?variableUnits } .

    FILTER (
        (!BOUND(?filterTimeRangeStart) || ?endDate >= ?filterTimeRangeStart) &&
        (!BOUND(?filterTimeRangeEnd) || ?startDate <= ?filterTimeRangeEnd)
    ) .

    FILTER (
        !BOUND(?pattern) ||
        regex(
            CONCAT(
                ?datasetDescription, " ",
                ?variableName, " ",
                ?variableLabel, " ",
                ?variableFieldName, " ",
                IF(BOUND(?variableStandardName), STR(?variableStandardName), ""), " ",
                IF(BOUND(?variableDescription), ?variableDescription, "")),
            ?pattern,
            "is"
        )
    ) .

    BIND (STRAFTER(STR(?dataset), "__KG_NAMESPACE__") AS ?dataset_id) .
    BIND (STRAFTER(STR(?variable), "__KG_NAMESPACE__") AS ?variable_id) .
    BIND (STRAFTER(STR(?dim), "__KG_NAMESPACE__") AS ?dim_id) .
}
GROUP BY ?dataset_id ?variable_id
""".replace("__KG_NAMESPACE__", KG_NAMESPACE)
)

GET_DATASET_FOR_VARIABLE_QUERY = prepareQuery(
    """
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <https://schema.org/>

SELECT
    (STRAFTER(STR(?dataset), "__KG_NAMESPACE__") AS ?dataset_id)
    ?datasetLabel
    ?description
    ?landingPage
    ?sourceUri
    ?sourceType
    ?startDate
    ?endDate
    ?temporalResolution
WHERE {
    ?dataset a schema:Dataset ;
        a dcat:Dataset ;
        schema:variableMeasured ?variable ;
        rdfs:label ?datasetLabel ;
        dcat:distribution ?distribution ;
        dcat:startDate ?startDate  ;
        dcat:endDate ?endDate  .

    ?distribution dcat:accessURL ?sourceUri .
    ?distribution dcat:mediaType ?sourceType .

    ?dataset schema:description ?description  .

    OPTIONAL { ?dataset schema:url ?landingPage } .
    OPTIONAL { ?dataset dcat:temporalResolution ?temporalResolution } .
}
""".replace("__KG_NAMESPACE__", KG_NAMESPACE)
)

FETCH_CANONICAL_DIMENSIONS_QUERY = prepareQuery(
    """
PREFIX : <http://kmnlp-demo.noaa.gov/>
PREFIX qb: <http://purl.org/linked-data/cube#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <https://schema.org/>

SELECT
    ?dimension_id
    ?cfStandardName
    (GROUP_CONCAT(?alternateName; SEPARATOR=",") AS ?alternateNames)
WHERE {
    ?dimension a qb:DimensionProperty .

    BIND (STRAFTER(STR(?dimension), "__KG_NAMESPACE__") AS ?dimension_id) .

    OPTIONAL { ?dimension :cf_standard_name ?cfStandardName } .
    OPTIONAL { ?dimension schema:alternateName ?alternateName } .
}
GROUP BY ?dimension_id
""".replace("__KG_NAMESPACE__", KG_NAMESPACE)
)


class _ResultRowModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _DatasetResultRow(_ResultRowModel):
    dataset_id: str
    datasetLabel: str
    description: str
    landingPage: str | None = None
    sourceUri: str
    sourceType: str
    startDate: str
    endDate: str
    temporalResolution: str | None = None


class _VariableResultRow(_ResultRowModel):
    variable_id: str
    variableName: str | None = None
    datasetLabel: str
    variableFieldName: str
    variableStandardName: str | None = None
    variableDescription: str | None = None
    variableUnits: str | None = None
    dimIds: str  # semicolon-separated list of dimension IDs
    dimFieldNames: str  # semicolon-separated list of dimension field names


class _CanonicalDimensionResultRow(_ResultRowModel):
    dimension_id: str
    cfStandardName: str | None = None
    alternateNames: str | None = None


def _parse_datasets_query_result_row(
    row: _DatasetResultRow, variables: list[Variable] | None = None
) -> Dataset:
    """Map single result row from fetch-datasets query to Dataset model."""
    return Dataset(
        id=row.dataset_id,
        source=DatasetSource(uri=row.sourceUri, format=basename(row.sourceType)),  # pyright: ignore[reportArgumentType]
        label=row.datasetLabel,
        description=row.description,
        landing_page=row.landingPage,
        # rely on pydantic to implicitly parse these to datetimes
        temporal_coverage=(row.startDate, row.endDate),  # type: ignore[reportArgumentType]
        # rely on pydantic to implicitly parse this to timedelta
        temporal_resolution=row.temporalResolution,  # type: ignore[reportArgumentType]
        variables=variables if variables is not None else [],
    )


def _parse_variables_query_result_row(
    row: _VariableResultRow, dimensions: dict[str, CanonicalDimension]
) -> Variable:
    """Map single result row from fetch-variables query to Variable model."""
    canon_dims = [dimensions[dim_id] for dim_id in row.dimIds.split(";")]
    field_names = row.dimFieldNames.split(";")
    dims = [
        Dimension.model_validate(
            {
                **canon_dim.model_dump(),
                "field_name": field_name,
            }
        )
        for canon_dim, field_name in zip(canon_dims, field_names, strict=True)
    ]
    return Variable(
        id=row.variable_id,
        name=row.variableName or row.variableFieldName,
        dataset_name=row.datasetLabel,
        field_name=row.variableFieldName,
        description=row.variableDescription,
        standard_name=row.variableStandardName,
        units=row.variableUnits,
        dimensions=dims,
    )


def _parse_canonical_dimensions_query_result_row(
    row: _CanonicalDimensionResultRow,
) -> CanonicalDimension:
    """Map single result row from fetch-canonical-dimensions query to CanonicalDimension model."""
    return CanonicalDimension(
        id=row.dimension_id,
        cf_standard_name=basename(row.cfStandardName) if row.cfStandardName else None,
        alternate_names=frozenset(row.alternateNames.split(",") if row.alternateNames else []),
    )


class KnowledgeGraphDAO:
    """Data Access Object for accessing the knowledge graph."""

    graph: Graph
    namespace: str = KG_NAMESPACE

    def __init__(self, kg_paths: Sequence[str | Traversable] = KG_TTL_PATHS) -> None:
        """Initializer for knowledge graph.

        Args:
            kg_paths (Sequence[str | Traversable]): Paths to serialized representations of
                the knowledge graph as e.g. Turtle or JSON-LD etc. files.
                N.B. Paths can be str or `pathlib.Path`.
        """
        self.graph = Graph()
        for kg_path in kg_paths:
            if isinstance(kg_path, str):
                self.graph.parse(kg_path)
            else:
                with as_file(kg_path) as path:
                    self.graph.parse(path)

    def id_to_iri(self, path: str) -> str:
        """Convert a path to an IRI within this KG's namespace."""
        return join(self.namespace, path)

    def iri_to_id(self, iri: str) -> str:
        """Convert an IRI within this KG's namespace to an ID by stripping the namespace."""
        return iri.removeprefix(self.namespace)

    @observe(as_type="retriever")
    def fetch_all_triples(self) -> list[tuple[str, str, str]]:
        """Fetch all triples present in the graph.

        Intended largely for debugging purposes
        """
        return [
            (triple.s, triple.p, triple.o)  # type: ignore[reportUnknownMember]
            for triple in self.graph.query(FETCH_ALL_NODES_QUERY)
        ]

    @observe(as_type="retriever")
    def fetch_datasets(
        self,
        *,
        ids: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        regex_pattern: str | None = None,
    ) -> list[Dataset]:
        """Fetch datasets represented in the knowledge graph.

        Args:
            ids (list[str] | None): If provided, return only datasets with
                these IDs.
            start_time (datetime | None): If provided, this
                function returns only variables where the datasets to which
                they belong start at or before the provided time.
            end_time (datetime | None): If provided, this function
                returns only variables where the datasets to which they belong
                end at or after the provided time.
            regex_pattern (str | None): Filter to datasets and variables whose
                metadata matches the provided regex pattern.

        Returns:
            A list of datasets satisfying the provided criteria.
        """
        if ids is not None and len(ids) == 0:
            return []

        bindings: dict[str, RDFLiteral | URIRef] = {
            k: RDFLiteral(v)
            for k, v in {
                "filterTimeRangeStart": start_time,
                "filterTimeRangeEnd": end_time,
                "pattern": regex_pattern,
            }.items()
            if v is not None
        }

        if ids is not None:
            # Use initBindings per ID for efficient lookup (avoids slow VALUES clause)
            all_rows: list[dict[str, object]] = []
            for id_ in ids:
                id_bindings = {**bindings, "dataset": URIRef(self.id_to_iri(id_))}
                rows = self.graph.query(FETCH_DATASETS_QUERY, initBindings=id_bindings)
                all_rows.extend(cast("ResultRow", row).asdict() for row in rows)  # type: ignore[reportArgumentType]
        else:
            result = self.graph.query(
                FETCH_DATASETS_QUERY, initBindings=bindings if len(bindings) > 0 else None
            )
            all_rows = [cast("ResultRow", row).asdict() for row in result]  # type: ignore[reportAssignmentType]

        grouped_rows = groupby(all_rows, key=lambda r: r["dataset_id"])
        datasets: list[Dataset] = []
        for _, group_rows in grouped_rows:
            group_row_dicts = list(group_rows)
            first_row = group_row_dicts[0]
            ds_vars = [
                _parse_variables_query_result_row(
                    _VariableResultRow.model_validate(row), self.canonical_dimensions
                )
                for row in group_row_dicts
            ]
            ds = _parse_datasets_query_result_row(
                _DatasetResultRow.model_validate(first_row), variables=ds_vars
            )
            datasets.append(ds)

        return datasets

    @cached_property
    @observe(as_type="retriever")
    def canonical_dimensions(self) -> dict[str, CanonicalDimension]:
        """Fetch all dimensions represented in the knowledge graph.

        Returns:
            A dictionary mapping dimension IDs to dimensions.
        """
        dimensions: dict[str, CanonicalDimension] = {}
        for row in self.graph.query(FETCH_CANONICAL_DIMENSIONS_QUERY):
            row = cast("ResultRow", row)
            dim = _parse_canonical_dimensions_query_result_row(
                _CanonicalDimensionResultRow.model_validate(row.asdict())
            )
            dimensions[dim.id] = dim
        return dimensions

    @observe(as_type="retriever")
    def get_variable_by_id(self, var_id: str) -> Variable | None:
        """Fetch a variable by ID.

        Args:
            var_id (str): The ID of the variable to be fetched from the graph

        Returns:
            The `Variable` model based on fetched graph data, or None if not
                found.

        Raises:
            pydantic.ValidationError: If the IRI is found in the graph, but
                expected relationships for that IRI are not present.
        """
        iri_ref = URIRef(self.id_to_iri(var_id))

        query_result = self.graph.query(FETCH_VARIABLES_QUERY, initBindings={"variable": iri_ref})
        try:
            first_result_row = cast("ResultRow", next(iter(query_result)))
        except StopIteration:
            return None

        return _parse_variables_query_result_row(
            _VariableResultRow.model_validate(first_result_row.asdict()), self.canonical_dimensions
        )

    @observe(as_type="retriever")
    def get_dataset_by_variable_id(self, var_id: str) -> Dataset | None:
        """Fetch the dataset to which a given variable belongs.

        Args:
            var_id (str): The ID of the variable whose dataset to fetch from the
                graph.

        Returns:
            The `Dataset` model based on fetched graph data, or None if not
                found.

        Raises:
            pydantic.ValidationError: If the IRI is found in the graph, but
                expected relationships for that IRI are not present.
        """
        query_result = self.graph.query(
            GET_DATASET_FOR_VARIABLE_QUERY,
            initBindings={"variable": URIRef(self.id_to_iri(var_id))},
        )
        try:
            first_result_row = cast("ResultRow", next(iter(query_result)))
        except StopIteration:
            return None

        ds = _parse_datasets_query_result_row(
            _DatasetResultRow.model_validate(first_result_row.asdict())
        )

        # populate variables
        ds_vars_results = self.graph.query(
            FETCH_VARIABLES_QUERY, initBindings={"dataset": URIRef(self.id_to_iri(ds.id))}
        )
        for row in ds_vars_results:
            row = cast("ResultRow", row)
            ds.variables.append(
                _parse_variables_query_result_row(
                    _VariableResultRow.model_validate(row.asdict()), self.canonical_dimensions
                )
            )

        return ds

    @observe(as_type="retriever")
    def execute_read_only_query(self, query: str) -> list[str]:
        """Execute the provided query in a read-only context.

        Args:
            query (str): A SPARQL read query

        Returns:
            dict[str, Any]: The query results
        """
        return [str(row) for row in self.graph.query(query)]
