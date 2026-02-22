from collections.abc import Sequence
from datetime import datetime
from importlib.resources import as_file
from importlib.resources.abc import Traversable
from itertools import groupby
from os.path import basename, join
from string import Template
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

FETCH_DATASETS_QUERY = Template(
    """
PREFIX : <http://kmnlp-demo.noaa.gov/>
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX qb: <http://purl.org/linked-data/cube#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <https://schema.org/>

SELECT
    ?dataset_id
    ?label
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
    ?variableDescription
    ?variableUnits
    (GROUP_CONCAT(?dim_id; SEPARATOR=";") AS ?dimIds)
    (GROUP_CONCAT(?dimFieldName; SEPARATOR=";") AS ?dimFieldNames)
WHERE {
    ${values_clause}

    ?dataset a schema:Dataset ;
        a dcat:Dataset ;
        rdfs:label ?label ;
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

    ?dataStructDef a qb:DataStructureDefinition ;
        qb:component ?dataStructDefComp .
    ?dataStructDefComp qb:measure ?variable .

    ?variable a qb:MeasureProperty ;
        schema:name ?variableName ;
        schema:alternateName ?variableFieldName .

    ?dataStructDef qb:component ?dataStructDefCompDim .
    ?dataStructDefCompDim qb:dimension ?dim ;
        :field_name ?dimFieldName .

    OPTIONAL { ?variable schema:description ?variableDescription } .
    OPTIONAL { ?variable schema:unitText ?variableUnits } .

    FILTER (
        (!BOUND(?filterTimeRangeStart) || ?endDate >= ?filterTimeRangeStart) &&
        (!BOUND(?filterTimeRangeEnd) || ?startDate <= ?filterTimeRangeEnd)
    ) .

    FILTER (
        !BOUND(?pattern) ||
        (
            regex(?label, ?pattern, "i") ||
            regex(?description, ?pattern, "i") ||
            regex(?variableName, ?pattern, "i") ||
            regex(?variableFieldName, ?pattern, "i") ||
            (!BOUND(?variableDescription) || regex(?variableDescription, ?pattern, "i"))
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
    ?variable_id
    ?variableName
    ?variableFieldName
    ?variableDescription
    ?variableUnits
    (GROUP_CONCAT(?dim_id; SEPARATOR=";") AS ?dimIds)
    (GROUP_CONCAT(?dimFieldName; SEPARATOR=";") AS ?dimFieldNames)
WHERE {
    ?dataset a schema:Dataset ;
        a dcat:Dataset ;
        schema:variableMeasured ?variable ;
        qb:structure ?dataStructDef .

    ?dataStructDef a qb:DataStructureDefinition ;
        qb:component ?dataStructDefComp .
    ?dataStructDefComp qb:measure ?variable .

    ?dataStructDef qb:component ?dataStructDefCompDim .
    ?dataStructDefCompDim qb:dimension ?dim ;
        :field_name ?dimFieldName .

    ?variable a qb:MeasureProperty ;
        schema:name ?variableName ;
        schema:alternateName ?variableFieldName .

    OPTIONAL { ?variable schema:description ?variableDescription } .
    OPTIONAL { ?variable schema:unitText ?variableUnits } .

    FILTER (
        (!BOUND(?filterTimeRangeStart) || ?endDate >= ?filterTimeRangeStart) &&
        (!BOUND(?filterTimeRangeEnd) || ?startDate <= ?filterTimeRangeEnd)
    ) .

    FILTER (
        !BOUND(?pattern) ||
        (
            regex(?datasetDescription, ?pattern, "i") ||
            regex(?variableName, ?pattern, "i") ||
            regex(?variableFieldName, ?pattern, "i") ||
            (!BOUND(?variableDescription) || regex(?variableDescription, ?pattern, "i"))
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
    ?label
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
        rdfs:label ?label ;
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
    label: str
    description: str
    landingPage: str | None = None  # noqa: N815
    sourceUri: str  # noqa: N815
    sourceType: str  # noqa: N815
    startDate: str  # noqa: N815
    endDate: str  # noqa: N815
    temporalResolution: str | None = None  # noqa: N815


class _VariableResultRow(_ResultRowModel):
    variable_id: str
    variableName: str  # noqa: N815
    variableFieldName: str  # noqa: N815
    variableDescription: str | None = None  # noqa: N815
    variableUnits: str | None = None  # noqa: N815
    dimIds: str  # semicolon-separated list of dimension IDs# noqa: N815
    dimFieldNames: str  # semicolon-separated list of dimension field names# noqa: N815


class _CanonicalDimensionResultRow(_ResultRowModel):
    dimension_id: str
    cfStandardName: str | None = None  # noqa: N815
    alternateNames: str | None = None  # noqa: N815


def _parse_datasets_query_result_row(
    row: _DatasetResultRow, variables: list[Variable] | None = None
) -> Dataset:
    """Map single result row from fetch-datasets query to Dataset model."""
    return Dataset(
        id=row.dataset_id,
        source=DatasetSource(uri=row.sourceUri, format=basename(row.sourceType)),  # pyright: ignore[reportArgumentType]
        label=row.label,
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
        name=row.variableName,
        field_name=row.variableFieldName,
        description=row.variableDescription,
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
        if ids is not None and len(ids) > 0:
            # Convert the list into a SPARQL-compatible VALUES clause
            # Format: VALUES ?dataset { <iri1> <iri2> ... }
            iris = [self.id_to_iri(id_) for id_ in ids]
            iris_formatted = " ".join(f"<{iri}>" for iri in iris)
            values_clause = f"VALUES ?dataset {{ {iris_formatted} }}"
        else:
            values_clause = ""
        query_str = FETCH_DATASETS_QUERY.safe_substitute(values_clause=values_clause)

        bindings: dict[str, RDFLiteral] | None = {
            k: RDFLiteral(v)
            for k, v in {
                "filterTimeRangeStart": start_time,
                "filterTimeRangeEnd": end_time,
                "pattern": regex_pattern,
            }.items()
            if v is not None
        }

        dimensions = self.get_canonical_dimensions()
        dim_id_to_dim = {dim.id: dim for dim in dimensions}

        rows = self.graph.query(query_str, initBindings=bindings if len(bindings) > 0 else None)
        grouped_rows = groupby(rows, key=lambda r: cast("ResultRow", r).dataset_id)
        datasets: list[Dataset] = []
        for _, group_rows in grouped_rows:
            group_row_dicts = [cast("ResultRow", row).asdict() for row in group_rows]
            first_row = group_row_dicts[0]
            ds_vars = [
                _parse_variables_query_result_row(
                    _VariableResultRow.model_validate(row), dim_id_to_dim
                )
                for row in group_row_dicts
            ]
            ds = _parse_datasets_query_result_row(
                _DatasetResultRow.model_validate(first_row), variables=ds_vars
            )
            datasets.append(ds)

        return datasets

    @observe(as_type="retriever")
    def get_canonical_dimensions(self) -> list[CanonicalDimension]:
        """Fetch all dimensions represented in the knowledge graph.

        Returns:
            A list of dimensions.
        """
        dimensions: list[CanonicalDimension] = []
        for row in self.graph.query(FETCH_CANONICAL_DIMENSIONS_QUERY):
            row = cast("ResultRow", row)
            dim = _parse_canonical_dimensions_query_result_row(
                _CanonicalDimensionResultRow.model_validate(row.asdict())
            )
            dimensions.append(dim)
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

        dimensions = self.get_canonical_dimensions()
        dim_id_to_dim = {dim.id: dim for dim in dimensions}

        return _parse_variables_query_result_row(
            _VariableResultRow.model_validate(first_result_row.asdict()), dim_id_to_dim
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

        dimensions = self.get_canonical_dimensions()
        dim_id_to_dim = {dim.id: dim for dim in dimensions}

        # populate variables
        ds_vars_results = self.graph.query(
            FETCH_VARIABLES_QUERY, initBindings={"dataset": URIRef(self.id_to_iri(ds.id))}
        )
        for row in ds_vars_results:
            row = cast("ResultRow", row)
            ds.variables.append(
                _parse_variables_query_result_row(
                    _VariableResultRow.model_validate(row.asdict()), dim_id_to_dim
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
