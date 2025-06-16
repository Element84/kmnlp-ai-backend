from collections.abc import Iterable
from importlib.resources import as_file
from importlib.resources.abc import Traversable
from typing import TYPE_CHECKING, cast

from rdflib import Graph, URIRef
from rdflib.plugins.sparql import prepareQuery

if TYPE_CHECKING:
    from rdflib.query import ResultRow

from llm_agent.model.dataset import Dataset
from llm_agent.model.variable import Variable

FETCH_ALL_NODES_QUERY = """
SELECT ?s ?p ?o
WHERE {
    ?s ?p ?o
}
"""

FETCH_DATASETS_QUERY = prepareQuery("""
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <https://schema.org/>

SELECT DISTINCT
    ?dataset
    ?label
    ?description
    ?source
    ?startDate
    ?endDate
    ?temporalResolution
WHERE {
    ?dataset a schema:Dataset ;
        a dcat:Dataset ;
        rdfs:label ?label ;
        dcat:distribution ?distribution ;
        dcat:startDate ?startDate  ;
        dcat:endDate ?endDate  .

    ?distribution dcat:accessURL ?source .

    OPTIONAL { ?dataset schema:description ?description } .
    OPTIONAL { ?dataset dcat:temporalResolution ?temporalResolution } .
}
""")

FETCH_VARIABLES_QUERY = prepareQuery("""
PREFIX : <http://kmnlp-demo.noaa.gov/>
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX qb: <http://purl.org/linked-data/cube#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <https://schema.org/>

SELECT
    ?dataset
    ?variable
    ?name
    ?fieldName
    ?description
    ?units
    (GROUP_CONCAT(?dimName; SEPARATOR=",") AS ?dimNames)
WHERE {
    ?dataset a schema:Dataset ;
        a dcat:Dataset ;
        schema:variableMeasured ?variable ;
        qb:structure ?dataStructDef .

    ?dataStructDef a qb:DataStructureDefinition ;
        qb:component ?dataStructDefComp .
    ?dataStructDefComp qb:measure ?variable .

    ?dataStructDef qb:component ?dataStructDefCompDim .
    ?dataStructDefCompDim qb:dimension ?dim .
    ?dim schema:name ?dimName .

    ?variable a qb:MeasureProperty ;
        schema:name ?name ;
        schema:alternateName ?fieldName .

    OPTIONAL { ?variable schema:description ?description } .
    OPTIONAL { ?variable schema:unitText ?units } .
}
GROUP BY ?variable
""")

GET_DATASET_FOR_VARIABLE_QUERY = prepareQuery("""
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <https://schema.org/>

SELECT
    ?dataset
    ?label
    ?description
    ?source
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

    ?distribution dcat:accessURL ?source .

    OPTIONAL { ?dataset schema:description ?description } .
    OPTIONAL { ?dataset dcat:temporalResolution ?temporalResolution } .
}
""")


def _parse_datasets_query_result_row(row: "ResultRow") -> Dataset:
    """Map single result row from fetch-datasets query to Dataset model."""
    return Dataset(
        iri=row.dataset,
        source=row.source,
        label=row.label,
        description=row.description,
        # rely on pydantic to implicitly parse these to datetimes
        temporal_coverage=(row.startDate, row.endDate),  # type: ignore[reportArgumentType]
        # rely on pydantic to implicitly parse this to timedelta
        temporal_resolution=row.temporalResolution,  # type: ignore[reportArgumentType]
        variables=[],
    )


def _parse_variables_query_result_row(row: "ResultRow") -> Variable:
    """Map single result row from fetch-variables query to Variable model."""
    return Variable(
        iri=row.variable,
        name=row.name,
        field_name=row.fieldName,
        description=row.description,
        units=row.units,
        dimensions=frozenset(row.dimNames.split(",")),
    )


class KnowledgeGraphDAO:
    """Data Access Object for accessing the knowledge graph."""

    graph: Graph

    def __init__(self, kg_path: str | Traversable) -> None:
        """Initializer for knowledge graph.

        Args:
            kg_path (str | Traversable): Path to a serialized representation of
                the knowledge graph as e.g. a Turtle or JSON-LD etc. file.
                N.B. This arg can be a `pathlib.Path`.
        """
        self.graph = Graph()
        if isinstance(kg_path, str):
            self.graph.parse(kg_path)
        else:
            with as_file(kg_path) as path:
                self.graph.parse(path)

    def fetch_all_triples(self) -> list[tuple[str, str, str]]:
        """Fetch all triples present in the graph.

        Intended largely for debugging purposes
        """
        return [
            (triple.s, triple.p, triple.o)  # type: ignore[reportUnknownMember]
            for triple in self.graph.query(FETCH_ALL_NODES_QUERY)
        ]

    def _fetch_datasets(self) -> list[Dataset]:
        """Fetch all the datasets represented in the knowledge graph."""
        return [
            _parse_datasets_query_result_row(cast("ResultRow", row))
            for row in self.graph.query(FETCH_DATASETS_QUERY)
        ]

    def fetch_variables(self) -> list[Dataset]:
        """Fetch all the variables represented in the knowledge graph.

        Returns:
            Variables found, related to their datasets.
        """
        datasets = self._fetch_datasets()
        ds_iri_to_ds: dict[str, Dataset] = {
            dataset.iri: dataset for dataset in datasets
        }

        for row in self.graph.query(FETCH_VARIABLES_QUERY):
            row = cast("ResultRow", row)
            var = _parse_variables_query_result_row(row)
            ds_iri = str(row.dataset)
            ds_iri_to_ds[ds_iri].variables.append(var)

        return datasets

    def get_variable_by_iri(self, iri: str) -> Variable | None:
        """Fetch a variable by IRI.

        Args:
            iri (str): The IRI of the variable to be fetched from the graph

        Returns:
            The `Variable` model based on fetched graph data, or None if not
                found.

        Raises:
            pydantic.ValidationError: If the IRI is found in the graph, but
                expected relationships for that IRI are not present.
        """
        iri_ref = URIRef(iri)

        query_result = self.graph.query(
            FETCH_VARIABLES_QUERY, initBindings={"variable": iri_ref}
        )
        try:
            first_result_row = cast("ResultRow", next(iter(query_result)))
        except StopIteration:
            return None

        return _parse_variables_query_result_row(first_result_row)

    def get_dataset_by_variable_iri(self, iri: str) -> Dataset | None:
        """Fetch the dataset to which a given variable belongs.

        Args:
            iri (str): The IRI of the variable whose dataset to fetch from the
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
            initBindings={"variable": URIRef(iri)},
        )
        try:
            first_result_row = cast("ResultRow", next(iter(query_result)))
        except StopIteration:
            return None

        ds = _parse_datasets_query_result_row(first_result_row)

        # populate variables
        ds_vars_results = self.graph.query(
            FETCH_VARIABLES_QUERY, initBindings={"dataset": URIRef(ds.iri)}
        )
        for row in ds_vars_results:
            row = cast("ResultRow", row)
            ds.variables.append(_parse_variables_query_result_row(row))

        return ds

    def get_datasets_by_iris(self, iris: Iterable[str]) -> list[Dataset]:
        """Fetch datasets with the given IRIs.

        Args:
            iris (list[str]): The IRIs of the datasets to fetch.

        Returns:
            list[Dataset]: Datasets.
        """
        iris_set = set(iris)
        all_datasets = self.fetch_variables()
        filtered_datasets = [d for d in all_datasets if d.iri in iris_set]
        return filtered_datasets
