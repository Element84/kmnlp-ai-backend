from collections.abc import Callable
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from natural_language_geocoding.tests.canned_place_lookup import CannedPlaceLookup

from llm_agent.context import LLMAgentContext
from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO

if TYPE_CHECKING:
    from dask.distributed import Client
    from shapely.geometry.base import BaseGeometry


@pytest.fixture(scope="session")
def default_knowledge_graph_dao() -> KnowledgeGraphDAO:
    return KnowledgeGraphDAO()


@pytest.fixture(scope="session")
def canned_place_lookup(
    black_sea_geometry: "BaseGeometry",
    gulf_of_mexico_geometry: "BaseGeometry",
) -> CannedPlaceLookup:
    place_lookup = CannedPlaceLookup()
    place_lookup.name_to_geom["gulf of mexico"] = gulf_of_mexico_geometry
    place_lookup.name_to_geom["black sea"] = black_sea_geometry
    return place_lookup


@pytest.fixture
def global_context(
    default_knowledge_graph_dao: KnowledgeGraphDAO, canned_place_lookup: CannedPlaceLookup
) -> Callable[..., LLMAgentContext]:
    """Context for unit tests."""

    def _make_context(knowledge_graph_dao: KnowledgeGraphDAO | None = None) -> LLMAgentContext:
        kg = knowledge_graph_dao if knowledge_graph_dao else default_knowledge_graph_dao
        return LLMAgentContext(
            place_lookup=canned_place_lookup,
            s3_client=Mock(),
            knowledge_graph_dao=kg,
            dask_client=Mock(),
        )

    return _make_context


@pytest.fixture
def global_context_integration(
    dask_client: "Client",
    default_knowledge_graph_dao: KnowledgeGraphDAO,
    canned_place_lookup: CannedPlaceLookup,
) -> LLMAgentContext:
    """Context for integration tests, with live Dask client."""
    return LLMAgentContext(
        place_lookup=canned_place_lookup,
        s3_client=Mock(),
        knowledge_graph_dao=default_knowledge_graph_dao,
        dask_client=dask_client,
    )
