import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import boto3
import botocore.config
import pytest
from chromadb import EphemeralClient
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from natural_language_geocoding.tests.canned_place_lookup import CannedPlaceLookup
from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from llm_agent.context import AgentDeps, ChatContext, LLMAgentContext, ProgressCallback
from llm_agent.dao.chromadb_embedding_dao import ChromaDBEmbeddingDAO
from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from llm_agent.repository.embedding_index_repository import EmbeddingIndexRepository
from llm_agent.tools.agent_tool import AgentTool
from tests.utils.mocks.context import MockChatContext

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
) -> LLMAgentContext:
    """Fixture for a default global context."""
    return LLMAgentContext(
        place_lookup=canned_place_lookup,
        knowledge_graph_dao=default_knowledge_graph_dao,
        embedding_repo=None,
        dask_client=Mock(),
        bedrock_client=Mock(),
    )


@pytest.fixture
def chat_context(global_context: LLMAgentContext) -> MockChatContext:
    """Fixture for a default ChatContext."""
    return MockChatContext(global_context=global_context)


@pytest.fixture
def make_chat_context(
    default_knowledge_graph_dao: KnowledgeGraphDAO, canned_place_lookup: CannedPlaceLookup
) -> Callable[..., MockChatContext]:
    """Fixture for a mock ChatContext."""

    def _make_chat_context(
        knowledge_graph_dao: KnowledgeGraphDAO | None = None,
        embedding_repo: EmbeddingIndexRepository | None = None,
    ) -> MockChatContext:
        kg = knowledge_graph_dao or default_knowledge_graph_dao
        global_context = LLMAgentContext(
            place_lookup=canned_place_lookup,
            knowledge_graph_dao=kg,
            embedding_repo=embedding_repo,
            dask_client=Mock(),
            bedrock_client=Mock(),
        )
        return MockChatContext(global_context=global_context)

    return _make_chat_context


@pytest.fixture
def make_run_context() -> Callable[..., RunContext[AgentDeps]]:
    """Factory for a Pydantic AI ``RunContext`` for unit-testing ``AgentTool``s.

    Tools reach their dependencies via ``ctx.deps``; ``tool_call_id`` must be
    non-``None`` for progress updates to be surfaced, so it defaults to a stub id.
    The ``model`` and ``usage`` fields are required by ``RunContext`` but unused
    by the tools, so they are filled with throwaway values.
    """

    def _make_run_context(
        chat_context: ChatContext,
        *,
        progress_callback: ProgressCallback | None = None,
        tool_call_id: str = "test-tool-call-id",
    ) -> RunContext[AgentDeps]:
        deps = AgentDeps(chat_context=chat_context, progress_callback=progress_callback)
        return RunContext(
            deps=deps,
            model=TestModel(),
            usage=RunUsage(),
            tool_call_id=tool_call_id,
        )

    return _make_run_context


@pytest.fixture
def invoke_tool_safely() -> Callable[..., Awaitable[Any]]:
    """Invoke an ``AgentTool`` through its agent-facing safety net.

    Calling a tool's ``__call__`` directly runs the *pure* logic, which may
    raise. Agents instead invoke the registered :meth:`AgentTool.as_pydantic_tool`
    wrapper, which catches unhandled exceptions and returns an ``ErrorResponse``.
    This helper exercises that wrapper so tests can assert the catch-all behavior.
    """

    async def _invoke(
        tool: AgentTool[Any, Any], ctx: RunContext[AgentDeps], tool_input: BaseModel
    ) -> Any:  # noqa: ANN401
        return await tool.as_pydantic_tool().function(ctx, tool_input)

    return _invoke


@pytest.fixture
def proxied_bedrock_client() -> BedrockRuntimeClient:
    return boto3.client(  # type: ignore[reportUnknownMemberType]
        "bedrock-runtime",
        config=botocore.config.Config(read_timeout=300, connect_timeout=1),
        endpoint_url=os.getenv("LITE_LLM_PROXY_URL"),
    )


@pytest.fixture
def global_context_integration(
    dask_client: "Client",
    default_knowledge_graph_dao: KnowledgeGraphDAO,
    canned_place_lookup: CannedPlaceLookup,
    proxied_bedrock_client: BedrockRuntimeClient,
) -> LLMAgentContext:
    """Context for integration tests, with live Dask client."""
    return LLMAgentContext(
        place_lookup=canned_place_lookup,
        knowledge_graph_dao=default_knowledge_graph_dao,
        embedding_repo=EmbeddingIndexRepository(
            embedding_dao=ChromaDBEmbeddingDAO(
                client=EphemeralClient(), collection_name="datasets"
            ),
            datasets=default_knowledge_graph_dao.fetch_datasets(),
        ),
        dask_client=dask_client,
        bedrock_client=proxied_bedrock_client,
    )


@pytest.fixture
def chat_context_integration(global_context_integration: LLMAgentContext) -> MockChatContext:
    """ChatContext for integration tests, with live Dask client."""
    return MockChatContext(global_context=global_context_integration)
