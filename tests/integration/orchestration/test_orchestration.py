import json
import logging
import os
from collections import Counter
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pytest
from chromadb import EphemeralClient
from langfuse import observe  # pyright: ignore[reportUnknownVariableType]
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from natural_language_geocoding.tests.canned_place_lookup import CannedPlaceLookup
from pydantic_ai import AgentRunResult, AgentRunResultEvent, AgentStreamEvent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
)

from llm_agent import (
    ALGORITHM_AGENT_DEFAULT_LLM_MODEL_ID,
    DATASET_PICKING_AGENT_DEFAULT_LLM_MODEL_ID,
    ORCHESTRATION_AGENT_DEFAULT_LLM_MODEL_ID,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.base import (
    ComputedVariableUpdate,
    ComputingVariableUpdate,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.data_array import (
    DaskExecutionProgressUpdate,
    FilterByPolygonUpdate,
    GeocodingSuccessUpdate,
)
from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import DatasetPickingAgent
from llm_agent.agents.orchestration_agent.interface import UserMessage
from llm_agent.agents.orchestration_agent.orchestration_agent import OrchestrationAgent
from llm_agent.context import ChatContext, LLMAgentContext, ProgressCallback, ToolProgressUpdate
from llm_agent.dao.chromadb_embedding_dao import ChromaDBEmbeddingDAO
from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from llm_agent.dao.utils import get_kg_ttl_paths
from llm_agent.model_factory import make_bedrock_model
from llm_agent.repository.embedding_index_repository import EmbeddingIndexRepository
from llm_agent.tools.agent_plan_tool import AgentPlanOrUpdate, agent_plan_tool
from llm_agent.tools.data_analysis_tool.data_analysis_tool import (
    DataAnalysisTool,
    DataAnalysisToolOutput,
)
from llm_agent.tools.data_analysis_tool.types import GeometryRef, GeoTIFFRef
from llm_agent.tools.data_search_tool import (
    DataSearchTool,
    SearchCompletedUpdate,
    SearchInitiatedUpdate,
)
from llm_agent.tools.fetch_datasets_tool import KnowledgeGraphQueryUpdate, QueryResultsUpdate
from llm_agent.tools.nl_geocoding_tool import (
    GeocodingToolSuccessUpdate,
    NaturalLanguageGeocodingTool,
)
from llm_agent.tools.resolve_data_ref_tool import ResolveDataRefTool
from llm_agent.tools.tracking import TrackingTool
from tests.utils.mocks.context import MockChatContext

if TYPE_CHECKING:
    from dask.distributed import Client

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


async def _run_stream_collect(
    agent: OrchestrationAgent,
    chat_context: ChatContext,
    user_message: UserMessage,
    *,
    message_history: list[ModelMessage] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[AgentStreamEvent | AgentRunResultEvent[str]], AgentRunResult[str]]:
    """Run the orchestration agent, collecting every event and the final result."""
    events: list[AgentStreamEvent | AgentRunResultEvent[str]] = [
        event
        async for event in agent.run_stream(
            chat_context,
            user_message,
            message_history=message_history,
            progress_callback=progress_callback,
        )
    ]
    result_event = next(event for event in events if isinstance(event, AgentRunResultEvent))
    return events, result_event.result


def _tool_calls(
    events: list[AgentStreamEvent | AgentRunResultEvent[str]], tool_name: str
) -> list[FunctionToolCallEvent]:
    return [
        event
        for event in events
        if isinstance(event, FunctionToolCallEvent) and event.part.tool_name == tool_name
    ]


def _tool_results(
    events: list[AgentStreamEvent | AgentRunResultEvent[str]], tool_name: str
) -> list[FunctionToolResultEvent]:
    return [
        event
        for event in events
        if isinstance(event, FunctionToolResultEvent) and event.part.tool_name == tool_name
    ]


@pytest.fixture
def chat_context_integration_icechunk(
    dask_client: "Client",
    canned_place_lookup: CannedPlaceLookup,
    proxied_bedrock_client: BedrockRuntimeClient,
) -> Generator[MockChatContext, Any, Any]:
    """ChatContext for integration tests using icechunk stores."""
    # Set env var to use icechunk stores
    os.environ["USE_ICECHUNK_STORES"] = "true"
    try:
        # Get icechunk KG paths
        kg_dao = KnowledgeGraphDAO(get_kg_ttl_paths())

        global_context = LLMAgentContext(
            place_lookup=canned_place_lookup,
            knowledge_graph_dao=kg_dao,
            embedding_repo=EmbeddingIndexRepository(
                embedding_dao=ChromaDBEmbeddingDAO(
                    client=EphemeralClient(), collection_name="datasets"
                ),
                datasets=kg_dao.fetch_datasets(),
            ),
            dask_client=dask_client,
            bedrock_client=proxied_bedrock_client,
        )
        yield MockChatContext(global_context=global_context)  # pyright: ignore[reportReturnType]
    finally:
        # Reset env var to default
        os.environ["USE_ICECHUNK_STORES"] = "false"


@pytest.mark.integration
class TestOrchestration:
    @pytest.fixture
    def orchestration_agent(
        self, proxied_bedrock_client: BedrockRuntimeClient
    ) -> OrchestrationAgent:
        dataset_picking_agent = DatasetPickingAgent(
            make_bedrock_model(
                proxied_bedrock_client, model_id=DATASET_PICKING_AGENT_DEFAULT_LLM_MODEL_ID
            )
        )
        return OrchestrationAgent(
            make_bedrock_model(
                proxied_bedrock_client, model_id=ORCHESTRATION_AGENT_DEFAULT_LLM_MODEL_ID
            ),
            tools=[
                TrackingTool(DataSearchTool(dataset_picking_agent=dataset_picking_agent)),
                TrackingTool(
                    DataAnalysisTool(
                        make_bedrock_model(
                            proxied_bedrock_client,
                            model_id=ALGORITHM_AGENT_DEFAULT_LLM_MODEL_ID,
                        )
                    )
                ),
                TrackingTool(NaturalLanguageGeocodingTool()),
                ResolveDataRefTool(),
            ],
            response_shaping_tools=[agent_plan_tool],
        )

    @observe()
    @pytest.mark.asyncio
    async def test_multi_turn_convo(
        self, orchestration_agent: OrchestrationAgent, chat_context_integration: MockChatContext
    ):
        agent = orchestration_agent

        _, first = await _run_stream_collect(
            agent,
            chat_context_integration,
            UserMessage(user_message="Debugging check: list the names of your tools."),
        )

        _, second = await _run_stream_collect(
            agent,
            chat_context_integration,
            UserMessage(
                user_message=(
                    "Debugging check: is this the first message in the conversation? "
                    'Reply with one word -- either "yes" or "no".'
                )
            ),
            message_history=first.all_messages(),
        )

        assert "no" in second.output.lower()

    @observe()
    @pytest.mark.asyncio
    async def test_sst_happy_path(
        self, orchestration_agent: OrchestrationAgent, chat_context_integration: MockChatContext
    ):
        progress_updates: list[ToolProgressUpdate[Any]] = []

        def progress_callback(update: ToolProgressUpdate[Any]) -> None:
            progress_updates.append(update)

        agent = orchestration_agent
        _, result = await _run_stream_collect(
            agent,
            chat_context_integration,
            UserMessage(
                user_message=(
                    "Compute the mean sea surface temperature in the Black Sea on "
                    "June 01, 2024 to 2 decimal places. (Do not create any extra outputs. "
                    "Do not retry on tool failure.)"
                )
            ),
            progress_callback=progress_callback,
        )

        assert "20.04" in result.output

        def _get_update_type(update: ToolProgressUpdate[Any]) -> type[Any]:
            if not isinstance(update.update, ToolProgressUpdate):
                return type(update.update)  # pyright: ignore[reportUnknownVariableType]
            return _get_update_type(update.update)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

        update_type_counts: Counter[type[Any]] = Counter(
            _get_update_type(update) for update in progress_updates
        )

        assert update_type_counts[SearchInitiatedUpdate] >= 1
        assert update_type_counts[SearchCompletedUpdate] >= 1
        assert update_type_counts[KnowledgeGraphQueryUpdate] >= 1
        assert update_type_counts[QueryResultsUpdate] >= 1

        assert update_type_counts[ComputingVariableUpdate] >= 1
        assert update_type_counts[ComputedVariableUpdate] >= 1
        assert (
            update_type_counts[GeocodingSuccessUpdate] >= 1
            or update_type_counts[GeocodingToolSuccessUpdate] >= 1
        )
        assert update_type_counts[FilterByPolygonUpdate] >= 1
        assert update_type_counts[DaskExecutionProgressUpdate] >= 1

    @observe()
    @pytest.mark.asyncio
    async def test_sst_with_user_provided_geometry_happy_path(
        self, orchestration_agent: OrchestrationAgent, chat_context_integration: MockChatContext
    ):
        test_geojson = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-94.88867187499919, 23.235290501732123],
                    [-89.26367187499899, 17.972464521768543],
                    [-98.36035156249882, 18.556699932883774],
                    [-97.17382812499929, 22.668766337218543],
                    [-94.88867187499919, 23.235290501732123],
                ]
            ],
            "bbox": None,
        }
        key = chat_context_integration.persist_data(json.dumps(test_geojson), data_type="geojson")
        ref = GeometryRef(key=key, name="aoi")

        agent = orchestration_agent
        _, result = await _run_stream_collect(
            agent,
            chat_context_integration,
            UserMessage(
                user_message=(
                    "What was the mean sea surface temperature in the provided "
                    "area on June 01, 2024? (Do not create any extra outputs. Do "
                    "not retry on tool failure.)"
                ),
                attached_data=[ref],
            ),
        )

        assert "29.77" in result.output

    @observe()
    @pytest.mark.asyncio
    async def test_heatflux_happy_path(
        self, orchestration_agent: OrchestrationAgent, chat_context_integration: MockChatContext
    ):
        test_geojson = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-94.88867187499919, 23.235290501732123],
                    [-89.26367187499899, 17.972464521768543],
                    [-98.36035156249882, 18.556699932883774],
                    [-97.17382812499929, 22.668766337218543],
                    [-94.88867187499919, 23.235290501732123],
                ]
            ],
            "bbox": None,
        }
        key = chat_context_integration.persist_data(json.dumps(test_geojson), data_type="geojson")
        ref = GeometryRef(key=key, name="aoi")

        agent = orchestration_agent
        _, result = await _run_stream_collect(
            agent,
            chat_context_integration,
            UserMessage(
                user_message=(
                    "What was the mean latent heatflux in the provided area for the first time "
                    "step of July 15, 2019? "
                    "(Do not create any extra outputs. Do not retry on tool failure.)"
                ),
                attached_data=[ref],
            ),
        )

        assert "176.51" in result.output

    @observe()
    @pytest.mark.asyncio
    async def test_heatflux_happy_path_icechunk(
        self,
        orchestration_agent: OrchestrationAgent,
        chat_context_integration_icechunk: MockChatContext,
    ):
        """Test heatflux computation using icechunk zarr stores."""
        test_geojson = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-94.88867187499919, 23.235290501732123],
                    [-89.26367187499899, 17.972464521768543],
                    [-98.36035156249882, 18.556699932883774],
                    [-97.17382812499929, 22.668766337218543],
                    [-94.88867187499919, 23.235290501732123],
                ]
            ],
            "bbox": None,
        }
        key = chat_context_integration_icechunk.persist_data(
            json.dumps(test_geojson), data_type="geojson"
        )
        ref = GeometryRef(key=key, name="aoi")

        agent = orchestration_agent
        events, result = await _run_stream_collect(
            agent,
            chat_context_integration_icechunk,
            UserMessage(
                user_message=(
                    "What was the mean latent heatflux in the provided area for the first time "
                    "step of July 15, 2019? "
                    "(Do not create any extra outputs. Do not retry on tool failure.)"
                ),
                attached_data=[ref],
            ),
        )

        # Assert same result as non-icechunk test
        assert "176.51" in result.output

        # Assert that icechunk store was used
        # Get the dataset from the knowledge graph query results
        search_results = _tool_results(events, f"{DataSearchTool.name}WithTracking")
        assert len(search_results) >= 1

        # The dataset info should show icechunk zarr store
        kg_dao = chat_context_integration_icechunk.global_context.knowledge_graph_dao
        dataset = kg_dao.fetch_datasets(ids=["noaa-cdr-ocean-heatflux-pds_daily"])[0]

        # Verify icechunk store is being used
        assert "icechunk" in dataset.source.uri, (
            f"Expected icechunk store but got: {dataset.source.uri}"
        )
        assert dataset.source.format == "icechunk-zarr", (
            f"Expected zarr format but got: {dataset.source.format}"
        )

    @observe()
    @pytest.mark.asyncio
    async def test_planning_tool(
        self, orchestration_agent: OrchestrationAgent, chat_context_integration: MockChatContext
    ):
        agent = orchestration_agent
        events, _ = await _run_stream_collect(
            agent,
            chat_context_integration,
            UserMessage(
                user_message=(
                    "Debugging session: attempt to use the planning tool, don't retry on failure."
                )
            ),
        )

        plan_calls = _tool_calls(events, agent_plan_tool.name)
        assert len(plan_calls) >= 1
        # The structured plan payload rides on the tool-call args.
        AgentPlanOrUpdate.model_validate(plan_calls[0].part.args_as_dict())

    @observe()
    @pytest.mark.asyncio
    async def test_create_raster(
        self, orchestration_agent: OrchestrationAgent, chat_context_integration: MockChatContext
    ):
        agent = orchestration_agent
        events, _ = await _run_stream_collect(
            agent,
            chat_context_integration,
            UserMessage(
                user_message=(
                    "Show the mean sea surface temperature in the Black Sea on "
                    "June 01, 2024 on the map."
                )
            ),
        )

        analysis_results = _tool_results(events, f"{DataAnalysisTool.name}WithTracking")
        assert len(analysis_results) == 1

        output = DataAnalysisToolOutput.model_validate(analysis_results[0].part.content)
        assert output.outputs is not None
        assert len(output.outputs) == 1

        # The key name for the value we're reading is generated by the LLM and observably changes.
        geotiff_ref = GeoTIFFRef.model_validate(next(iter(output.outputs.values())))
        assert geotiff_ref.name in {"analysed_sst", "sea_surface_temperature"}
        assert geotiff_ref.units == "degrees_Celsius"

        geotiff_path = chat_context_integration.retrieve_data(geotiff_ref)
        assert isinstance(geotiff_path, str)
        assert geotiff_path.endswith(".tif")

    @observe()
    @pytest.mark.asyncio
    async def test_resolve_image_ref(
        self, orchestration_agent: OrchestrationAgent, chat_context_integration: MockChatContext
    ):
        agent = orchestration_agent
        events, _ = await _run_stream_collect(
            agent,
            chat_context_integration,
            UserMessage(
                user_message=(
                    "Create a line plot showing monthly mean SST in the Black Sea in "
                    "the period from January 2024 to June 2024 on the map, then look at it "
                    "and describe any trends you see. (Do not create any extra outputs. "
                    "Do not retry on tool failure.)"
                )
            ),
        )

        resolve_calls = _tool_calls(events, ResolveDataRefTool.name)
        assert len(resolve_calls) == 1
        call_id = resolve_calls[0].part.tool_call_id

        resolve_results = [
            event
            for event in events
            if isinstance(event, FunctionToolResultEvent) and event.part.tool_call_id == call_id
        ]
        assert len(resolve_results) == 1
