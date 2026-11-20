import logging
from typing import Any

import pytest
from langfuse import observe  # pyright: ignore[reportUnknownVariableType]
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from pydantic_ai.models import Model

from llm_agent import DATASET_PICKING_AGENT_DEFAULT_LLM_MODEL_ID
from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import (
    DatasetPickingAgent,
    DatasetSearchQuery,
    DatasetSearchResponse,
)
from llm_agent.context import ToolProgressUpdate
from llm_agent.model_factory import make_bedrock_model
from llm_agent.tools.execute_sparql_query_tool import ExecuteSparqlQueryTool
from llm_agent.tools.fetch_datasets_tool import (
    FetchDatasetsTool,
    KnowledgeGraphQueryUpdate,
    QueryResultsUpdate,
)
from tests.utils.mocks.context import MockChatContext

log = logging.getLogger(__name__)


execute_sparql_query_tool = ExecuteSparqlQueryTool()
fetch_datasets_tool = FetchDatasetsTool()


@pytest.mark.integration
class TestFindDatasets:
    @pytest.fixture
    def model(self, proxied_bedrock_client: BedrockRuntimeClient) -> Model:
        return make_bedrock_model(
            proxied_bedrock_client, model_id=DATASET_PICKING_AGENT_DEFAULT_LLM_MODEL_ID
        )

    @observe()
    @pytest.mark.asyncio
    async def test_refuses_to_answer_unrelated(self, model: Model, chat_context: MockChatContext):
        agent = DatasetPickingAgent(model)
        response = await agent.run(
            chat_context,
            DatasetSearchQuery(query="What is the airspeed velocity of an unladen swallow?"),
        )

        # A refusal carries no result and no tool calls.
        assert isinstance(response, DatasetSearchResponse)
        assert response.result is None
        assert len(response.explanation) > 0
        assert response.is_refusal

    @observe()
    @pytest.mark.asyncio
    async def test_finds_all_sst_datasets_with_fetch_datasets_tool(
        self, model: Model, chat_context: MockChatContext
    ):
        progress_updates: list[ToolProgressUpdate[Any]] = []

        def progress_callback(update: ToolProgressUpdate[Any]) -> None:
            progress_updates.append(update)

        agent = DatasetPickingAgent(model, tools=[fetch_datasets_tool])
        response = await agent.run(
            chat_context,
            DatasetSearchQuery(query="Find ALL datasets containing sea surface temperature."),
            progress_callback=progress_callback,
        )

        assert isinstance(response, DatasetSearchResponse)
        assert response.result is not None
        assert len(response.result) > 0
        assert not response.is_refusal

        # The result should be a list of datasets w/ at least the correct SST
        # datasets. We do not penalize extra datasets in the response that the
        # LLM might include because it interprets them to be related to SST.
        dataset_list = response.result
        assert set(dataset_list).issuperset(
            {
                "crw_5km_sst_daily",
                "crw_5km_sst_monthly_mean",
                "crw_5km_sst_monthly_min",
                "crw_5km_sst_monthly_max",
                "crw_5km_sst_annual_mean",
                "crw_5km_sst_annual_min",
                "crw_5km_sst_annual_max",
                "noaa-cdr-sea-surface-temp-whoi-pds_daily",
            }
        )

        assert len(progress_updates) >= 2
        assert {type(update.update) for update in progress_updates} == {
            KnowledgeGraphQueryUpdate,
            QueryResultsUpdate,
        }

    @observe()
    @pytest.mark.asyncio
    async def test_finds_all_sst_datasets_with_execute_sparql_tool(
        self, model: Model, chat_context: MockChatContext
    ):
        agent = DatasetPickingAgent(model, tools=[execute_sparql_query_tool])
        response = await agent.run(
            chat_context,
            DatasetSearchQuery(query="Find ALL datasets containing sea surface temperature."),
        )

        assert isinstance(response, DatasetSearchResponse)
        assert response.result is not None
        assert not response.is_refusal

        # The result should be a list of datasets w/ at least the correct SST
        # datasets. We do not penalize extra datasets in the response that the
        # LLM might include because it interprets them to be related to SST.
        dataset_list = response.result
        assert set(dataset_list).issuperset(
            {
                "crw_5km_sst_daily",
                "crw_5km_sst_monthly_mean",
                "crw_5km_sst_monthly_min",
                "crw_5km_sst_monthly_max",
                "crw_5km_sst_annual_mean",
                "crw_5km_sst_annual_min",
                "crw_5km_sst_annual_max",
                "noaa-cdr-sea-surface-temp-whoi-pds_daily",
            }
        )

    @observe()
    @pytest.mark.asyncio
    async def test_finds_crw_sst_dataset_only_based_on_temporal_coverage(
        self, model: Model, chat_context: MockChatContext
    ):
        agent = DatasetPickingAgent(model)
        response = await agent.run(
            chat_context,
            DatasetSearchQuery(
                query="Which SST datasets have daily temporal resolution and cover June 01, 2024?"
            ),
        )

        assert isinstance(response, DatasetSearchResponse)
        assert response.result is not None
        assert not response.is_refusal

        # And that list's lone element should be the CRW SST Dataset
        assert response.result == ["crw_5km_sst_daily"]
