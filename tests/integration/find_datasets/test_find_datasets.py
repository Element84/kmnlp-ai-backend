import logging
from collections.abc import Callable
from json import JSONDecodeError
from typing import cast

import pytest
from e84_geoai_common.llm.models import BedrockClaudeLLM
from langfuse import observe  # pyright: ignore[reportUnknownVariableType]

from llm_agent import DATASET_PICKING_AGENT_DEFAULT_LLM_MODEL_ID
from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import (
    DatasetPickingAgent,
    DatasetResponseTextContent,
)
from llm_agent.context import LLMAgentContext
from llm_agent.tools.execute_sparql_query_tool import ExecuteSparqlQueryTool
from llm_agent.tools.fetch_datasets_tool import FetchDatasetsTool
from llm_agent.utils.prompt_as_service import PromptService, PromptServiceAdapter
from tests.utils.fixtures.token_limiting_transformer import shared_token_limiting_transformer

log = logging.getLogger(__name__)


execute_sparql_query_tool = ExecuteSparqlQueryTool()
fetch_datasets_tool = FetchDatasetsTool()


@pytest.mark.integration
class TestFindDatasets:
    prompt_service: PromptService = ...  # type: ignore[reportAssignmentType]

    @classmethod
    def setup_class(cls) -> None:
        llm = BedrockClaudeLLM(
            model_id=DATASET_PICKING_AGENT_DEFAULT_LLM_MODEL_ID,
        )

        prompt_as_service = PromptServiceAdapter(llm.prompt)

        cls.prompt_service = shared_token_limiting_transformer().and_then_service(prompt_as_service)

    @observe()
    @pytest.mark.asyncio
    async def test_refuses_to_answer_unrelated(
        self, global_context: Callable[..., LLMAgentContext]
    ):
        agent = DatasetPickingAgent(self.prompt_service)
        try:
            assistant_messages = [
                message
                async for message in agent.run(
                    global_context(), "What is the airspeed velocity of an unladen swallow?"
                )
            ]
        except JSONDecodeError as e:
            log.exception("JSON doc was: %s", e.doc)
            raise

        # Should only be one message, no tool calls to do a refusal
        assert len(assistant_messages) == 1

        # Response message should have single parsed content object
        response_contents = assistant_messages[0].parsed_content
        assert len(response_contents) == 1

        # Parsed content should be a formatted message refusing to answer.
        parsed_content = response_contents[0]
        parsed_content = cast("DatasetResponseTextContent", response_contents[0])

        log.debug("response content: %s", parsed_content)

        assert parsed_content.result is None
        assert len(parsed_content.explanation) > 10
        assert parsed_content.is_refusal

    @observe()
    @pytest.mark.asyncio
    async def test_finds_all_sst_datasets_with_fetch_datasets_tool(
        self, global_context: Callable[..., LLMAgentContext]
    ):
        agent = DatasetPickingAgent(self.prompt_service, tools=[fetch_datasets_tool])
        try:
            assistant_messages = [
                message
                async for message in agent.run(
                    global_context(), "Find ALL datasets containing sea surface temperature."
                )
            ]
        except JSONDecodeError as e:
            log.exception("JSON doc was: %s", e.doc)
            raise

        last_message = assistant_messages[-1]

        # Last message should have 1 content element
        assert len(last_message.parsed_content) == 1

        # And that lone content element should be of the expected format
        assert isinstance(last_message.parsed_content[0], DatasetResponseTextContent)

        # Last response result should be a list of datasets w/ at least the 2
        # correct SST datasets. We do not penalize extra datasets in the
        # response that the LLM might include because it interprets them to be
        # related to SST.
        dataset_list = last_message.parsed_content[0].result
        assert isinstance(dataset_list, list)
        assert set(dataset_list).issuperset(
            {
                "crw_daily_5km",
                "noaa-cdr-sea-surface-temp-whoi-pds_daily",
            }
        )

    @observe()
    @pytest.mark.asyncio
    async def test_finds_all_sst_datasets_with_execute_sparql_tool(
        self, global_context: Callable[..., LLMAgentContext]
    ):
        agent = DatasetPickingAgent(self.prompt_service, tools=[execute_sparql_query_tool])
        try:
            assistant_messages = [
                message
                async for message in agent.run(
                    global_context(), "Find ALL datasets containing sea surface temperature."
                )
            ]
        except JSONDecodeError as e:
            log.exception("JSON doc was: %s", e.doc)
            raise

        last_message = assistant_messages[-1]

        # Last message should have 1 content element
        assert len(last_message.parsed_content) == 1

        # And that lone content element should be of the expected format
        assert isinstance(last_message.parsed_content[0], DatasetResponseTextContent)

        # Last response result should be a list of datasets w/ at least the 2
        # correct SST datasets. We do not penalize extra datasets in the
        # response that the LLM might include because it interprets them to be
        # related to SST.
        dataset_list = last_message.parsed_content[0].result
        assert isinstance(dataset_list, list)
        assert set(dataset_list).issuperset(
            {
                "crw_daily_5km",
                "noaa-cdr-sea-surface-temp-whoi-pds_daily",
            }
        )

    @observe()
    @pytest.mark.asyncio
    async def test_finds_crw_sst_dataset_only_based_on_temporal_coverage(
        self, global_context: Callable[..., LLMAgentContext]
    ):
        agent = DatasetPickingAgent(self.prompt_service)
        try:
            assistant_messages = [
                message
                async for message in agent.run(
                    global_context(),
                    "What dataset would show me the mean sea surface "
                    "temperature in the Black Sea on June 01, 2024?",
                )
            ]
        except JSONDecodeError as e:
            log.exception("JSON doc was: %s", e.doc)
            raise

        last_message = assistant_messages[-1]

        # Last message should have 1 content element
        assert len(last_message.parsed_content) == 1

        # And that lone content element should be of the expected format
        assert isinstance(last_message.parsed_content[0], DatasetResponseTextContent)

        # Last response result should be list of datasets with a single element
        dataset_list = last_message.parsed_content[0].result
        assert isinstance(dataset_list, list)
        assert (len(dataset_list)) == 1

        # And that list's lone element should be the CRW SST Dataset
        assert dataset_list[0] == "crw_daily_5km"

    @observe()
    @pytest.mark.asyncio
    async def test_uses_temporal_coverage_correctly(
        self, global_context: Callable[..., LLMAgentContext]
    ):
        agent = DatasetPickingAgent(self.prompt_service)
        try:
            assistant_messages = [
                message
                async for message in agent.run(
                    global_context(), "sea surface temperature SST Black Sea daily data 2024"
                )
            ]
        except JSONDecodeError as e:
            log.exception("JSON doc was: %s", e.doc)
            raise

        last_message = assistant_messages[-1]

        # Last message should have 1 content element
        assert len(last_message.parsed_content) == 1

        # And that lone content element should be of the expected format
        assert isinstance(last_message.parsed_content[0], DatasetResponseTextContent)

        # Last response result should be list of datasets with a single element
        dataset_list = last_message.parsed_content[0].result
        assert isinstance(dataset_list, list)
        assert (len(dataset_list)) == 1

        # And that list's lone element should be the CRW SST Dataset
        assert dataset_list[0] == "crw_daily_5km"
