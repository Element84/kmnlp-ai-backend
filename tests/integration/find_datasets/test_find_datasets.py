import logging
from json import JSONDecodeError
from typing import cast

import pytest
from e84_geoai_common.llm.models.claude import (
    CLAUDE_BEDROCK_MODEL_IDS,
    BedrockClaudeLLM,
)

from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import (
    DatasetPickingAgent,
    DatasetResponseTextContent,
)
from llm_agent.tools.data_search_tool import DEFAULT_MODEL_NAME
from tests.integration.base_integration_test import (
    BaseIntegrationTest,
)

log = logging.getLogger(__name__)


class TestFindDatasets(BaseIntegrationTest):
    @classmethod
    def setup_class(cls) -> None:
        cls.llm = BedrockClaudeLLM(
            model_id=CLAUDE_BEDROCK_MODEL_IDS[DEFAULT_MODEL_NAME],
        )

    @pytest.mark.asyncio
    async def test_refuses_to_answer_unrelated(self):
        agent = DatasetPickingAgent(self.llm)
        try:
            assistant_messages = [
                message
                async for message in agent.run(
                    "What is the airspeed velocity of an unladen swallow?"
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
        parsed_content = cast(
            "DatasetResponseTextContent", response_contents[0]
        )

        log.debug("response content: %s", parsed_content)

        assert parsed_content.result is None
        assert len(parsed_content.explanation) > 10
        assert parsed_content.is_refusal

    @pytest.mark.asyncio
    async def test_finds_all_sst_datasets(self):
        agent = DatasetPickingAgent(self.llm)
        try:
            assistant_messages = [
                message
                async for message in agent.run("Find all SST datasets.")
            ]
        except JSONDecodeError as e:
            log.exception("JSON doc was: %s", e.doc)
            raise

        last_message = assistant_messages[-1]

        # Last message should have 1 content element
        assert len(last_message.parsed_content) == 1

        # And that lone content element should be of the expected format
        assert isinstance(
            last_message.parsed_content[0], DatasetResponseTextContent
        )

        # Last response result should be list of datasets with 2 elements
        dataset_list = last_message.parsed_content[0].result
        assert isinstance(dataset_list, list)
        assert (len(dataset_list)) == 2
        assert set(dataset_list) == {
            "http://kmnlp-demo.noaa.gov/crw_daily_5km",
            "http://kmnlp-demo.noaa.gov/noaa-cdr-sea-surface-temp-whoi-pds_daily",
        }

    @pytest.mark.asyncio
    async def test_finds_crw_sst_dataset_only_based_on_temporal_coverage(self):
        agent = DatasetPickingAgent(self.llm)
        try:
            assistant_messages = [
                message
                async for message in agent.run(
                    "What dataset would show me the mean sea surface "
                    "temperature in the Black Sea on June 01, 2024?"
                )
            ]
        except JSONDecodeError as e:
            log.exception("JSON doc was: %s", e.doc)
            raise

        last_message = assistant_messages[-1]

        # Last message should have 1 content element
        assert len(last_message.parsed_content) == 1

        # And that lone content element should be of the expected format
        assert isinstance(
            last_message.parsed_content[0], DatasetResponseTextContent
        )

        # Last response result should be list of datasets with a single element
        dataset_list = last_message.parsed_content[0].result
        assert isinstance(dataset_list, list)
        assert (len(dataset_list)) == 1

        # And that list's lone element should be the CRW SST Dataset
        assert dataset_list[0] == "http://kmnlp-demo.noaa.gov/crw_daily_5km"
