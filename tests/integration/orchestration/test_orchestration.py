import logging
from json import JSONDecodeError
from typing import cast

import pytest
from e84_geoai_common.llm.core import LLMMessage, TextContent

from llm_agent.agents.orchestration_agent.orchestration_agent import (
    OrchestrationAgent,
)
from llm_agent.tools.data_analysis_tool import data_analysis_tool
from llm_agent.tools.data_search_tool import data_search_tool
from tests.integration.base_integration_test import BaseIntegrationTest

log = logging.getLogger(__name__)


class TestOrchestration(BaseIntegrationTest):
    @pytest.mark.asyncio
    async def test_sst_happy_path(self):
        agent = OrchestrationAgent(
            self.llm, tools=[data_search_tool, data_analysis_tool]
        )
        try:
            responses = agent.run(
                LLMMessage(
                    role="user",
                    content=(
                        "What was the mean sea surface temperature in the "
                        "Black Sea on June 01, 2024? (Do not use coordinates "
                        "from memory; let the data analysis tool lookup the "
                        "exact polygon.)"
                    ),
                )
            )
        except JSONDecodeError as e:
            log.exception("JSON doc was: %s", e.doc)
            raise

        materialized_responses = [response async for response in responses]
        last_response = cast("LLMMessage", materialized_responses[-1])
        text_contents: list[str] = [
            content.text
            for content in last_response.content
            if isinstance(content, TextContent)
        ]
        assert len(text_contents) == 1
        assert "20.03" in text_contents[0]

    @pytest.mark.asyncio
    async def test_atmo_happy_path(self):
        agent = OrchestrationAgent(
            self.llm, tools=[data_search_tool, data_analysis_tool]
        )
        try:
            responses = agent.run(
                LLMMessage(
                    role="user",
                    content=(
                        "What was the mean wind speed in the Black Sea on "
                        "June 01, 2021? (Do not use coordinates from memory; "
                        "let the data analysis tool lookup the exact polygon.)"
                    ),
                )
            )
        except JSONDecodeError as e:
            log.exception("JSON doc was: %s", e.doc)
            raise

        materialized_responses = [response async for response in responses]
        last_response = cast("LLMMessage", materialized_responses[-1])
        text_contents: list[str] = [
            content.text
            for content in last_response.content
            if isinstance(content, TextContent)
        ]
        assert len(text_contents) == 1
        assert "7.8" in text_contents[0]
