import logging
import os
from json import JSONDecodeError
from typing import Any, assert_type, cast

import pytest
from dask.distributed import LocalCluster
from e84_geoai_common.llm.core import (
    ExecutableLLMTool,
    JSONContent,
    LLMMessage,
    LLMToolResultContent,
    LLMToolUseContent,
    LLMUserMessage,
    TextContent,
)
from e84_geoai_common.llm.models.claude import BedrockClaudeLLM
from langfuse import observe  # type: ignore[reportUnknownVariableType]

from llm_agent import (
    DATASET_PICKING_AGENT_DEFAULT_LLM_MODEL_ID,
    ORCHESTRATION_AGENT_DEFAULT_LLM_MODEL_ID,
)
from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import DatasetPickingAgent
from llm_agent.agents.orchestration_agent.orchestration_agent import (
    OrchestrationAgent,
)
from llm_agent.agents.tools_config import ToolsConfig
from llm_agent.context import LLMAgentContext, ToolCallContext
from llm_agent.tools.data_analysis_tool.data_analysis_tool import (
    DataAnalysisTool,
    DataAnalysisToolOutput,
)
from llm_agent.tools.data_analysis_tool.types import GeoTIFFRef
from llm_agent.tools.data_search_tool import DataSearchTool
from llm_agent.tools.llm_tool_with_tracking import LLMToolWithTracking
from llm_agent.tools.response_shaping_tools import end_turn_tool
from llm_agent.tools.response_shaping_tools.agent_plan_tool import (
    AgentPlanOrUpdate,
    agent_plan_tool,
)
from llm_agent.tools.response_shaping_tools.text_response_tool import (
    TextResponse,
    text_response_tool,
)
from llm_agent.utils.prompt_as_service import PromptService, PromptServiceAdapter
from llm_agent.utils.utils import EnrichedLLMToolResultContent
from tests.utils.fixtures.token_limiting_transformer import shared_token_limiting_transformer

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@pytest.mark.integration
class TestOrchestration:
    prompt_service: PromptService = ...  # type: ignore[reportAssignmentType]

    @classmethod
    def setup_class(cls) -> None:
        llm = BedrockClaudeLLM(
            model_id=ORCHESTRATION_AGENT_DEFAULT_LLM_MODEL_ID,
        )

        base_service = PromptServiceAdapter(llm.prompt)
        limited_service = shared_token_limiting_transformer().and_then_service(base_service)

        assert_type(limited_service, PromptService)

        cls.prompt_service = limited_service

        assert_type(cls.prompt_service, PromptService)

        # Initialize tools
        data_search_tool = LLMToolWithTracking(
            DataSearchTool(
                dataset_picking_agent=DatasetPickingAgent(
                    prompt_service=shared_token_limiting_transformer().and_then_service(
                        PromptServiceAdapter(
                            BedrockClaudeLLM(
                                model_id=DATASET_PICKING_AGENT_DEFAULT_LLM_MODEL_ID
                            ).prompt
                        )
                    )
                )
            )
        )
        data_analysis_tool = LLMToolWithTracking(DataAnalysisTool())

        default_tools: list[ExecutableLLMTool[ToolCallContext]] = [
            data_search_tool,
            data_analysis_tool,
            text_response_tool,
            agent_plan_tool,
        ]

        cls.default_tools_config = ToolsConfig(
            tools=default_tools, end_turn_tool=end_turn_tool, max_allowed_tool_calls_per_turn=16
        )

        # Setup dask cluster if one is not already running
        if os.getenv("DASK_ADDRESS") is None:
            cls.cluster = LocalCluster()
            os.environ["DASK_ADDRESS"] = cls.cluster.scheduler_address
            log.info("Using local dask cluster: %s", cls.cluster)

    @observe()
    @pytest.mark.asyncio
    async def test_multi_turn_convo(self, global_context_integration: LLMAgentContext):
        agent = OrchestrationAgent(self.prompt_service, tools_config=self.default_tools_config)
        conversation: list[LLMMessage] = [
            LLMUserMessage(
                content=("Debugging check: list the names of your tools."),
            )
        ]
        try:
            responses = agent.run(global_context_integration, conversation)

            # "await" the AsyncGenerator
            async for _ in responses:
                pass

            conversation.append(
                LLMUserMessage(
                    content=(
                        "Debugging check: is this the first message in the conversation? "
                        'Reply with one word -- either "yes" or "no".'
                    ),
                )
            )
            responses = agent.run(global_context_integration, conversation)

        except JSONDecodeError as e:
            log.exception("JSON doc was: %s", e.doc)
            raise

        materialized_responses = [response async for response in responses]
        for r in materialized_responses[::-1]:
            if isinstance(r, EnrichedLLMToolResultContent) and isinstance(r.data, TextResponse):  # pyright: ignore[reportUnknownMemberType]
                assert "no" in r.data.text.lower()
                break
        else:
            pytest.fail("No TextResponse found in responses")

    @observe()
    @pytest.mark.asyncio
    async def test_sst_happy_path(self, global_context_integration: LLMAgentContext):
        agent = OrchestrationAgent(self.prompt_service, tools_config=self.default_tools_config)
        try:
            responses = agent.run(
                global_context_integration,
                conversation=[
                    LLMUserMessage(
                        content=(
                            "What was the mean sea surface temperature in the Black Sea on June "
                            "01, 2024? (Do not create any extra outputs. Do not retry on tool "
                            "failure.)"
                        ),
                    )
                ],
            )
        except JSONDecodeError as e:
            log.exception("JSON doc was: %s", e.doc)
            raise

        materialized_responses = [response async for response in responses]
        for r in materialized_responses[::-1]:
            if isinstance(r, EnrichedLLMToolResultContent) and isinstance(r.data, TextResponse):  # pyright: ignore[reportUnknownMemberType]
                assert "20.03" in r.data.text.lower()
                break
        else:
            pytest.fail("No TextResponse found in responses")

    @observe()
    @pytest.mark.asyncio
    async def test_sst_with_user_provided_geometry_happy_path(
        self, global_context_integration: LLMAgentContext
    ):
        test_geojson = """{
        "type": "Polygon",
        "coordinates": [
            [
                [
                    -94.88867187499919,
                    23.235290501732123
                ],
                [
                    -89.26367187499899,
                    17.972464521768543
                ],
                [
                    -98.36035156249882,
                    18.556699932883774
                ],
                [
                    -97.17382812499929,
                    22.668766337218543
                ],
                [
                    -94.88867187499919,
                    23.235290501732123
                ]
            ]
        ],
        "bbox": null
    }"""
        agent = OrchestrationAgent(self.prompt_service, tools_config=self.default_tools_config)
        try:
            responses = agent.run(
                global_context_integration,
                conversation=[
                    LLMUserMessage(
                        content=[
                            TextContent(
                                text=(
                                    "What was the mean sea surface temperature in the provided "
                                    "area on June 01, 2024? (Do not create any extra outputs. Do "
                                    "not retry on tool failure.)"
                                )
                            ),
                            TextContent(text=f"Provided area: {test_geojson}"),
                        ],
                    )
                ],
            )
        except JSONDecodeError as e:
            log.exception("JSON doc was: %s", e.doc)
            raise

        materialized_responses = [response async for response in responses]
        for r in materialized_responses[::-1]:
            if isinstance(r, EnrichedLLMToolResultContent) and isinstance(r.data, TextResponse):  # pyright: ignore[reportUnknownMemberType]
                assert "29.77" in r.data.text.lower()
                break
        else:
            pytest.fail("No TextResponse found in responses")

    @observe()
    @pytest.mark.asyncio
    async def test_planning_tool(self, global_context_integration: LLMAgentContext):
        agent = OrchestrationAgent(self.prompt_service, tools_config=self.default_tools_config)
        try:
            responses = agent.run(
                global_context_integration,
                conversation=[
                    LLMUserMessage(
                        content="Debugging session: attempt to use the planning tool, "
                        "don't retry on failure.",
                    )
                ],
            )
        except JSONDecodeError as e:
            log.exception("JSON doc was: %s", e.doc)
            raise

        materialized_responses = [response async for response in responses]
        for r in materialized_responses:
            if isinstance(r, EnrichedLLMToolResultContent) and isinstance(
                r.data,  # pyright: ignore[reportUnknownMemberType]
                AgentPlanOrUpdate,
            ):
                break
        else:
            pytest.fail("No AgentPlanOrUpdate found in responses")

    @observe()
    @pytest.mark.asyncio
    async def test_create_raster(self, global_context_integration: LLMAgentContext):
        agent = OrchestrationAgent(self.prompt_service, tools_config=self.default_tools_config)
        try:
            responses = agent.run(
                global_context_integration,
                [
                    LLMMessage(
                        role="user",
                        content=(
                            "Please provide a raster visualization of the mean sea surface "
                            "temperature in the Black Sea on June 01, 2024."
                        ),
                    )
                ],
            )

        except JSONDecodeError as e:
            log.exception("JSON doc was: %s", e.doc)
            raise

        materialized_responses = [response async for response in responses]
        log.debug("materialized responses: %s", materialized_responses)

        # Responses should contain a reference to a GeoTIFF
        def _is_data_analysis_tool_output(
            response: LLMToolUseContent | LLMToolResultContent,
        ) -> bool:
            match response:
                case LLMToolUseContent():
                    return False
                case LLMToolResultContent():
                    log.debug("response content: %s", response.content)
                    return (
                        isinstance(response, EnrichedLLMToolResultContent)
                        and isinstance(response.content[0], JSONContent)
                        and "data_outputs" in response.content[0].data
                    )

        data_analysis_tool_outputs = [
            r for r in materialized_responses if _is_data_analysis_tool_output(r)
        ]

        log.debug("filtered responses: %s", data_analysis_tool_outputs)

        assert len(data_analysis_tool_outputs) == 1

        # Assert the single response with a GeoTIFF reference is as expected
        data_analysis_tool_output = cast(
            "EnrichedLLMToolResultContent[DataAnalysisToolOutput]", data_analysis_tool_outputs[0]
        )
        data_dict: dict[str, Any] = cast(
            "dict[str, Any]",
            data_analysis_tool_output.content[0].data,  # type: ignore[reportAttributeAccessIssue]
        )
        data_outputs = cast("dict[str, dict[str, Any]]", data_dict["data_outputs"])
        assert len(data_outputs) == 1
        # The key name for the value we're reading is generated by the LLM and observably changes.
        geotiff_ref = GeoTIFFRef.model_validate(next(iter(data_outputs.values())))

        assert geotiff_ref.url.startswith(
            "https://test-geotiff-bucket.s3.us-east-1.amazonaws.com/transformed_data_array"
        )
        assert geotiff_ref.url.endswith(".tif")

        assert geotiff_ref.label == "analysed_sst"
        assert geotiff_ref.units == "degrees_Celsius"
