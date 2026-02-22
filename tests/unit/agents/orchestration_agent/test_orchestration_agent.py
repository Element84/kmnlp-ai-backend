from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest
from e84_geoai_common.llm.core import (
    JSONContent,
    LLMToolResultContent,
    LLMToolUseContent,
    LLMUserMessage,
)
from e84_geoai_common.llm.models.claude import BedrockClaudeLLM
from e84_geoai_common.llm.tests.mock_bedrock_runtime import (
    _MockBedrockRuntimeClient,  # type: ignore[reportPrivateUsage]
    claude_response_with_content,
    make_test_bedrock_runtime_client,
)
from rich import print as rich_print

from llm_agent import ORCHESTRATION_AGENT_DEFAULT_LLM_MODEL_ID
from llm_agent.agents.orchestration_agent.orchestration_agent import (
    OrchestrationAgent,
)
from llm_agent.agents.tools_config import ToolsConfig
from llm_agent.context import LLMAgentContext, ToolCallContext
from llm_agent.model.dataset import Dataset, DatasetSource
from llm_agent.tools.data_analysis_tool.data_analysis_tool import (
    DataAnalysisTool,
    DataAnalysisToolOutput,
    _algo_outputs_to_tool_output,  # pyright: ignore[reportPrivateUsage]
)
from llm_agent.tools.data_search_tool import DataSearchTool, DataSearchToolResult
from llm_agent.tools.response_shaping_tools import end_turn_tool
from llm_agent.tools.response_shaping_tools.text_response_tool import (
    TextResponse,
    text_response_tool,
)
from llm_agent.utils.prompt_as_service import PromptServiceAdapter
from llm_agent.utils.utils import EnrichedLLMToolResultContent

stub_responses = [
    # response to initial question
    claude_response_with_content(
        [
            {
                "id": "tool_use_01",
                "name": "DataSearchTool",
                "input": {
                    "variable_description": "Sea surface temperature",
                    "time_range": {
                        "time_min": "2024-06-01",
                        "time_max": "2024-06-01",
                    },
                },
            },
        ],
        overrides={"stop_reason": "tool_use"},
    ),
    # response to tool data search result
    claude_response_with_content(
        [
            {
                "id": "tool_use_02",
                "name": "DataAnalysisTool",
                "input": {
                    "problem_statement": "Compute the mean sea surface temperature in the Black Sea on June 01, 2024.",  # noqa: E501
                    "variable_ids": ["coraltemp-v3.1-sst"],
                    "outputs_requested": [
                        {
                            "name": "sst_mean",
                            "variable_type": "scalar_float",
                            "description": "Mean SST in the Black Sea on June 01, 2024.",
                        },
                    ],
                },
            },
        ],
        overrides={"stop_reason": "tool_use"},
    ),
    # response to tool data analysis result
    claude_response_with_content(
        [
            {
                "id": "tool_use_03",
                "name": "TextResponseTool",
                "input": {"text": "The mean SST was 20.03 deg C."},
            },
        ],
        overrides={"stop_reason": "tool_use"},
    ),
    # end turn
    claude_response_with_content(
        [
            {
                "id": "tool_use_04",
                "name": "EndTurnTool",
                "input": {},
            },
        ],
        overrides={"stop_reason": "tool_use"},
    ),
]

client = make_test_bedrock_runtime_client(stub_responses)
llm = BedrockClaudeLLM(model_id=ORCHESTRATION_AGENT_DEFAULT_LLM_MODEL_ID, client=client)


class StubDataSearchTool(DataSearchTool):
    def execute(
        self,
        context: ToolCallContext,  # noqa: ARG002
        tool_use_request: LLMToolUseContent,
    ) -> LLMToolResultContent:
        return LLMToolResultContent(
            id=tool_use_request.id,
            content=[
                JSONContent(
                    data=DataSearchToolResult(
                        datasets=[
                            Dataset(
                                id="http://fake/id",
                                source=DatasetSource(format="zarr", uri="http://source"),
                                label="label",
                                description="description",
                                landing_page="http://example.com/landing_page",
                                temporal_coverage=(
                                    datetime(year=2020, month=1, day=1, tzinfo=UTC),
                                    datetime(
                                        year=2024,
                                        month=11,
                                        day=30,
                                        hour=12,
                                        tzinfo=UTC,
                                    ),
                                ),
                                temporal_resolution=timedelta(days=1),
                                variables=[],
                            )
                        ]
                    ).model_dump(mode="json")
                )
            ],
            status="success",
        )


class StubDataAnalysisTool(DataAnalysisTool):
    def execute(
        self,
        context: ToolCallContext,  # noqa: ARG002
        tool_use_request: LLMToolUseContent,
    ) -> EnrichedLLMToolResultContent[DataAnalysisToolOutput]:
        out = _algo_outputs_to_tool_output(
            algo_outputs={"black_sea_mean_scalar": 20.03},
            pseudocode="__pseudocode__",
        )
        return EnrichedLLMToolResultContent(
            id=tool_use_request.id, content=[JSONContent(data=out.model_dump())], data=out
        )


stub_data_search_tool = StubDataSearchTool(dataset_picking_agent=Mock())
stub_data_analysis_tool = StubDataAnalysisTool()


@pytest.mark.asyncio
async def test_orchestration_agent(global_context: Callable[..., LLMAgentContext]) -> None:
    agent = OrchestrationAgent(
        PromptServiceAdapter(llm.prompt),
        tools_config=ToolsConfig(
            tools=[text_response_tool, stub_data_search_tool, stub_data_analysis_tool],
            end_turn_tool=end_turn_tool,
            max_allowed_tool_calls_per_turn=16,
        ),
    )
    responses = agent.run(
        global_context(),
        conversation=[
            LLMUserMessage(
                content="What was the mean sea surface temperature in the Black Sea on June 01, 2024?",  # noqa: E501
            )
        ],
    )
    responses = [response async for response in responses]

    # print responses if using a real LLM
    if not isinstance(client, _MockBedrockRuntimeClient):
        rich_print(responses)

    for r in responses:
        if isinstance(r, TextResponse):
            assert "20.03" in r.text
