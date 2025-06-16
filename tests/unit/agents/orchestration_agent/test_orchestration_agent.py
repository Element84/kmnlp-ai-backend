from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from e84_geoai_common.llm.core import (
    JSONContent,
    LLMMessage,
    LLMToolResultContent,
    LLMToolUseContent,
)
from e84_geoai_common.llm.models.claude import (
    CLAUDE_BEDROCK_MODEL_IDS,
    BedrockClaudeLLM,
)
from e84_geoai_common.llm.tests.mock_bedrock import (
    _MockBedrockRuntimeClient,  # type: ignore[reportPrivateUsage]
    claude_response_with_content,
    make_test_bedrock_client,
)
from rich import print as rich_print

from llm_agent.agents.orchestration_agent.orchestration_agent import (
    OrchestrationAgent,
)
from llm_agent.model.dataset import Dataset
from llm_agent.tools.data_analysis_tool import (
    algo_outputs_to_tool_result_contents,
    data_analysis_tool,
)
from llm_agent.tools.data_search_tool import (
    DataSearchToolResult,
    data_search_tool,
)
from llm_agent.utils import DEFAULT_MODEL_NAME

stub_responses = [
    # response to initial question
    claude_response_with_content(
        [
            {
                "text": "I will first use the DataSearchTool to find the right variable."  # noqa: E501
            },
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
                "text": "We found a variable! Now using the DataAnalysisTool ..."  # noqa: E501
            },
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
                            "description": "Mean SST in the Black Sea on June 01, 2024.",  # noqa: E501
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
            {"text": "The mean SST was 20.03 deg C."},
        ]
    ),
]

client = make_test_bedrock_client(stub_responses)
llm = BedrockClaudeLLM(
    model_id=CLAUDE_BEDROCK_MODEL_IDS[DEFAULT_MODEL_NAME], client=client
)


def stub_exec_data_search_tool(
    tool_use_request: LLMToolUseContent,
) -> LLMToolResultContent:
    return LLMToolResultContent(
        id=tool_use_request.id,
        content=[
            JSONContent(
                data=DataSearchToolResult(
                    datasets=[
                        Dataset(
                            iri="http://fake/iri",
                            source="http://source",
                            label="label",
                            temporal_coverage=(
                                datetime(
                                    year=2020, month=1, day=1, tzinfo=UTC
                                ),
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


def stub_exec_data_analysis_tool(
    tool_use_request: LLMToolUseContent,
) -> LLMToolResultContent:
    content = LLMToolResultContent(
        id=tool_use_request.id,
        content=algo_outputs_to_tool_result_contents(
            algo_outputs={"black_sea_mean_scalar": 20.03},
            pseudocode="__pseudocode__",
        ),
        status="success",
    )
    return content


stub_data_search_tool = data_search_tool.model_copy(
    update={"execution_func": stub_exec_data_search_tool}
)
stub_data_analysis_tool = data_analysis_tool.model_copy(
    update={"execution_func": stub_exec_data_analysis_tool}
)


@pytest.mark.asyncio
async def test_orchestration_agent() -> None:
    agent = OrchestrationAgent(
        llm, tools=[stub_data_search_tool, stub_data_analysis_tool]
    )
    responses = agent.run(
        LLMMessage(
            role="user",
            content="What was the mean sea surface temperature in the "
            "Black Sea on June 01, 2024?",
        )
    )
    responses = [response async for response in responses]

    # print responses if using a real LLM
    if not isinstance(client, _MockBedrockRuntimeClient):
        rich_print(responses)

    assert "20.03" in cast("LLMMessage", responses[-1]).to_text_only()
