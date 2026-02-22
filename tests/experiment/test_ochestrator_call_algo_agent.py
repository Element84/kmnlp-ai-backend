# ruff: noqa: E501
import logging
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest
from e84_geoai_common.llm.core import (
    ExecutableLLMTool,
    JSONContent,
    LLMAssistantMessage,
    LLMMessage,
    LLMToolResultContent,
    LLMToolUseContent,
    LLMUserMessage,
    TextContent,
)
from e84_geoai_common.llm.models import BedrockClaudeLLM
from langfuse.experiment import ExperimentItem, TaskFunction

from llm_agent import (
    ORCHESTRATION_AGENT_DEFAULT_LLM_MODEL_ID,
)
from llm_agent.agents.orchestration_agent.orchestration_agent import OrchestrationAgent
from llm_agent.agents.tools_config import ToolsConfig
from llm_agent.tools.data_analysis_tool.data_analysis_tool import (
    DataAnalysisTool,
)
from llm_agent.tools.data_search_tool import DataSearchTool
from llm_agent.tools.llm_tool_with_tracking import LLMToolWithTracking
from llm_agent.tools.response_shaping_tools import (
    agent_plan_tool,
    end_turn_tool,
    text_response_tool,
)
from llm_agent.utils.prompt_as_service import PromptServiceAdapter
from tests import shared_test_token_limiting_transformer
from tests.experiment.langfuse_test import LangfuseTest
from tests.experiment.util.evals import (
    RUN_EVAL_FRACTION_EXPECTED_CALL_EVAL_NAME,
    RUN_EVAL_FRACTION_NO_BOUNDING_BOX_EVAL_NAME,
    eval_has_data_analysis_tool_call,
    eval_problem_statement_does_not_pass_bounding_box,
    run_eval_fraction_expected_call,
)

if TYPE_CHECKING:
    from llm_agent.context import ToolCallContext

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class TestOrchestratorCallAlgoAgent(LangfuseTest):
    @pytest.mark.integration
    async def test_basic_dataset(self):
        dataset = self.langfuse.get_dataset("Orchestration Agent Single Turn")

        llm = BedrockClaudeLLM(model_id=ORCHESTRATION_AGENT_DEFAULT_LLM_MODEL_ID)
        data_search_tool = LLMToolWithTracking(
            DataSearchTool(
                dataset_picking_agent=Mock()  # We don't actually make the tool call
            )
        )
        data_analysis_tool = LLMToolWithTracking(DataAnalysisTool())

        default_tools: list[ExecutableLLMTool[ToolCallContext]] = [
            data_search_tool,
            data_analysis_tool,
            text_response_tool,
            agent_plan_tool,
        ]

        prompt_service = shared_test_token_limiting_transformer.and_then_service(
            PromptServiceAdapter(llm.prompt)
        )

        orchestration_agent = OrchestrationAgent(
            prompt_service,
            tools_config=ToolsConfig(
                tools=default_tools, end_turn_tool=end_turn_tool, max_allowed_tool_calls_per_turn=1
            ),
        )

        result = dataset.run_experiment(
            name="orchestration_agent_single_turn",
            task=SinglePromptTask(orchestration_agent),
            evaluators=[
                eval_has_data_analysis_tool_call,
                eval_problem_statement_does_not_pass_bounding_box,
            ],
            run_evaluators=[run_eval_fraction_expected_call],
            max_concurrency=8,
        )

        assert len(result.run_evaluations) == 2
        expected_tool_call = next(
            cast("float", evaluation.value)
            for evaluation in result.run_evaluations
            if evaluation.name == RUN_EVAL_FRACTION_EXPECTED_CALL_EVAL_NAME
        )
        expected_args_call = next(
            cast("float", evaluation.value)
            for evaluation in result.run_evaluations
            if evaluation.name == RUN_EVAL_FRACTION_NO_BOUNDING_BOX_EVAL_NAME
        )

        assert float(expected_tool_call) == 1.0
        assert float(expected_args_call) > 0.85


class SinglePromptTask(TaskFunction):
    """Build a curried callable with an orchestration agent."""

    orchestration_agent: OrchestrationAgent

    def __init__(self, orchestration_agent: OrchestrationAgent) -> None:
        self.orchestration_agent = orchestration_agent

    async def __call__(
        self, *, item: ExperimentItem, **_: OrchestrationAgent
    ) -> LLMAssistantMessage:
        """Run a single item in the dataset and return the result.

        Use the limiter from the parent method.

        This method's signature must adhere to `TaskFunction`.

        Args:
            item (ExperimentItem): The item in the dataset to run.
            kwargs (dict[str, Any]): Arbitrary kwargs passed to this function. (Not
                used.)

        Returns:
            The output from running the experiment with the given input. n.b.
            This is the output from the LLM that will be evaluated later
            (optionally with the expected output of the `ExperimentItem`.)
        """
        user_query = item.input["user_query"]  # type: ignore[reportAttributeAccessIssue]
        assert isinstance(user_query, str)
        prompt_messages = _state_to_test(user_query)
        return await self.orchestration_agent._generate_llm_response_to_conversation(  # type: ignore[reportPrivateUsage]
            prompt_messages,
        )


def _state_to_test(user_query: str) -> list[LLMMessage]:
    """Build the state to test for the conversation.

    The `ExperimentItem` being run has arguments. Those arguments are meant to
    be easy to construct. In the case of this particular experiment, the items
    have a single argument representing some user's input. This is the method
    where we build the corresponding conversation list used as a prompt to the
    LLM.

    Args:
        user_query (str): The hypothetical user input from the experiment item.

    Returns:
        A conversation state based on the user's input with up to a response
        from the data search tool.
    """
    return [
        LLMUserMessage(content=[TextContent(text=user_query)]),
        LLMAssistantMessage(
            content=[
                LLMToolUseContent(
                    id="toolu_bdrk_01E2h7rhJSRzqKPoxqyMfhCA",
                    name="AgentPlanTool",
                    input={
                        "plan_or_update": {
                            "content_type": "plan",
                            "id": "gulf_mexico_sst_plan",
                            "title": user_query,
                            "goal": "Retrieve and analyze sea surface temperature data for the Gulf of Mexico on June 1, 2023",
                            "steps": [
                                {
                                    "id": "search_sst_data",
                                    "title": "Search for sea surface temperature datasets",
                                    "status": "pending",
                                },
                                {
                                    "id": "analyze_sst_data",
                                    "title": "Extract SST values for Gulf of Mexico on June 1, 2023",
                                    "status": "pending",
                                },
                            ],
                        }
                    },
                ),
                LLMToolUseContent(
                    id="toolu_bdrk_015JgRTn6Rrifc3wNwZmD6Fj",
                    name="DataSearchToolWithTracking",
                    input={
                        "tracking": {
                            "plan_id": "gulf_mexico_sst_plan",
                            "step_id": "search_sst_data",
                        },
                        "tool_input": {
                            "natural_language_query": "sea surface temperature Gulf of Mexico June 2023"
                        },
                    },
                ),
            ]
        ),
        LLMUserMessage(
            content=[
                LLMToolResultContent(
                    id="toolu_bdrk_01E2h7rhJSRzqKPoxqyMfhCA",
                    content=[JSONContent(data={})],
                    status="success",
                ),
                LLMToolResultContent(
                    id="toolu_bdrk_015JgRTn6Rrifc3wNwZmD6Fj",
                    content=[
                        JSONContent(
                            data={
                                "datasets": [
                                    {
                                        "id": "crw_daily_5km",
                                        "source": "s3://data-c6c22a2e42294c11b52ee7f0c792c071/crw/daily_5km/zarr_reference_final.parquet",
                                        "label": "NOAA Coral Reef Watch Daily Global 5km Satellite Sea Surface Temperature (CoralTemp)",
                                        "description": "CoralTemp is derived from three different but related 5km daily gap-free SST data sets and provides an internally consistent SST product that stretches from 1985 to present: OSTIA Sea Surface Temperature Reanalysis (1985-2002), Geo-Polar Blended Night-only Sea Surface Temperature Reanalysis (2002-2016), Geo-Polar Blended Night-only Sea Surface Temperature Near Real-Time (2017 to present).",
                                        "temporal_coverage": [
                                            "2020-01-01T00:00:00Z",
                                            "2024-12-01T00:00:00Z",
                                        ],
                                        "temporal_resolution": "P1D",
                                        "spatial_coverage": "global",
                                        "variables": [
                                            {
                                                "id": "sea_ice_area_fraction_crw_daily_5km",
                                                "name": "sea_ice_area_fraction",
                                                "field_name": "sea_ice_fraction",
                                                "description": "Sea Ice Fraction",
                                                "units": "1",
                                                "dimensions": ["lat", "lon", "time"],
                                            },
                                            {
                                                "id": "sea_surface_temperature_crw_daily_5km",
                                                "name": "sea_surface_temperature",
                                                "field_name": "analysed_sst",
                                                "description": "analysed sea surface temperature",
                                                "units": "degrees_Celsius",
                                                "dimensions": ["lat", "lon", "time"],
                                            },
                                        ],
                                    }
                                ]
                            }
                        )
                    ],
                    status="success",
                ),
            ]
        ),
    ]
