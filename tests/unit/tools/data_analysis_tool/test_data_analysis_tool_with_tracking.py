from collections.abc import Callable
from typing import Any
from unittest.mock import Mock, patch

from anyio import to_thread
from e84_geoai_common.llm.core import JSONContent, LLMToolUseContent
from shapely.geometry import box

from llm_agent.agents.algorithm_agent import AlgorithmProblemSpec, AlgorithmSolutionSpec
from llm_agent.agents.algorithm_agent.algorithm.base import (
    make_function_call_model,
)
from llm_agent.agents.algorithm_agent.algorithm.exec import (
    GeocodingSuccessUpdate,
)
from llm_agent.agents.algorithm_agent.interface import input_vars_to_function_set
from llm_agent.context import LLMAgentContext, ToolCallContext, ToolProgressUpdate
from llm_agent.tools.data_analysis_tool.data_analysis_tool import (
    DataAnalysisTool,
    DataAnalysisToolInput,
    DataAnalysisToolProblemFormulatedUpdate,
    DataAnalysisToolSolutionGeneratedUpdate,
    _prepare_problem,  # type: ignore[reportPrivateUsage]
)
from llm_agent.tools.llm_tool_with_tracking import LLMToolWithTracking
from llm_agent.utils.utils import EnrichedLLMToolResultContent
from tests.unit.agents.algorithm_agent.utils import make_test_data_array_spatiotemporal
from tests.unit.tools.data_analysis_tool.test_data_analysis_tool import STUB_SOLUTION, TOOL_INPUT


class TestDataAnalysisToolWithTracking:
    async def test_data_analysis_tool(self, global_context: Callable[..., LLMAgentContext]):
        # Tool to call
        data_analysis_tool = DataAnalysisTool()
        data_analysis_tool_with_tracking = LLMToolWithTracking(data_analysis_tool)

        # Prepare input
        tool_input = {
            "tracking": {
                "plan_id": "plan-1",
                "step_id": "step-1",
            },
            "tool_input": TOOL_INPUT,
        }
        tool_use_request = LLMToolUseContent(id="abc123", name="DataAnalysisTool", input=tool_input)

        parsed_data_analysis_tool_input = DataAnalysisToolInput.model_validate(
            tool_input["tool_input"]
        )

        # Setup types for algorithm execution works
        problem = _prepare_problem(
            global_context().knowledge_graph_dao,
            AlgorithmProblemSpec,
            tool_input=parsed_data_analysis_tool_input,
        )
        assert isinstance(problem, AlgorithmProblemSpec)

        function_set = input_vars_to_function_set(problem.inputs_provided)
        function_call_model = make_function_call_model(function_set.functions)
        solution_spec = AlgorithmSolutionSpec[function_call_model]

        progress_updates: list[ToolProgressUpdate[Any]] = []

        def progress_callback(update: ToolProgressUpdate[Any]) -> None:
            progress_updates.append(update)

        tool_call_context = ToolCallContext(
            tool_call_id=tool_use_request.id,
            global_context=global_context(),
            progress_callback=progress_callback,
        )

        # mock BedrockClaudeLLM() and AlgorithmAgent.run() to avoid Bedrock call
        # mock get_variable_data() to avoid S3 access
        # mock geocode_natural_language_description() to avoid Bedrock call
        with (
            patch(
                "llm_agent.tools.data_analysis_tool.data_analysis_tool.BedrockClaudeLLM",
                return_value=Mock(),
            ),
            patch(
                "llm_agent.tools.data_analysis_tool.data_analysis_tool._construct_solution",
                return_value=solution_spec.model_validate(STUB_SOLUTION),
            ),
            patch(
                "llm_agent.tools.data_analysis_tool.data_analysis_tool._get_variable_data",
                return_value=make_test_data_array_spatiotemporal(
                    start_date="2024-01-01", num_time_steps=365
                ),
            ),
            patch(
                "llm_agent.agents.algorithm_agent.algorithm.exec.geocode_natural_language_description",
                return_value=box(-10, -10, 10, 10),
            ),
        ):
            tool_result = await to_thread.run_sync(
                data_analysis_tool_with_tracking.execute, tool_call_context, tool_use_request
            )

        assert isinstance(tool_result, EnrichedLLMToolResultContent)
        expected_content_types = {JSONContent}
        assert {type(c) for c in tool_result.content} == expected_content_types

        assert len(progress_updates) == 3
        assert isinstance(progress_updates[0].update, DataAnalysisToolProblemFormulatedUpdate)
        assert isinstance(progress_updates[1].update, DataAnalysisToolSolutionGeneratedUpdate)
        assert isinstance(progress_updates[2].update, GeocodingSuccessUpdate)
