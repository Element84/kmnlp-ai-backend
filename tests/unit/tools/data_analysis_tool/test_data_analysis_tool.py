from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
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
from llm_agent.utils.utils import EnrichedLLMToolResultContent
from tests.unit.agents.algorithm_agent.utils import make_test_data_array_spatiotemporal

data_analysis_tool = DataAnalysisTool()

TOOL_INPUT = {
    "problem_statement": "Plot the sea surface temperature in the Black "
    "Sea on June 01, 2024 and compute its mean.",
    "variable_ids": ["sea_surface_temperature_crw_daily_5km"],
    "outputs_requested": [
        {
            "name": "black_sea_plot",
            "variable_type": "raster_plot",
            "description": "Plot of the SST in the Black Sea on June 01, 2024.",
        },
        {
            "name": "black_sea_mean",
            "variable_type": "scalar_float",
            "description": "Mean SST in the Black Sea on June 01, 2024.",
        },
    ],
}

STUB_SOLUTION = {
    "thoughts": "__thoughts__",
    "pseudocode": "__pseudocode__",
    "algorithm": {
        "statements": [
            {
                "type": "VariableAssignment",
                "name": "black_sea_polygon",
                "value": {
                    "type": "FunctionCall",
                    "func": {
                        "type": "Function",
                        "func_name": "GeocodeNaturalLanguageDescription",
                        "text": "Black Sea",
                    },
                },
                "description": "Polygon representing the Black Sea area",
            },
            {
                "type": "VariableAssignment",
                "name": "sst_filtered_time",
                "value": {
                    "type": "FunctionCall",
                    "func": {
                        "type": "Function",
                        "func_name": "SliceDataArray_LatLonTime",
                        "data_array": {"type": "VariableReference", "name": "analysed_sst"},
                        "dim_slices": {"time": ["2024-06-01", "2024-06-01"]},
                        "dim_index_slices": {},
                    },
                },
                "description": "SST data filtered to June 01, 2024",
            },
            {
                "type": "VariableAssignment",
                "name": "sst_black_sea",
                "value": {
                    "type": "FunctionCall",
                    "func": {
                        "type": "Function",
                        "func_name": "FilterDataArrayByPolygon",
                        "data_array": {"type": "VariableReference", "name": "sst_filtered_time"},
                        "polygon": {"type": "VariableReference", "name": "black_sea_polygon"},
                    },
                },
                "description": "SST data for the Black Sea on June 01, 2024",
            },
            {
                "type": "VariableAssignment",
                "name": "sst_mean_dataarray",
                "value": {
                    "type": "FunctionCall",
                    "func": {
                        "type": "Function",
                        "func_name": "ReduceDataArray_LatLonTime",
                        "data_array": {"type": "VariableReference", "name": "sst_black_sea"},
                        "op": "mean",
                        "dim": None,
                    },
                },
                "description": "Mean SST value for the Black Sea on June 01, 2024 as a DataArray",
            },
            {
                "type": "VariableAssignment",
                "name": "sst_mean_scalar",
                "value": {
                    "type": "FunctionCall",
                    "func": {
                        "type": "Function",
                        "func_name": "ZeroDimDataArrayToScalar",
                        "data_array": {"type": "VariableReference", "name": "sst_mean_dataarray"},
                    },
                },
                "description": "Mean SST value for the Black Sea on June 01, 2024 as a scalar",
            },
            {
                "type": "VariableAssignment",
                "name": "sst_plot",
                "value": {
                    "type": "FunctionCall",
                    "func": {
                        "type": "Function",
                        "func_name": "PlotDataArraysAsRaster",
                        "data_arrays": [{"type": "VariableReference", "name": "sst_black_sea"}],
                        "subplot_grid_shape": [1, 1],
                        "title": "Sea Surface Temperature in the Black Sea on June 01, 2024",
                    },
                },
                "description": "Plot of the SST in the Black Sea on June 01, 2024",
            },
        ],
        "output_variables": [
            {"type": "VariableReference", "name": "sst_mean_scalar"},
            {"type": "VariableReference", "name": "sst_plot"},
        ],
    },
    "explanation": "__explanation__",
    "error_message": None,
}


class TestDataAnalysisToolSolutionGeneratedUpdate:
    def test_serialization(self, global_context: Callable[..., LLMAgentContext]) -> None:
        problem = _prepare_problem(
            global_context().knowledge_graph_dao,
            AlgorithmProblemSpec,
            tool_input=DataAnalysisToolInput.model_validate(TOOL_INPUT),
        )
        assert isinstance(problem, AlgorithmProblemSpec)
        function_set = input_vars_to_function_set(problem.inputs_provided)
        function_call_model = make_function_call_model(function_set.functions)
        solution_spec = AlgorithmSolutionSpec[function_call_model]
        solution = solution_spec.model_validate(STUB_SOLUTION)

        result = DataAnalysisToolSolutionGeneratedUpdate[solution_spec](solution=solution)
        result_serialized = result.model_dump(mode="json")
        assert result_serialized == {
            "solution": STUB_SOLUTION,
        }


class TestDataAnalysisTool:
    @classmethod
    def setup_class(cls) -> None:
        cls.tool = data_analysis_tool
        cls.tool_input = TOOL_INPUT
        cls.tool_input_validated = DataAnalysisToolInput.model_validate(cls.tool_input)

    async def test_execution(self, global_context: Callable[..., LLMAgentContext]):
        tool_use_request = LLMToolUseContent(
            id="abc123", name="DataAnalysisTool", input=self.tool_input
        )

        problem = _prepare_problem(
            global_context().knowledge_graph_dao,
            AlgorithmProblemSpec,
            tool_input=self.tool_input_validated,
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
                self.tool.execute, tool_call_context, tool_use_request
            )

        assert isinstance(tool_result, EnrichedLLMToolResultContent)
        expected_content_types = {JSONContent}
        assert {type(c) for c in tool_result.content} == expected_content_types

        assert len(progress_updates) == 3
        assert isinstance(progress_updates[0].update, DataAnalysisToolProblemFormulatedUpdate)
        assert isinstance(progress_updates[1].update, DataAnalysisToolSolutionGeneratedUpdate)
        assert isinstance(progress_updates[2].update, GeocodingSuccessUpdate)

    async def test_unhandled_exception(self, global_context: Callable[..., LLMAgentContext]):
        tool_use_request = LLMToolUseContent(
            id="abc123", name="DataAnalysisTool", input=self.tool_input
        )
        tool_call_context = ToolCallContext(
            tool_call_id=tool_use_request.id, global_context=global_context()
        )
        # mock BedrockClaudeLLM() and AlgorithmAgent.run() to avoid Bedrock call,
        # and raise unhandled exception
        # mock get_variable_data() to avoid S3 access
        # mock geocode_natural_language_description() to avoid Bedrock call
        with (
            patch(
                "llm_agent.tools.data_analysis_tool.data_analysis_tool.BedrockClaudeLLM",
                return_value=Mock(),
            ),
            patch(
                "llm_agent.tools.data_analysis_tool.data_analysis_tool._construct_solution",
            ) as agent_run,
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
            agent_run.side_effect = ZeroDivisionError
            tool_result = await to_thread.run_sync(
                self.tool.execute, tool_call_context, tool_use_request
            )

        assert tool_result.model_dump(mode="json") == {
            "id": "abc123",
            "content": [
                {
                    "data": {
                        "message": ("An unhandled exception was encountered: `ZeroDivisionError`"),
                    }
                }
            ],
            "status": "error",
        }

    def test_no_error_on_repr(self):
        with does_not_raise():
            _ = repr(self.tool)
