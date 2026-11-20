import json
from collections.abc import Awaitable, Callable
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from pydantic_ai import RunContext
from shapely.geometry import box, mapping

from llm_agent.agents.algorithm_agent import AlgorithmProblemSpec, AlgorithmSolutionSpec
from llm_agent.agents.algorithm_agent.algorithm.base import make_function_call_model
from llm_agent.agents.algorithm_agent.algorithm.exec.base import (
    ComputedVariableUpdate,
    ComputingVariableUpdate,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.data_array import (
    FilterByPolygonUpdate,
)
from llm_agent.agents.algorithm_agent.interface import input_vars_to_function_set
from llm_agent.context import AgentDeps, ToolProgressUpdate
from llm_agent.model_factory import make_bedrock_model
from llm_agent.tools.agent_tool import ErrorResponse
from llm_agent.tools.data_analysis_tool.data_analysis_tool import (
    DataAnalysisTool,
    DataAnalysisToolInput,
    DataAnalysisToolOutput,
    DataAnalysisToolProblemFormulatedUpdate,
    DataAnalysisToolSolutionGeneratedUpdate,
    _prepare_problem,  # type: ignore[reportPrivateUsage]
)
from llm_agent.tools.data_analysis_tool.utils import algorithm_to_pseudocode
from tests.unit.agents.algorithm_agent.utils import make_test_data_array_spatiotemporal
from tests.utils.mocks.context import MockChatContext

TOOL_INPUT = {
    "problem_statement": "Plot the sea surface temperature in the Black "
    "Sea on June 01, 2024 and compute its mean.",
    "input_data": [
        "sea_surface_temperature_crw_5km_sst_daily",
        {
            "data_ref_type": "geometry",
            "key": "black_sea_geometry_key",
            "name": "black_sea_polygon",
            "description": "Polygon representing the Black Sea area",
            "geometry_type": "Polygon",
        },
    ],
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
    "analysis": "__analysis__",
    "pseudocode": "__pseudocode__",
    "algorithm": {
        "statements": [
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
                        "method": "exact",
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
                "name": "sst_mean_scalar",
                "value": {
                    "type": "FunctionCall",
                    "func": {
                        "type": "Function",
                        "func_name": "ReduceDataArray_LatLonTime",
                        "data_array": {"type": "VariableReference", "name": "sst_black_sea"},
                        "op": "mean",
                        "dim": None,
                        "q": None,
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
                        "func_name": "MakePlot",
                        "plot_spec": {
                            "spec_name": "SSTPlotSpec",
                            "sst_data_array": {
                                "type": "VariableReference",
                                "name": "sst_black_sea",
                            },
                            "sea_ice_data_array": None,
                            "ice_threshold": 0.15,
                            "map_projection": {
                                "projection": "PlateCarree",
                                "central_longitude": 0.0,
                            },
                            "colorbar_min_and_max": [
                                None,
                                None,
                            ],
                            "title": "Sea Surface Temperature in the Black Sea on June 01, 2024",
                        },
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
    def test_serialization(self, chat_context: MockChatContext) -> None:
        problem = _prepare_problem(
            chat_context.global_context.knowledge_graph_dao,
            AlgorithmProblemSpec,
            tool_input=DataAnalysisToolInput.model_validate(TOOL_INPUT),
        )
        assert not isinstance(problem, ErrorResponse)
        problem, _ = problem
        assert isinstance(problem, AlgorithmProblemSpec)
        function_set = input_vars_to_function_set(problem.inputs_provided)
        function_call_model = make_function_call_model(function_set.functions)
        solution_spec = AlgorithmSolutionSpec[function_call_model]
        solution = solution_spec.model_validate(STUB_SOLUTION)

        result = DataAnalysisToolSolutionGeneratedUpdate[solution_spec](
            solution=solution, pseudocode=algorithm_to_pseudocode(solution.algorithm)
        )
        result_serialized = result.model_dump(mode="json")

        pseudocode: str = result_serialized.pop("pseudocode")
        assert len(pseudocode.split("\n")) > 1

        assert result_serialized == {
            "solution": STUB_SOLUTION,
        }


class TestDataAnalysisTool:
    @classmethod
    def setup_class(cls) -> None:
        cls.tool = DataAnalysisTool(make_bedrock_model(Mock(), "__model_id__"))
        cls.tool_input = TOOL_INPUT
        cls.tool_input_validated = DataAnalysisToolInput.model_validate(cls.tool_input)

    async def test_execution(
        self,
        chat_context: MockChatContext,
        make_run_context: Callable[..., RunContext[AgentDeps]],
    ):
        problem = _prepare_problem(
            chat_context.global_context.knowledge_graph_dao,
            AlgorithmProblemSpec,
            tool_input=self.tool_input_validated,
        )
        assert not isinstance(problem, ErrorResponse)
        problem, _ = problem
        assert isinstance(problem, AlgorithmProblemSpec)
        function_set = input_vars_to_function_set(problem.inputs_provided)
        function_call_model = make_function_call_model(function_set.functions)
        solution_spec = AlgorithmSolutionSpec[function_call_model]

        progress_updates: list[ToolProgressUpdate[Any]] = []
        ctx = make_run_context(chat_context, progress_callback=progress_updates.append)

        # the geometry is resolved upstream and passed to the tool as an input, so pre-store
        # the geometry the GeometryRef points to.
        chat_context.data_store["black_sea_geometry_key"] = json.dumps(
            mapping(box(-10, -10, 10, 10))
        )

        # mock get_variable_data() to avoid S3 access
        with (
            patch(
                "llm_agent.tools.data_analysis_tool.data_analysis_tool._construct_solution",
                new_callable=AsyncMock,
                return_value=solution_spec.model_validate(STUB_SOLUTION),
            ),
            patch(
                "llm_agent.tools.data_analysis_tool.data_analysis_tool._get_variable_data",
                new_callable=AsyncMock,
                return_value=make_test_data_array_spatiotemporal(
                    start_date="2024-01-01", num_time_steps=365
                ),
            ),
            patch("cartopy.mpl.geoaxes.GeoAxes.add_feature"),
            patch("cartopy.mpl.geoaxes.GeoAxes.coastlines"),
            patch("cartopy.mpl.geoaxes.GeoAxes.stock_img"),
        ):
            result = await self.tool(ctx, self.tool_input_validated)

        assert isinstance(result, DataAnalysisToolOutput)
        assert result.outputs is not None

        assert {type(update.update) for update in progress_updates} == {
            DataAnalysisToolProblemFormulatedUpdate,
            DataAnalysisToolSolutionGeneratedUpdate,
            FilterByPolygonUpdate,
            ComputingVariableUpdate,
            ComputedVariableUpdate,
        }

    async def test_unhandled_exception(
        self,
        chat_context: MockChatContext,
        make_run_context: Callable[..., RunContext[AgentDeps]],
        invoke_tool_safely: Callable[..., Awaitable[Any]],
    ):
        ctx = make_run_context(chat_context)
        # mock _construct_solution() to raise an unhandled exception
        # mock get_variable_data() to avoid S3 access
        with (
            patch(
                "llm_agent.tools.data_analysis_tool.data_analysis_tool._construct_solution",
                new_callable=AsyncMock,
                side_effect=ZeroDivisionError,
            ),
            patch(
                "llm_agent.tools.data_analysis_tool.data_analysis_tool._get_variable_data",
                new_callable=AsyncMock,
                return_value=make_test_data_array_spatiotemporal(
                    start_date="2024-01-01", num_time_steps=365
                ),
            ),
        ):
            result = await invoke_tool_safely(self.tool, ctx, self.tool_input_validated)

        assert isinstance(result, ErrorResponse)
        assert result.message == "An unhandled exception was encountered: `ZeroDivisionError`"

    def test_no_error_on_repr(self):
        with does_not_raise():
            _ = repr(self.tool)
