from typing import cast
from unittest.mock import patch

import pytest
from anyio import to_thread
from e84_geoai_common.llm.core import (
    Base64ImageContent,
    JSONContent,
    LLMToolUseContent,
    TextContent,
)
from shapely.geometry import box

from llm_agent.agents.algorithm_agent import (
    AlgorithmAgent,
    AlgorithmSolutionSpec,
    DataArrayFunctionSet,
    GeneralPlottingFunctionSet,
)
from llm_agent.tools.data_analysis_tool import (
    data_analysis_tool,
)
from tests.unit.agents.algorithm_agent.utils import make_dummy_data_array

_stub_algo_solution = {
    "thoughts": "__thoughts__",
    "pseudocode": "__pseudocode__",
    "algorithm": {
        "statements": [
            {
                "name": "black_sea_polygon",
                "value": {
                    "func": {
                        "func_name": "GeocodeNaturalLanguageDescription",
                        "text": "Black Sea",
                    }
                },
                "description": "Polygon representing the Black Sea area",
            },
            {
                "name": "sst_filtered_time",
                "value": {
                    "func": {
                        "func_name": "FilterDataArray",
                        "data_array": {"name": "analysed_sst"},
                        "time_range": ["2024-06-01", "2024-06-01"],
                    }
                },
                "description": "SST data filtered to June 01, 2024",
            },
            {
                "name": "sst_black_sea",
                "value": {
                    "func": {
                        "func_name": "FilterDataArrayByPolygon",
                        "data_array": {"name": "sst_filtered_time"},
                        "polygon": {"name": "black_sea_polygon"},
                    }
                },
                "description": "SST data for the Black Sea on June 01, 2024",
            },
            {
                "name": "sst_mean_dataarray",
                "value": {
                    "func": {
                        "func_name": "ReduceDataArray",
                        "data_array": {"name": "sst_black_sea"},
                        "op": "mean",
                    }
                },
                "description": "Mean SST value for the Black Sea on June 01, 2024 as a DataArray",  # noqa: E501
            },
            {
                "name": "sst_mean_scalar",
                "value": {
                    "func": {
                        "func_name": "ZeroDimDataArrayToScalar",
                        "data_array": {"name": "sst_mean_dataarray"},
                    }
                },
                "description": "Mean SST value for the Black Sea on June 01, 2024 as a scalar",  # noqa: E501
            },
            {
                "name": "sst_plot",
                "value": {
                    "func": {
                        "func_name": "PlotDataArraysAsRaster",
                        "data_arrays": [{"name": "sst_black_sea"}],
                        "subplot_grid_shape": [1, 1],
                        "title": "Sea Surface Temperature in the Black Sea on June 01, 2024",  # noqa: E501
                    }
                },
                "description": "Plot of the SST in the Black Sea on June 01, 2024",  # noqa: E501
            },
        ],
        "output_variables": [
            {"name": "sst_mean_scalar"},
            {"name": "sst_plot"},
        ],
    },
    "explanation": "__explanation__",
}


def test_name_set_correctly():
    assert data_analysis_tool.name == "DataAnalysisTool"


def test_description_set_correctly():
    assert data_analysis_tool.description == (
        "A tool that can perform complex data analysis to compute "
        "quantitative results from data. It can also create data "
        "visualizations. This tool simply constructs a solution, executes it, "
        "and returns the results. It has no ability to inspect the outputs it "
        "produces, nor does it remember context from previous calls."
    )


@pytest.mark.asyncio
async def test_data_analysis_tool():
    tool_input = {
        "problem_statement": "Plot the sea surface temperature in the Black "
        "Sea on June 01, 2024 and compute its mean.",
        "variable_ids": [
            "http://kmnlp-demo.noaa.gov/sea_surface_temperature_crw_daily_5km"
        ],
        "outputs_requested": [
            {
                "name": "black_sea_plot",
                "variable_type": "raster_plot",
                "description": "Plot of the SST in the Black Sea on June 01, 2024.",  # noqa: E501
            },
            {
                "name": "black_sea_mean",
                "variable_type": "scalar_float",
                "description": "Mean SST in the Black Sea on June 01, 2024.",
            },
        ],
    }
    tool_use_request = LLMToolUseContent(
        id="abc123", name="DataAnalysisTool", input=tool_input
    )
    # mock AlgorithmAgent.run() to avoid Bedrock call
    # mock get_variable_data() to avoid S3 access
    # mock geocode_natural_language_description() to avoid Bedrock call
    solution_spec = AlgorithmSolutionSpec[
        DataArrayFunctionSet | GeneralPlottingFunctionSet
    ]
    with (
        patch.object(
            AlgorithmAgent,
            "run",
            return_value=solution_spec.model_validate(_stub_algo_solution),
        ),
        patch(
            "llm_agent.tools.data_analysis_tool._get_variable_data",
            return_value=make_dummy_data_array(
                start_date="2024-01-01", num_time_steps=365
            ),
        ),
        patch(
            "llm_agent.agents.algorithm_agent.algorithm.exec.geocode_natural_language_description",
            return_value=box(-10, -10, 10, 10),
        ),
    ):
        tool_result = await to_thread.run_sync(
            data_analysis_tool.execute, tool_use_request
        )

    expected_content_types = {JSONContent, Base64ImageContent}
    assert {type(c) for c in tool_result.content} == expected_content_types


@pytest.mark.asyncio
async def test_data_analysis_tool_error_if_invalid_dims():
    tool_input = {
        "problem_statement": "Plot historical yearly total solar irradiance.",
        "variable_ids": [
            "http://kmnlp-demo.noaa.gov/solar_irradiance_per_unit_wavelength_noaa-cdr-solar-spectral-irradiance-pds_yearly"
        ],
        "outputs_requested": [
            {
                "name": "tsi_plot",
                "variable_type": "line_plot",
                "description": "TSI plot.",
            },
        ],
    }
    tool_use_request = LLMToolUseContent(
        id="abc123", name="DataAnalysisTool", input=tool_input
    )

    with patch.object(
        AlgorithmAgent,
        "run",
        side_effect=RuntimeError(
            "Variable with invalid dims not caught in time."
        ),
    ):
        tool_result = await to_thread.run_sync(
            data_analysis_tool.execute, tool_use_request
        )

    assert len(tool_result.content) == 1
    assert isinstance(tool_result.content[0], JSONContent)
    error_message = cast("str", tool_result.content[0].data["error_message"])
    assert "dims" in error_message.lower()


@pytest.mark.asyncio
async def test_data_analysis_tool_unhandled_exception():
    tool_input = {
        "problem_statement": "Plot the sea surface temperature in the Black "
        "Sea on June 01, 2024 and compute its mean.",
        "variable_ids": [
            "http://kmnlp-demo.noaa.gov/sea_surface_temperature_crw_daily_5km"
        ],
        "outputs_requested": [
            {
                "name": "black_sea_plot",
                "variable_type": "raster_plot",
                "description": "Plot of the SST in the Black Sea on June 01, 2024.",  # noqa: E501
            },
            {
                "name": "black_sea_mean",
                "variable_type": "scalar_float",
                "description": "Mean SST in the Black Sea on June 01, 2024.",
            },
        ],
    }
    tool_use_request = LLMToolUseContent(
        id="abc123", name="DataAnalysisTool", input=tool_input
    )
    # mock AlgorithmAgent.run() to avoid Bedrock call, and raise unhandled
    #   exception
    # mock get_variable_data() to avoid S3 access
    # mock geocode_natural_language_description() to avoid Bedrock call
    with (
        patch.object(
            AlgorithmAgent,
            "run",
        ) as agent_run,
        patch(
            "llm_agent.tools.data_analysis_tool._get_variable_data",
            return_value=make_dummy_data_array(
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
            data_analysis_tool.execute, tool_use_request
        )

    assert tool_result.id == "abc123"
    assert tool_result.status == "error"
    assert tool_result.content == [
        TextContent(
            text="An unhandled exception was encountered. That exception was: "
            "ZeroDivisionError()"
        )
    ]
