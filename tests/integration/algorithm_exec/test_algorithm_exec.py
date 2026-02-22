import logging
from collections.abc import Callable

import pytest
import xarray as xr
from langfuse import observe  # pyright: ignore[reportUnknownVariableType]

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    make_function_call_model,
)
from llm_agent.agents.algorithm_agent.algorithm.exec import execute_algorithm
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
    SpatioTemporalDataArrayFunctionSet,
)
from llm_agent.context import LLMAgentContext, ToolCallContext
from llm_agent.tools.data_analysis_tool.types import GeoTIFFRef
from llm_agent.tools.data_analysis_tool.utils import normalized_data_array

log = logging.getLogger(__name__)


@observe()
@pytest.mark.asyncio
@pytest.mark.integration
async def test_algorithm_exec(global_context: Callable[..., LLMAgentContext]) -> None:
    algo = Algorithm[
        make_function_call_model(SpatioTemporalDataArrayFunctionSet().functions)
    ].model_validate(
        {
            "statements": [
                {
                    "type": "VariableAssignment",
                    "name": "gulf_polygon",
                    "value": {
                        "type": "FunctionCall",
                        "func": {
                            "type": "Function",
                            "func_name": "GeocodeNaturalLanguageDescription",
                            "text": "Gulf of Mexico",
                        },
                    },
                    "description": "Polygon defining the Gulf of Mexico region",
                },
                {
                    "type": "VariableAssignment",
                    "name": "sst_may1_2020",
                    "value": {
                        "type": "FunctionCall",
                        "func": {
                            "type": "Function",
                            "func_name": "SliceDataArray_LatLonTime",
                            "data_array": {"type": "VariableReference", "name": "sst"},
                            "dim_slices": {"time": ["2020-05-01", "2020-05-01"]},
                        },
                    },
                    "description": "Wind speed data filtered to May 1, 2020",
                },
                {
                    "type": "VariableAssignment",
                    "name": "sst_may1_avg",
                    "value": {
                        "type": "FunctionCall",
                        "func": {
                            "type": "Function",
                            "func_name": "ReduceDataArray_LatLonTime",
                            "data_array": {"type": "VariableReference", "name": "sst_may1_2020"},
                            "op": "mean",
                            "dim": "time",
                        },
                    },
                    "description": "Daily average wind speed for May 1, 2020",
                },
                {
                    "type": "VariableAssignment",
                    "name": "gulf_sst_data",
                    "value": {
                        "type": "FunctionCall",
                        "func": {
                            "type": "Function",
                            "func_name": "FilterDataArrayByPolygon",
                            "data_array": {"type": "VariableReference", "name": "sst_may1_avg"},
                            "polygon": {"type": "VariableReference", "name": "gulf_polygon"},
                        },
                    },
                    "description": "Wind speed data filtered to Gulf of Mexico region",
                },
                {
                    "type": "VariableAssignment",
                    "name": "gulf_sst_map",
                    "value": {
                        "type": "FunctionCall",
                        "func": {
                            "type": "Function",
                            "func_name": "PlotDataArraysAsRaster",
                            "data_arrays": [{"type": "VariableReference", "name": "gulf_sst_data"}],
                            "subplot_grid_shape": [1, 1],
                            "title": "Wind Speeds in Gulf of Mexico - May 1, 2020",
                        },
                    },
                    "description": "Raster plot of wind speeds in Gulf of Mexico",
                },
            ],
            "output_variables": [{"type": "VariableReference", "name": "gulf_sst_map"}],
        }
    )

    ds = xr.open_dataset(  # pyright: ignore[reportUnknownMemberType]
        "s3://data-c6c22a2e42294c11b52ee7f0c792c071/crw/daily_5km/zarr_reference_final.parquet",
        engine="kerchunk",
        storage_options={
            "remote_protocol": "s3",
            "target_protocol": "s3",
        },
    )
    input_vars = {"sst": normalized_data_array(ds["analysed_sst"])}

    tool_call_context = ToolCallContext(
        tool_call_id="some-id", global_context=global_context(), progress_callback=lambda _: None
    )

    result = await execute_algorithm(tool_call_context, algo, input_vars)

    assert isinstance(result["gulf_sst_map"], GeoTIFFRef)


@observe()
@pytest.mark.asyncio
@pytest.mark.integration
async def test_algorithm_exec_with_user_provided_geometry(
    global_context: Callable[..., LLMAgentContext], gulf_of_mexico_geojson_dict: str
) -> None:
    algo = Algorithm[
        make_function_call_model(SpatioTemporalDataArrayFunctionSet().functions)
    ].model_validate(
        {
            "statements": [
                {
                    "type": "VariableAssignment",
                    "name": "gulf_polygon",
                    "value": {
                        "type": "FunctionCall",
                        "func": {
                            "type": "Function",
                            "func_name": "AssignUserProvidedGeometry",
                            "geometry": gulf_of_mexico_geojson_dict,
                        },
                    },
                    "description": "Polygon defining the Gulf of Mexico region",
                },
                {
                    "type": "VariableAssignment",
                    "name": "sst_may1_2020",
                    "value": {
                        "type": "FunctionCall",
                        "func": {
                            "type": "Function",
                            "func_name": "SliceDataArray_LatLonTime",
                            "data_array": {"type": "VariableReference", "name": "sst"},
                            "dim_slices": {"time": ["2020-05-01", "2020-05-01"]},
                        },
                    },
                    "description": "Wind speed data filtered to May 1, 2020",
                },
                {
                    "type": "VariableAssignment",
                    "name": "sst_may1_avg",
                    "value": {
                        "type": "FunctionCall",
                        "func": {
                            "type": "Function",
                            "func_name": "ReduceDataArray_LatLonTime",
                            "data_array": {"type": "VariableReference", "name": "sst_may1_2020"},
                            "op": "mean",
                            "dim": "time",
                        },
                    },
                    "description": "Daily average wind speed for May 1, 2020",
                },
                {
                    "type": "VariableAssignment",
                    "name": "gulf_sst_data",
                    "value": {
                        "type": "FunctionCall",
                        "func": {
                            "type": "Function",
                            "func_name": "FilterDataArrayByPolygon",
                            "data_array": {"type": "VariableReference", "name": "sst_may1_avg"},
                            "polygon": {"type": "VariableReference", "name": "gulf_polygon"},
                        },
                    },
                    "description": "Wind speed data filtered to Gulf of Mexico region",
                },
                {
                    "type": "VariableAssignment",
                    "name": "gulf_sst_map",
                    "value": {
                        "type": "FunctionCall",
                        "func": {
                            "type": "Function",
                            "func_name": "PlotDataArraysAsRaster",
                            "data_arrays": [{"type": "VariableReference", "name": "gulf_sst_data"}],
                            "subplot_grid_shape": [1, 1],
                            "title": "Wind Speeds in Gulf of Mexico - May 1, 2020",
                        },
                    },
                    "description": "Raster plot of wind speeds in Gulf of Mexico",
                },
            ],
            "output_variables": [{"type": "VariableReference", "name": "gulf_sst_map"}],
        }
    )

    ds = xr.open_dataset(  # pyright: ignore[reportUnknownMemberType]
        "s3://data-c6c22a2e42294c11b52ee7f0c792c071/crw/daily_5km/zarr_reference_final.parquet",
        engine="kerchunk",
        storage_options={
            "remote_protocol": "s3",
            "target_protocol": "s3",
        },
    )
    input_vars = {"sst": normalized_data_array(ds["analysed_sst"])}

    tool_call_context = ToolCallContext(
        tool_call_id="some-id", global_context=global_context(), progress_callback=lambda _: None
    )

    result = await execute_algorithm(tool_call_context, algo, input_vars)

    assert isinstance(result["gulf_sst_map"], GeoTIFFRef)
