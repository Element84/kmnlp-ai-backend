import logging
from typing import Any

import pytest
import xarray as xr
from langfuse import observe  # pyright: ignore[reportUnknownVariableType]
from shapely.geometry import shape

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    make_function_call_model,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.base import ExecutionConfig, execute_algorithm
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
    SpatioTemporalDataArrayFunctionSet,
)
from llm_agent.context import ToolCallContext
from llm_agent.tools.data_analysis_tool.types import GeoTIFFRef
from llm_agent.tools.data_analysis_tool.utils import normalized_data_array
from tests.utils.mocks.context import MockChatContext

log = logging.getLogger(__name__)


@observe()
@pytest.mark.asyncio
@pytest.mark.integration
async def test_algorithm_exec_with_user_provided_geometry(
    chat_context: MockChatContext, gulf_of_mexico_geojson_dict: dict[str, Any]
) -> None:
    algo = Algorithm[
        make_function_call_model(SpatioTemporalDataArrayFunctionSet().functions)
    ].model_validate(
        {
            "statements": [
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
                    "description": "SST data filtered to May 1, 2020",
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
                    "description": "Daily average SST for May 1, 2020",
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
                    "description": "SST data filtered to Gulf of Mexico region",
                },
                {
                    "type": "VariableAssignment",
                    "name": "gulf_sst_geotiff",
                    "value": {
                        "type": "FunctionCall",
                        "func": {
                            "type": "Function",
                            "func_name": "CreateGeoTIFF",
                            "data_array": {"type": "VariableReference", "name": "gulf_sst_data"},
                        },
                    },
                    "description": "GeoTIFF of SST in Gulf of Mexico",
                },
            ],
            "output_variables": [{"type": "VariableReference", "name": "gulf_sst_geotiff"}],
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
    input_vars = {
        "sst": normalized_data_array(ds["analysed_sst"]),
        "gulf_polygon": shape(gulf_of_mexico_geojson_dict),  # pyright: ignore[reportUnknownMemberType]
    }

    tool_call_context = ToolCallContext(
        tool_call_id="some-id", chat_context=chat_context, progress_callback=lambda _: None
    )

    result = await execute_algorithm(
        tool_call_context, algo, input_vars, exec_config=ExecutionConfig()
    )

    assert isinstance(result["gulf_sst_geotiff"], GeoTIFFRef)
