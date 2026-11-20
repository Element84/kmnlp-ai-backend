"""Integration test for end-to-end cancellation propagation."""

import asyncio
import logging
from typing import Any

import numpy as np
import pytest
import xarray as xr
from anyio import from_thread, to_thread
from distributed import Client

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    make_function_call_model,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.base import ExecutionConfig, execute_algorithm
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
    SpatioTemporalDataArrayFunctionSet,
)
from llm_agent.context import ToolCallContext
from llm_agent.tools.data_analysis_tool.utils import normalized_data_array
from tests.utils.mocks.context import MockChatContext

log = logging.getLogger(__name__)

# Global tracking for where cancellation was detected
cancellation_detected_at: list[str] = []


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cancellation_propagates_through_execute_algorithm(
    dask_client: Client,  # noqa: ARG001
    chat_context_integration: MockChatContext,
) -> None:
    """Test that cancelling execute_algorithm propagates to Dask."""
    # Reset tracking
    cancellation_detected_at.clear()

    # Create a large dask array that will take time to compute
    rng = np.random.default_rng(seed=42)
    data = rng.random((200, 200, 200))

    ds = xr.Dataset(
        {
            "test_var": xr.DataArray(
                data=data,
                dims=("time", "lat", "lon"),
                coords={
                    "time": np.arange(200),
                    "lat": np.linspace(-90, 90, 200),
                    "lon": np.linspace(-180, 180, 200),
                },
            )
        }
    )

    # Chunk it to make it a Dask array
    ds = ds.chunk({"time": 10, "lat": 10, "lon": 10})  # type: ignore[reportUnknownMemberType]

    # Create an algorithm that does a slow computation
    # CreateGeoTIFF triggers compute_dask_array
    algo = Algorithm[
        make_function_call_model(SpatioTemporalDataArrayFunctionSet().functions)
    ].model_validate(
        {
            "statements": [
                {
                    "type": "VariableAssignment",
                    "name": "slow_result",
                    "value": {
                        "type": "FunctionCall",
                        "func": {
                            "type": "Function",
                            "func_name": "ReduceDataArray_LatLonTime",
                            "data_array": {"type": "VariableReference", "name": "test_var"},
                            "op": "mean",
                            "dim": "time",
                        },
                    },
                    "description": "Slow reduction",
                },
                {
                    "type": "VariableAssignment",
                    "name": "geotiff",
                    "value": {
                        "type": "FunctionCall",
                        "func": {
                            "type": "Function",
                            "func_name": "CreateGeoTIFF",
                            "data_array": {"type": "VariableReference", "name": "slow_result"},
                        },
                    },
                    "description": "Create GeoTIFF (triggers compute)",
                },
            ],
            "output_variables": [{"type": "VariableReference", "name": "geotiff"}],
        }
    )

    input_vars = {"test_var": normalized_data_array(ds["test_var"])}

    tool_call_context = ToolCallContext(
        tool_call_id="test-cancellation",
        chat_context=chat_context_integration,
        progress_callback=lambda _: None,
    )

    # Start execution in a task
    task = asyncio.create_task(
        execute_algorithm(tool_call_context, algo, input_vars, exec_config=ExecutionConfig())
    )

    # Give it time to start computing
    await asyncio.sleep(0.2)

    log.info("Cancelling task from test")
    # Cancel the task (simulates HTTP request cancellation)
    task.cancel()

    # Verify that cancellation happened
    with pytest.raises(asyncio.CancelledError):
        await task

    log.info("Cancellation test completed - task was cancelled as expected")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cancellation_through_full_tool_execution_path(
    dask_client: Client,  # noqa: ARG001
    chat_context_integration: MockChatContext,
) -> None:
    """Test cancellation through the full tool execution path with thread boundaries.

    This test simulates the actual execution path:
    - Async task (like HTTP request handler)
    - → to_thread.run_sync (sync tool.execute)
    - → from_thread.run (async _a_execute_data_analysis_tool)
    - → execute_algorithm
    - → compute_dask_array
    """
    cancellation_detected_at.clear()

    # Create test data
    rng = np.random.default_rng(seed=42)
    data = rng.random((200, 200, 200))

    ds = xr.Dataset(
        {
            "test_var": xr.DataArray(
                data=data,
                dims=("time", "lat", "lon"),
                coords={
                    "time": np.arange(200),
                    "lat": np.linspace(-90, 90, 200),
                    "lon": np.linspace(-180, 180, 200),
                },
            )
        }
    )
    ds = ds.chunk({"time": 10, "lat": 10, "lon": 10})  # type: ignore[reportUnknownMemberType]

    algo = Algorithm[
        make_function_call_model(SpatioTemporalDataArrayFunctionSet().functions)
    ].model_validate(
        {
            "statements": [
                {
                    "type": "VariableAssignment",
                    "name": "reduced",
                    "value": {
                        "type": "FunctionCall",
                        "func": {
                            "type": "Function",
                            "func_name": "ReduceDataArray_LatLonTime",
                            "data_array": {"type": "VariableReference", "name": "test_var"},
                            "op": "mean",
                            "dim": "time",
                        },
                    },
                    "description": "Reduce array",
                },
                {
                    "type": "VariableAssignment",
                    "name": "result",
                    "value": {
                        "type": "FunctionCall",
                        "func": {
                            "type": "Function",
                            "func_name": "CreateGeoTIFF",
                            "data_array": {"type": "VariableReference", "name": "reduced"},
                        },
                    },
                    "description": "Create GeoTIFF (triggers compute)",
                },
            ],
            "output_variables": [{"type": "VariableReference", "name": "result"}],
        }
    )

    input_vars = {"test_var": normalized_data_array(ds["test_var"])}

    tool_call_context = ToolCallContext(
        tool_call_id="test-full-path",
        chat_context=chat_context_integration,
        progress_callback=lambda _: None,
    )

    def sync_wrapper() -> Any:  # noqa: ANN401
        """Sync function that calls async code - simulates tool.execute pattern."""
        log.info("In sync_wrapper (simulating tool.execute in worker thread)")
        # This simulates the from_thread.run call in DataAnalysisTool._execute_typed
        result = from_thread.run(
            execute_algorithm, tool_call_context, algo, input_vars, ExecutionConfig()
        )
        log.info("sync_wrapper completed")
        return result

    # Simulate the full path: async → sync (thread) → async
    async def run_through_thread_boundary() -> Any:  # noqa: ANN401
        """Simulates the to_thread.run_sync call in tool_only_agent.py."""
        log.info("Starting execution through thread boundary")
        result = await to_thread.run_sync(sync_wrapper)
        log.info("Completed execution through thread boundary")
        return result

    # Start the task
    task = asyncio.create_task(run_through_thread_boundary())

    # Give it time to start
    await asyncio.sleep(0.2)

    log.info("Cancelling task with thread boundary")
    task.cancel()

    # Verify cancellation propagates through the thread boundary
    with pytest.raises(asyncio.CancelledError):
        await task

    log.info("Thread boundary cancellation test completed")
