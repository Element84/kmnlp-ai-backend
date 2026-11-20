import asyncio

import numpy as np
import pytest
import xarray as xr
from dask.distributed import Client

from llm_agent.utils.dask import compute_dask_array


async def test_compute_dask_array_cancellation(dask_client: Client):  # noqa: ARG001
    """Feature test: verify that Dask computations can be cancelled mid-execution."""
    # Create a large chunked data array that will take time to compute
    # Use a computation-heavy operation to ensure cancellation has time to occur
    rng = np.random.default_rng(seed=42)
    data = rng.random((200, 200, 200))

    da = xr.DataArray(
        data=data,
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(200),
            "lat": np.linspace(-90, 90, 200),
            "lon": np.linspace(-180, 180, 200),
        },
    )

    # Chunk the array to make it a Dask array
    da = da.chunk({"time": 10, "lat": 10, "lon": 10})  # type: ignore[reportUnknownMemberType]
    # Create a slow computation by chaining multiple operations
    slow_computation = da * 2.0 + da.mean()

    # Start the computation in a task so we can cancel it
    task = asyncio.create_task(compute_dask_array(slow_computation))

    # Give it a moment to start computing
    await asyncio.sleep(0.1)

    # Cancel the task
    task.cancel()

    # Verify that cancellation happened
    try:
        await task
        # If we get here, cancellation failed
        pytest.fail("Expected CancelledError but computation completed")
    except asyncio.CancelledError:
        # This is what we expect. Indicates that cancellation worked.
        pass
