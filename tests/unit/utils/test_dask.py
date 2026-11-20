import asyncio
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd  # type: ignore[reportMissingStubs]
import pytest
import xarray as xr
from dask.distributed import Future
from xarray.core.dataarray import DataArray

from llm_agent.utils.dask import (
    SPATIAL_VALID_FRACTION_ATTR,
    DataSizeTooLargeError,
    compute_dask_array,
    validate_data_array_size_within_limit,
)


def build_test_data_array() -> xr.DataArray:
    """Returns a (10, 10, 10) shaped (time, lat, lon) data array."""
    # '2025-01-01', '2025-01-02', ..., '2025-01-10'
    time = pd.date_range("2025-01-01", periods=10, freq="D")  # type: ignore[reportUnknownMemberType]
    # -90, -70, -50, -30, -10,  10,  30,  50,  70,  90
    lat = np.linspace(-90, 90, 10)
    # -180, -140, -100,  -60,  -20,   20,   60,  100,  140, 180
    lon = np.linspace(-180, 180, 10)

    rng = np.random.default_rng()
    random_data = rng.random((len(time), len(lat), len(lon)))

    da = xr.DataArray(
        data=random_data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="dummy_data",
    )
    return da


def test_validate_size_without_attr_uses_full_nbytes():
    da = build_test_data_array()
    validate_data_array_size_within_limit(da, size_limit=da.nbytes)  # exactly at the limit is ok
    with pytest.raises(DataSizeTooLargeError):
        validate_data_array_size_within_limit(da, size_limit=da.nbytes - 1)


def test_validate_size_respects_spatial_valid_fraction():
    da = build_test_data_array()
    limit = da.nbytes // 2
    # Full bounding box exceeds the limit...
    with pytest.raises(DataSizeTooLargeError):
        validate_data_array_size_within_limit(da, size_limit=limit)
    # ...but only a quarter of the cells are valid, so the effective size fits.
    da.attrs[SPATIAL_VALID_FRACTION_ATTR] = 0.25
    validate_data_array_size_within_limit(da, size_limit=limit)


async def test_compute_dask_array_cancellation():
    """Test that cancelling compute_dask_array also cancels the Dask future."""
    da = build_test_data_array()
    # Make it a dask array by chunking it
    da = da.chunk({"time": 2, "lat": 2, "lon": 2})  # type: ignore[reportUnknownMemberType]

    # Mock the dask client and future
    mock_client = Mock()
    mock_client.asynchronous = True
    mock_future = Mock(spec=Future)

    # Create a coroutine that will be cancelled
    async def slow_computation() -> DataArray:
        # Sleep long enough that we know we can cancel it.
        await asyncio.sleep(10)
        pytest.fail("Got to end of slow computation")

    mock_future.result.return_value = slow_computation()
    mock_client.compute.return_value = mock_future

    with (
        patch("llm_agent.utils.dask.get_client", return_value=mock_client),
        patch("llm_agent.utils.dask._AsyncProgressLogger") as mock_progress_cls,
    ):
        mock_progress_cls.return_value.listen = AsyncMock()
        # Create a task that we'll cancel
        task = asyncio.create_task(compute_dask_array(da))

        # Give it a moment to start
        await asyncio.sleep(0.01)

        # Cancel the task
        task.cancel()

        # Verify that CancelledError is raised
        with pytest.raises(asyncio.CancelledError):
            await task

    # Verify that the Dask future's cancel method was called
    mock_future.cancel.assert_called_once()


async def test_compute_dask_array_sync_client_error():
    """Test that compute_dask_array raises RuntimeError for sync Dask client."""
    da = build_test_data_array()
    # Make it a dask array by chunking it
    da = da.chunk({"time": 2, "lat": 2, "lon": 2})  # type: ignore[reportUnknownMemberType]

    # Mock a synchronous dask client
    mock_client = Mock()
    mock_client.asynchronous = False

    with (
        patch("llm_agent.utils.dask.get_client", return_value=mock_client),
        pytest.raises(RuntimeError, match="asynchronous=True"),
    ):
        await compute_dask_array(da)
