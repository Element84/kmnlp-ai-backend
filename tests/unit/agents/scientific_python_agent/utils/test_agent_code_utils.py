from unittest.mock import Mock, patch

import numpy as np
import pandas as pd  # type: ignore[reportMissingStubs]
import pytest
import xarray as xr
from natural_language_geocoding.tests.canned_place_lookup import CannedPlaceLookup
from requests.exceptions import ConnectTimeout, ReadTimeout
from shapely.geometry import box

from llm_agent.agents.scientific_python_agent.utils.agent_code_utils import (
    filter_data_array,
    geocode_natural_language_description,
    geographic_coords_to_decimal_coords,
    mask_data_array_with_polygon,
    max_coords,
    min_coords,
    polygon_to_mask,
)
from llm_agent.utils.dask import SPATIAL_VALID_FRACTION_ATTR

place_lookup = CannedPlaceLookup()


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


def test_filter_data_array():
    da = build_test_data_array()

    da_filtered = filter_data_array(da, time=slice(None, "2025-01-05"))
    assert da_filtered.shape == (5, 10, 10)

    da_filtered = filter_data_array(da, lat=slice(-50, 50), lon=slice(-30, 30))
    assert da_filtered.shape == (10, 6, 2)


async def test_max_coords():
    da = build_test_data_array()
    da.data[2, 1, 2] = 100.0
    expected_output = {
        "time": "2025-01-03T00:00:00.000000000",
        "lat": -70.0,
        "lon": -100.0,
        "value": 100.0,
    }
    assert await max_coords(da) == expected_output


async def test_min_coords():
    da = build_test_data_array()
    da.data[3, 4, 1] = -100
    expected_output = {
        "time": "2025-01-04T00:00:00.000000000",
        "lat": -10.0,
        "lon": -140.0,
        "value": -100.0,
    }
    assert await min_coords(da) == expected_output


def test_geographic_coords_to_decimal_coords():
    assert geographic_coords_to_decimal_coords("20N") == 20.0
    assert geographic_coords_to_decimal_coords("20S") == -20.0
    assert geographic_coords_to_decimal_coords("50E") == 50.0
    assert geographic_coords_to_decimal_coords("50W") == -50.0


def test_polygon_to_mask():
    da = build_test_data_array()
    _, h, w = da.shape
    polygon = box(-60, 10, 60, 50)
    mask = polygon_to_mask(polygon, da)
    expected_mask = np.zeros((1, h, w), dtype=np.bool_)
    expected_mask[:, 5:8, 3:7] = True
    assert np.all(mask == expected_mask)


def test_polygon_to_mask_2d():
    da = build_test_data_array()
    _, h, w = da.shape
    polygon = box(-60, 10, 60, 50)
    mask = polygon_to_mask(polygon, da.isel(time=0))
    expected_mask = np.zeros((h, w), dtype=np.bool_)
    expected_mask[5:8, 3:7] = True
    assert np.all(mask == expected_mask)


def test_mask_data_array_with_polygon_2d():
    """Test mask_data_array_with_polygon on 2D (lat, lon) data array."""
    da_3d = build_test_data_array()
    # Create a 2D data array by selecting a single time slice
    da = da_3d.isel(time=0)
    _, h, w = da_3d.shape

    # Create a polygon that covers a specific region
    polygon = box(-60, 10, 60, 50)

    # Test with cropping
    result_cropped = mask_data_array_with_polygon(da, polygon)
    assert set(result_cropped.dims) == {"lat", "lon"}
    # Cropped result should be smaller or equal in dimensions
    assert result_cropped.shape[0] <= h
    assert result_cropped.shape[1] <= w
    # All non-NaN values should be within the cropped region
    assert result_cropped.sizes["lat"] <= h
    assert result_cropped.sizes["lon"] <= w


def test_mask_data_array_with_polygon_sets_valid_fraction():
    da = build_test_data_array().isel(time=0)
    polygon = box(-60, 10, 60, 50)  # covers only part of the grid
    result = mask_data_array_with_polygon(da, polygon)

    fraction = result.attrs[SPATIAL_VALID_FRACTION_ATTR]
    assert 0.0 < fraction < 1.0
    # Matches the fraction of lat/lon cells the polygon actually keeps.
    mask = polygon_to_mask(polygon, da)
    assert np.isclose(fraction, float(mask.mean()))


# this test takes a few seconds to complete because of the exponential backoff
async def test_geocode_natural_language_description_retry():
    with (
        patch(
            "llm_agent.agents.scientific_python_agent.utils.agent_code_utils.extract_geometry_from_text",
            # Simulate: ReadTimeout -> ConnectTimeout -> Success
            side_effect=[
                ReadTimeout("First attempt timeout"),
                ConnectTimeout("Second attempt timeout"),
                box(-10, -10, 10, 10),
            ],
        ) as mock_func,
        patch(
            "llm_agent.agents.scientific_python_agent.utils.agent_code_utils.BedrockClaudeLLM",
            return_value=Mock(),
        ),
    ):
        result = await geocode_natural_language_description(place_lookup, Mock(), "")
        assert result == box(-10, -10, 10, 10)
        assert mock_func.call_count == 3


# this test takes a few seconds to complete because of the exponential backoff
async def test_geocode_natural_language_description_max_retries():
    with (
        patch(
            "llm_agent.agents.scientific_python_agent.utils.agent_code_utils.extract_geometry_from_text",
            # Simulate: ReadTimeout on every call
            side_effect=ReadTimeout(),
        ) as mock_func,
        patch(
            "llm_agent.agents.scientific_python_agent.utils.agent_code_utils.BedrockClaudeLLM",
            return_value=Mock(),
        ),
    ):
        with pytest.raises(ReadTimeout):
            _ = await geocode_natural_language_description(place_lookup, Mock(), "")
        assert mock_func.call_count == 3
