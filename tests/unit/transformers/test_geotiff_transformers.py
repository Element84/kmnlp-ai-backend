import logging
import os
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import rioxarray
import xarray as xr

from llm_agent.transformers.geotiff_transformers import (
    InvalidShapeForGeoTIFFError,
    _preprocess_data_array,  # pyright: ignore[reportPrivateUsage]
    _save_data_array_as_geotiff,  # pyright: ignore[reportPrivateUsage]
    data_array_to_geotiff_ref,
)
from tests.utils.fixtures.context import MockChatContext

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@pytest.mark.asyncio
def test_save_data_array_as_geotiff(lat_lon_data_array: xr.DataArray):
    with NamedTemporaryFile(prefix="geotiff_", suffix=".tif") as tmpfile:
        _save_data_array_as_geotiff(lat_lon_data_array, tmpfile.name)
        assert os.path.exists(tmpfile.name)
        assert tmpfile.name.endswith(".tif")

        # Read back with rioxarray, reproject to match the original grid, then compare
        data_back = rioxarray.open_rasterio(tmpfile.name).squeeze("band", drop=True)  # pyright: ignore[reportUnknownVariableType, reportAttributeAccessIssue, reportUnknownMemberType]
        data_back_matched = data_back.rio.reproject_match(lat_lon_data_array)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        np.testing.assert_allclose(  # pyright: ignore[reportUnknownMemberType]
            data_back_matched.values,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportUnknownArgumentType]
            lat_lon_data_array.values.astype(np.float32),
            rtol=1e-3,
        )


@pytest.mark.asyncio
async def test_data_array_to_geotiff_ref(
    lat_lon_data_array: xr.DataArray, chat_context: MockChatContext
):
    # Create Stub Data Array
    data_array = lat_lon_data_array

    # Create GeoTIFF
    ref = await data_array_to_geotiff_ref(data_array, chat_context)

    # Assert Ref as expected
    assert ref.name == "some name"
    assert ref.units == "example unit"
    assert ref.min_value == 0.0
    assert ref.max_value == 99.0

    geotiff_path = chat_context.retrieve_data(ref)
    assert isinstance(geotiff_path, str)
    assert geotiff_path.endswith(".tif")


####################################
# Tests for _preprocess_data_array #
####################################


def test_preprocess_data_array_2d_array():
    """Test that 2D arrays are passed through without modification (except dtype)."""
    # Create a 2D data array
    numpy_array = np.arange(100).reshape(10, 10).astype(float)
    lon_coords = np.linspace(0, 10, 10)
    lat_coords = np.linspace(0, 10, 10)

    data_array: xr.DataArray = xr.DataArray(
        numpy_array,
        coords={"lat": lat_coords, "lon": lon_coords},
        dims=("lat", "lon"),
        name="test_2d",
    )

    result = _preprocess_data_array(data_array)

    # Should maintain 2D shape
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert "lat" in result.dims
    assert "lon" in result.dims


def test_preprocess_data_array_float64_to_float32():
    """Test that float64 arrays are converted to float32."""
    numpy_array = np.arange(100, dtype=np.float64).reshape(10, 10)
    lon_coords = np.linspace(0, 10, 10)
    lat_coords = np.linspace(0, 10, 10)

    data_array: xr.DataArray = xr.DataArray(
        numpy_array,
        coords={"lat": lat_coords, "lon": lon_coords},
        dims=("lat", "lon"),
        name="test_float64",
    )

    assert data_array.dtype == np.float64

    result = _preprocess_data_array(data_array)

    # Should be converted to float32
    assert result.dtype == np.float32
    # Values should be approximately the same
    np.testing.assert_allclose(result.values, numpy_array.astype(np.float32))


def test_preprocess_data_array_float32_unchanged():
    """Test that float32 arrays are not modified."""
    numpy_array = np.arange(100, dtype=np.float32).reshape(10, 10)
    lon_coords = np.linspace(0, 10, 10)
    lat_coords = np.linspace(0, 10, 10)

    data_array: xr.DataArray = xr.DataArray(
        numpy_array,
        coords={"lat": lat_coords, "lon": lon_coords},
        dims=("lat", "lon"),
        name="test_float32",
    )

    result = _preprocess_data_array(data_array)

    # Should remain float32
    assert result.dtype == np.float32


def test_preprocess_data_array_invalid_1d():
    """Test that 1D arrays raise ValueError."""
    numpy_array = np.arange(10)

    data_array: xr.DataArray = xr.DataArray(
        numpy_array,
        coords={"x": np.arange(10)},
        dims=("x",),
        name="test_1d",
    )

    with pytest.raises(InvalidShapeForGeoTIFFError, match="Expected a DataArray squeezable to 2D"):
        _preprocess_data_array(data_array)


def test_preprocess_data_array_valid_squeezable():
    rng = np.random.default_rng(42)
    numpy_array = rng.random((1, 1, 10, 10))

    data_array: xr.DataArray = xr.DataArray(
        numpy_array,
        coords={
            "x": np.arange(1),
            "y": np.arange(1),
            "lat": np.linspace(0, 10, 10),
            "lon": np.linspace(0, 10, 10),
        },
        dims=("x", "y", "lat", "lon"),
        name="test",
    )

    data_array_out = _preprocess_data_array(data_array)
    assert data_array_out.shape == (10, 10)
    assert set(data_array_out.dims) == {"lat", "lon"}
    assert data_array_out.dtype == np.float32


def test_preprocess_data_array_invalid_4d():
    """Test that 4D arrays raise ValueError."""
    rng = np.random.default_rng(42)
    numpy_array = rng.random((2, 5, 10, 10))

    data_array: xr.DataArray = xr.DataArray(
        numpy_array,
        dims=("a", "b", "c", "d"),
        name="test_4d",
    )

    with pytest.raises(InvalidShapeForGeoTIFFError, match="Expected a DataArray squeezable to 2D"):
        _preprocess_data_array(data_array)


def test_preprocess_data_array_preserves_attributes():
    """Test that data array attributes are preserved."""
    numpy_array = np.arange(100).reshape(10, 10).astype(float)
    lon_coords = np.linspace(0, 10, 10)
    lat_coords = np.linspace(0, 10, 10)

    data_array: xr.DataArray = xr.DataArray(
        numpy_array,
        coords={"lat": lat_coords, "lon": lon_coords},
        dims=("lat", "lon"),
        name="test_attrs",
        attrs={"units": "meters", "description": "test data"},
    )

    result = _preprocess_data_array(data_array)

    # Attributes should be preserved
    assert result.attrs["units"] == "meters"
    assert result.attrs["description"] == "test data"
