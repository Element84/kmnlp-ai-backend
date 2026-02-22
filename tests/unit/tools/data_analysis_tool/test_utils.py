import numpy as np
import xarray as xr

from llm_agent.tools.data_analysis_tool.utils import normalized_data_array


def test_leaves_normal_data_array_alone(lat_lon_data_array: xr.DataArray):
    initial_data_array = lat_lon_data_array
    modified_data_array = normalized_data_array(initial_data_array)
    assert initial_data_array.equals(modified_data_array)


def test_normalizes_data_array_with_strange_args():
    # Stub data array, 2D
    numpy_array = np.arange(100).reshape(10, 10).astype(float)  # min=0.0, max=99.0

    # Define spatial coordinates
    x_coords = np.linspace(0, 360, 10)  # longitude
    y_coords = np.linspace(-90, 90, 10)  # latitude
    y_coords = y_coords[::-1]  # Reverse to be in descending order, to test sorting

    # Create DataArray with spatial dimensions and coordinates
    initial_data_array: xr.DataArray = xr.DataArray(
        numpy_array,
        coords={"latitude": y_coords, "longitude": x_coords},
        dims=("latitude", "longitude"),
        name="some name",
        attrs={"units": "example unit"},
    )

    # Set CRS and spatial information using rioxarray
    initial_data_array.rio.write_crs("EPSG:4326", inplace=True)
    initial_data_array.rio.set_spatial_dims(x_dim="latitude", y_dim="longitude", inplace=True)
    initial_data_array.rio.write_coordinate_system(inplace=True)

    modified_data_array = normalized_data_array(initial_data_array)

    assert "lat" in modified_data_array.dims
    assert "lon" in modified_data_array.dims

    assert "latitude" not in modified_data_array.dims
    assert "longitude" not in modified_data_array.dims

    assert modified_data_array.lat.min().item() == -90
    assert modified_data_array.lat.max().item() == 90

    assert modified_data_array.lon.min().item() == -160
    assert modified_data_array.lon.max().item() == 160

    # Check coords sorted
    np.testing.assert_array_equal(
        np.sort(modified_data_array.lat.values), modified_data_array.lat.values
    )
    np.testing.assert_array_equal(
        np.sort(modified_data_array.lon.values), modified_data_array.lon.values
    )
